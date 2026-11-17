from __future__ import annotations

import hashlib
import itertools
import math
import string
from collections import OrderedDict
from functools import lru_cache
import time
from pathlib import Path

from .matching import softmax




class HFScorer:
    is_neural = True

    def __init__(self, model_path, device="cuda:0", dtype="auto", trust_remote_code=False,
                 max_context_tokens=2048, cpu_threads=4, attention_implementation="model"):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        directory = Path(model_path).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError("--model-path must point to an existing local model directory")
        if device == "cpu":
            torch.set_num_threads(cpu_threads)
        if dtype == "auto":
            dtype = "float32" if device == "cpu" else "float16"
        dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype not in dtypes:
            raise ValueError("Unsupported dtype")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
                                                       trust_remote_code=trust_remote_code)
        model_options = {}
        if attention_implementation != "model":
            config = AutoConfig.from_pretrained(str(directory), local_files_only=True,
                                                trust_remote_code=trust_remote_code)
            # InternLM2's custom implementation reads the public attribute;
            # Transformers' private _attn_implementation alone does not select it.
            config.attn_implementation = attention_implementation
            model_options.update(config=config, attn_implementation=attention_implementation)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=trust_remote_code,
            torch_dtype=dtypes[dtype], device_map="auto" if device == "auto" else {"": device},
            **model_options)
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = self.model.get_input_embeddings().weight.device
        model_limit = getattr(self.model.config, "max_position_embeddings", max_context_tokens)
        self.max_context_tokens = min(max_context_tokens, model_limit)
        config_file = directory / "config.json"
        self.identity = {"backend": "huggingface", "model_path": str(directory), "dtype": dtype,
                         "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
                         "frozen": True, "device": device}
        self.identity["attention_implementation"] = getattr(
            self.model.config, "attn_implementation", getattr(self.model.config, "_attn_implementation", None))
        self.identity["device_map"] = {str(k): str(v) for k, v in getattr(self.model, "hf_device_map", {}).items()}
        self.calls = self.prompt_tokens = self.answer_tokens = 0
        self._codes = []
        self.score_seconds = 0.0
        self.score_cache_hits = 0
        self._score_cache = OrderedDict()
        # The checked local InternLM2 forward is decoder -> linear head -> float.
        # A small final block avoids T x vocabulary logits, retaining a GEMM
        # rather than the different bfloat16 single-row GEMV kernel.
        self.last_logit_only = (self.model.__class__.__name__ == "InternLM2ForCausalLM"
                                and getattr(self.model.config, "pretraining_tp", 1) == 1)
        self.identity["last_logit_only"] = self.last_logit_only
        self.identity["logit_tail_block"] = 32 if self.last_logit_only else None

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=True)

    @lru_cache(maxsize=256)
    def token_count(self, text):
        return len(self._encode(text))

    def codes(self, n):
        if len(self._codes) >= n:
            return self._codes[:n]
        prefix = self._encode("Answer:\n")
        pool = itertools.chain((str(i) for i in range(1000)), string.ascii_uppercase,
                               string.ascii_lowercase,
                               (a + b for a in string.ascii_uppercase for b in string.ascii_uppercase))
        codes, seen_tokens = [], set()
        for code in pool:
            encoded = self._encode("Answer:\n" + code)
            if encoded[:len(prefix)] != prefix:
                continue
            suffix = encoded[len(prefix):]
            if len(suffix) == 1 and suffix[0] not in seen_tokens:
                codes.append(code)
                seen_tokens.add(suffix[0])
            if len(codes) >= n:
                self._codes = codes
                return codes
        # Full teacher-forced sequence scoring handles tokenizers without enough atomic labels.
        return ["C{:04d}".format(i) for i in range(n)]

    def _answer_tokens(self, prompt, answer, prefix):
        full = self._encode(prompt + answer)
        if full[:len(prefix)] != prefix:
            raise ValueError("Prompt/answer token boundary merged; use the canonical Answer:\\n suffix")
        suffix = full[len(prefix):]
        if not suffix:
            raise ValueError("Empty answer token sequence")
        return full, suffix

    def score_cached(self, prompt, answers):
        """Bounded cache for exactly repeated frozen-model prompts (e.g. DCRM)."""
        key = (prompt, tuple(answers))
        if key in self._score_cache:
            self.score_cache_hits += 1
            self._score_cache.move_to_end(key)
            return list(self._score_cache[key])
        result = self.score(prompt, answers)
        self._score_cache[key] = tuple(result)
        if len(self._score_cache) > 2048:
            self._score_cache.popitem(last=False)
        return result

    def score(self, prompt, answers):
        started = time.perf_counter()
        result = self._score(prompt, answers)
        self.score_seconds = getattr(self, "score_seconds", 0.0) + time.perf_counter() - started
        return result

    def _score(self, prompt, answers):
        if not answers or len(set(answers)) != len(answers):
            raise ValueError("A nonempty unique answer vocabulary is required")
        prefix = self._encode(prompt)
        prepared = [self._answer_tokens(prompt, answer, prefix) for answer in answers]
        atomic = all(len(suffix) == 1 for _, suffix in prepared)
        if not atomic:
            prepared = [self._answer_tokens(prompt, answer + "\n", prefix) for answer in answers]
        if max(len(full) for full, _ in prepared) > self.max_context_tokens:
            raise ValueError("Prompt and answer exceed --max-context-tokens")
        torch = self.torch
        self.calls += 1
        self.prompt_tokens += len(prefix)
        self.answer_tokens += sum(len(suffix) for _, suffix in prepared)
        with torch.inference_mode():
            if atomic:
                input_ids = torch.tensor([prefix], device=self.device)
                if getattr(self, "last_logit_only", False):
                    states = self.model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                              use_cache=False, return_dict=True)[0]
                    # Invoke the module normally so Accelerate's device hooks still run.
                    last_logits = self.model.get_output_embeddings()(states[:, -32:, :])[0, -1]
                    del states
                else:
                    output = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
                    last_logits = output.logits[0, -1]
                # Closed one-token answer vocabulary: one forward pass scores every candidate exactly.
                indices = torch.tensor([suffix[0] for _, suffix in prepared], device=last_logits.device)
                logits = last_logits.index_select(0, indices).float()
                result = torch.softmax(logits, dim=-1).cpu().tolist()
                return result
            log_scores = []
            for full, suffix in prepared:
                input_ids = torch.tensor([full], device=self.device)
                output = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
                logits = output.logits[0, len(prefix) - 1:len(full) - 1].float()
                targets = torch.tensor(suffix, device=logits.device)
                score = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).sum().item()
                log_scores.append(score)
                del output
            return softmax(log_scores)

    def generate(self, prompt, max_new_tokens):
        torch = self.torch
        prefix = self._encode(prompt)
        if len(prefix) + max_new_tokens > self.max_context_tokens:
            raise ValueError("Schema prompt exceeds context budget; reduce --schema-candidates or generation length")
        self.calls += 1
        self.prompt_tokens += len(prefix)
        input_ids = torch.tensor([prefix], device=self.device)
        with torch.inference_mode():
            output = self.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                         max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0, len(prefix):]
        self.answer_tokens += len(generated)
        return self.tokenizer.decode(generated, skip_special_tokens=True)

"""Same frozen BF16 decoder, selected candidate output dot products in FP32.

This is a numerical implementation control, not a transaction mechanism.
Only one decoder invocation is made per fresh nonempty score request.
"""
import time
from types import MethodType
from t2e.data import digest

POLICY = 'bf16_decoder_fp32_selected_output_dot_softmax_tf32_disabled_v1'


def selected_logits(hidden, weight, token_ids, bias=None):
    import torch
    if hidden.ndim != 1 or weight.ndim != 2 or hidden.shape[0] != weight.shape[1]:
        raise ValueError('Expected one final hidden state and linear vocabulary weights')
    indices = torch.tensor(token_ids, dtype=torch.long, device=weight.device)
    rows = weight.index_select(0, indices).float()
    b = None if bias is None else bias.index_select(0, indices).float()
    return torch.nn.functional.linear(hidden.to(weight.device).float(), rows, b)


def atomic_inputs(scorer, prompt, answers):
    if not answers or len(set(answers)) != len(answers):
        raise ValueError('Nonempty unique answers required')
    prefix = scorer._encode(prompt)
    prepared = [scorer._answer_tokens(prompt, a, prefix) for a in answers]
    if any(len(suffix) != 1 for _, suffix in prepared):
        raise ValueError('Audited atomic candidate labels required; no mixed-precision fallback')
    if max(len(full) for full, _ in prepared) > scorer.max_context_tokens:
        raise ValueError('Prompt and answer exceed context limit')
    ids = [suffix[0] for _, suffix in prepared]
    if len(set(ids)) != len(ids):
        raise ValueError('Distinct candidate labels must have distinct token IDs')
    return prefix, ids


def install(scorer, writer):
    torch = scorer.torch
    if not scorer.last_logit_only or not isinstance(scorer.model.get_output_embeddings(), torch.nn.Linear):
        raise ValueError('Audited InternLM2 decoder and linear head required')
    if any(p.device.type != 'cuda' for p in scorer.model.parameters()):
        raise RuntimeError('All model parameters must remain CUDA resident')
    torch.backends.cuda.matmul.allow_tf32 = False
    scorer.identity = dict(scorer.identity, candidate_projection=POLICY,
                          output_compute_dtype='float32', softmax_dtype='float32',
                          decoder_dtype='bfloat16', allow_tf32=False,
                          logit_tail_block=None)
    scorer.fp32_probe_seconds = 0.0

    def score(self, prompt, answers):
        prefix, token_ids = atomic_inputs(self, prompt, answers)
        self.calls += 1; self.prompt_tokens += len(prefix); self.answer_tokens += len(token_ids)
        with torch.inference_mode():
            input_ids = torch.tensor([prefix], device=self.device)
            states = self.model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                      use_cache=False, return_dict=True)[0]
            head = self.model.get_output_embeddings()
            if states.device.type != 'cuda' or head.weight.device.type != 'cuda':
                raise RuntimeError('CUDA decoder states and output weights required')
            logits = selected_logits(states[0, -1], head.weight, token_ids, head.bias)
            probabilities = torch.softmax(logits, dim=-1)
            record = dict(input_hash=digest([prompt, answers]), answer_tokens=token_ids,
                          logits=logits.cpu().tolist(), probabilities=probabilities.cpu().tolist(),
                          hidden_dtype=str(states.dtype), output_dtype=str(logits.dtype),
                          hidden_device=str(states.device), output_device=str(logits.device),
                          decoder_calls=1, policy=POLICY)
            # Engineering verification on first three fresh inputs, independent of labels.
            # Same hidden state: no second decoder or scoring request.
            if self.calls <= 3:
                tick = time.perf_counter()
                ix = torch.tensor(token_ids, device=head.weight.device)
                weights64 = head.weight.index_select(0, ix).double()
                ref = (weights64 * states[0, -1].to(head.weight.device).double()).sum(dim=1)
                if head.bias is not None: ref += head.bias.index_select(0, ix).double()
                tolerance = 2e-4 + 1e-6 * ref.abs()
                error = (logits.double() - ref).abs()
                if not bool(torch.all(error <= tolerance)):
                    raise AssertionError('FP32 projection differs from independent FP64 product sum')
                old = head(states[:, -32:, :])[0, -1].index_select(0, ix).float()
                record['numerical_probe'] = dict(fp64_reference=ref.cpu().tolist(),
                    max_abs_error=error.max().item(), atol=2e-4, rtol=1e-6,
                    bf16_head_logits=old.cpu().tolist(),
                    bf16_vs_fp32_max_abs=(old-logits).abs().max().item(),
                    seconds=time.perf_counter()-tick, passed=True)
                self.fp32_probe_seconds += record['numerical_probe']['seconds']
            writer(record)
            return record['probabilities']

    scorer._score = MethodType(score, scorer)


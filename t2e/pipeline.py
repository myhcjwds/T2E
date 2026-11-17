"""Dataset-to-prediction entry point with isolated, resumable run artifacts."""
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

from .data import batches, queries_and_labels, digest, write_json
from .metrics import Metrics
from .statistics import CanonicalDataset, fit_rule_precision


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def emit(stream, value):
    stream.write(json.dumps(value, separators=(',', ':')) + '\n')
    stream.flush()


def run(args, project, config=None):
    import numpy as np
    from transformers import AutoTokenizer
    if config is None:
        from .configuration import load_config
        config = load_config(project, args)
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else (project / Path(config['model_path']).expanduser()).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError('Model directory not found: ' + str(model_path))
    data = CanonicalDataset(project / 'data' / args.dataset)
    data.relations = {k: v.replace(' ', '_') for k, v in data.relations.items()}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(dataset=data.audit(), config=config, model_path=str(model_path),
        source={str(p.relative_to(project)): sha256(p) for p in sorted((project/'t2e').glob('*.py'))},
        weights={p.name: sha256(p) for p in sorted(model_path.iterdir())
                 if p.is_file() and p.suffix in ('.bin','.safetensors','.py','.json','.model')})
    identity['source']['run.py'] = sha256(project/'run.py')
    identity_path = output/'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('Output belongs to different inputs or code; choose a new --output directory')
    else:
        if any(output.iterdir()):
            raise ValueError('Output must be empty for a new run')
        write_json(identity_path, identity)
    complete = output/'metrics.json'
    if complete.exists():
        print(complete.read_text(), flush=True)
        return
    import fcntl
    control = (output/'run.lock').open('a')
    fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.perf_counter()
    total = sum(len(queries_and_labels(b)[0]) for _, b in batches(data.splits['test']))

    def progress(stage, done=0, **extra):
        value = dict(stage=stage, queries=done, total_queries=total, pid=os.getpid(),
                     updated_unix=time.time(), elapsed_seconds=time.perf_counter()-started, **extra)
        write_json(output/'progress.json', value)
        print(json.dumps(value), flush=True)

    try:
        progress('historical_statistics')
        counts_file = output/'rules.npz'
        if not counts_file.exists():
            np.savez_compressed(str(counts_file), **fit_rule_precision(data))
        precision = np.load(str(counts_file), allow_pickle=False)['precision']
        plan_file = output/'plan.jsonl'
        prepared = output/'prepared.json'
        if not prepared.exists():
            progress('evidence_preparation')
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
            from .retrieval import plans
            temporary = output/'plan.partial.jsonl'
            with temporary.open('w', encoding='utf-8') as stream:
                for n, plan in enumerate(plans(data, config, precision, tokenizer), 1):
                    emit(stream, plan)
                    if n % 100 == 0:
                        progress('evidence_preparation', n)
            assert n == total
            temporary.replace(plan_file)
            write_json(prepared, dict(queries=n, sha256=sha256(plan_file)))
        assert json.loads(prepared.read_text())['sha256'] == sha256(plan_file)
        if args.prepare_only:
            progress('inputs_ready', total)
            return

        progress('waiting_for_gpu')
        lock_path = Path(os.environ.get('T2E_GPU_LOCK', '/tmp/t2e-gpu.lock'))
        gpu_lock = lock_path.open('a')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        progress('loading_model')
        from .scoring import HFScorer
        from .numeric_tokens import install as install_tokens
        from .fp32_candidate_readout import install as install_precision
        from .renewal_memory import RenewalMemory
        scorer = HFScorer(str(model_path), device='auto', dtype='bfloat16',
                          trust_remote_code=True, max_context_tokens=config['max_context_tokens'])
        install_tokens(scorer)
        readouts = (output/'readouts.jsonl').open('a', encoding='utf-8')
        install_precision(scorer, lambda value: emit(readouts, value))
        write_json(output/'model.json', scorer.identity)
        clock = RenewalMemory(data.prefix('test'))
        metrics = Metrics()
        database = sqlite3.connect(str(output/'scores.sqlite'))
        database.execute('CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        plan_iter = iter(rows(plan_file))
        fresh = cached = empty = 0
        # Regenerate rows from exact within-run score cache on resume.
        with (output/'predictions.jsonl').open('w', encoding='utf-8') as stream:
            n = 0
            for _, batch in batches(data.splits['test']):
                queries, labels = queries_and_labels(batch)
                for q in queries:
                    p = next(plan_iter)
                    assert p['query_id'] == q.qid and p['targets'] == sorted(labels[q])
                    cs = p['candidate_ids']
                    prompt = p['prompts']['ledger']['text']
                    answers = list(map(str, range(len(cs))))
                    key = digest([scorer.identity, identity['weights'], prompt, answers])
                    if cs:
                        saved = database.execute('SELECT value FROM scores WHERE key=?', (key,)).fetchone()
                        if saved is None:
                            neural = scorer.score(prompt, answers)
                            database.execute('INSERT INTO scores VALUES (?,?)', (key, json.dumps(neural)))
                            database.commit()
                            fresh += 1
                        else:
                            neural = json.loads(saved[0]); cached += 1
                    else:
                        neural = []; empty += 1
                    temporal = [max(0., v) for v in clock.values(q, cs)[0]]
                    scores = [math.log(v+p['floor']) + p['terms']['recurrence'][i]
                              + p['terms']['symbolic'][i] + temporal[i] for i, v in enumerate(neural)]
                    ranking, ranks = metrics.add(cs, scores, labels[q])
                    emit(stream, dict(query_id=q.qid, candidates=cs, neural=neural, clock=temporal,
                         scores=scores, ranking=ranking, ranks=ranks, plan_hash=digest(p)))
                    n += 1
                    if n % 20 == 0:
                        progress('prediction', n, fresh=fresh, cache=cached, empty=empty,
                                 hits1=100*metrics.hits[1]/metrics.count)
                clock.observe(batch)
        assert n == total and next(plan_iter, None) is None
        assert metrics.count == 2*len(data.splits['test'])
        result = dict(dataset=args.dataset, queries=n, targets=metrics.count,
            hits=metrics.hits, hits_percent={str(k):100*v/metrics.count for k,v in metrics.hits.items()},
            covered=metrics.covered, fresh=fresh, cache=cached, empty=empty,
            elapsed_seconds=time.perf_counter()-started, predictions_sha256=sha256(output/'predictions.jsonl'))
        write_json(complete, result)
        readouts.close(); database.close(); gpu_lock.close()
        progress('completed', n, result=result)
    except BaseException as exc:
        progress('failed', error=repr(exc))
        raise
    finally:
        control.close()

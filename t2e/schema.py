"""Map physical interaction identities to each dataset's native relation IDs."""
from collections import Counter
import importlib


def configure(relations, config):
    names = {name: key for key, name in relations.items()}
    symmetric_names = ('Consult', 'Engage_in_negotiation', 'Discuss_by_telephone',
                       "Meet_at_a_'third'_location")
    symmetric = {names[n] for n in symmetric_names}
    visit, host = names['Make_a_visit'], names['Host_a_visit']

    def physical(e):
        if e.r == host:
            return (e.t, 'visit', e.s, e.o)
        if e.r == visit:
            return (e.t, 'visit', e.o, e.s)
        if e.r in symmetric:
            return (e.t, e.r, min(e.s, e.o), max(e.s, e.o))
        return (e.t, e.r, e.s, e.o)

    for name in ('ownership_history', 'context_history', 'event_ledger',
                 'provenance_memory', 'projected_memory', 'parallel_memory',
                 'witness_snapshot_memory', 'transaction_day_memory', 'stable_read_memory'):
        module = importlib.import_module('t2e.' + name)
        if hasattr(module, 'interaction_key'):
            module.interaction_key = physical
    from . import context_history, witness_snapshot_memory, stable_read_memory, projected_memory
    context_history.MIRROR = {r: r for r in symmetric}
    context_history.MIRROR.update({visit: host, host: visit})
    witness_snapshot_memory.role = lambda e, endpoint: (
        0 if e.r in symmetric else 1 if physical(e)[2] == endpoint else -1)
    witness_snapshot_memory.WitnessSnapshotHistory.physical_window = 24 * config['physical_witness_days']
    stable_read_memory.TAU = config['tau']
    projected_memory.PACKET_CAP = config['packet_cap']
    return physical


def ordered_candidates(q, store, pool):
    store.assert_past(q.t)
    events = store.incident.get(q.known, ())
    witnessed = {e.o if e.s == q.known else e.s for e in events}
    assert set(pool) <= witnessed
    counts = Counter(e.o if q.direction == 'tail' else e.s for e in store.recent(q, 200))
    candidates = sorted(pool, key=lambda c: (-counts[c], c))
    proofs = {}
    for e in reversed(events):
        other = e.o if e.s == q.known else e.s
        if other in pool and other not in proofs:
            proofs[other] = e.eid
        if len(proofs) == len(pool):
            break
    return candidates, proofs

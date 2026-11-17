"""Read short observations that share an observed accepted branch, with fallback.

A positive alternative context is a retrieval distinction, not a false fact,
completed transaction, or real-world identity judgment. Empty contexts stay.
"""
from collections import defaultdict
from .event_ledger import render


def branch_day_contexts(index, memory, q, short, trace):
    supported = {tuple(sorted((q.known, p['candidate']))) for p in trace['packets']}
    result = {}
    for pair in {e.pair for e in short} & supported:
        memory.versions(index, pair)
        for day in {e.t for e in short if e.pair == pair}:
            result[pair, day] = tuple(sorted(memory.lineage_contexts[pair].get(day, {})))
    return result


def scope_short(q, short, long, trace, day_contexts, events, physical):
    assert all(e.t < q.t for e in short + long)
    accepted = defaultdict(set); entries = defaultdict(list)
    protected = set(trace.get('protected', []))
    for p in trace['packets']:
        pair = tuple(sorted((q.known, p['candidate'])))
        accepted[pair].add(tuple(p['context'])); entries[pair].append(events[p['seed']].t)
        protected.update(p['facts']); protected.update(p['roots']); protected.add(p['seed'])
    for name in ('first_hop_witness_reads', 'ordinary_witness_control_reads'):
        for p in trace.get(name, []): protected.update(p['facts'])
    protected.update(e.eid for e in short if e.r == q.r)
    protected_keys = {physical(events[e]) for e in protected}
    groups = defaultdict(list); latest = {}
    for e in short:
        groups[e.pair, e.t].append(e); latest[e.pair] = max(latest.get(e.pair, e.t), e.t)
    removed = set(); proof = []
    for (pair, day), facts in sorted(groups.items()):
        actual = set(day_contexts.get((pair, day), ()))
        if (pair not in accepted or not actual or actual & accepted[pair]
                or day >= min(entries[pair]) or day == latest[pair]
                or any(physical(e) in protected_keys for e in facts)):
            continue
        removed.update(facts)
        proof.append(dict(pair=list(pair), day=day, facts=[e.eid for e in facts],
                          observed_contexts=[list(c) for c in sorted(actual)],
                          accepted_contexts=[list(c) for c in sorted(accepted[pair])],
                          earliest_entry=min(entries[pair])))
    if not removed: return short, []
    result = [e for e in short if e not in removed]
    assert {e.pair for e in result} == {e.pair for e in short}
    assert all(e in result for e in short if physical(e) in protected_keys)
    return result, proof


def scoped_plan(q, plan, short, long, contexts, events, physical, token_count):
    selected, proof = scope_short(q, short, long, plan['trace'], contexts, events, physical)
    if not proof: return plan
    text = render(q, selected, long, plan['candidate_ids'], plan['trace'], False)
    tokens = token_count(text); assert tokens + 16 <= 8192
    return dict(plan, short=[e.eid for e in selected],
                short_branch_scope=dict(original=plan['short'], excluded_groups=proof,
                    original_readout=plan.get('readout'),
                    semantics='positive_disjoint_observed_lineages_before_all_accepted_entries'),
                readout=dict(plan.get('readout',dict(policy='original_chronological_long_evidence',
                    order=plan['long']['ledger'])),short_suffix_identical=False,
                    short_projection='whole_positive_disjoint_pair_day_groups'),
                prompts=dict(ledger=dict(text=text, tokens=tokens)))

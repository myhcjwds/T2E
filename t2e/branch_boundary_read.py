"""Expose the first observed primary-pair day without the original witness epoch.

The boundary is an observation, not a claim of termination or real identity.
Only real first-hop facts are rendered; no absence label enters the model.
"""
from .post_entry_pair_read import fill_pair_days
from .event_ledger import render


def boundary_day(q, pack, contexts, events, pairs, physical):
    seed = events[pack['seed']]
    epoch = tuple(pack['context'])
    assert seed.t < q.t and q.known in seed.pair and epoch in contexts[seed.t]
    prefix = []
    for day in sorted(t for t in contexts if seed.t < t < q.t):
        actual = contexts[day]
        if epoch in actual:
            prefix.append(dict(day=day, witnesses=[e.eid for e in actual[epoch]]))
            continue
        unique = {}
        for e in pairs[seed.pair]:
            if e.t == day:
                unique.setdefault(physical(e), e)
        assert unique
        return dict(candidate=pack['candidate'], seed=pack['seed'], roots=pack['roots'],
            context=pack['context'], read_day=day, observations=prefix,
            boundary_contexts=[dict(context=list(c), witnesses=[e.eid for e in actual[c]]) for c in sorted(actual)],
            facts=[e.eid for e in sorted(unique.values())])
    return None


def boundary_plan(q, plan, selected, accepted, events, token_count):
    if not accepted:
        return plan
    text = render(q, [events[e] for e in plan['short']], selected,
                  plan['candidate_ids'], plan['trace'], False)
    tokens = token_count(text)
    assert tokens + 16 <= 8192
    order = [e.eid for e in selected]
    return dict(plan, long=dict(plan['long'], ledger=order),
        branch_boundary_read=dict(source_long=plan['long']['ledger'], reads=accepted,
            limit=len(plan.get('long_branch_scope', {}).get('original', plan['long']['ledger'])),
            semantics='first_actual_pair_day_without_original_entry_witness_epoch'),
        readout=dict(plan.get('readout', {}), order=order,
            long_projection='first_observed_branch_boundary_days', short_suffix_identical_to_source=True),
        prompts=dict(ledger=dict(text=text, tokens=tokens)))

"""Read later observed primary-pair days while the entry branch stays present.

An observed missing branch stops this read path. Neither continuation nor a
stop asserts real-world identity, fulfillment, or termination.
"""
from collections import defaultdict
from .event_ledger import render


def post_entry_day(q, pack, contexts, events, pairs, physical):
    seed=events[pack['seed']];pair=seed.pair;context=tuple(pack['context'])
    assert seed.t<q.t and q.known in pair
    assert context in contexts[seed.t]
    steps=[]
    for day in sorted(t for t in contexts if seed.t<t<q.t):
        actual=contexts[day]
        if context not in actual:break
        steps.append(dict(day=day,witnesses=[e.eid for e in actual[context]]))
    if not steps:return None
    latest=steps[-1]['day'];unique={}
    for e in pairs[pair]:
        if e.t==latest:unique.setdefault(physical(e),e)
    return dict(candidate=pack['candidate'],seed=pack['seed'],roots=pack['roots'],
        context=pack['context'],read_day=latest,observations=steps,
        facts=[e.eid for e in sorted(unique.values())])


def fill_pair_days(q, plan, options, events, physical):
    source=[events[e] for e in plan['long']['ledger']];short=[events[e] for e in plan['short']]
    limit=len(plan.get('long_branch_scope',{}).get('original',plan['long']['ledger']))
    assert len(source)<=limit<=100
    forbidden={physical(events[e]) for name in ('short_branch_scope','long_branch_scope')
        for group in plan.get(name,{}).get('excluded_groups',[]) for e in group['facts']}
    used={physical(e) for e in source+short};by=defaultdict(list)
    for option in options:
        assert option['candidate'] in plan['candidate_ids'];by[option['candidate']].append(option)
    for c in by:by[c].sort(key=lambda x:(-x['read_day'],tuple(x['context']),x['seed']))
    seen=set();accepted=[];chosen=list(source)
    for depth in range(max((len(v) for v in by.values()),default=0)):
        for c in plan['candidate_ids']:
            if depth>=len(by[c]):continue
            option=by[c][depth];facts=[events[e] for e in option['facts']]
            assert all(events[option['seed']].t<e.t<q.t and e.pair==tuple(sorted((q.known,c))) for e in facts)
            keys=frozenset(physical(e) for e in facts)
            if keys in seen:continue
            seen.add(keys)
            if keys & forbidden:continue
            missing={physical(e):e for e in facts if physical(e) not in used}
            if not missing or len(chosen)+len(missing)>limit:continue
            chosen.extend(missing.values());used.update(missing)
            accepted.append(dict(option,added=[e.eid for e in sorted(missing.values())]))
    return sorted(chosen),accepted


def pair_read_plan(q,plan,selected,accepted,events,token_count):
    if not accepted:return plan
    text=render(q,[events[e] for e in plan['short']],selected,plan['candidate_ids'],plan['trace'],False)
    tokens=token_count(text);assert tokens+16<=8192
    order=[e.eid for e in selected]
    return dict(plan,long=dict(plan['long'],ledger=order),
        post_entry_pair_read=dict(source_long=plan['long']['ledger'],reads=accepted,
            limit=len(plan.get('long_branch_scope',{}).get('original',plan['long']['ledger'])),
            semantics='latest_actual_pair_day_before_first_observed_entry_branch_absence'),
        readout=dict(plan.get('readout',{}),order=order,long_projection='post_entry_primary_pair_days',
            short_suffix_identical_to_source=True),prompts=dict(ledger=dict(text=text,tokens=tokens)))

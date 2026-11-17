"""Query-facing observed-event candidate admission with complete physical pair days.

The original candidate numbering and all displayed evidence are retained. This
is a candidate-and-evidence intervention, not a fixed-candidate ablation.
"""
from .event_ledger import render
from .context_history import facing


def query_supported_options(q, plan, index, physical):
    index.store.assert_past(q.t)
    neighbors={e.o if e.s==q.known else e.s for e in index.store.incident.get(q.known,())}
    options=[]
    for c in sorted(neighbors-set(plan['candidate_ids'])):
        history=index.store.pairs[tuple(sorted((q.known,c)))]
        anchors=[]
        for e in history:
            v=facing(e,q)
            endpoint=v.s if q.direction=='tail' else v.o
            if v.r==q.r and endpoint==q.known:anchors.append(e)
        if not anchors:continue
        latest=max(e.t for e in anchors);assert latest<q.t
        day=sorted(e for e in history if e.t==latest);unique={}
        for e in day:unique.setdefault(physical(e),e)
        options.append(dict(candidate=c,entry_day=latest,seed=max(e for e in anchors if e.t==latest).eid,query_anchors=[e.eid for e in sorted(anchors) if e.t==latest],
            facts=[e.eid for e in sorted(unique.values())]))
    return options


def admit_candidates(plan, options):
    limit=len(plan.get('long_branch_scope',{}).get('original',plan['long']['ledger']))
    assert len(plan['long']['ledger'])<=limit<=100
    remaining=limit-len(plan['long']['ledger']);accepted=[];seen=set(plan['candidate_ids'])
    for option in sorted(options,key=lambda z:(-z['entry_day'],z['candidate'])):
        assert option['candidate'] not in seen and option['facts']
        seen.add(option['candidate'])
        if len(option['facts'])<=remaining:
            accepted.append(option);remaining-=len(option['facts'])
    return accepted,limit


def query_supported_plan(q, plan, options, accepted, limit, events, physical, calibrate, token_count):
    if not accepted:return plan
    cs=plan['candidate_ids']+[z['candidate'] for z in accepted]
    assert len(cs)==len(set(cs))
    long=[events[e] for e in plan['long']['ledger']]
    occupied={physical(events[e]) for e in plan['short']+plan['long']['ledger']}
    forbidden={physical(events[e]) for field in ('short_branch_scope','long_branch_scope')
        for group in plan.get(field,{}).get('excluded_groups',[]) for e in group['facts']}
    for z in accepted:
        assert all(events[e].pair==tuple(sorted((q.known,z['candidate']))) and events[e].t==z['entry_day']<q.t for e in z['facts'])
        assert not occupied&{physical(events[e]) for e in z['facts']}
        assert not forbidden&{physical(events[e]) for e in z['facts']}
        long.extend(events[e] for e in z['facts']);occupied.update(physical(events[e]) for e in z['facts'])
    long.sort();assert len(long)<=limit<=100
    text=render(q,[events[e] for e in plan['short']],long,cs,plan['trace'],False)
    oldlines=plan['prompts']['ledger']['text'].splitlines(keepends=True)
    assert text.splitlines(keepends=True)[len(long):]==oldlines[len(plan['long']['ledger']):]
    tokens=token_count(text);assert tokens+16<=8192
    terms,floor=calibrate(cs)
    witnesses=dict(plan['candidate_witnesses']);witnesses.update({str(z['candidate']):z['seed'] for z in accepted})
    order=[e.eid for e in long]
    return dict(plan,candidate_ids=cs,candidate_witnesses=witnesses,terms=terms,floor=floor,
        long=dict(plan['long'],ledger=order),
        query_supported_candidate_read=dict(source_candidates=plan['candidate_ids'],source_long=plan['long']['ledger'],
            limit=limit,options=options,accepted=accepted,semantics='latest_observed_query_facing_pair_day_candidate_admission'),
        readout=dict(plan.get('readout',{}),order=order,long_projection='query_facing_observed_candidate_whole_day',short_suffix_identical_to_source=True),
        prompts=dict(ledger=dict(text=text,tokens=tokens)))

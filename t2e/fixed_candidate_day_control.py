"""Fixed v38 candidate pool, ordinary latest pair-day read-set control."""
from .event_ledger import render


def latest_options(q, plan, index, physical):
    index.store.assert_past(q.t)
    meta=plan.get('query_supported_candidate_read',{})
    options=[]
    for source in meta.get('accepted',[]):
        c=source['candidate'];history=index.store.pairs[tuple(sorted((q.known,c)))]
        latest=max(e.t for e in history);unique={}
        for e in sorted(history):
            if e.t==latest:unique.setdefault(physical(e),e)
        options.append(dict(candidate=c,entry_day=latest,seed=max(e for e in history if e.t==latest).eid,
                            facts=[e.eid for e in sorted(unique.values())]))
    return options


def decide(plan, options, events, physical):
    meta=plan.get('query_supported_candidate_read')
    if not meta:return 'no_admissions',list(plan['long']['ledger'])
    assert [z['candidate'] for z in options]==[z['candidate'] for z in meta['accepted']]
    base=list(meta['source_long']);facts=[e for z in options for e in z['facts']]
    assert len(facts)==len(set(facts)) and not set(base)&set(facts)
    forbidden={physical(events[e]) for field in ('short_branch_scope','long_branch_scope')
        for group in plan.get(field,{}).get('excluded_groups',[]) for e in group['facts']}
    if forbidden&{physical(events[e]) for e in facts}:return 'protected_fallback',list(plan['long']['ledger'])
    if len(base)+len(facts)>meta['limit']:return 'capacity_fallback',list(plan['long']['ledger'])
    wanted=sorted(base+facts,key=lambda e:events[e])
    return ('unchanged' if wanted==plan['long']['ledger'] else 'changed'),wanted


def controlled_plan(q, plan, options, events, physical, token_count):
    reason,wanted=decide(plan,options,events,physical)
    if reason!='changed':return plan,reason
    meta=plan['query_supported_candidate_read']
    assert len(wanted)<=meta['limit']<=100
    assert all(events[e].t<q.t and q.known in events[e].pair for e in wanted)
    text=render(q,[events[e] for e in plan['short']],[events[e] for e in wanted],plan['candidate_ids'],plan['trace'],False)
    oldlines=plan['prompts']['ledger']['text'].splitlines(keepends=True);newlines=text.splitlines(keepends=True)
    assert newlines[len(wanted):]==oldlines[len(plan['long']['ledger']):]
    oldtext=dict(zip(plan['long']['ledger'],oldlines));newtext=dict(zip(wanted,newlines))
    assert all(oldtext[e]==newtext[e] for e in meta['source_long'])
    tokens=token_count(text);assert tokens+16<=8192
    return dict(plan,long=dict(plan['long'],ledger=wanted),
        fixed_candidate_day_control=dict(source_long=plan['long']['ledger'],base_long=meta['source_long'],
            source_readout=plan.get('readout',{}),source_reads=meta['accepted'],ordinary_reads=options,limit=meta['limit'],
            semantics='fixed_candidate_pool_all_or_nothing_latest_pair_day_read_set'),
        readout=dict(plan.get('readout',{}),order=wanted,long_projection='fixed_query_supported_candidates_latest_pair_days',short_suffix_identical_to_source=True),
        prompts=dict(ledger=dict(text=text,tokens=tokens))),reason

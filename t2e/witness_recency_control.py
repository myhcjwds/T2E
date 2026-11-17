"""Same-cohort reading control: ordinary latest pair day before packet entrance.

This is conditional on v21's already accepted witness opportunities. It removes
the witness-date choice, not the original stable transaction mechanism.
"""
from .observed_successor_read import pair_local_reads


def ordinary_days(q, accepted, candidates, events, pairs, physical):
    result = []
    for source in accepted:
        c = source['candidate']
        assert c in candidates and c != q.known
        entrance = events[source['seed']]
        assert entrance.t < q.t and source['anchor_day'] < entrance.t
        history = [e for e in pairs[tuple(sorted((q.known, c)))] if e.t < entrance.t]
        assert history, 'An accepted witness must have a past pair observation'
        day = max(e.t for e in history)
        facts = {}
        for e in sorted(history):
            if e.t == day:
                facts.setdefault(physical(e), e)
        result.append(dict(candidate=c, context=source['context'], seed=source['seed'],
                           roots=source['roots'], anchor_day=day, successor_day=day,
                           facts=[e.eid for e in sorted(facts.values())],
                           source_witness_event=source['witness_event'],
                           source_witness_day=source['anchor_day'],
                           supported_candidate=source['supported_candidate'],
                           packet_seed=source['packet_seed'],
                           semantics='ordinary_latest_pair_day_before_same_entrance'))
    return result


def extend_ordinary_days(q, short, reference, trace, options, events, physical):
    result, extra = pair_local_reads(q, short, reference, trace, options, events, physical)
    if result == reference:
        return reference, trace
    return result, dict(trace, ordinary_witness_control_reads=extra['observed_successors'],
                        ordinary_witness_control_replacements=extra['successor_replacements'])

"""Expose already-routed physical witness days that are first-hop candidates.

The other endpoint's witness edge remains routing provenance only. No new
candidate, second-hop display edge, ownership label, or scoring call is added.
"""
from .observed_successor_read import pair_local_reads


def first_hop_witness_days(q, packet, candidates, events, pairs, physical):
    entrance = events[packet['seed']]
    assert entrance.t < q.t
    allowed = set(candidates)
    days = {}
    options = []
    for eid in packet.get('routing_facts', []):
        e = events[eid]
        assert e.t < q.t
        if q.known not in e.pair:
            continue
        other = e.o if e.s == q.known else e.s
        if other == packet['candidate'] or other not in allowed or e.t >= entrance.t:
            continue
        assert other == packet['context'][0] and other != q.known
        key = (e.pair, e.t)
        if key not in days:
            facts = {}
            for f in sorted(pairs[e.pair]):
                if f.t == e.t:
                    facts.setdefault(physical(f), f)
            assert facts and physical(e) in facts
            days[key] = [f.eid for f in sorted(facts.values())]
        options.append(dict(candidate=other, context=packet['context'],
                            seed=packet['seed'], roots=packet['roots'],
                            anchor_day=e.t, successor_day=e.t, facts=days[key],
                            witness_event=eid, supported_candidate=packet['candidate'],
                            packet_seed=packet['seed'],
                            semantics='known_endpoint_raw_witness_day_existing_candidate'))
    return options


def extend_first_hop_witness(q, short, reference, trace, options, events, physical):
    result, extra = pair_local_reads(q, short, reference, trace, options, events, physical)
    if result == reference:
        return reference, trace
    return result, dict(trace, first_hop_witness_reads=extra['observed_successors'],
                        first_hop_witness_replacements=extra['successor_replacements'])

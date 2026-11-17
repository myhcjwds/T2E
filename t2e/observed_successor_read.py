"""Read an earlier observed continuation of a repeated branch observation.

No transition probability, completion state, semantic rule or target is used.
An equal physical pair-day bundle identifies a historical analogy, not a case ID.
"""
from collections import defaultdict


def historical_successor(packet, pair_days, context_days, physical):
    seed_day = int(packet['seed'].split(':')[0])
    first_root = min(int(e.split(':')[0]) for e in packet['roots'])
    context = tuple(packet['context'])
    days = sorted(t for t in pair_days
                  if first_root <= t <= seed_day and context in context_days.get(t, {}))
    if seed_day not in days:
        return None
    signature = lambda t: {physical(e)[1:] for e in pair_days[t]}
    current = signature(seed_day)
    matches = [(a, b) for a, b in zip(days, days[1:])
               if b < seed_day and signature(a) == current]
    if not matches:
        return None
    anchor, successor = max(matches)
    facts = {}
    for t in (anchor, successor):
        for e in sorted(pair_days[t]):
            facts.setdefault(physical(e), e)
    return dict(candidate=packet['candidate'], context=packet['context'],
                seed=packet['seed'], roots=packet['roots'], anchor_day=anchor,
                successor_day=successor, facts=[e.eid for e in sorted(facts.values())],
                semantics='next_previously_observed_branch_day_after_equal_pair_day')


def pair_local_reads(q, short, reference, trace, options, events, physical):
    """Only replace ordinary fill inside the same candidate's existing quota."""
    assert len(reference) <= 100 and all(e.t < q.t for e in short + reference)
    short_keys = {physical(e) for e in short}
    groups = defaultdict(list)
    untouched = []
    for e in reference:
        if q.known not in (e.s, e.o):
            untouched.append(e)
        else:
            groups[e.o if e.s == q.known else e.s].append(e)
    protected = set(trace.get('protected', []))
    protected.update(e for p in trace['packets'] for e in p['facts'])
    protected.update(e.eid for e in reference if e.r == q.r)
    protected.update(max(es).eid for es in groups.values())
    protected_keys = {physical(events[e]) for e in protected}
    bycandidate = defaultdict(list)
    for option in options:
        bycandidate[option['candidate']].append(option)
    selected = list(untouched)
    accepted, replacements = [], []
    for c, old in groups.items():
        quota = len(old)
        keep = {e.eid: e for e in old if physical(e) in protected_keys}
        used = short_keys | {physical(e) for e in keep.values()}
        chosen, seen = [], set()
        choices = sorted(bycandidate[c], key=lambda p: (-p['anchor_day'], -p['successor_day'], tuple(p['context']), p['seed']))
        for option in choices:
            alias = tuple(option['facts'])
            if alias in seen:
                continue
            seen.add(alias)
            fs = [events[e] for e in option['facts']]
            assert all(e.pair == tuple(sorted((q.known, c))) and e.t < int(option['seed'].split(':')[0]) < q.t for e in fs)
            extra = {physical(e): e for e in fs if physical(e) not in used}
            if not extra or len(keep) + len(extra) > quota:
                continue
            keep.update({e.eid: e for e in extra.values()})
            used.update(extra)
            chosen.append(option)
        for e in sorted(old, reverse=True):
            if len(keep) == quota:
                break
            if physical(e) not in used:
                keep[e.eid] = e
                used.add(physical(e))
        new = sorted(keep.values())
        if len(new) != quota:
            new, chosen = old, []
        selected.extend(new)
        if set(new) != set(old):
            accepted.extend(chosen)
            replacements.append(dict(candidate=c, before=[e.eid for e in old],
                                     after=[e.eid for e in new],
                                     protected=sorted(e.eid for e in old if physical(e) in protected_keys)))
    if not replacements:
        return reference, trace
    return sorted(selected), dict(trace, observed_successors=accepted,
                                  successor_replacements=replacements)

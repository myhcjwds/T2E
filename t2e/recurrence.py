"""Past-only recurrence calibration; no fitted model parameters.

Entity-context and query-relation counts form separate normalized distributions.
This is an optional statistical control/calibrator, not an instance-state claim.
"""
import math


def relation_history(query, store, limit):
    store.assert_past(query.t)
    events = (e for e in store.incident.get(query.known, ())
              if e.r == query.r and ((query.direction == 'tail' and e.s == query.known)
                                    or (query.direction == 'head' and e.o == query.known)))
    return list(events)[-limit:] if limit else []


def select_history(query, store, history_len, relation_quota=0):
    """Reserve a fixed part of the unchanged fact budget for the query relation."""
    if not 0 <= relation_quota <= history_len:
        raise ValueError('Relation history quota must be between zero and history length')
    recent = store.recent(query, history_len)
    if not relation_quota:
        return recent
    selected = set(relation_history(query, store, relation_quota))
    for event in recent:
        if len(selected) >= history_len:
            break
        selected.add(event)
    return sorted(selected, reverse=True)


def distribution(events, query, half_life):
    if not events:
        return {}
    latest = max(e.t for e in events)
    result = {}
    for event in events:
        entity = event.o if query.direction == 'tail' else event.s
        # Common absolute-age factor cancels after normalization. Subtraction
        # from latest avoids total underflow on sparse, very old histories.
        weight = 2.0**(-(latest-event.t)/half_life) if half_life else 1.0
        result[entity] = result.get(entity, 0.0) + weight
    total = sum(result.values())
    return {entity: weight/total for entity, weight in result.items() if weight > 0}


def recurrence_distribution(query, store, history_len=100, half_life=720.0, relation_weight=.75):
    if not math.isfinite(half_life) or half_life < 0:
        raise ValueError('Recurrence half-life must be finite and nonnegative')
    if not 0 <= relation_weight <= 1:
        raise ValueError('Recurrence relation weight must be in [0,1]')
    store.assert_past(query.t)
    recent = store.recent(query, history_len)
    same_relation = relation_history(query, store, history_len)
    entity_probs = distribution(recent, query, half_life)
    relation_probs = distribution(same_relation, query, half_life)
    if not entity_probs:
        candidates = store.fallback(query, 128)
        counts = store.role_counts.get((query.r, query.direction), {})
        values = [counts.get(c, 0) for c in candidates]
        if not sum(values):
            values = [1.0]*len(candidates)
        total = sum(values)
        entity_probs = dict(zip(candidates, (v/total for v in values))) if total else {}
    if not relation_probs:
        relation_probs = entity_probs
    candidates = sorted(set(entity_probs) | set(relation_probs))
    probabilities = {c: (1-relation_weight)*entity_probs.get(c,0.) + relation_weight*relation_probs.get(c,0.)
                     for c in candidates}
    return {c:p for c,p in probabilities.items() if p>0}, {
        'entity_history_count':len(recent), 'relation_history_count':len(same_relation),
        'latest_entity_time':max((e.t for e in recent),default=None),
        'latest_relation_time':max((e.t for e in same_relation),default=None)}


def mix_recurrence(query, store, candidates, probabilities, weight, history_len=100,
                   half_life=720.0, relation_weight=.75):
    if not 0 <= weight <= 1:
        raise ValueError('Recurrence mixture weight must be in [0,1]')
    if not weight:
        return candidates, probabilities, {'enabled':False}
    prior, trace = recurrence_distribution(query, store, history_len, half_life, relation_weight)
    if not prior:
        return candidates, probabilities, {'enabled':False,'reason':'no_past_support'}
    # Preserve the entire neural candidate pool and its probabilities. Newly
    # recalled entities receive only the explicit statistical branch's mass.
    combined = list(candidates) + sorted(set(prior)-set(candidates))
    base = dict(zip(candidates,probabilities))
    values = [(1-weight)*base.get(c,0.)+weight*prior.get(c,0.) for c in combined]
    trace.update(enabled=True,weight=weight,half_life=half_life,relation_weight=relation_weight,
                 candidate_ids=combined,probabilities=[prior.get(c,0.) for c in combined],
                 extra_candidates=len(combined)-len(candidates))
    return combined, values, trace

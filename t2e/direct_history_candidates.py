"""Deterministic one-hop choices, without a preliminary neural ranking.

Retains the historical supports of the frozen base candidate union. Removes
the winner-dependent legacy transaction supplement. No scorer/label API.
"""
from t2e.compact import frequency_candidates
from t2e.recurrence import select_history, recurrence_distribution
from t2e.calibration import backoff_distribution, symbolic_distribution


def history_candidates(query, store, spec, precision, relation_count):
    store.assert_past(query.t)
    version = store.version
    supports = {}
    views = [("base_history", 100, 50)]
    for name in ("second_view", "long_view"):
        view = spec.get(name)
        if not view or not view['weight']:
            continue
        if view.get('gate', 'always') != 'always':
            raise ValueError('Neural-dependent candidate gates are forbidden')
        views.append((name, view['history_len'], view['relation_quota']))
    for name, length, quota in views:
        facts = select_history(query, store, length, quota)
        supports[name] = frequency_candidates(facts, query) or store.fallback(query, 128)
    fixed, _ = recurrence_distribution(query, store, 100, 720., .75)
    prior = spec['prior']
    if prior['type'] == 'fixed':
        supports['prior'] = sorted(fixed)
    elif prior['type'] == 'backoff':
        supports['prior'] = sorted(backoff_distribution(
            query, store, fixed, prior['half_life'], prior['pseudocount']))
    else:
        raise ValueError('Unknown historical prior')
    if spec['calibration'] == 'product':
        supports['symbolic'] = sorted(symbolic_distribution(
            query, store, precision, relation_count) or fixed)
    elif spec['calibration'] != 'arithmetic':
        raise ValueError('Unknown calibration')
    neighbors = {e.o if e.s == query.known else e.s
                 for e in store.incident.get(query.known, ())}
    supports = {name: sorted(set(cs) & neighbors) for name, cs in supports.items()}
    pool = sorted(set().union(*(set(cs) for cs in supports.values())))
    assert version == store.version
    return pool, supports

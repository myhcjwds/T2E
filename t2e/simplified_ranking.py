"""Fixed reduced ranking: recurrence, symbolic evidence and pair recency.

No popularity or activity-window statistics are computed here. The remaining
formula and constants are identical to the corresponding legacy score terms.
"""
import math

from t2e.recurrence import recurrence_distribution
from t2e.calibration import backoff_distribution, symbolic_distribution


def pair_recency(query, store, candidates):
    store.assert_past(query.t)
    values = []
    for candidate in candidates:
        age = None
        for event in reversed(store.pairs.get(tuple(sorted((query.known, candidate))), ())):
            if event.r != query.r:
                continue
            subject, obj = ((query.known, candidate) if query.direction == "tail"
                            else (candidate, query.known))
            if (event.s, event.o) == (subject, obj):
                age = query.t - event.t
                break
        values.append(math.log1p(9. * 2. ** (-age / 168.)) if age is not None else 0.)
    return values


def calibration_terms(query, store, candidates, selection, precision, relation_count):
    store.assert_past(query.t)
    fixed, _ = recurrence_distribution(query, store, 100, 720., .75)
    spec = selection["fusion_spec"]["prior"]
    prior = backoff_distribution(query, store, fixed, spec["half_life"], spec["pseudocount"])
    symbolic = symbolic_distribution(query, store, precision, relation_count) or fixed
    floor = 1e-4 / max(1, len(candidates))
    terms = {
        "recurrence": [math.log(prior.get(c, 0.) + floor) for c in candidates],
        "symbolic": [selection['symbolic_weight'] * math.log(symbolic.get(c, 0.) + floor) for c in candidates],
        "pair7": [selection["partner_spec"]["pair7"] * x for x in pair_recency(query, store, candidates)],
    }
    return terms, floor


def score(neural_probabilities, terms, floor):
    n = len(neural_probabilities)
    if set(terms) != {"recurrence", "symbolic", "pair7"} or any(len(x) != n for x in terms.values()):
        raise ValueError("Invalid simplified calibration terms")
    logits = [math.log(p + floor) + (terms["recurrence"][i] + terms["symbolic"][i] + terms["pair7"][i])
              for i, p in enumerate(neural_probabilities)]
    if not logits:
        return []
    peak = max(logits)
    values = [math.exp(x - peak) for x in logits]
    total = sum(values)
    return [x / total for x in values]

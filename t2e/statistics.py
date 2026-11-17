from .data import Dataset, Event, EventStore, batches, queries_and_labels, digest

MULTIPLIER = {"ICEWS14":1,"ICEWS05-15":24,"ICEWS18":1}

class CanonicalDataset(Dataset):
    """Keep one day = 24 units, matching all frozen temporal constants."""
    def __init__(self, directory):
        super().__init__(directory)
        self.raw_data_hash = self.data_hash
        self.multiplier = MULTIPLIER[self.name]
        if self.multiplier != 1:
            self.splits = {s: tuple(Event(e.t*self.multiplier, e.s, e.r, e.o) for e in es)
                           for s, es in self.splits.items()}
            self.train_hash = digest([e.eid for e in self.splits['train']])
            self.data_hash = digest({s: [e.eid for e in es] for s, es in self.splits.items()})

    def audit(self):
        return dict(super().audit(), raw_data_hash=self.raw_data_hash,
                    timestamp_multiplier=self.multiplier, canonical_units_per_day=24)

def fit_rule_precision(data):
    """Same historical-body estimator as audit/validation_rules.py."""
    import numpy as np
    nr = max(data.relations)+1
    numerator = np.zeros((2*nr, 2*nr), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    store = EventStore()
    for _, batch in batches(data.splits['train']):
        queries, labels = queries_and_labels(batch)
        cache = {}
        for q in queries:
            if q.known not in cache:
                cache[q.known] = {
                    (e.r+(nr if e.o == q.known and e.s != q.known else 0),
                     e.o if e.s == q.known else e.s)
                    for e in store.incident.get(q.known, ())[-200:]}
            head = q.r+(nr if q.direction == 'head' else 0)
            for body, candidate in cache[q.known]:
                denominator[head, body] += 1
                numerator[head, body] += candidate in labels[q]
        store.observe(batch)
    prior = (numerator.sum(1)+1)/(denominator.sum(1)+20)
    precision = (numerator+10*prior[:, None])/(denominator+10)
    precision[denominator < 3] = 0
    return dict(numerator=numerator, denominator=denominator, precision=precision)

from __future__ import annotations


class Metrics:
    def __init__(self):
        self.count = self.covered = 0
        self.hits = {1: 0, 3: 0, 10: 0}

    def add(self, candidates, probabilities, targets, filtering="time-aware"):
        if len(candidates) != len(probabilities) or len(set(candidates)) != len(candidates):
            raise ValueError("Invalid candidate score alignment")
        ranking = [c for c, _ in sorted(zip(candidates, probabilities), key=lambda p: (-p[1], p[0]))]
        ranks = {}
        for target in sorted(targets):
            self.count += 1
            if target not in ranking:
                ranks[target] = None
                continue
            self.covered += 1
            eligible = [c for c in ranking if filtering != "time-aware" or c == target or c not in targets]
            rank = eligible.index(target) + 1
            ranks[target] = rank
            for k in self.hits:
                self.hits[k] += rank <= k
        return ranking, ranks

    def dump(self):
        """The same progress-bar field names and target denominator as ONSEP."""
        denom = max(self.count, 1)
        return {"total": self.count, "hit1": self.hits[1] / denom,
                "hit3": self.hits[3] / denom, "hit10": self.hits[10] / denom}

    def summary(self):
        denom = max(self.count, 1)
        return {"metric_scope": "fixed_candidate_ranking_with_missing_targets_as_misses",
                "target_count": self.count, "candidate_recall": self.covered / denom,
                "candidate_hits": {str(k): v / denom for k, v in self.hits.items()}}

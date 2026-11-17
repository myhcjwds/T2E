"""Past-only history retrieval; no outcome tables, rule induction, or LLM calls.

A context is an observed (third participant, role of endpoint A, role of endpoint
B), not a real-world case ID. One pair-day can belong to several contexts.
Daily snapshots are unordered: contexts are built only after a whole day commits.
"""
from collections import defaultdict, deque
from dataclasses import dataclass

from .data import EventStore, batches


@dataclass(frozen=True)
class Context:
    anchor: int
    left_role: int
    right_role: int


class TransactionHistory:
    def __init__(self, relation_count, context_window=168, contexts=True):
        self.nr = relation_count
        self.window = context_window
        self.contexts_enabled = contexts
        self.store = EventStore()
        self.directed = defaultdict(list)
        self.active = defaultdict(dict)
        self.expirations = deque()
        self.day_events = {}
        self.day_contexts = {}
        self.routes = defaultdict(list)

    def extend(self, events):
        for _, group in batches(events):
            self.observe(group)

    def observe(self, events):
        events = tuple(sorted(set(events)))
        if not events:
            return
        self.store.observe(events)  # Checks chronology before touching indexes.
        t = events[0].t
        for e in events:
            self.directed[(e.s, "tail")].append(e)
            self.directed[(e.o, "head")].append(e)
        if not self.contexts_enabled:
            return
        while self.expirations and self.expirations[0][0] < t - self.window:
            old_t, node, key = self.expirations.popleft()
            if self.active[node].get(key, (None,))[0] == old_t:
                del self.active[node][key]
        pairs = defaultdict(list)
        for e in events:
            pairs[e.pair].append(e)
            for node, neighbor, role in ((e.s, e.o, e.r),
                                         (e.o, e.s, e.r + self.nr)):
                key = neighbor, role
                self.active[node][key] = (t, e.eid)
                self.expirations.append((t, node, key))
        # All same-day events are visible together, without inventing their order.
        by_neighbor = {}
        for node in {n for pair in pairs for n in pair}:
            lookup = defaultdict(list)
            for (neighbor, role), (_, eid) in self.active[node].items():
                lookup[neighbor].append((role, eid))
            by_neighbor[node] = lookup
        for pair, facts in pairs.items():
            left, right = pair
            key = pair, t
            contexts = {}
            if left != right:
                la, ra = by_neighbor[left], by_neighbor[right]
                for anchor in (set(la) & set(ra)) - set(pair):
                    for lr, le in la[anchor]:
                        for rr, re in ra[anchor]:
                            context = Context(anchor, lr, rr)
                            contexts[context] = (le, re)
                            self.routes[(pair, context)].append(t)
            self.day_events[key] = tuple(facts)
            self.day_contexts[key] = contexts

    def short(self, query, length):
        """Exactly the latest L role-consistent events, irrespective of relation."""
        self.store.assert_past(query.t)
        return list(self.directed.get((query.known, query.direction), ())[-length:]) if length else []

    def ordinary_long(self, query, short, candidates, length):
        self.store.assert_past(query.t)
        if not short or not length:
            return []
        cutoff = min(e.t for e in short)
        allowed = set(candidates)
        result = []
        for e in reversed(self.directed.get((query.known, query.direction), ())):
            other = e.o if query.direction == "tail" else e.s
            if e.t < cutoff and other in allowed:
                result.append(e)
                if len(result) == length:
                    break
        return sorted(result)

    def transaction_long(self, query, short, candidates, length):
        """Round-robin over observed contexts, followed by an explicit fallback.

        Each candidate's latest pair-day inside the short history supplies seeds.
        If the queried relation is observed inside that short history, its latest
        day is used instead. Old events must precede the *entire* short window.
        """
        self.store.assert_past(query.t)
        if not short or not length:
            return [], {"matched_fact_ids": [], "assignments": {}, "fallback_fact_ids": []}
        cutoff = min(e.t for e in short)
        by_candidate = defaultdict(list)
        for e in short:
            other = e.o if query.direction == "tail" else e.s
            by_candidate[other].append(e)
        streams = []
        assignments = defaultdict(list)
        for candidate in candidates:
            facts = by_candidate[candidate]
            if not facts:
                continue
            pair = tuple(sorted((query.known, candidate)))
            seeds = [e for e in facts if e.r == query.r] or facts
            seed_t = max(e.t for e in seeds)
            current = self.day_contexts.get((pair, seed_t), {})
            for context in sorted(current, key=lambda c: (c.anchor, c.left_role, c.right_role)):
                eligible = []
                for day in reversed(self.routes.get((pair, context), ())):
                    if day >= cutoff:
                        continue
                    for e in reversed(self.day_events[(pair, day)]):
                        if ((query.direction == "tail" and e.s != query.known)
                                or (query.direction == "head" and e.o != query.known)):
                            continue
                        eligible.append(e)
                        assignments[e.eid].append({
                            "anchor": context.anchor, "pair": list(pair),
                            "roles": [context.left_role, context.right_role],
                            "seed_timestamp": seed_t,
                            "current_witnesses": list(current[context]),
                            "old_witnesses": list(self.day_contexts[(pair, day)][context]),
                        })
                if eligible:
                    streams.append(deque(eligible))
        selected, seen = [], set()
        # Keep a prolific pair/context from consuming the entire evidence budget.
        while streams and len(selected) < length:
            remaining = []
            for stream in streams:
                while stream and stream[0].eid in seen:
                    stream.popleft()
                if stream and len(selected) < length:
                    event = stream.popleft()
                    selected.append(event)
                    seen.add(event.eid)
                if stream:
                    remaining.append(stream)
            streams = remaining
        matched = [e.eid for e in selected]
        fallback = []
        for e in reversed(self.ordinary_long(query, short, candidates, length)):
            if len(selected) == length:
                break
            if e.eid not in seen:
                selected.append(e)
                seen.add(e.eid)
                fallback.append(e.eid)
        return sorted(selected), {
            "matched_fact_ids": matched,
            "assignments": {eid: assignments[eid] for eid in matched},
            "fallback_fact_ids": fallback,
        }

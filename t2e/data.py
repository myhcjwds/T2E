from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True, order=True)
class Event:
    t: int
    s: int
    r: int
    o: int

    @property
    def eid(self):
        return "{}:{}:{}:{}".format(self.t, self.s, self.r, self.o)

    @property
    def pair(self):
        return tuple(sorted((self.s, self.o)))

    def as_dict(self):
        return {"id": self.eid, "subject": self.s, "relation": self.r,
                "object": self.o, "timestamp": self.t}


@dataclass(frozen=True, order=True)
class Query:
    t: int
    known: int
    r: int
    direction: str

    @property
    def qid(self):
        return "{}:{}:{}:{}".format(self.t, self.known, self.r, self.direction)

    def as_dict(self):
        return {"timestamp": self.t, "known_entity": self.known,
                "relation": self.r, "direction": self.direction}


def read_dictionary(path):
    result = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        name, idx = line.rsplit("\t", 1)
        idx = int(idx)
        if idx in result:
            raise ValueError("Duplicate dictionary ID at {}:{}".format(path, n))
        result[idx] = name
    return result


def read_events(path, entities, relations):
    rows, seen = [], set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 4:
            raise ValueError("Expected s r o t at {}:{}".format(path, n))
        s, r, o, t = map(int, fields[:4])
        if s not in entities or o not in entities or r not in relations:
            raise ValueError("Unknown ID at {}:{}".format(path, n))
        event = Event(t, s, r, o)
        if event not in seen:
            rows.append(event)
            seen.add(event)
    return tuple(sorted(rows))


class Dataset:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.name = self.directory.name
        self.entities = read_dictionary(self.directory / "entity2id.txt")
        self.relations = read_dictionary(self.directory / "relation2id.txt")
        self.splits = {split: read_events(self.directory / (split + ".txt"),
                                         self.entities, self.relations)
                       for split in ("train", "valid", "test")}
        self.train_hash = digest([e.eid for e in self.splits["train"]])
        self.relation_hash = digest(self.relations)
        self.data_hash = digest({s: [e.eid for e in events]
                                 for s, events in self.splits.items()})

    def prefix(self, split):
        if split not in ("valid", "test"):
            raise ValueError("Evaluation split must be valid or test")
        prefix = self.splits["train"]
        if split == "test":
            prefix = tuple(sorted(set(prefix + self.splits["valid"])))
        evaluation = self.splits[split]
        if not evaluation:
            raise ValueError("Empty evaluation split")
        if prefix and prefix[-1].t >= evaluation[0].t:
            raise ValueError("History/evaluation timestamps overlap; chronological split required")
        return prefix

    def audit(self):
        return {"dataset": self.name, "data_hash": self.data_hash,
                "entities": len(self.entities), "relations": len(self.relations),
                "splits": {s: {"events": len(es), "timestamps": len(set(e.t for e in es)),
                               "first": es[0].t if es else None,
                               "last": es[-1].t if es else None}
                           for s, es in self.splits.items()}}


def batches(events):
    for timestamp, values in groupby(sorted(events), key=lambda e: e.t):
        yield timestamp, tuple(values)


def queries_and_labels(events):
    """Labels are returned separately and never placed inside Query."""
    labels = defaultdict(set)
    for event in events:
        labels[Query(event.t, event.s, event.r, "tail")].add(event.o)
        labels[Query(event.t, event.o, event.r, "head")].add(event.s)
    return sorted(labels), {q: frozenset(v) for q, v in labels.items()}


def relative(event, known):
    views = []
    if event.s == known:
        views.append((event.r, "tail", event.o))
    if event.o == known:
        views.append((event.r, "head", event.s))
    return views


class EventStore:
    def __init__(self, events=()):
        self.events = {}
        self.incident = defaultdict(list)
        self.pairs = defaultdict(list)
        self.pair_versions = Counter()
        self.role_counts = defaultdict(Counter)
        self.entity_counts = Counter()
        self.latest = None
        self.version = 0
        for _, batch in batches(events):
            self.observe(batch)

    def observe(self, events):
        events = tuple(sorted(set(events)))
        if not events:
            return
        if len(set(e.t for e in events)) != 1:
            raise ValueError("Commit one complete timestamp at a time")
        t = events[0].t
        if self.latest is not None and t <= self.latest:
            raise ValueError("Observation timestamps must strictly increase")
        touched = set()
        for e in events:
            if e.eid in self.events:
                raise ValueError("Event already committed: " + e.eid)
            self.events[e.eid] = e
            self.incident[e.s].append(e)
            if e.o != e.s:
                self.incident[e.o].append(e)
            self.pairs[e.pair].append(e)
            touched.add(e.pair)
            self.role_counts[(e.r, "tail")][e.o] += 1
            self.role_counts[(e.r, "head")][e.s] += 1
            self.entity_counts[e.s] += 1
            self.entity_counts[e.o] += 1
        for pair in touched:
            self.pair_versions[pair] += 1
        self.latest = t
        self.version += 1

    def assert_past(self, timestamp):
        if self.latest is not None and self.latest >= timestamp:
            raise ValueError("Snapshot contains current/future observations")

    def recent(self, q, limit):
        self.assert_past(q.t)
        relevant = [e for e in self.incident.get(q.known, ())
                    if (q.direction == "tail" and e.s == q.known)
                    or (q.direction == "head" and e.o == q.known)]
        return list(reversed(relevant[-limit:])) if limit else []

    def fallback(self, q, limit):
        counts = self.role_counts.get((q.r, q.direction)) or self.entity_counts
        return [entity for entity, _ in sorted(counts.items(), key=lambda p: (-p[1], p[0]))[:limit]]

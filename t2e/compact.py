from collections import Counter

def target_of(event, query):
    return event.o if event.s == query.known else event.s

def frequency_candidates(events, query):
    counts = Counter(target_of(e, query) for e in sorted(set(events)))
    return sorted(counts, key=lambda e: -counts[e])
"""Render isolated query-event evidence with the original chronological policy."""
from t2e.event_ledger import render


def chronological_plan(q, plan, short, long, trace, token_count):
    ids = [e.eid for e in long]
    if ids == plan['long']['ledger']:
        return plan
    assert long == sorted(long)
    text = render(q, short, long, plan['candidate_ids'], trace, False)
    tokens = token_count(text)
    assert tokens + 16 <= 8192
    return dict(plan, long=dict(plan['long'], ledger=ids), trace=trace,
                prompts=dict(ledger=dict(text=text, tokens=tokens)),
                readout=dict(policy='original_chronological_long_evidence', order=ids,
                             short_suffix_identical=True, identity_claim=False))

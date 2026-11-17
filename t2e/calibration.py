import math
from t2e.recurrence import relation_history, distribution

def symbolic_distribution(query,store,precision,nr):
    store.assert_past(query.t)
    events=store.incident.get(query.known,())[-200:]
    latest=max((e.t for e in events),default=query.t)
    head=query.r+(nr if query.direction=='head' else 0);result={}
    for event in events:
        body=event.r+(nr if event.o==query.known and event.s!=query.known else 0)
        candidate=event.o if event.s==query.known else event.s
        value=float(precision[head,body])**2*2.**(-(latest-event.t)/168.)
        if value>0:result[candidate]=result.get(candidate,0.)+value
    total=sum(result.values())
    return {c:v/total for c,v in result.items()} if total else {}

def backoff_distribution(query,store,fallback,half_life,pseudocount):
    if half_life<0 or not math.isfinite(half_life) or pseudocount<=0 or not math.isfinite(pseudocount):
        raise ValueError('Invalid count-backoff parameters')
    store.assert_past(query.t)
    generic=distribution(store.recent(query,200),query,half_life) or fallback
    specific={}
    for event in relation_history(query,store,200):
        c=event.o if query.direction=='tail' else event.s
        weight=2.**(-(query.t-event.t)/half_life) if half_life else 1.
        specific[c]=specific.get(c,0.)+weight
    total=sum(specific.values())+pseudocount
    return {c:(specific.get(c,0.)+pseudocount*generic.get(c,0.))/total for c in set(specific)|set(generic)}
def interaction_key(e):
    """Identify coded views, preserving host/visitor roles and distinct issues.

    Host(A,B) and Visit(B,A) are two views of a visit hosted by A. Visit(A,B)
    is a different interaction. Intentions, requests and executions never merge.
    """
    if e.r==5:return (e.t,'visit',e.s,e.o)
    if e.r==4:return (e.t,'visit',e.o,e.s)
    if e.r in (1,12,19,27):return (e.t,e.r,min(e.s,e.o),max(e.s,e.o))
    return (e.t,e.r,e.s,e.o)

def event_line(q,e,labels):
    # Head prediction uses the same relative encoding as the frozen baseline.
    forward=(e.s==q.known) if q.direction=='tail' else (e.o==q.known)
    other=e.o if e.s==q.known else e.s
    relation=str(e.r) if forward else str(e.r)+'^-1'
    return '{}:[{},{},{}.{}]\n'.format(e.t,q.known,relation,labels.get(other,'_'),other)

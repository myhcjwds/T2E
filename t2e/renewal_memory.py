"""Censored event-stream clocks, replacing the fixed seven-day pair bonus.

Each observed directed triple keeps its own clock. Another event type cannot
reset it, so coexisting strands on one pair are not collapsed. Historical
inter-arrival intervals estimate a decreasing continuation hazard: unresolved
silence weakens evidence; it never proves completion. The shape constraint
pools unsupported fluctuations rather than tuning a decay weight on test data.
"""
from collections import Counter,defaultdict
import math
from .data import batches

def decreasing_hazard(hits,risk,background):
    blocks=[]
    for age in range(1,max(risk,default=1)+1):
        # One background observation avoids zero hazards without privileging
        # rare old intervals. Block merging is the binomial isotonic MLE.
        blocks.append([age,age,hits.get(age,0)+background,risk.get(age,0)+1.])
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][3]<blocks[-1][2]/blocks[-1][3]:
            b=blocks.pop();a=blocks.pop();blocks.append([a[0],b[1],a[2]+b[2],a[3]+b[3]])
    out={}
    for lo,hi,h,n in blocks:
        for age in range(lo,hi+1):out[age]=h/n
    return out

class RenewalMemory:
    def __init__(self,events):
        streams=defaultdict(list);self.latest={};self.last=None;self.hits=defaultdict(Counter);self.risk=defaultdict(Counter)
        for e in events:streams[e.s,e.r,e.o].append(e.t)
        end=max(e.t for e in events)
        for (s,r,o),ts in streams.items():
            for a,b in zip(ts,ts[1:]):self.hits[r][(b-a)//24]+=1
            for gap in [(b-a)//24 for a,b in zip(ts,ts[1:])]+[(end-ts[-1])//24]:
                for age in range(1,gap+1):self.risk[r][age]+=1
        self.background={r:(sum(self.hits[r].values())+.5)/(sum(risk.values())+1.) for r,risk in self.risk.items()}
        self.curves={r:decreasing_hazard(self.hits[r],risk,self.background[r]) for r,risk in self.risk.items()}
        for _,es in batches(events):self.observe(es)

    def observe(self,es):
        if not es:return
        t=es[0].t
        if len({e.t for e in es})!=1 or (self.last is not None and t<=self.last):raise ValueError('Complete strictly increasing timestamp required')
        for e in es:self.latest[e.s,e.r,e.o]=e.t
        self.last=t

    def values(self,q,candidates,shape='decreasing'):
        if self.last is not None and self.last>=q.t:raise ValueError('Future snapshot')
        base=self.background.get(q.r);curve=self.curves.get(q.r,{})
        result=[];trace=[]
        for c in candidates:
            s,o=(q.known,c) if q.direction=='tail' else (c,q.known);t=self.latest.get((s,q.r,o))
            if t is None or not curve:result.append(0.);trace.append(None);continue
            age=(q.t-t)//24
            if shape=='decreasing':hazard=curve.get(age,curve[max(curve)])
            elif shape=='empirical':hazard=(self.hits[q.r][age]+base)/(self.risk[q.r][age]+1.)
            else:raise ValueError(shape)
            result.append(math.log(max(hazard,1e-15)/base));trace.append(dict(last_event=t,age_days=age,hazard=hazard,background=base))
        return result,trace

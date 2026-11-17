"""Project transaction joins to answer-bearing events without losing coverage.

Witnesses remain in the auditable ownership index. They do not consume display
slots when a transaction retrieval succeeds. Every candidate with direct facts
in the source long memory retains its latest such fact. Membership alternatives
and new intentions are inherited unchanged from ProvenanceMemory.
"""
from .provenance_memory import ProvenanceMemory
from .ownership_history import interaction_key

PACKET_CAP = None


class ProjectedMemory(ProvenanceMemory):
    def retrieve(self,index,q,short,candidates,baseline,length=100,policy='replace'):
        index.store.assert_past(q.t)
        allowed=set(candidates)
        short_keys={interaction_key(e) for e in short}
        selected={};used=set(short_keys)

        def candidate(e):
            if e.s==q.known:return e.o
            if e.o==q.known:return e.s
            return None

        def insert(e):
            k=interaction_key(e)
            if k in used or len(selected)>=length:return False
            selected[e.eid]=e;used.add(k);return True

        # Preserve candidate coverage from the exact same source memory. The
        # invariant is independent of target labels, neural scores and weights.
        protected={};latest={}
        for e in sorted(baseline):
            c=candidate(e)
            if c not in allowed or interaction_key(e) in short_keys:continue
            latest[c]=e
            if e.r==q.r:protected[e.eid]=e
        protected.update({e.eid:e for e in latest.values()})
        for e in sorted(protected.values()):insert(e)

        available={}
        for c in candidates:
            available[c]=[]
            pair=tuple(sorted((q.known,c)))
            for p in self.packets(index,q,c,short):
                direct=[e for e in p['facts'] if e.pair==pair]
                available[c].append({**p,'facts':direct,
                    'routing_facts':[e.eid for e in p['facts']]})
        accepted=[];depth=0;accepted_by_candidate={c:0 for c in candidates}
        while any(depth<len(ps) for ps in available.values()):
            for c in candidates:
                if depth>=len(available[c]):continue
                if PACKET_CAP is not None and accepted_by_candidate[c]>=PACKET_CAP:continue
                p=available[c][depth]
                add={interaction_key(e):e for e in p['facts'] if interaction_key(e) not in used}
                if not add or len(selected)+len(add)>length:continue
                for e in add.values():insert(e)
                accepted.append(p)
                accepted_by_candidate[c]+=1
            depth+=1
            if len(selected)>=length:break
        if not accepted:return [e for e in baseline if q.known in (e.s,e.o)],dict(packets=[],policy=policy)

        # Fill the remaining slots with direct first-hop history; no witness
        # edge is silently substituted for an event involving the query entity.
        for pool in (baseline,index.directed.get((q.known,q.direction),())):
            for e in reversed(pool):
                if len(selected)>=length:break
                if candidate(e) in allowed:insert(e)
        trace=[{**{k:v for k,v in p.items() if k!='facts'},
                'facts':[e.eid for e in p['facts']]} for p in accepted]
        self.stats['projected_queries']+=1
        self.stats['protected_candidate_facts']+=len(latest)
        return sorted(selected.values()),dict(packets=trace,policy=policy,
            projection='pair_events_with_witness_routing',
            protected=[e.eid for e in sorted(protected.values())])

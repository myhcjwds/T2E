"""Keep all physical witness observations from a neighbor's latest day.

Simultaneous incidence roles are alternatives, never an ordering or case ID.
Only the provenance index changes; projected first-hop reads use the old budget.
"""
from collections import defaultdict
from .context_history import ContextHistory
from .parallel_memory import ParallelMemory
from .ownership_history import interaction_key
from .event_ledger import initiates


def role(event, endpoint):
    k=interaction_key(event)
    return 0 if k[1] in (1,12,19,27) else (1 if k[2]==endpoint else -1)


class WitnessSnapshotHistory(ContextHistory):
    physical_window = 168
    def __init__(self, relation_count):
        super().__init__(relation_count)
        self.witness_days=defaultdict(dict)

    def observe(self, events):
        events=tuple(sorted(set(events)))
        if not events:return
        ContextHistory.observe(self,events)
        bypair=defaultdict(dict)
        for e in events:bypair[e.pair].setdefault(interaction_key(e),e)
        for (a,b),physical in bypair.items():
            bundle=tuple(sorted(physical.values()))
            self.witness_days[a][b]=bundle;self.witness_days[b][a]=bundle
        t=events[0].t
        for a,b in bypair:
            if a==b:continue
            options={}
            for z in sorted((self.witness_days[a].keys()&self.witness_days[b].keys())-{a,b}):
                left=self.witness_days[a][z];right=self.witness_days[b][z]
                if min(left[0].t,right[0].t)<t-self.physical_window:continue
                buckets=[]
                for endpoint,bundle in ((a,left),(b,right)):
                    d=defaultdict(list)
                    for e in bundle:d[role(e,endpoint)].append(e)
                    buckets.append(d)
                for ra,la in sorted(buckets[0].items()):
                    for rb,lb in sorted(buckets[1].items()):
                        options[(z,ra,rb)]=tuple(sorted(set(la+lb)))
            self.observations[(a,b)][-1]['witness_options']=options


class WitnessSnapshotMemory(ParallelMemory):
    def versions(self,index,pair):
        observations=index.observations.get(pair,())
        if not observations:return [],{}
        stamp=observations[-1]['time']
        if pair in self.cache and self.cache[pair][0]==stamp:return self.cache[pair][1:]
        roots=[];memberships={};previous={};contexts_by_day={};links=[]
        for obs in observations:
            contexts={}
            for option,ws in obs['witness_options'].items():
                z,ra,rb=option;support={interaction_key(e) for e in ws};old=previous.get(option)
                shared=support&old['support'] if old else set()
                lineage=old['lineage'] if shared else (z,obs['time'],3*(ra+1)+rb+1)
                contexts[lineage]=ws
                if shared:links.append(dict(participant=z,before=old['time'],after=obs['time'],
                    lineage=list(lineage),shared=[list(k) for k in sorted(shared,key=str)]))
                previous[option]=dict(support=support,lineage=lineage,time=obs['time'])
            contexts_by_day[obs['time']]=contexts
            before=list(roots)
            for e in obs['events']:
                possible=[v for v in before if v['root'].t<e.t and set(v['contexts'])&set(contexts)]
                memberships[e.eid]=[v['root'].eid for v in possible]
                if initiates(self.relations[e.r]):
                    v=dict(root=e,contexts=contexts,updates=[],previous=memberships[e.eid]);roots.append(v)
                    self.nodes[e.eid]=dict(event=e,initiation=True,parents=v['previous'],
                        effect='new_intent_observation',uncertain=bool(possible),key=[],
                        witnesses=[w.eid for ws in contexts.values() for w in ws])
                else:
                    for v in possible:
                        common=set(v['contexts'])&set(contexts)
                        v['updates'].append(dict(event=e,contexts={k:contexts[k] for k in common}))
                    self.nodes[e.eid]=dict(event=e,initiation=False,parents=memberships[e.eid],
                        effect='unresolved_observation',uncertain=bool(possible),key=[],
                        witnesses=[w.eid for ws in contexts.values() for w in ws])
        self.lineage_contexts[pair]=contexts_by_day;self.lineage_links[pair]=links
        self.stats['pair_snapshots']+=1;self.stats['versions_indexed']+=len(roots)
        self.stats['multi_owner_observations']+=sum(len(x)>1 for x in memberships.values())
        self.stats['witness_continuity_links']+=len(links)
        self.cache[pair]=(stamp,roots,memberships)
        return roots,memberships

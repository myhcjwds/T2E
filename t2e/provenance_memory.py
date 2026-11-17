"""Separate possible matters by continuity of actual witness observations.

A recurring third participant is insufficient to join historical intentions.
For each pair/participant, consecutive witnessed observations share a lineage
only when at least one physical witness record survives. Different participant
lineages never merge when an event has several possible memberships. Lineage
termination means missing continuity evidence, not completion or cancellation.
"""
from .event_ledger import EventLedger,initiates
from .ownership_history import interaction_key

class ProvenanceMemory(EventLedger):
    def __init__(self,relations):
        super().__init__(relations)
        self.lineage_contexts={}
        self.lineage_links={}
        self.packet_cache={};self.cache_day=None

    def active_context(self,index,pair,observation):
        self.versions(index,pair)
        return self.lineage_contexts[pair].get(observation['time'],{})

    def compatible_witnesses(self,pair,previous,current):
        return True

    def versions(self,index,pair):
        observations=index.observations.get(pair,())
        if not observations:return [],{}
        stamp=observations[-1]['time']
        if pair in self.cache and self.cache[pair][0]==stamp:return self.cache[pair][1:]
        roots=[];memberships={};previous={};contexts_by_day={};links=[]
        for obs in observations:
            contexts={}
            for z,ws in obs['witness'].items():
                support={interaction_key(e) for e in ws};old=previous.get(z)
                shared=support & old['support'] if old else set()
                if shared and not self.compatible_witnesses(pair,old['witness'],ws):shared=set()
                lineage=old['lineage'] if shared else (z,obs['time'],0)
                contexts[lineage]=ws
                if shared:links.append(dict(participant=z,before=old['time'],after=obs['time'],lineage=list(lineage),shared=[list(k) for k in sorted(shared,key=str)]))
                previous[z]=dict(support=support,lineage=lineage,time=obs['time'],witness=ws)
            contexts_by_day[obs['time']]=contexts
            before=list(roots)
            for e in obs['events']:
                possible=[v for v in before if v['root'].t<e.t and set(v['contexts'])&set(contexts)]
                memberships[e.eid]=[v['root'].eid for v in possible]
                if initiates(self.relations[e.r]):
                    v=dict(root=e,contexts=contexts,updates=[],previous=memberships[e.eid])
                    roots.append(v)
                    self.nodes[e.eid]=dict(event=e,initiation=True,parents=v['previous'],effect='new_intent_observation',uncertain=bool(possible),key=[],witnesses=[w.eid for ws in contexts.values() for w in ws])
                else:
                    for v in possible:
                        common=set(v['contexts'])&set(contexts)
                        v['updates'].append(dict(event=e,contexts={k:contexts[k] for k in common}))
                    self.nodes[e.eid]=dict(event=e,initiation=False,parents=memberships[e.eid],effect='unresolved_observation',uncertain=bool(possible),key=[],witnesses=[w.eid for ws in contexts.values() for w in ws])
        self.lineage_contexts[pair]=contexts_by_day;self.lineage_links[pair]=links
        self.stats['pair_snapshots']+=1;self.stats['versions_indexed']+=len(roots)
        self.stats['multi_owner_observations']+=sum(len(x)>1 for x in memberships.values())
        self.stats['witness_continuity_links']+=len(links)
        self.cache[pair]=(stamp,roots,memberships)
        return roots,memberships

    def packets(self,index,q,c,short):
        pair=tuple(sorted((q.known,c)));seed=self.seed_event(index,q,c,short)
        if seed is None:return []
        if self.cache_day!=q.t:self.packet_cache.clear();self.cache_day=q.t
        key=(q.known,q.direction,c,seed.eid)
        if key in self.packet_cache:
            self.stats['packet_cache_hits']+=1
            return self.packet_cache[key]
        roots,memberships=self.versions(index,pair)
        active=self.lineage_contexts.get(pair,{}).get(seed.t,{})
        result=[]
        for lineage,now in active.items():
            owners=[v for v in roots if v['root'].t<=seed.t and lineage in v['contexts']]
            if not owners:continue
            # One observation may support C and D, but that does not identify
            # their matters as one. Each lineage retains every possible root;
            # a root with more contexts cannot eliminate a weaker alternative.
            fs={e.eid:e for e in (seed,)+now}
            for v in owners:
                for e in (v['root'],)+v['contexts'][lineage]:fs[e.eid]=e
                updates=[u for u in v['updates'] if lineage in u['contexts'] and u['event'].t<=seed.t]
                if updates:
                    u=updates[-1]
                    for e in (u['event'],)+u['contexts'][lineage]:fs[e.eid]=e
                preceding=[u for earlier in owners if earlier['root'].t<v['root'].t for u in earlier['updates'] if lineage in u['contexts'] and u['event'].t<v['root'].t]
                if preceding:
                    u=max(preceding,key=lambda x:x['event'])
                    for e in (u['event'],)+u['contexts'][lineage]:fs[e.eid]=e
            edges=[]
            for e in sorted(fs.values()):
                parents=[rid for rid in memberships.get(e.eid,()) if rid in fs and index.store.events[rid].t<e.t]
                if parents:edges.append(dict(event=e.eid,parents=parents,effect='possible_membership',key=[]))
            root_ids=[v['root'].eid for v in owners];earliest=min(v['root'].t for v in owners)
            result.append(dict(seed=seed.eid,root=owners[-1]['root'].eid,roots=root_ids,candidate=c,context=list(lineage),support=[list(lineage)],effect='possible_owners_with_witness_continuity',facts=sorted(fs.values()),edges=edges,alternatives=root_ids,selection_key=None,
                continuity_links=[v for v in self.lineage_links[pair] if tuple(v['lineage'])==lineage and earliest<=v['before']<v['after']<=seed.t]))
        result.sort(key=lambda p:tuple(map(int,p['root'].split(':'))),reverse=True)
        self.stats['lineage_packets']+=len(result);self.stats['multi_version_packets']+=sum(len(p['roots'])>1 for p in result)
        self.packet_cache[key]=result
        return result

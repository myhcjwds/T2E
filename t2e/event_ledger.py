"""Evidence-linked intention versions; identity and completion remain unknown.

Only observable facts update memory. There is no relation-family table, learned
transition model, or per-record semantic classifier. One frozen prediction call
reads the resulting evidence. Context links are possible memberships, not case IDs.
"""
from collections import Counter
from .ownership_history import interaction_key
from .context_history import render as flat_render

VERSION='event-ledger-6-observable'

def initiates(name):
    return name.startswith(('Express_intent_', 'Appeal_', 'Demand')) or name=='Make_an_appeal_or_request'

def context_key(pair,z,witness):
    return (z, int(witness[0].s==pair[0]), int(witness[1].s==pair[1]))

class EventLedger:
    def __init__(self,relations):
        self.relations=relations;self.cache={};self.nodes={};self.stats=Counter()

    def active_context(self,index,pair,observation):
        return {context_key(pair,z,w):w for z,w in observation['witness'].items()
                if 0<index.anchor_days[pair][z]<index.context_days[pair]}

    def seed_event(self,index,q,c,short):
        pair=tuple(sorted((q.known,c)))
        return max((e for e in short if e.pair==pair),default=None)

    def versions(self,index,pair):
        observations=index.observations.get(pair,())
        if not observations:return [],{}
        stamp=observations[-1]['time']
        if pair in self.cache and self.cache[pair][0]==stamp:return self.cache[pair][1:]
        roots=[];memberships={}
        for obs in observations:
            contexts={context_key(pair,z,w):w for z,w in obs['witness'].items()}
            before=list(roots)
            for e in obs['events']:
                possible=[v for v in before if v['root'].t<e.t and set(v['contexts'])&set(contexts)]
                memberships[e.eid]=[v['root'].eid for v in possible]
                if initiates(self.relations[e.r]):
                    v=dict(root=e,contexts=contexts,updates=[],previous=[v['root'].eid for v in possible])
                    roots.append(v)
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
        self.stats['pair_snapshots']+=1;self.stats['versions_indexed']+=len(roots)
        self.stats['multi_owner_observations']+=sum(len(x)>1 for x in memberships.values())
        self.cache[pair]=(stamp,roots,memberships)
        return roots,memberships

    def packets(self,index,q,c,short):
        pair=tuple(sorted((q.known,c)))
        seed=self.seed_event(index,q,c,short)
        if seed is None:return []
        observations=index.observations.get(pair,())
        current=next((o for o in reversed(observations) if o['time']==seed.t),None)
        if not current:return []
        roots,memberships=self.versions(index,pair)
        active=self.active_context(index,pair,current)
        result=[]
        for v in reversed(roots):
            root=v['root']
            if root.t>seed.t:continue
            shared=set(v['contexts'])&set(active)
            if not shared:continue
            k=min(shared,key=lambda x:(index.anchor_days[pair][x[0]],x))
            fs={e.eid:e for e in (root,seed)+v['contexts'][k]+active[k]}
            updates=[u for u in v['updates'] if k in u['contexts'] and u['event'].t<=seed.t]
            latest=updates[-1] if updates else None
            if latest:
                for e in (latest['event'],)+latest['contexts'][k]:fs[e.eid]=e
            previous=[x for x in roots if x['root'].eid in v['previous'] and k in x['contexts']]
            if previous:
                prev=previous[-1]
                preceding=[u for u in prev['updates'] if k in u['contexts'] and u['event'].t<root.t]
                if preceding:
                    for e in (prev['root'],preceding[-1]['event'])+prev['contexts'][k]+preceding[-1]['contexts'][k]:fs[e.eid]=e
            edges=[]
            for e in sorted(fs.values()):
                parents=[p for p in memberships.get(e.eid,()) if p in fs and index.store.events[p].t<e.t]
                if parents:edges.append(dict(event=e.eid,parents=parents,effect='possible_membership',key=[]))
            result.append(dict(seed=seed.eid,root=root.eid,candidate=c,context=list(k),
                effect='unresolved_observations' if updates else 'intention_observed',
                facts=sorted(fs.values()),edges=edges,
                alternatives=memberships.get(latest['event'].eid,[]) if latest else v['previous'],
                selection_key=None))
        return result

    def retrieve(self,index,q,short,candidates,baseline,length=100,policy='replace'):
        index.store.assert_past(q.t)
        selected={e.eid:e for e in baseline if e.r==q.r and q.known in (e.s,e.o)}
        used={interaction_key(e) for e in short+list(selected.values())}
        available={c:self.packets(index,q,c,short) for c in candidates}
        accepted=[];depth=0
        # Round-robin preserves complete evidence packets across candidates.
        while any(depth<len(ps) for ps in available.values()):
            for c in candidates:
                ps=available[c]
                if depth>=len(ps):continue
                p=ps[depth];add={interaction_key(e):e for e in p['facts'] if interaction_key(e) not in used}
                if not add or len(selected)+len(add)>length:continue
                selected.update({e.eid:e for e in add.values()});used.update(add);accepted.append(p)
            depth+=1
            if len(selected)>=length:break
        if not accepted:return baseline,dict(packets=[],policy=policy)
        for e in reversed(baseline):
            if len(selected)>=length:break
            if interaction_key(e) not in used:selected[e.eid]=e;used.add(interaction_key(e))
        for e in reversed(index.directed.get((q.known,q.direction),())):
            if len(selected)>=length:break
            other=e.o if q.direction=='tail' else e.s
            if other in candidates and interaction_key(e) not in used:
                selected[e.eid]=e;used.add(interaction_key(e))
        trace=[{**{k:v for k,v in p.items() if k!='facts'},'facts':[e.eid for e in p['facts']]} for p in accepted]
        return sorted(selected.values()),dict(packets=trace,policy=policy)

def render(q,short,long,candidates,trace,annotated=False):
    text=flat_render(q,short,long,candidates)
    if not annotated or not trace['packets']:return text
    visible={e.eid:e for e in short+long};notes=[]
    for p in trace['packets']:
        if not set(p['facts'])<=set(visible):continue
        notes.append('Intention version {}: candidate {}; observed facts [{}]. Possible shared context {}, ownership and completion unknown.'.format(
            p['root'],p['candidate'],'; '.join(p['facts']),p['context'][0]))
    return ('\n'.join(notes)+'\n' if notes else '')+text

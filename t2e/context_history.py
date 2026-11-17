"""Observable, overlapping participant episodes. No fitted relation rules.

Whole-day updates only. A context is a witnessed hypothesis, not a case ID.
The answer pool stays one-hop; the evidence includes candidate--participant
edges. All those edges consume the same 100-fact long-memory budget.
"""
from collections import Counter, defaultdict

from .data import Event, batches, digest
from .ownership_history import interaction_key, event_line
from .transaction_history import TransactionHistory

MIRROR = {4: 5, 5: 4, 1: 1, 12: 12, 19: 19, 27: 27}


def facing(e, q, equivalent=True):
    forward = e.s == q.known if q.direction == 'tail' else e.o == q.known
    if not forward and q.known in (e.s, e.o) and equivalent and e.r in MIRROR:
        return Event(e.t, e.o, MIRROR[e.r], e.s)
    return e


def witness_signature(e, endpoint):
    """Date, relation and role, independent of the third participant's ID."""
    if e.s==endpoint:return (e.t,e.r,'out')
    if e.r in MIRROR:return (e.t,MIRROR[e.r],'out')
    return (e.t,e.r,'in')


class ContextHistory(TransactionHistory):
    def __init__(self, relation_count, window=168, shared_short=False, same_event_control=False):
        super().__init__(relation_count, contexts=False)
        self.window = window
        self.shared_short = shared_short
        self.same_event_control = same_event_control
        self.neighbors = defaultdict(dict)
        self.observations = defaultdict(list)
        self.runs = {}
        self.anchor_days = defaultdict(Counter)
        self.context_days = Counter()
        self.packet_cache = {}

    def common(self, a, b, at, inclusive=False):
        result = {}
        for z in self.neighbors[a].keys() & self.neighbors[b].keys():
            if z in (a, b):
                continue
            ea, eb = self.neighbors[a][z], self.neighbors[b][z]
            # Same-day ties are deterministic views, never temporal order.
            if min(ea.t, eb.t) >= at - self.window and (
                    max(ea.t, eb.t) <= at if inclusive else max(ea.t, eb.t) < at):
                result[z] = (ea, eb)
        return result

    def observe(self, events):
        events = tuple(sorted(set(events)))
        if not events:
            return
        super().observe(events)
        self.packet_cache.clear()
        t = events[0].t
        pairs = defaultdict(list)
        for e in events:
            pairs[e.pair].append(e)
        for (a, b), es in pairs.items():
            # Retain one source row for a neighbor pair at this date. Its ID
            # remains auditable; selecting a view does not invent an event.
            representative = min(es, key=lambda e: (str(interaction_key(e)), e.eid))
            self.neighbors[a][b] = representative
            self.neighbors[b][a] = representative
        for pair, es in pairs.items():
            if pair[0] == pair[1]:
                continue
            witness = self.common(*pair, at=t, inclusive=True)
            owners = {}
            for z in witness:
                old = self.runs.get((pair, z))
                start = old[0] if old and t - old[1] <= self.window else t
                self.runs[pair, z] = (start, t)
                owners[z] = start
                self.anchor_days[pair][z] += 1
            if witness:
                self.context_days[pair] += 1
            self.observations[pair].append(dict(time=t, events=tuple(es),
                                                witness=witness, owners=owners))

    def candidate_packets(self, q, c, short_keys, mode='connected'):
        cache_key = (q.t, q.known, q.direction, c, mode)
        if cache_key in self.packet_cache:
            return self.packet_cache[cache_key]
        pair = tuple(sorted((q.known, c)))
        now = self.common(*pair, at=q.t)
        active = {}
        for z, evidence in now.items():
            count = self.anchor_days[pair][z]
            # An anchor present on every observed contextual date cannot
            # distinguish this pair's competing contexts. Missing contexts
            # are not counted as negative observations.
            if not 0 < count < self.context_days[pair]:
                continue
            old = self.runs.get((pair, z))
            if old and (mode == 'connected' and q.t - old[1] <= self.window):
                active[z] = old[0]
            elif old and mode == 'anchored':
                active[z] = old[0]
        if not active:
            self.packet_cache[cache_key] = []
            return []
        result = []
        records = self.observations[pair]
        for obs in reversed(records):
            zs = sorted(z for z, start in active.items()
                        if z in obs['owners'] and obs['owners'][z] == start)
            if not zs:
                continue
            seen = set()
            for e in sorted(obs['events'], key=lambda e: (facing(e, q).s != q.known, e)):
                v = facing(e, q)
                if (v.s if q.direction == 'tail' else v.o) != q.known:
                    continue
                key = interaction_key(e)
                if key in seen or (key in short_keys and not self.shared_short):
                    continue
                seen.add(key)
                # A physical fact may have several possible memberships.
                for z in zs:
                    owner = (c, z, active[z])
                    old_witness = obs['witness'][z]
                    same = dict(main=e, facts=(e,) + old_witness + now[z],
                                owner=owner, old_time=obs['time'],
                                historical_witness=old_witness, current_witness=now[z])
                    if self.same_event_control:
                        signature=tuple(witness_signature(w,endpoint) for w,endpoint in zip(old_witness,pair))
                        options=[]
                        for zz,ws in obs['witness'].items():
                            if zz in now or not 0 < self.anchor_days[pair][zz] < self.context_days[pair]:continue
                            if tuple(witness_signature(w,endpoint) for w,endpoint in zip(ws,pair))!=signature:continue
                            options.append((zz,ws,obs['owners'][zz]))
                        if options:
                            zz,ws,start=min(options)
                            same['foil']=dict(main=e,facts=(e,)+ws+now[z],owner=(c,zz,start),
                                historical_witness=ws,current_witness=now[z],old_time=e.t)
                        result.append(same)
                        continue
                    foils = []
                    for other in records:
                        if not other['owners'] or set(other['owners']) & set(active):
                            continue
                        for f in other['events']:
                            fv = facing(f, q)
                            if (fv.s, fv.r, fv.o) != (v.s, v.r, v.o):
                                continue
                            if interaction_key(f) in short_keys:
                                continue
                            for zz in other['owners']:
                                if not 0 < self.anchor_days[pair][zz] < self.context_days[pair]:
                                    continue
                                foils.append((abs(f.t-e.t), -f.t, zz, f,
                                              other['witness'][zz], other['owners'][zz]))
                    if foils:
                        _, _, zz, f, ws, start = min(foils)
                        same['foil'] = dict(main=f, facts=(f,) + ws + now[z],
                            owner=(c, zz, start), historical_witness=ws,
                            current_witness=now[z], old_time=f.t)
                    result.append(same)
        # One packet per actual interaction, retaining all ownership choices
        # in the index; the least ubiquitous anchor supplies the first view.
        result.sort(key=lambda p: (-p['main'].t,
                    self.anchor_days[pair][p['owner'][1]], p['owner']))
        self.packet_cache[cache_key] = result
        return result

    def retrieve(self, q, short, candidates, length=100, mode='connected'):
        self.store.assert_past(q.t)
        ordinary = self.ordinary_long(q, short, candidates, length)
        short_keys = {interaction_key(e) for e in short}
        protected = {e for e in ordinary if e.r == q.r}
        same = set(protected)
        sk = {interaction_key(e) for e in same} | short_keys
        packets = {c: self.candidate_packets(q, c, short_keys, mode) for c in candidates}
        selected = []
        memberships = defaultdict(set)
        definitions = {}
        pending = []

        def fresh(facts, used):
            chosen = {}
            for e in sorted(set(facts)):
                key = interaction_key(e)
                if key not in used:
                    chosen.setdefault(key, e)
            return set(chosen.values())

        depth = 0
        while any(depth < len(ps) for ps in packets.values()):
            for c in candidates:
                ps = packets[c]
                if depth >= len(ps):
                    continue
                packet = ps[depth]
                add = fresh(packet['facts'], sk)
                if not add or len(same) + len(add) > length:
                    continue
                same.update(add)
                sk.update(interaction_key(e) for e in add)
                owner = '{}:{}:{}'.format(*packet['owner'])
                definitions[owner] = dict(candidate=c, participant=packet['owner'][1],
                                          observed_start=packet['owner'][2])
                # Map equivalent source views to the row actually selected.
                chosen_by_key = {interaction_key(e): e for e in same}
                for e in packet['facts']:
                    stored = chosen_by_key.get(interaction_key(e))
                    if stored:
                        memberships[stored.eid].add(owner)
                pending.append(packet)
            depth += 1
            if len(same) >= length:
                break
        if not pending:
            return ordinary, ordinary, dict(packets=[],memberships={},definitions={},protected=[])
        allowed = set(candidates)
        cutoff = min(e.t for e in short) if short else q.t
        for e in reversed(self.directed.get((q.known,q.direction),())):
            other = e.o if q.direction == 'tail' else e.s
            key = interaction_key(e)
            if e.t >= cutoff or other not in allowed or key in sk:
                continue
            if len(same) >= length:
                break
            same.add(e);sk.add(key)
        # Build the negative control AFTER the main result is frozen. Only
        # exclusive packet facts can be replaced, so another packet's actual
        # evidence and the protected ordinary facts cannot be removed.
        different=set(same);dk={interaction_key(e) for e in different}|short_keys
        usage=Counter(k for p in pending for k in {interaction_key(e) for e in p['facts']})
        protected_keys={interaction_key(e) for e in protected}|short_keys
        control_witnesses=set()
        encode = lambda p: dict(main=p['main'].eid,owner=list(p['owner']),
            facts=[e.eid for e in p['facts']],
            historical_witness=[e.eid for e in p['historical_witness']],
            current_witness=[e.eid for e in p['current_witness']])
        for packet in pending:
            foil=packet.get('foil');matched=False
            oldkeys={interaction_key(e) for e in (packet['main'],)+packet['historical_witness']}
            oldkeys-={interaction_key(e) for e in packet['current_witness']}
            if self.same_event_control and foil:
                oldkeys-={interaction_key(e) for e in foil['facts']}
            main_replace_ok=self.same_event_control or interaction_key(packet['main']) in oldkeys
            if (foil and oldkeys and main_replace_ok
                    and not oldkeys & (protected_keys|control_witnesses)
                    and oldkeys<=dk and all(usage[k]==1 for k in oldkeys)):
                add=fresh(foil['facts'],dk-oldkeys)
                if len(add)==len(oldkeys):
                    different={e for e in different if interaction_key(e) not in oldkeys}
                    different.update(add);dk={interaction_key(e) for e in different}|short_keys
                    control_witnesses.update(interaction_key(e) for e in foil['facts'])
                    matched=True
            selected.append(dict(same=encode(packet),different=encode(foil if matched else packet),matched=matched))
        assert len(same) == len(different) <= length
        return sorted(same), sorted(different), dict(packets=selected,
            memberships={k:sorted(v) for k,v in memberships.items()},definitions=definitions,
            protected=[e.eid for e in sorted(protected)])


def shuffled_memberships(qid, memberships):
    ids = sorted(k for k,v in memberships.items() if v)
    values = [memberships[k] for k in ids]
    if len({tuple(v) for v in values}) < 2:
        return dict(memberships)
    # Permute membership assignments, not just the names of group IDs.
    order = sorted(range(len(ids)),key=lambda i:digest([qid,ids[i],'ownership-control']))
    rotated = values[:]
    for a,b in zip(order,order[1:]+order[:1]):
        rotated[a] = values[b]
    return dict(zip(ids,rotated))


def render(q, short, long, candidates, trace=None, attribution=False, shuffle=False,
           equivalent=True):
    labels = {c:i for i,c in enumerate(candidates)}
    trace = trace or dict(memberships={},definitions={})
    assignments = trace['memberships']
    if shuffle:
        assignments = shuffled_memberships(q.qid, assignments)
    names = {key:'T'+str(i) for i,key in enumerate(sorted(trace['definitions']))}
    text = []
    if attribution and names:
        for key,name in names.items():
            d = trace['definitions'][key]
            text.append('{}: entities {},{}; participant {}; observed since {}.\n'.format(
                name,q.known,d['candidate'],d['participant'],d['observed_start']))
    for e in sorted(long):
        v = facing(e,q,equivalent)
        if q.known in (v.s,v.o):
            line = event_line(q,v,labels).rstrip()
        else:
            a,b = (v.s,v.o) if q.direction == 'tail' else (v.o,v.s)
            line = '{}:[{},{},{}.{}]'.format(v.t,a,v.r,labels.get(b,'_'),b)
        if attribution and e.eid in assignments:
            line += ' ['+'|'.join(names[k] for k in assignments[e.eid])+']'
        text.append(line+'\n')
    # Short history stays byte-identical, including its original views.
    text.extend(event_line(q,e,labels) for e in sorted(short))
    text.append('{}:[{},{},'.format(q.t,q.known,q.r))
    return ''.join(text)

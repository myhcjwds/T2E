"""Read the intersection of observation days across possible intention roots.

Possible ownership remains in the provenance ledger. Reading a day does not
claim fulfillment; omitting it from this query does not delete its facts.
"""
from .transaction_day_memory import TransactionDayMemory
from .event_ledger import initiates
from collections import Counter
from fractions import Fraction

TAU = "0.75"


def alternative_day_reads(pack,relations):
    roots=sorted({int(r.split(':')[0]) for r in pack['roots']})
    seed_time=int(pack['seed'].split(':')[0])
    pair=next(e.pair for e in pack['facts'] if e.eid==pack['seed'])
    update_days={e.t for e in pack['facts'] if e.pair==pair and not initiates(relations[e.r])}
    paths=[]
    for start in roots:
        dates={start,seed_time}
        after=[t for t in update_days if start<t<=seed_time]
        before=[t for t in update_days if roots[0]<t<start]
        if after:dates.add(max(after))
        if before:dates.add(max(before))
        paths.append(dict(root_day=start,days=sorted(dates)))
    assert paths and set().union(*(set(p['days']) for p in paths))==set(pack['read_days'])
    ratio=Fraction(str(TAU))
    counts=Counter(day for path in paths for day in set(path['days']))
    stable={day for day,n in counts.items() if n*ratio.denominator>=ratio.numerator*len(paths)}
    return paths,stable,pair


class StableReadMemory(TransactionDayMemory):
    def packets(self,index,q,c,short):
        result=[]
        for pack in super().packets(index,q,c,short):
            paths,stable,pair=alternative_day_reads(pack,self.relations)
            epoch=tuple(pack['context']);fs={}
            for e in pack['facts']:
                if e.pair==pair and e.t in stable:fs[e.eid]=e
            for day in stable:
                for e in self.lineage_contexts[pair][day][epoch]:fs[e.eid]=e
            result.append(dict(pack,facts=sorted(fs.values()),
                possible_routing_facts=[e.eid for e in pack['facts']],
                alternative_day_reads=paths,stable_read_days=sorted(stable),
                read_semantics='quorum_of_possible_root_day_reads',root_day_quorum=str(TAU)))
        return result

"""Read complete observed days at a possible transaction's selected boundaries.

Rows on the same day have no known temporal ordering. All are possible
co-observations, not one asserted real-world matter or an inferred state.
The unchanged allocator either fits the entire packet or skips it.
"""
from collections import defaultdict
from .witness_snapshot_memory import WitnessSnapshotMemory


class TransactionDayMemory(WitnessSnapshotMemory):
    def __init__(self, relations):
        super().__init__(relations)
        self.day_read_cache={};self.day_read_time=None

    def packets(self,index,q,c,short):
        if self.day_read_time!=q.t:
            self.day_read_cache.clear();self.day_read_time=q.t
        pair=tuple(sorted((q.known,c)))
        key=(q.known,q.direction,c,tuple(e.eid for e in short if e.pair==pair))
        if key in self.day_read_cache:return self.day_read_cache[key]
        packets=super().packets(index,q,c,short)
        byday=defaultdict(list)
        for e in index.store.pairs.get(pair,()):
            assert e.t<q.t
            byday[e.t].append(e)
        result=[]
        for pack in packets:
            days=sorted({e.t for e in pack['facts'] if e.pair==pair})
            fs={e.eid:e for e in pack['facts']}
            for day in days:
                for e in byday[day]:fs[e.eid]=e
            result.append(dict(pack,facts=sorted(fs.values()),read_days=days,
                read_semantics='complete_coobservations_without_within_day_order'))
        self.day_read_cache[key]=result
        return result

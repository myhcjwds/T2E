"""A new interaction in one context does not erase another context's entry.

The short history is unchanged. Each witnessed lineage represented in it keeps
its latest pair interaction as retrieval seed. This is observable coverage, not
a claim that every earlier real-world matter remains active or unfinished.
"""
from .projected_memory import ProjectedMemory


class ParallelMemory(ProjectedMemory):
    def __init__(self,relations):
        super().__init__(relations)
        self.parallel_cache={};self.parallel_day=None

    def packets(self,index,q,c,short):
        if self.parallel_day!=q.t:
            self.parallel_cache.clear();self.parallel_day=q.t
        pair=tuple(sorted((q.known,c)))
        seeds=sorted((e for e in short if e.pair==pair),reverse=True)
        key=(q.known,q.direction,c,tuple(e.eid for e in seeds))
        if key in self.parallel_cache:return self.parallel_cache[key]
        seen=set();result=[]
        for seed in seeds:
            for p in super().packets(index,q,c,[seed]):
                context=tuple(p['context'])
                if context in seen:continue
                seen.add(context);result.append(p)
        self.parallel_cache[key]=result
        self.stats['parallel_context_packets']+=len(result)
        return result

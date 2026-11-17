"""Build the complete evidence input in one chronological pass."""
from .data import batches, queries_and_labels
from .schema import configure, ordered_candidates
from .direct_history_candidates import history_candidates
from .simplified_ranking import calibration_terms


def plans(data, config, precision, tokenizer):
    physical = configure(data.relations, config)
    from .context_history import ContextHistory
    from .witness_snapshot_memory import WitnessSnapshotHistory
    from .stable_read_memory import StableReadMemory
    from .event_ledger import render
    from .first_hop_witness_read import first_hop_witness_days, extend_first_hop_witness
    from .chronological_episode_plan import chronological_plan
    from .witness_recency_control import ordinary_days, extend_ordinary_days
    from .short_branch_scope import branch_day_contexts, scoped_plan
    from .long_branch_scope import long_day_contexts, long_scoped_plan
    from .post_entry_pair_read import post_entry_day, fill_pair_days, pair_read_plan
    from .branch_boundary_read import boundary_day, boundary_plan
    from .query_supported_candidate import query_supported_options, admit_candidates, query_supported_plan
    from .fixed_candidate_day_control import latest_options, controlled_plan

    ordinary = ContextHistory(len(data.relations))
    index = WitnessSnapshotHistory(len(data.relations))
    ordinary.extend(data.prefix('test'))
    index.extend(data.prefix('test'))
    memory = StableReadMemory(data.relations)
    token_count = lambda text: len(tokenizer.encode(text, add_special_tokens=True))
    events = index.store.events

    for _, batch in batches(data.splits['test']):
        queries, labels = queries_and_labels(batch)
        for q in queries:
            pool, _ = history_candidates(q, index.store, config['fusion_spec'], precision, len(data.relations))
            candidates, witnesses = ordered_candidates(q, index.store, set(pool))
            short = index.short(q, 100)
            baseline = ordinary.retrieve(q, short, candidates, 100, 'connected')[0]
            long, trace = memory.retrieve(index, q, short, candidates, baseline, 100, 'replace')
            text = render(q, short, long, candidates, trace, False)
            terms, floor = calibration_terms(q, index.store, candidates, config, precision, len(data.relations))
            p = dict(query_id=q.qid, query=q.as_dict(), candidate_ids=candidates,
                     candidate_witnesses=witnesses, short=[e.eid for e in short], terms=terms,
                     floor=floor, targets=sorted(labels[q]),
                     long=dict(baseline=[e.eid for e in baseline], ledger=[e.eid for e in long]),
                     trace=trace, prompts=dict(ledger=dict(text=text, tokens=token_count(text))))

            # Retain the same accepted witness slots, then read ordinary pair days.
            options = []
            for packet in trace['packets']:
                options.extend(first_hop_witness_days(q, packet, candidates, events, index.store.pairs, physical))
            expanded, extra = extend_first_hop_witness(q, short, long, trace, options, events, physical)
            first = chronological_plan(q, p, short, expanded, extra, token_count)
            options = ordinary_days(q, extra.get('first_hop_witness_reads', []), candidates,
                                    events, index.store.pairs, physical)
            expanded, extra = extend_ordinary_days(q, short, long, trace, options, events, physical)
            n = chronological_plan(q, p, short, expanded, extra, token_count)
            if n['long'] == first['long'] and n['prompts'] == first['prompts']:
                n = first

            # Keep compatible pair-day groups in both evidence budgets.
            short = [events[e] for e in n['short']]
            long = [events[e] for e in n['long']['ledger']]
            contexts = branch_day_contexts(index, memory, q, short, n['trace'])
            n = scoped_plan(q, n, short, long, contexts, events, physical, token_count)
            short = [events[e] for e in n['short']]
            long = [events[e] for e in n['long']['ledger']]
            contexts = long_day_contexts(index, memory, q, long, n['trace'])
            n = long_scoped_plan(q, n, short, long, contexts, events, physical, token_count)

            for option_builder, make_plan in ((post_entry_day, pair_read_plan), (boundary_day, boundary_plan)):
                options = []
                for pack in n['trace']['packets']:
                    pair = tuple(sorted((q.known, pack['candidate'])))
                    memory.versions(index, pair)
                    option = option_builder(q, pack, memory.lineage_contexts[pair], events, index.store.pairs, physical)
                    if option is not None:
                        options.append(option)
                selected, accepted = fill_pair_days(q, n, options, events, physical)
                n = make_plan(q, n, selected, accepted, events, token_count)

            options = query_supported_options(q, n, ordinary, physical)
            accepted, limit = admit_candidates(n, options)
            n = query_supported_plan(q, n, options, accepted, limit, events, physical,
                lambda cs: calibration_terms(q, index.store, cs, config, precision, len(data.relations)), token_count)
            options = latest_options(q, n, ordinary, physical)
            n, _ = controlled_plan(q, n, options, events, physical, token_count)

            assert len(n['short']) <= 100 and len(n['long']['ledger']) <= 100
            assert n['prompts']['ledger']['tokens'] + 16 <= config['max_context_tokens']
            assert all(events[e].t < q.t for e in n['short'] + n['long']['ledger'])
            assert all(q.known in events[e].pair for e in n['long']['ledger'])
            yield n
        ordinary.observe(batch)
        index.observe(batch)

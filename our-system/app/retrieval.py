"""Raw FTS, exact user-local Dense, and bounded RRF with explicit degradation."""

from __future__ import annotations

import inspect
import logging
import json
import time
from dataclasses import dataclass
from dataclasses import replace
from typing import Protocol
from pathlib import Path

from app.config import Settings
from app.embeddings import EmbeddingFailure, EmbeddingProvider, normalize_vector, provider_version
from app.index import build_match_query
from app.models import EvidenceCandidate, SearchHit, SearchRequest, SearchResponse
from app.safety import filter_evidence
from app.governance import temporal_view
from app.store import MessageStore
from app.jobs import NotReady, SnapshotChanged


class LifecycleSnapshotBusy(RuntimeError):
    """A stable semantic selection/source snapshot could not be obtained."""

logger = logging.getLogger("aml.retrieval")


@dataclass(frozen=True)
class QueryPlan:
    user_id: str
    query_text: str
    options: tuple[str, ...]
    temporal_intent: str = "none"


class GraphCandidateProvider(Protocol):
    def search(self, plan: QueryPlan, user_id: str) -> list[EvidenceCandidate]:
        """Append-only graph candidates. Phase A does not call this."""


def parse_query(request: SearchRequest) -> QueryPlan:
    options = tuple(request.options or ())
    return QueryPlan(user_id=request.user_id, query_text=request.query, options=options)


def fuse_lexical(candidates: list[EvidenceCandidate]) -> list[EvidenceCandidate]:
    """Keep lexical order for callers using the single route."""

    return list(candidates)


def reciprocal_rank_fusion(lexical: list[EvidenceCandidate], dense: list[EvidenceCandidate], k: int):
    candidates: dict[str, EvidenceCandidate] = {}
    ranks: dict[str, dict[str, int]] = {}
    for route, hits in (("lexical", lexical), ("dense", dense)):
        seen: set[str] = set()
        for rank, hit in enumerate(hits, 1):
            if hit.id in seen:
                continue
            seen.add(hit.id)
            candidates.setdefault(hit.id, hit)
            ranks.setdefault(hit.id, {})[route] = rank
    fused = [hit.model_copy(update={"score": sum(1.0 / (k + rank) for rank in ranks[hit.id].values())})
             for hit in candidates.values()]
    fused.sort(key=lambda hit: (-hit.score, hit.message_index, hit.id))
    trace = [{"raw_id": hit.id, "ranks": ranks[hit.id], "score": hit.score} for hit in fused]
    return fused, trace


def _complete_raw(hit: EvidenceCandidate) -> bool:
    return hit.representation == "raw" or (
        hit.id == hit.source_raw_id and hit.span_start is None and hit.span_end is None)


def _span_inside_raw(raw: EvidenceCandidate, span: EvidenceCandidate) -> bool:
    start, end = span.span_start, span.span_end
    if start is None or end is None or not (0 <= start < end <= len(raw.content)):
        return False
    return raw.content[start:end] == span.content


def aggregate_source_occupancy(candidates: list[EvidenceCandidate]):
    """After route fusion, one selected complete Raw occupies its covered spans.

    Uncovered spans of the same source stay. Distinct sources are independent.
    Covered spans donate their ranking scores to the Raw and stop taking a slot.
    """
    groups: dict[str, list[EvidenceCandidate]] = {}
    for hit in candidates:
        groups.setdefault(hit.source_raw_id or hit.id, []).append(hit)
    drop: set[str] = set()
    extra_score: dict[str, float] = {}
    for hits in groups.values():
        raws = [hit for hit in hits if _complete_raw(hit)]
        if not raws:
            continue
        raws.sort(key=lambda hit: (-hit.score, hit.id))
        primary = raws[0]
        extra = 0.0
        for other in raws[1:]:
            drop.add(other.id)
            extra += other.score
        for span in hits:
            if span.id == primary.id or _complete_raw(span):
                continue
            if _span_inside_raw(primary, span):
                drop.add(span.id)
                extra += span.score
        if extra:
            extra_score[primary.id] = extra
    if not drop:
        return list(candidates), {"dropped_ids": [], "kept": len(candidates)}
    kept = []
    for hit in candidates:
        if hit.id in drop:
            continue
        extra = extra_score.get(hit.id, 0.0)
        kept.append(hit if not extra else hit.model_copy(update={"score": hit.score + extra}))
    kept.sort(key=lambda hit: (-hit.score, hit.message_index, hit.id))
    return kept, {"dropped_ids": [hit.id for hit in candidates if hit.id in drop], "kept": len(kept)}


def _revocation(exc, **fields):
    trace = dict(getattr(exc, "dispatch_trace", None) or {})
    trace.update(fields)
    trace.setdefault("inflight_recall", False)
    exc.dispatch_trace = trace
    return exc


def budget_prefix(candidates: list[EvidenceCandidate], max_items: int, char_limit: int):
    kept: list[EvidenceCandidate] = []
    chars = 0
    for candidate in candidates:
        if len(kept) >= max_items or chars + len(candidate.content) > char_limit:
            break
        kept.append(candidate)
        chars += len(candidate.content)
    return kept, chars


def pack_evidence(candidates,max_items,char_limit,policy="prefix",token_limit=None,renderer=None):
    if policy=="prefix" and token_limit is None and renderer is None:
        kept,chars=budget_prefix(candidates,max_items,char_limit)
        return kept,chars,{"skipped_budget":0,"policy":policy}
    from app.segments import token_count
    kept,chars,tokens,skipped=[],0,0,0
    for candidate in candidates:
        if len(kept)>=max_items:
            break
        text=renderer(candidate) if renderer else candidate.content
        count=token_count(text) if token_limit is not None else 0
        if chars+len(text)>char_limit or (token_limit is not None and tokens+count>token_limit):
            skipped+=1
            if policy=="prefix":
                break
            continue
        kept.append(candidate);chars+=len(text);tokens+=count
    return kept,chars,{"skipped_budget":skipped,"policy":policy,"packed_content_tokens":tokens if token_limit is not None else None}


class RetrievalService:
    def __init__(self, store: MessageStore, settings: Settings | None = None,
                 embedding_provider: EmbeddingProvider | None = None) -> None:
        self.store = store
        self.settings = settings or Settings(db_path=getattr(store, "db_path", Path(".")))
        self.embedding_provider = embedding_provider
        self.coordinator = None

    def search(self, request: SearchRequest) -> SearchResponse:
        response, _trace = self.search_with_trace(request)
        return response

    def search_with_trace(self, request: SearchRequest) -> tuple[SearchResponse, dict[str, object]]:
        from app.embedding_accounting import embedding_scope
        from uuid import uuid4
        with embedding_scope(request.user_id, uuid4().hex, 'search'):
            return self._search_scoped(request)

    def _search_scoped(self, request: SearchRequest) -> tuple[SearchResponse, dict[str, object]]:
        if self.coordinator is None:
            return self._search_once(request)
        discarded = []
        request_generation = None
        for attempt in range(self.coordinator.profile.identity['search_attempts']):
            stamp = self.coordinator.require_ready(request.user_id)
            if request_generation is None:
                request_generation = stamp[0]
            elif stamp[0] != request_generation:
                # A retry may reuse its query vector only within the original
                # configuration generation, including A -> B -> A changes.
                raise NotReady('query_config_generation_changed')
            try:
                return self._search_once(request, stamp=stamp, formal_attempt=attempt+1,
                                         prior_revocations=discarded)
            except SnapshotChanged as exc:
                record = dict(getattr(exc, "dispatch_trace", None) or {})
                record.setdefault("reason", str(exc))
                record.setdefault("inflight_recall", False)
                discarded.append(record)
                continue
        raise LifecycleSnapshotBusy("lifecycle changed during final evidence check")

    def _search_once(self, request, *, stamp=None, formal_attempt=1, prior_revocations=()):
        started = time.perf_counter()
        plan = parse_query(request)
        match_query = build_match_query(plan.query_text)
        settings, provider = self.settings, self.embedding_provider
        from app.queryplan import allocate_index_budget,plan_query,scale_option_queries,typed_order
        capability_plan=plan_query(plan.query_text,plan.options)
        adaptive_trace=None
        if settings.route_policy in {"adaptive_options","adaptive_options_dense"}:
            if settings.profile_id!="span_dense_sync":raise NotReady("adaptive_requires_complete_dense_profile")
            selected=bool(capability_plan.option_queries)
            semantic_mode="raw_dense" if settings.route_policy=="adaptive_options_dense" else "raw_hybrid_rrf"
            adaptive_trace={"input":"query_and_bounded_options_only","selected_mode":"raw_fts" if selected else semantic_mode,
                            "selected_policy":"options_typed" if selected else "none","catalog_or_gold_read":False}
            settings=replace(settings,route_policy=adaptive_trace["selected_policy"])
        effective_mode=adaptive_trace["selected_mode"] if adaptive_trace else settings.route_mode
        options_active=settings.route_policy in {"options","options_typed","pareto"} and bool(capability_plan.option_queries)
        option_total=sum(quota for _,quota in capability_plan.option_queries) if options_active else 0
        planned_hits=allocate_index_budget(settings.candidate_limit,options_active=options_active,
            sentence=settings.sentence_index,bridge=bool(settings.bridge_hops),atom=settings.enable_extractor,
            option_total=option_total)
        original_limit=planned_hits["original"]
        sentence_quota=planned_hits["sentence"]
        bridge_quota=planned_hits["bridge"]
        atom_quota=planned_hits["atom"]
        option_plan=scale_option_queries(capability_plan.option_queries,planned_hits["options"]) if options_active else ()
        lexical_quota=dense_quota=None
        if effective_mode=="raw_hybrid_rrf" and (options_active or settings.profile_id.startswith("cycle2_")):
            lexical_quota=original_limit//2
            dense_quota=original_limit-lexical_quota
        version = provider_version(provider) if provider is not None else "unconfigured"
        dimensions = provider.config.dimensions if provider is not None else 0
        query_vector = None
        reason = None
        usage: dict[str, object] = {"requests": 0, "monetary_cost": None}
        mode = effective_mode
        if mode != "raw_fts":
            if provider is None:
                reason = "missing_configuration"
            total, ready = self.store.vector_coverage(plan.user_id, version, dimensions,
                **({"span":True} if settings.fragment_policy == "span_exact_v1" else {}))
            if total:
                if provider is None:
                    reason = "missing_configuration"
                elif ready != total:
                    reason = "incomplete_current_version"
                else:
                    try:
                        if self.coordinator is not None:
                            def check_dispatch():
                                # Only the caller's query leaves here, never a
                                # saved memory. An intervening Add must not make
                                # this source-independent computation stale.
                                # Readiness/configuration checks still run before
                                # every dispatch (including provider retries).
                                nonlocal stamp
                                fresh_stamp = self.coordinator.require_ready(plan.user_id)
                                if fresh_stamp[0] != stamp[0]:
                                    raise NotReady('query_config_generation_changed')
                                stamp = fresh_stamp
                            check_dispatch()
                            batch = provider.embed_guarded([plan.query_text], before_dispatch=check_dispatch) if hasattr(provider, "embed_guarded") else provider.embed([plan.query_text])
                        else:
                            batch = provider.embed([plan.query_text])
                        usage = batch.usage
                        if len(batch.vectors) != 1:
                            raise EmbeddingFailure("response_count_mismatch", usage)
                        query_vector = normalize_vector(batch.vectors[0], dimensions)
                    except EmbeddingFailure as exc:
                        reason, usage = exc.code, exc.usage or usage
        if self.coordinator is not None and reason:
            raise NotReady(reason)
        if self.coordinator is not None:
            # Start the evidence snapshot after the slow remote computation.
            # Add200 before this point must be visible; changes while evidence
            # is selected still fail the unchanged final/source checks below.
            fresh_stamp = self.coordinator.require_ready(plan.user_id)
            if fresh_stamp[0] != stamp[0]:
                raise NotReady('query_config_generation_changed')
            stamp = fresh_stamp
        snapshot = self.store.search_snapshot(
            user_id=plan.user_id, match_query=match_query, query_vector=query_vector,
            version=version, dimensions=dimensions, mode=mode,
            candidate_limit=original_limit,
            lexical_quota=settings.lexical_quota if lexical_quota is None else lexical_quota,
            dense_quota=settings.dense_quota if dense_quota is None else dense_quota, degraded_reason=reason,
            **({"span":True} if settings.fragment_policy == "span_exact_v1" else {}),
            **({'exact_acceleration':True} if settings.exact_dense_acceleration else {}),
        )
        if self.coordinator is not None and snapshot.degraded_reason:
            if snapshot.degraded_reason == "query_vector_unavailable" and query_vector is None:
                # The user was empty during coverage checking, then its first
                # Add committed. Retry against that published route and obtain
                # a query vector; never turn it into a fake empty/FTS success.
                raise SnapshotChanged("first_add_during_query_preparation")
            raise NotReady(snapshot.degraded_reason)
        unknown_dropped=0
        if settings.unknown_dense_floor is not None:
            allowed=[h for h in snapshot.dense if h.score>=settings.unknown_dense_floor]
            unknown_dropped=len(snapshot.dense)-len(allowed)
            snapshot=replace(snapshot,dense=allowed)
        fusion_trace = []
        if snapshot.actual_mode == "raw_hybrid_rrf":
            fused, fusion_trace = reciprocal_rank_fusion(snapshot.lexical, snapshot.dense, settings.rrf_k)
        else:
            fused = list(snapshot.dense if snapshot.actual_mode == "raw_dense" else snapshot.lexical)
        option_trace=[]
        actual_hits={"original":len(snapshot.lexical)+len(snapshot.dense)+unknown_dropped,
                     "sentence":0,"bridge":0,"atom":0,"options":0}
        sentence_trace=[]
        atom_trace={'enabled':False}
        if atom_quota:
            extractor_version=self.coordinator.profile.identity['atom_extractor_version']
            fts_limit=atom_quota if query_vector is None else max(1,atom_quota//2)
            results=[self.store.atom_candidates(plan.user_id,plan.query_text,fts_limit,extractor_version)]
            if query_vector is not None and atom_quota>fts_limit:
                results.append(self.store.atom_dense_candidates(plan.user_id,query_vector,version,dimensions,
                    atom_quota-fts_limit,extractor_version))
            atom_trace={'enabled':True,'routes':[],'set_sufficient':'unknown'}
            for result in results:
                actual_hits["atom"]+=len(result.trace['matched_atoms'])+len(result.trace['invalid_atom_ids'])
                fused,_=reciprocal_rank_fusion(fused,result.candidates,settings.rrf_k)
                atom_trace['routes'].append({'route':result.trace['route'],'returned_raw_ids':result.trace['returned_raw_ids'],
                    'matched_atoms':[{'atom_id':a['atom_id'],'kind':a['kind'],
                        'source_valid':a['source_valid'],'field_supported':a['field_supported']} for a in result.trace['matched_atoms']],
                    'invalid_atom_ids':result.trace['invalid_atom_ids']})
        if sentence_quota:
            sentence_hits,sentence_trace=self.store.sentence_candidates(plan.user_id,match_query,sentence_quota)
            actual_hits["sentence"]+=len(sentence_trace)
            fused,_=reciprocal_rank_fusion(fused,sentence_hits,settings.rrf_k)
        bridge_trace=[]
        if bridge_quota and 'multi_hop' in capability_plan.intents:
            from app.evidence_navigation import bridge_seeds
            seen_seeds=[];hop_sources=fused
            remaining=bridge_quota
            for hop in range(settings.bridge_hops):
                if remaining<1:break
                legal_policy=self.store.policy_snapshot(plan.user_id,hop_sources)
                legal_seeds,_=filter_evidence(plan.user_id,hop_sources,legal_policy)
                seeds=bridge_seeds(legal_seeds,plan.query_text,seen_seeds)
                if not seeds:break
                # One source-grounded query per hop; no recent/all-user fallback.
                names=[seed[0] for seed in seeds];seen_seeds.extend(names)
                quota=max(1,remaining//(settings.bridge_hops-hop))
                extra=self.store.search_snapshot(user_id=plan.user_id,match_query=build_match_query(' '.join(names)),
                    query_vector=None,version=version,dimensions=dimensions,mode='raw_fts',candidate_limit=quota,
                    lexical_quota=quota,dense_quota=quota,span=True)
                actual_hits["bridge"]+=len(extra.lexical);remaining-=len(extra.lexical)
                fused,_=reciprocal_rank_fusion(fused,extra.lexical,settings.rrf_k)
                hop_sources=extra.lexical
                bridge_trace.append({'hop':hop+1,'seed_source_ids':[x[1] for x in seeds],
                    'seed_hashes':[__import__('hashlib').sha256(x.encode()).hexdigest() for x in names],
                    'quota':quota,'hit_ids':[h.id for h in extra.lexical]})
        if option_plan:
            auxiliary=[]
            for option_query,quota in option_plan:
                extra=self.store.search_snapshot(user_id=plan.user_id,match_query=build_match_query(option_query),
                    query_vector=None,version=version,dimensions=dimensions,mode="raw_fts",candidate_limit=quota,
                    lexical_quota=quota,dense_quota=quota,span=True)
                actual_hits["options"]+=len(extra.lexical)
                auxiliary,_=reciprocal_rank_fusion(auxiliary,extra.lexical,settings.rrf_k)
                option_trace.append({"quota":quota,"hit_ids":[h.id for h in extra.lexical]})
            fused,option_fusion=reciprocal_rank_fusion(fused,auxiliary,settings.rrf_k)
        fused,source_occupancy=aggregate_source_occupancy(fused)
        index_hits=sum(actual_hits.values())
        route_budget={"unit":"index_hit","candidate_limit":settings.candidate_limit,
                      "planned":planned_hits,"planned_total":sum(planned_hits.values()),
                      "actual":actual_hits,"actual_total":index_hits}
        if index_hits>settings.candidate_limit:raise NotReady("index_hit_budget_exceeded")
        view = temporal_view(plan.query_text)
        projection={};typed_trace={};window_trace={"groups":[]}
        if settings.claim_projection:
            fused,projection=self.store.project_span_claims(plan.user_id,fused,view)
        if settings.route_policy in {"typed","options_typed","pareto"}:
            fused,typed_trace=typed_order(fused,capability_plan)
        independent_ids={h.id for h in fused}
        if settings.route_policy in {"witness_window","pareto"} and capability_plan.window_reason:
            # Base index hits retain their ranking; the reserved expansion budget
            # admits only adjacency witnesses of already recalled legal sources.
            fused,window_trace=self.store.expand_span_windows(plan.user_id,fused,settings.candidate_limit)
        if settings.context_neighbor_radius:
            if settings.profile_id.startswith('cycle2_'):
                fused,navigation_trace=self.store.expand_evidence(plan.user_id,fused,parents=settings.parent_evidence,
                    radius=settings.context_neighbor_radius,limit=settings.candidate_limit)
            else:
                fused = self.store.context_neighbors(plan.user_id,fused,settings.context_neighbor_radius,settings.candidate_limit,
                                                    historical=view.intent in {"history","as_of"})
                navigation_trace=[]
        elif settings.parent_evidence:
            fused,navigation_trace=self.store.expand_evidence(plan.user_id,fused,parents=True,limit=settings.candidate_limit)
        else:
            navigation_trace=[]
        if settings.claim_projection:
            # Expansion cannot bypass the selected source-time view. Adjacency
            # is navigation, so disable a window when the view removes a
            # member; retain the remaining independently recalled sources.
            fused,projection=self.store.project_span_claims(plan.user_id,fused,view)
            excluded={d['candidate_id'] for d in projection['decisions'] if d['decision']=='omit'}
            downgraded=[g for g in window_trace['groups'] if excluded.intersection(g['member_ids'])]
            window_trace['groups']=[g for g in window_trace['groups'] if not excluded.intersection(g['member_ids'])]
            window_trace['disabled_by_temporal_view']=downgraded
            authorized={sid for group in window_trace['groups'] for sid in group['member_ids']}
            fused=[hit for hit in fused if hit.id in independent_ids or hit.id in authorized]
        seeds = fused
        for snapshot_attempt in range(3):
            fused = self.store.governed_candidates(plan.user_id,seeds,view,settings.candidate_limit) if settings.enable_governance else seeds
            policy = self.store.policy_snapshot(plan.user_id,fused,historical=view.intent in {"history","as_of"})
            if not settings.enable_governance or fused.lifecycle_epoch == policy.epoch:
                break
        else:
            raise LifecycleSnapshotBusy("lifecycle changed during evidence selection")
        gated, gate_log = filter_evidence(plan.user_id, fused, policy)
        from app.governance import knowledge_time_gate
        gated, knowledge_time_trace = knowledge_time_gate(plan.query_text, gated, view)
        capability_trace = {'enabled':False}
        if settings.capability_views:
            from app.capability_views import build_views, rank_views
            views, capability_trace = build_views(gated, plan.user_id)
            gated, ranking = rank_views(gated, views, capability_plan.intents)
            capability_trace.update(enabled=True, ranking=ranking)
        rerank_trace={'status':'disabled'}
        if getattr(self,'reranker',None) is not None:
            # Re-check the stamp and this pool before any scorer sees text. Injected
            # rerankers that do not accept `authorize` still get this one check.
            authorize=None
            if self.coordinator is not None:
                try:
                    self.coordinator.final_check(plan.user_id,stamp,gated,policy)
                except SnapshotChanged as exc:
                    raise _revocation(exc,stage="before_rerank",batches_dispatched=0,
                        batches_cancelled=1 if gated else 0,queued_not_sent=True,
                        dispatched_results_discarded=False,inflight_recall=False,
                        dispatch_unit="rerank_call")
                def authorize(batch,user_id=plan.user_id,stamp=stamp,policy=policy):
                    self.coordinator.final_check(user_id,stamp,batch,policy)
            kwargs={"user_id":plan.user_id,"groups":window_trace["groups"]}
            if authorize is not None:
                try:
                    params=inspect.signature(self.reranker.rerank).parameters
                except (TypeError,ValueError):
                    params={}
                if "authorize" in params or any(item.kind is inspect.Parameter.VAR_KEYWORD for item in params.values()):
                    kwargs["authorize"]=authorize
            gated,rerank_trace=self.reranker.rerank(plan.query_text,gated,**kwargs)
        invalid_derived_ids = list(getattr(fused,"invalid_derived_ids",()))
        from app.rendering import render_span_source, render_source_date, should_show_source_dates
        renderer=render_span_source if settings.source_labels else None
        dates_active = settings.temporal_source_dates and should_show_source_dates(plan.query_text)
        if dates_active:
            renderer = render_source_date
        if settings.metadata_roles or settings.metadata_dates:
            from app.rendering import render_metadata
            renderer=lambda candidate:render_metadata(candidate,roles=settings.metadata_roles,dates=settings.metadata_dates)
        atom_display_trace={'enabled':False}
        display_groups=()
        if settings.atom_display:
            display=self.store.atom_display_groups(plan.user_id,gated,
                self.coordinator.profile.identity['atom_extractor_version'])
            display_groups=display.groups
            window_trace['groups'].extend(display.support_groups)
            atom_display_trace=display.trace
            from app.atomic_memory import render_atoms_with_raw
            rendered_atoms=render_atoms_with_raw(gated,display_groups,raw_renderer=renderer)
            renderer=lambda candidate:rendered_atoms[candidate.id]
        if settings.set_selection:
            from app.set_selection import select_evidence
            kept,chars,packing=select_evidence(gated,plan.query_text,min(request.top_k,settings.max_return_items),
                settings.return_char_budget,settings.return_token_budget,renderer,window_trace['groups'])
        elif window_trace["groups"]:
            from app.support_groups import pack_groups
            kept,chars,packing=pack_groups(gated,window_trace["groups"],min(request.top_k,settings.max_return_items),
                                          settings.return_char_budget,settings.return_token_budget,renderer)
        else:
            kept, chars,packing = pack_evidence(gated,min(request.top_k,settings.max_return_items),settings.return_char_budget,
                                              settings.packing_policy,settings.return_token_budget,renderer)
        if self.coordinator is not None:
            self.coordinator.final_check(plan.user_id, stamp, kept, policy)
        if display_groups:
            # Complete group packing guarantees identical rendering for kept Raw.
            from app.atomic_memory import render_atoms_with_raw
            final_rendered=render_atoms_with_raw(kept,display_groups,
                raw_renderer=(lambda c:render_metadata(c,roles=settings.metadata_roles,dates=settings.metadata_dates))
                if settings.metadata_roles or settings.metadata_dates else
                render_source_date if dates_active else render_span_source if settings.source_labels else None)
            if any(final_rendered[h.id]!=renderer(h) for h in kept):
                raise NotReady('incomplete_atom_display_group')
        trace = {
            "adaptive_route":adaptive_trace,
            "sentence_navigation":sentence_trace,"parent_navigation":navigation_trace,
            "bridge_navigation":bridge_trace,
            "atom_navigation":atom_trace,"reranking":rerank_trace,
            "atom_display":atom_display_trace,
            "requested_mode": mode, "actual_mode": snapshot.actual_mode,
            "degraded": snapshot.degraded_reason is not None or bool(invalid_derived_ids),
            "degraded_reason": snapshot.degraded_reason or ("invalid_derived_source" if invalid_derived_ids else None),
            "invalid_derived_ids":invalid_derived_ids,
            "embedding_version": version, "embedding_implementation": getattr(provider, "implementation_kind", "unavailable"),
            "active_rows": snapshot.active_count, "ready_rows": snapshot.ready_count,
            "candidate_limit": settings.candidate_limit,
            "lexical_quota": settings.lexical_quota, "dense_quota": settings.dense_quota,
            "lexical_ids": [item.id for item in snapshot.lexical],
            "dense_ids": [item.id for item in snapshot.dense],
            "candidates_before_dedup": len(snapshot.lexical) + len(snapshot.dense),
            "rrf_k": settings.rrf_k, "fusion": fusion_trace,
            "returned_ids": [item.id for item in kept], "output_characters": chars,
            "fragment_policy": settings.fragment_policy,
            "exact_dense_acceleration": settings.exact_dense_acceleration,
            "temporal_source_dates": {"enabled":settings.temporal_source_dates,"triggered":dates_active,
                                      "meaning":"source recording time, not inferred event time"},
            "capability_views": capability_trace,
            "knowledge_time_gate": knowledge_time_trace,
            "capability_plan":{"intents":capability_plan.intents,"temporal_intent":capability_plan.temporal_intent,
                               "options_reason":capability_plan.options_reason,"window_reason":capability_plan.window_reason},
            "optional_routes":{"policy":settings.route_policy,"options":option_trace,"index_hits":index_hits,
                               "route_budget":route_budget,
                               "typed":typed_trace,"windows":window_trace,"claims":projection,
                               "unknown_dense_dropped":unknown_dropped,"source_labels":settings.source_labels},
            "source_occupancy":source_occupancy,
            "evidence": [{key:value for key,value in item.model_dump().items() if key not in {"content","score"}}
                         for item in kept],
            "governance_enabled": settings.enable_governance, "temporal_view": view.intent,
            "context_neighbor_radius":settings.context_neighbor_radius,
            "lifecycle_epoch": policy.epoch, "gate_decisions": gate_log,
            "snapshot_attempts":formal_attempt if self.coordinator is not None else snapshot_attempt+1,
            "profile_id":settings.profile_id,
            "return_char_budget": settings.return_char_budget,
            "return_token_budget":settings.return_token_budget,"packing":packing,
            "messages_omitted": len(gated) - len(kept), "query_embedding_usage": usage,
            "query_embedding_usage_scope": "last_snapshot_attempt; all dispatches are in the embedding ledger",
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        if prior_revocations:
            trace["dispatch_revocation"] = list(prior_revocations)
        # Raw IDs and counters only. Never log input text, credentials or HTTP bodies.
        logger.info("retrieval_trace %s", json.dumps(trace, sort_keys=True))
        response = SearchResponse(
            data=[
                SearchHit(
                    id=candidate.id,
                    content=renderer(candidate) if renderer else candidate.content,
                    score=candidate.score,
                    created_at=candidate.created_at,
                )
                for candidate in kept
            ]
        )
        return response, trace

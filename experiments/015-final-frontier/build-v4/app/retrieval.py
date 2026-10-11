"""Raw FTS, exact user-local Dense, and bounded RRF with explicit degradation."""

from __future__ import annotations

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
        if self.coordinator is None:
            return self._search_once(request)
        for attempt in range(3):
            stamp = self.coordinator.require_ready(request.user_id)
            try:
                return self._search_once(request, stamp=stamp, formal_attempt=attempt+1)
            except SnapshotChanged:
                continue
        raise LifecycleSnapshotBusy("lifecycle changed during final evidence check")

    def _search_once(self, request, *, stamp=None, formal_attempt=1):
        started = time.perf_counter()
        plan = parse_query(request)
        match_query = build_match_query(plan.query_text)
        settings, provider = self.settings, self.embedding_provider
        from app.queryplan import plan_query,typed_order
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
        original_limit=100 if options_active else settings.candidate_limit
        lexical_quota=dense_quota=50 if options_active and effective_mode=="raw_hybrid_rrf" else None
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
                                if self.coordinator.require_ready(plan.user_id) != stamp:
                                    raise SnapshotChanged("query_dispatch_snapshot_changed")
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
        snapshot = self.store.search_snapshot(
            user_id=plan.user_id, match_query=match_query, query_vector=query_vector,
            version=version, dimensions=dimensions, mode=mode,
            candidate_limit=original_limit, lexical_quota=lexical_quota or settings.lexical_quota,
            dense_quota=dense_quota or settings.dense_quota, degraded_reason=reason,
            **({"span":True} if settings.fragment_policy == "span_exact_v1" else {}),
        )
        if self.coordinator is not None and snapshot.degraded_reason:
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
        option_trace=[];index_hits=len(snapshot.lexical)+len(snapshot.dense)+unknown_dropped
        if options_active:
            auxiliary=[]
            for option_query,quota in capability_plan.option_queries:
                extra=self.store.search_snapshot(user_id=plan.user_id,match_query=build_match_query(option_query),
                    query_vector=None,version=version,dimensions=dimensions,mode="raw_fts",candidate_limit=quota,
                    lexical_quota=quota,dense_quota=quota,span=True)
                index_hits+=len(extra.lexical)
                auxiliary,_=reciprocal_rank_fusion(auxiliary,extra.lexical,settings.rrf_k)
                option_trace.append({"quota":quota,"hit_ids":[h.id for h in extra.lexical]})
            fused,option_fusion=reciprocal_rank_fusion(fused,auxiliary,settings.rrf_k)
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
            fused = self.store.context_neighbors(plan.user_id,fused,settings.context_neighbor_radius,settings.candidate_limit,
                                                historical=view.intent in {"history","as_of"})
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
        invalid_derived_ids = list(getattr(fused,"invalid_derived_ids",()))
        from app.rendering import render_span_source
        renderer=render_span_source if settings.source_labels else None
        if window_trace["groups"]:
            from app.support_groups import pack_groups
            kept,chars,packing=pack_groups(gated,window_trace["groups"],min(request.top_k,settings.max_return_items),
                                          settings.return_char_budget,settings.return_token_budget,renderer)
        else:
            kept, chars,packing = pack_evidence(gated,min(request.top_k,settings.max_return_items),settings.return_char_budget,
                                              settings.packing_policy,settings.return_token_budget,renderer)
        if self.coordinator is not None:
            self.coordinator.final_check(plan.user_id, stamp, kept, policy)
        trace = {
            "adaptive_route":adaptive_trace,
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
            "capability_plan":{"intents":capability_plan.intents,"temporal_intent":capability_plan.temporal_intent,
                               "options_reason":capability_plan.options_reason,"window_reason":capability_plan.window_reason},
            "optional_routes":{"policy":settings.route_policy,"options":option_trace,"index_hits":index_hits,
                               "typed":typed_trace,"windows":window_trace,"claims":projection,
                               "unknown_dense_dropped":unknown_dropped,"source_labels":settings.source_labels},
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
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
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

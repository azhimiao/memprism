"""Runtime settings. FTS remains default; raw Dense/Hybrid are opt-in."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import math

from app.embeddings import EmbeddingConfig


def _default_db_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "memory.sqlite"


@dataclass(frozen=True)
class Settings:
    db_path: Path
    enable_extractor: bool = False
    enable_vector: bool = False
    enable_graph: bool = False
    enable_llm_reranker: bool = False
    enable_governance: bool = False
    context_neighbor_radius: int = 0
    retrieval_mode: str = "raw_fts"
    embedding: EmbeddingConfig | None = None
    candidate_limit: int = 200
    lexical_quota: int = 100
    dense_quota: int = 100
    max_return_items: int = 100
    return_char_budget: int = 32000
    rrf_k: int = 60
    auth_mode: str = "none"
    service_key_env: str = "AML_SERVICE_API_KEY"
    max_concurrent_requests: int = 16
    coordinator_lock_timeout_seconds: float = 0.1
    max_request_bytes: int = 10485760
    profile_id: str = "research"
    packing_policy: str = "prefix"
    return_token_budget: int | None = None
    route_policy: str = "none"
    source_labels: bool = False
    unknown_dense_floor: float | None = None
    claim_projection: bool = False
    exact_dense_acceleration: bool = False
    capability_views: bool = False
    set_selection: bool = False
    temporal_source_dates: bool = False
    sentence_index: bool = False
    parent_evidence: bool = False
    metadata_roles: bool = False
    metadata_dates: bool = False
    reranker_model: str | None = None
    reranker_device: str = "cpu"
    reranker_max_length: int = 512
    reranker_batch_size: int = 16
    atom_endpoint: str | None = None
    atom_key_env: str = "AML_ATOM_API_KEY"
    atom_model: str = "gpt-4o-mini"
    atom_batch_tokens: int = 6000
    atom_display: bool = False
    bridge_hops: int = 0

    @property
    def formal(self) -> bool:
        return self.profile_id != "research"

    @property
    def route_mode(self) -> str:
        if self.profile_id in {"dense_sync", "span_dense_sync", "cycle2_dense"} and self.retrieval_mode == "raw_fts":
            return "raw_hybrid_rrf"
        return self.retrieval_mode

    @property
    def required_routes(self) -> tuple[str, ...]:
        base = ("raw", "span", "fts") if self.fragment_policy == "span_exact_v1" else ("raw", "fts")
        if self.profile_id in {"dense_sync", "span_dense_sync", "cycle2_dense"}:
            base += ("dense",)
        if self.sentence_index:
            base += ("sentence",)
        if self.enable_extractor:
            base += ("atom",)
        return base

    @property
    def fragment_policy(self) -> str:
        return "span_exact_v1" if self.profile_id in {"span_sync", "span_dense_sync", "cycle2_raw", "cycle2_dense"} else "whole_raw_v1"

    @property
    def vector_enabled(self) -> bool:
        return self.enable_vector or self.retrieval_mode != "raw_fts"

    @classmethod
    def from_env(cls) -> Settings:
        raw_path = os.environ.get("AML_DB_PATH")
        db_path = Path(raw_path) if raw_path else _default_db_path()
        mode = os.environ.get("AML_RETRIEVAL_MODE", "raw_fts")
        enable = os.environ.get("AML_ENABLE_VECTOR", "false").lower() == "true"
        embedding = None
        profile = os.environ.get("AML_PROFILE", "research")
        if mode != "raw_fts" or enable or profile in {"dense_sync", "span_dense_sync", "cycle2_dense"}:
            names = ["AML_EMBEDDING_PROVIDER", "AML_EMBEDDING_ENDPOINT", "AML_EMBEDDING_DIMENSIONS"]
            if all(os.environ.get(name) for name in names):
                embedding = EmbeddingConfig(
                    provider=os.environ[names[0]], endpoint=os.environ[names[1]], dimensions=int(os.environ[names[2]]),
                    model=os.environ.get("AML_EMBEDDING_MODEL", "text-embedding-v4"),
                    revision=os.environ.get("AML_EMBEDDING_REVISION", "1"),
                    api_key_env=os.environ.get("AML_EMBEDDING_KEY_ENV", "AML_EMBEDDING_API_KEY"),
                    batch_size=int(os.environ.get("AML_EMBEDDING_BATCH_SIZE", "10")),
                    timeout_seconds=float(os.environ.get("AML_EMBEDDING_TIMEOUT_SECONDS", "30")),
                    max_retries=int(os.environ.get("AML_EMBEDDING_MAX_RETRIES", "0")),
                )
        return cls(db_path=db_path, retrieval_mode=mode, enable_vector=enable, embedding=embedding, profile_id=profile,
                   enable_governance=os.environ.get("AML_ENABLE_GOVERNANCE", "false").lower() == "true",
                   context_neighbor_radius=int(os.environ.get("AML_CONTEXT_NEIGHBOR_RADIUS", "0")),
                   candidate_limit=int(os.environ.get("AML_CANDIDATE_LIMIT", "200")),
                   lexical_quota=int(os.environ.get("AML_LEXICAL_QUOTA", "100")),
                   dense_quota=int(os.environ.get("AML_DENSE_QUOTA", "100")),
                   max_return_items=int(os.environ.get("AML_MAX_RETURN_ITEMS", "100")),
                   return_char_budget=int(os.environ.get("AML_RETURN_CHAR_BUDGET", "1000000000" if profile.startswith('cycle2_') else "32000")),
                   rrf_k=int(os.environ.get("AML_RRF_K", "60")),
                   auth_mode=os.environ.get("AML_AUTH_MODE", "none"),
                   service_key_env=os.environ.get("AML_SERVICE_KEY_ENV", "AML_SERVICE_API_KEY"),
                   max_concurrent_requests=int(os.environ.get("AML_MAX_CONCURRENT_REQUESTS", "16")),
                   coordinator_lock_timeout_seconds=float(
                       os.environ.get("AML_COORDINATOR_LOCK_TIMEOUT_SECONDS", "0.1")),
                   max_request_bytes=int(os.environ.get("AML_MAX_REQUEST_BYTES", "10485760")),
                   packing_policy=os.environ.get('AML_PACKING_POLICY','prefix'),
                   return_token_budget=int(os.environ['AML_RETURN_TOKEN_BUDGET']) if os.environ.get('AML_RETURN_TOKEN_BUDGET') else (65536 if profile.startswith('cycle2_') else None),
                   route_policy=os.environ.get('AML_ROUTE_POLICY','none'),
                   source_labels=os.environ.get('AML_SOURCE_LABELS','false').lower()=='true',
                   claim_projection=os.environ.get('AML_CLAIM_PROJECTION','false').lower()=='true',
                   exact_dense_acceleration=os.environ.get('AML_EXACT_DENSE_ACCELERATION','false').lower()=='true',
                   capability_views=os.environ.get('AML_CAPABILITY_VIEWS','false').lower()=='true',
                   set_selection=os.environ.get('AML_SET_SELECTION','false').lower()=='true',
                   temporal_source_dates=os.environ.get('AML_TEMPORAL_SOURCE_DATES','false').lower()=='true',
                   sentence_index=os.environ.get('AML_SENTENCE_INDEX','false').lower()=='true',
                   parent_evidence=os.environ.get('AML_PARENT_EVIDENCE','false').lower()=='true',
                   metadata_roles=os.environ.get('AML_METADATA_ROLES','false').lower()=='true',
                   metadata_dates=os.environ.get('AML_METADATA_DATES','false').lower()=='true',
                   reranker_model=os.environ.get('AML_RERANKER_MODEL') or None,
                   reranker_device=os.environ.get('AML_RERANKER_DEVICE','cpu'),
                   reranker_max_length=int(os.environ.get('AML_RERANKER_MAX_LENGTH','512')),
                   reranker_batch_size=int(os.environ.get('AML_RERANKER_BATCH_SIZE','16')),
                   enable_extractor=os.environ.get('AML_ENABLE_EXTRACTOR','false').lower()=='true',
                   atom_endpoint=os.environ.get('AML_ATOM_ENDPOINT') or None,
                   atom_key_env=os.environ.get('AML_ATOM_KEY_ENV','AML_ATOM_API_KEY'),
                   atom_model=os.environ.get('AML_ATOM_MODEL','gpt-4o-mini'),
                   atom_batch_tokens=int(os.environ.get('AML_ATOM_BATCH_TOKENS','6000')),
                   atom_display=os.environ.get('AML_ATOM_DISPLAY','false').lower()=='true',
                   bridge_hops=int(os.environ.get('AML_BRIDGE_HOPS','0')),
                   unknown_dense_floor=float(os.environ['AML_UNKNOWN_DENSE_FLOOR']) if os.environ.get('AML_UNKNOWN_DENSE_FLOOR') else None)

    def ensure_phase_a_defaults(self) -> None:
        if self.profile_id not in {"research", "raw_sync", "dense_sync", "span_sync", "span_dense_sync", "cycle2_raw", "cycle2_dense"}:
            raise ValueError("unknown profile")
        if self.profile_id in {"raw_sync", "span_sync"} and (self.vector_enabled or self.embedding is not None):
            raise ValueError("raw_sync requires raw_fts without optional vectors")
        cycle2 = self.profile_id in {"cycle2_raw", "cycle2_dense"}
        if self.formal and (self.enable_governance or self.context_neighbor_radius) and not cycle2:
            raise ValueError("K0 formal profiles use raw units and lifecycle controls only")
        enabled = [
            name
            for name, value in (
                ("enable_extractor", self.enable_extractor and not cycle2),
                ("enable_graph", self.enable_graph),
                ("enable_llm_reranker", self.enable_llm_reranker),
            )
            if value
        ]
        if enabled:
            joined = ", ".join(enabled)
            raise RuntimeError(
                "The raw baseline does not implement extractor, graph or LLM reranker. "
                f"Disable: {joined}."
            )
        if self.retrieval_mode not in {"raw_fts", "raw_dense", "raw_hybrid_rrf"}:
            raise ValueError("unknown retrieval mode")
        values = [self.candidate_limit, self.lexical_quota, self.dense_quota,
                  self.max_return_items, self.return_char_budget, self.rrf_k,
                  self.max_concurrent_requests,self.max_request_bytes]
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in values):
            raise ValueError("retrieval budgets and rrf_k must be positive integers")
        if self.route_mode == "raw_hybrid_rrf" and self.lexical_quota + self.dense_quota > self.candidate_limit:
            raise ValueError("hybrid route quotas exceed the total candidate budget")
        lock_timeout = self.coordinator_lock_timeout_seconds
        if (isinstance(lock_timeout, bool) or not isinstance(lock_timeout, (int, float)) or
                not math.isfinite(float(lock_timeout)) or not 0 < float(lock_timeout) <= 30):
            raise ValueError("coordinator lock timeout must be a finite float in (0, 30]")
        if self.auth_mode not in {"none","bearer","token","x-api-key"}:
            raise ValueError("unknown authentication mode")
        if self.auth_mode != "none" and not os.environ.get(self.service_key_env):
            raise ValueError("service authentication key is not configured")
        if self.packing_policy not in {"prefix","skip"}:
            raise ValueError("unknown packing policy")
        if self.return_token_budget is not None and (isinstance(self.return_token_budget,bool) or
                not isinstance(self.return_token_budget,int) or self.return_token_budget<1):
            raise ValueError("token budget must be a positive integer")
        if self.route_policy not in {"none","options","witness_window","typed","options_typed","adaptive_options","adaptive_options_dense","pareto"}:
            raise ValueError("unknown optional route policy")
        if (self.route_policy!="none" or self.source_labels or self.claim_projection) and self.fragment_policy!="span_exact_v1":
            raise ValueError("optional Pareto mechanisms require an explicit span profile")
        if self.unknown_dense_floor is not None and (isinstance(self.unknown_dense_floor,bool) or
                not isinstance(self.unknown_dense_floor,(int,float)) or not math.isfinite(self.unknown_dense_floor) or
                not -1<=self.unknown_dense_floor<=1):
            raise ValueError("unknown dense floor must be a finite cosine threshold")
        if self.route_policy in {"options","pareto"} and self.candidate_limit<200:
            raise ValueError("options allocation needs the registered 200-hit budget")
        if isinstance(self.context_neighbor_radius,bool) or self.context_neighbor_radius not in {0,1,2}:
            raise ValueError("context neighbor radius must be 0, 1 or 2")
        if (self.capability_views or self.set_selection or self.exact_dense_acceleration) and self.fragment_policy != 'span_exact_v1':
            raise ValueError('capability mechanisms require an explicit span profile')
        if self.exact_dense_acceleration:
            try:
                import numpy  # Explicit opt-in dependency; never silent fallback.
            except ImportError:
                raise ValueError('exact dense acceleration requires numpy') from None
        if self.temporal_source_dates and self.fragment_policy != 'span_exact_v1':
            raise ValueError('temporal source dates require an explicit span profile')
        if self.temporal_source_dates and self.source_labels:
            raise ValueError('temporal source dates and full source labels are mutually exclusive')
        if any((self.sentence_index,self.parent_evidence,self.metadata_roles,self.metadata_dates,
                self.reranker_model,self.enable_extractor,self.bridge_hops)) and not cycle2:
            raise ValueError('Cycle2 mechanisms require a cycle2 profile')
        if self.enable_extractor and (not self.atom_endpoint or self.atom_model != 'gpt-4o-mini'):
            raise ValueError('Atom extraction requires an explicit mini endpoint')
        if self.atom_display and not self.enable_extractor:
            raise ValueError('Atom display requires extraction')
        if type(self.bridge_hops) is not int or self.bridge_hops not in {0,1,2}:
            raise ValueError('bridge_hops must be 0, 1 or 2')
        if type(self.atom_batch_tokens) is not int or not 1024 <= self.atom_batch_tokens <= 8192:
            raise ValueError('Atom batch tokens must be 1024..8192')

"""K0 identities. Raw indexing/renderer/budget versions stay explicit."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from app.embeddings import provider_identity, provider_version


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Profile:
    profile_id: str
    required_routes: tuple[str, ...]
    config_hash: str
    model_version: str
    dimensions: int
    identity: dict

    @classmethod
    def build(cls, settings, provider):
        if settings.formal and provider is not None:
            if provider.config.timeout_seconds > 30 or provider.config.max_retries:
                raise ValueError("formal embedding timeout must be <=30s and retries must be zero")
            if settings.embedding is not None and settings.embedding != provider.config:
                raise ValueError("configured embedding and provider differ")
        identity = {
            "profile_id": settings.profile_id, "profile_version": "k0-v1",
            "required_routes": settings.required_routes, "mode": settings.route_mode,
            "fragment_policy": "whole_raw_v1", "index_version": "user_fts_unicode61_v1",
            "renderer": "exact_raw_v1", "budget_policy": "raw_prefix_v1",
            "model": provider_identity(provider) if provider is not None else None,
            "candidate_limit": settings.candidate_limit, "lexical_quota": settings.lexical_quota,
            "dense_quota": settings.dense_quota, "max_return_items": settings.max_return_items,
            "return_char_budget": settings.return_char_budget, "rrf_k": settings.rrf_k,
            "governance": settings.enable_governance, "neighbors": settings.context_neighbor_radius,
            "lease_ttl_seconds": 60, "renew_seconds": 20, "wait_seconds": 5,
            "external_timeout_seconds": 30, "search_attempts": 3,
        }
        if settings.fragment_policy == "span_exact_v1":
            from app.segments import fragment_identity
            identity.update(profile_version="k1-v1", fragment_policy=fragment_identity(),
                            index_version="user_span_fts_unicode61_v1", renderer="exact_span_v1")
        if settings.packing_policy != "prefix" or settings.return_token_budget is not None:
            identity.update(budget_policy=settings.packing_policy,return_token_budget=settings.return_token_budget)
        if settings.route_policy!="none" or settings.source_labels or settings.unknown_dense_floor is not None or settings.claim_projection:
            identity.update(optional_route_policy=settings.route_policy,source_labels=settings.source_labels,
                            unknown_dense_floor=settings.unknown_dense_floor,claim_projection=settings.claim_projection,
                            optional_mechanism_version="e30-candidate-v4")
        return cls(settings.profile_id, settings.required_routes, fingerprint(identity),
                   provider_version(provider) if provider is not None else "unconfigured",
                   provider.config.dimensions if provider is not None else 0, identity)

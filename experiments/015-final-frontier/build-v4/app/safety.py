"""Pure, stable source/lifecycle gate; no ranking, model calls or writes."""
from __future__ import annotations
from dataclasses import dataclass, field
from app.models import EvidenceCandidate


@dataclass(frozen=True)
class SourcePolicy:
    user_id: str
    raw_id: str
    version: int
    visibility: str = "visible"
    disclosure_policy: str = "within_user"
    retraction_state: str = "not_retracted"
    content_valid: bool = True


@dataclass(frozen=True)
class PolicySnapshot:
    epoch: int
    sources: dict[str, SourcePolicy]
    allow_historical: bool = False
    allowed_restricted_ids: frozenset[str] = field(default_factory=frozenset)


def filter_evidence(user_id: str, candidates: list[EvidenceCandidate],
                    snapshot: PolicySnapshot) -> tuple[list[EvidenceCandidate], list[dict]]:
    kept, decisions = [], []
    for candidate in candidates:
        source = snapshot.sources.get(candidate.source_raw_id)
        reason = None
        if candidate.user_id != user_id:
            reason = "wrong_user"
        elif candidate.deletion_state != "active" or not candidate.content.strip():
            reason = "inactive_candidate"
        elif source is None or source.user_id != user_id or source.raw_id != candidate.source_raw_id:
            reason = "invalid_source"
        elif not source.content_valid:
            reason = "source_body_mismatch"
        elif source.visibility != "visible":
            reason = "source_not_visible"
        elif source.disclosure_policy == "blocked":
            reason = "disclosure_blocked"
        elif source.disclosure_policy == "restricted" and source.raw_id not in snapshot.allowed_restricted_ids:
            reason = "restricted_without_permission"
        elif source.disclosure_policy not in {"within_user","restricted"}:
            reason = "unknown_disclosure_policy"
        elif source.retraction_state == "retracted" and not snapshot.allow_historical:
            reason = "retracted_current_support"
        elif source.retraction_state not in {"not_retracted","retracted"}:
            reason = "unknown_retraction_state"
        if reason is None:
            kept.append(candidate)
        decisions.append({"candidate_id":candidate.id,"source_id":candidate.source_raw_id,
                          "decision":"drop" if reason else "keep","reason":reason or "eligible",
                          "lifecycle_epoch":snapshot.epoch})
    return kept, decisions


def apply_gate(user_id: str, candidates: list[EvidenceCandidate]) -> list[EvidenceCandidate]:
    """Legacy unit-level filter. Production always calls filter_evidence with a DB snapshot."""
    return [c for c in candidates if c.user_id == user_id and c.deletion_state == "active"
            and c.source_raw_id and c.content.strip()]

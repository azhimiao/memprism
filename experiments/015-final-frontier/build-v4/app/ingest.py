"""Persist raw messages before Add returns success."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
import threading

from app.governance import GovernancePolicy, GovernancePlan, BoundedGovernance
from app.embeddings import EmbeddingProvider, prepare_embedding
from app.jobs import JobCoordinator, LeaseHeartbeat, LeasedEmbeddingProvider, NotReady, PayloadConflict
from app.profiles import Profile
from app.models import AddRequest, AddResponse, RawMessage
from app.store import CommitStatus, MessageStore

logger = logging.getLogger("aml.ingest")


class RequestPayloadConflict(RuntimeError):
    """Raised when a request_id is reused with a different Add payload."""


def canonical_payload_hash(request: AddRequest) -> str:
    payload = request.model_dump(mode="json")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class IngestService:
    def __init__(
        self,
        store: MessageStore,
        governance: GovernancePolicy,
        embedding_provider: EmbeddingProvider | None = None,
        vector_enabled: bool = False,
        settings=None,
        coordinator=None,
    ) -> None:
        self.store = store
        self.governance = governance
        self.embedding_provider = embedding_provider
        self.vector_enabled = vector_enabled
        self.settings = settings
        self.coordinator = coordinator
        self.profile = Profile.build(settings, embedding_provider) if settings else None
        # Bound memory and serialize same-key work in this service only. Separate
        # processes may duplicate HTTP cost; the DB transaction prevents writes.
        self._request_locks = [threading.Lock() for _ in range(64)]

    def add(self, request: AddRequest) -> AddResponse:
        if self.coordinator is not None:
            return self._formal_add(request)
        lock = self._request_locks[hash((request.user_id, request.request_id)) % len(self._request_locks)]
        with lock:
            return self._add(request)

    def _add(self, request: AddRequest) -> AddResponse:
        payload_hash = canonical_payload_hash(request)
        recorded_at = int(time.time() * 1000)
        messages = [
            RawMessage(
                id=_raw_message_id(request.user_id, request.request_id, index),
                user_id=request.user_id,
                session_id=request.session_id,
                request_id=request.request_id,
                message_index=index,
                role=message.role,
                content=message.content,
                source_timestamp=message.timestamp,
                recorded_at=recorded_at,
            )
            for index, message in enumerate(request.messages)
        ]
        response = AddResponse(
            success=True,
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        previous = self.store.lookup_add(messages, payload_hash=payload_hash, response=response.model_dump(mode="json"))
        if previous is not None:
            return _committed_response(previous)
        commands = self.governance.plan(messages)
        if commands and not isinstance(commands, GovernancePlan):
            raise RuntimeError("Fact governance commands are not applied in the raw baseline.")
        # Lifecycle controls and known erased text never leave for embedding.
        lifecycle_control = isinstance(commands, GovernancePlan) and bool(commands.mutations)
        blocked_replay = self.store.has_erased_statements(messages)
        prepared = prepare_embedding(self.embedding_provider, [message.content for message in messages]) if self.vector_enabled and not lifecycle_control and not blocked_replay else None
        if prepared is not None:
            # Record every attempted model batch, including work later discarded
            # by a concurrent winner or DB rollback. The committed job alone
            # would undercount such costs. No original text or secret is logged.
            logger.info("embedding_attempt %s", json.dumps({
                "attempt_id": uuid.uuid4().hex, "user_id": request.user_id,
                "request_id": request.request_id, "embedding_version": prepared.version,
                "implementation": getattr(self.embedding_provider, "implementation_kind", "unavailable"),
                "status": prepared.status, "error_code": prepared.error_code, "usage": prepared.usage,
            }, sort_keys=True))
        committed = self.store.commit_add(
            messages,
            payload_hash=payload_hash,
            response=response.model_dump(mode="json"),
            embedding=prepared,
            governance_plan=commands if isinstance(commands, GovernancePlan) else None,
            profile=self.profile,
        )
        logger.info(
            "add %s user_id=%s request_id=%s messages=%s",
            committed.status.value,
            request.user_id,
            request.request_id,
            len(messages),
        )
        return _committed_response(committed)

    def _formal_add(self, request):
        payload_hash = canonical_payload_hash(request)
        messages = [RawMessage(
            id=_raw_message_id(request.user_id, request.request_id, index),
            user_id=request.user_id, session_id=request.session_id, request_id=request.request_id,
            message_index=index, role=message.role, content=message.content,
            source_timestamp=message.timestamp, recorded_at=int(time.time() * 1000),
        ) for index, message in enumerate(request.messages)]
        response = AddResponse(success=True, request_id=request.request_id,
                               user_id=request.user_id, session_id=request.session_id)
        # K0 uses only existing exact lifecycle grammar, never claim extraction.
        parsed_plan=BoundedGovernance().plan(messages)
        plan = parsed_plan if self.settings.claim_projection else GovernancePlan(mutations=parsed_plan.mutations)
        pure_control = len(plan.mutations) == len(messages) and all(m.operation == "erase" for m in plan.mutations)
        coordinator = self.coordinator
        try:
            reserved = coordinator.reserve(request.user_id, request.request_id, payload_hash,
                pure_control=pure_control, messages=messages, response=response.model_dump(mode="json"))
            if isinstance(reserved, dict):
                return AddResponse.model_validate(reserved)
            lease = reserved
            try:
                with LeaseHeartbeat(coordinator, lease) as heartbeat:
                    active = coordinator.draft(lease, messages, plan)
                    span_pieces = None
                    if self.settings.fragment_policy == "span_exact_v1":
                        from app.segments import split_source
                        span_pieces = [piece for message in active for piece in split_source(message)]
                    prepared = None
                    if active and "dense" in self.profile.required_routes:
                        provider = LeasedEmbeddingProvider(self.embedding_provider, coordinator, lease,
                                                           heartbeat, messages, plan)
                        texts = [p["content"] for p in span_pieces] if span_pieces is not None else [m.content for m in active]
                        prepared = prepare_embedding(provider, texts)
                        if prepared.status != "ready":
                            raise NotReady(prepared.error_code or "required_embedding_failed")
                    heartbeat.check()
                    committed = self.store.commit_add(
                        messages, payload_hash=payload_hash, response=response.model_dump(mode="json"),
                        embedding=prepared, governance_plan=plan, coordinator=coordinator, lease=lease,
                        embedding_messages=active, profile=self.profile, pure_control=pure_control,span_pieces=span_pieces,
                        claim_projection=self.settings.claim_projection)
                    return _committed_response(committed)
            except Exception as exc:
                try:
                    coordinator.fail(lease, str(exc) if isinstance(exc, NotReady) else "preparation_failed")
                except NotReady:
                    pass
                raise
        except PayloadConflict as exc:
            raise RequestPayloadConflict(str(exc)) from exc


def _committed_response(committed) -> AddResponse:
    if committed.status is CommitStatus.CONFLICT:
        raise RequestPayloadConflict("request_id has already been used with a different Add payload")
    if committed.response is None:
        raise RuntimeError("Add transaction completed without a response")
    return AddResponse.model_validate(committed.response)


def _raw_message_id(user_id: str, request_id: str, message_index: int) -> str:
    identity = f"{user_id}\0{request_id}\0{message_index}"
    return f"raw_{uuid.uuid5(uuid.NAMESPACE_URL, identity).hex}"

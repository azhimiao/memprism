"""HTTP boundary for Add, Search, and Health."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response

from app.config import Settings
from app.embeddings import EmbeddingProvider, create_embedding_provider
from app.governance import GovernancePolicy, PassthroughGovernance, BoundedGovernance
from app.ingest import IngestService, RequestPayloadConflict
from app.models import AddRequest, AddResponse, SearchRequest, SearchResponse
from app.retrieval import RetrievalService, LifecycleSnapshotBusy
from app.store import SqliteStore
from app.transport import TransportGuard
from app.profiles import Profile
from app.jobs import JobCoordinator, NotReady
import sqlite3


def create_app(
    settings: Settings | None = None,
    governance: GovernancePolicy | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    atom_provider=None,
    reranker=None,
) -> FastAPI:
    resolved = settings or Settings.from_env()
    resolved.ensure_phase_a_defaults()
    store = SqliteStore(resolved.db_path)
    store.initialize()
    provider = None
    if resolved.vector_enabled or "dense" in resolved.required_routes:
        provider = embedding_provider or (create_embedding_provider(resolved.embedding) if resolved.embedding else None)
    profile = Profile.build(resolved, provider)
    if resolved.enable_extractor:
        from app.atomic_memory import AtomicMemoryConfig,HttpAtomProvider
        atom_provider=atom_provider or HttpAtomProvider(AtomicMemoryConfig.from_settings(resolved))
    if resolved.reranker_model and reranker is None:
        from app.reranking import EvidenceReranker,RerankConfig
        from pathlib import Path
        reranker=EvidenceReranker(RerankConfig(enabled=True,local_model_path=Path(resolved.reranker_model),
            device=resolved.reranker_device,max_length=resolved.reranker_max_length,batch_size=resolved.reranker_batch_size))
    coordinator = (
        JobCoordinator(store, profile, lock_timeout_seconds=resolved.coordinator_lock_timeout_seconds)
        if resolved.formal
        else None
    )
    if coordinator is not None:
        coordinator.activate()
    ingest = IngestService(
        store,
        governance or (BoundedGovernance() if resolved.enable_governance else PassthroughGovernance()),
        provider,
        resolved.vector_enabled,
        settings=resolved,
        coordinator=coordinator,
        atom_provider=atom_provider,
    )
    retrieval = RetrievalService(store, resolved, provider.for_query() if provider is not None and hasattr(provider,'for_query') else provider)
    retrieval.coordinator = coordinator
    retrieval.reranker=reranker
    app = FastAPI(title="AML Memory", version="0.3.0")
    app.state.store = store
    app.state.ingest = ingest
    app.state.retrieval = retrieval
    app.state.profile = profile
    app.state.coordinator = coordinator
    app.add_middleware(TransportGuard, settings=resolved)

    @app.get("/health")
    def health():
        if coordinator is not None:
            try:
                with coordinator.transaction(False) as db:
                    coordinator.control_ready_on(db)
                    users = [row[0] for row in db.execute("SELECT DISTINCT user_id FROM raw_messages")
                             if isinstance(row[0], str) and row[0].strip()]
                    for user in users:
                        coordinator.ready_on(db, user)
                return {"status": "ok", "control_ready": True, "data_route_ready": True,
                        "profile_id": profile.profile_id, "config_hash": profile.config_hash}
            except NotReady as exc:
                control = str(exc) != "control_not_ready"
                return {"status": "not_ready", "control_ready": control, "data_route_ready": False,
                        "reason": str(exc), "profile_id": profile.profile_id}
        return {"status": "ok"}

    @app.post("/add", response_model=AddResponse)
    def add_messages(body: AddRequest) -> AddResponse:
        try:
            return ingest.add(body)
        except RequestPayloadConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except NotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"}) from exc
        except sqlite3.Error as exc:
            if coordinator is None:
                raise
            raise HTTPException(status_code=503, detail="control_not_ready", headers={"Retry-After": "1"}) from exc
        except Exception as exc:
            if coordinator is None:
                raise
            raise HTTPException(status_code=503, detail="preparation_failed", headers={"Retry-After": "1"}) from None

    @app.post("/search", response_model=SearchResponse)
    def search_messages(body: SearchRequest, response: Response) -> SearchResponse:
        try:
            result, trace = retrieval.search_with_trace(body)
        except LifecycleSnapshotBusy as exc:
            raise HTTPException(status_code=503,detail="lifecycle snapshot busy",headers={"Retry-After":"1"}) from exc
        except NotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"}) from exc
        except sqlite3.Error as exc:
            if coordinator is None:
                raise
            raise HTTPException(status_code=503, detail="control_not_ready", headers={"Retry-After": "1"}) from exc
        except Exception as exc:
            if coordinator is None:
                raise
            raise HTTPException(status_code=503, detail="search_failed", headers={"Retry-After": "1"}) from None
        response.headers["X-AML-Requested-Mode"] = str(trace["requested_mode"])
        response.headers["X-AML-Actual-Mode"] = str(trace["actual_mode"])
        response.headers["X-AML-Degraded"] = str(trace["degraded"]).lower()
        response.headers["X-AML-Degraded-Reason"] = str(trace["degraded_reason"] or "none")
        return result

    return app


def __getattr__(name: str) -> FastAPI:
    if name == "app":
        return create_app()
    raise AttributeError(name)

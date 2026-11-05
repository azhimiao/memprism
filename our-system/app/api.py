"""HTTP boundary for Add, Search, and Health."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response

from app.config import Settings
from app.embeddings import EmbeddingProvider, create_embedding_provider
from app.governance import GovernancePolicy, PassthroughGovernance, BoundedGovernance
from app.ingest import IngestService, RequestPayloadConflict
from app.models import AddRequest, AddResponse, SearchRequest, SearchResponse
from app.retrieval import RetrievalService, LifecycleSnapshotBusy, InvalidSearchInput
from app.store import SqliteStore
from app.transport import TransportGuard
from app.profiles import Profile
from app.jobs import JobCoordinator, NotReady
import sqlite3
import asyncio
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path
import threading
from app import diagnostics as diagnostic


class _HealthReadiness:
    """Coalesce full audits; reuse only an unchanged observed SQLite database.

    HTTP waits are bounded independently of the full integrity audit. The
    retained, read-only connection's data_version detects other connections'
    commits, including out-of-band edits that do not update our watermarks.
    No source text, vectors, user ids, Env or provider responses are cached.
    """
    def __init__(self, coordinator, path, profile, wait_seconds=2.0):
        self.coordinator, self.path, self.profile = coordinator, Path(path), profile
        self.wait_seconds = wait_seconds
        self._lock = threading.Lock()
        self._future = None
        self._db = None
        self._inode = None
        self._validated = None
        self._closed = False

    def _not_ready(self, reason, control=False):
        return {"status": "not_ready", "control_ready": control, "data_route_ready": False,
                "reason": reason, "profile_id": self.profile.profile_id}

    def _file_identity(self):
        stat = self.path.stat()
        return (stat.st_dev, stat.st_ino)

    def _disconnect(self):
        if self._db is not None:
            try:
                self._db.close()
            except sqlite3.Error:
                pass
        self._db, self._inode, self._validated = None, None, None

    def _evaluate(self):
        if hasattr(self.coordinator, 'health_ready'):
            self.coordinator.health_ready()
            return {"status": "ok", "control_ready": True, "data_route_ready": True,
                    "profile_id": self.profile.profile_id, "config_hash": self.profile.config_hash}
        inode = self._file_identity()
        if self._db is None or self._inode != inode:
            self._disconnect()
            self._db = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True,
                                       timeout=0.2, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.isolation_level = None
            self._db.execute('PRAGMA query_only=ON')
            self._inode = inode
        db = self._db
        try:
            db.execute('BEGIN')
            self.coordinator.control_ready_on(db)
            version = db.execute('PRAGMA data_version').fetchone()[0]
            stamp = (inode, version)
            if self._validated != stamp:
                # Full original validation, in one snapshot. Publication proofs
                # alone do not replace validation of actual sources/FTS/vectors.
                for row in db.execute('SELECT DISTINCT user_id FROM raw_messages'):
                    user = row[0]
                    if isinstance(user, str) and user.strip():
                        self.coordinator.ready_on(db, user)
            db.execute('COMMIT')
            # A commit during the audit or a file replacement invalidates the
            # result. Never certify a snapshot that is already known to be old.
            if (self._file_identity() != inode
                    or db.execute('PRAGMA data_version').fetchone()[0] != version):
                self._validated = None
                return self._not_ready('health_snapshot_changed', control=True)
            self._validated = stamp
            return {"status": "ok", "control_ready": True, "data_route_ready": True,
                    "profile_id": self.profile.profile_id, "config_hash": self.profile.config_hash}
        finally:
            if db.in_transaction:
                db.execute('ROLLBACK')

    def _worker(self, future):
        try:
            result = self._evaluate()
        except NotReady as exc:
            self._validated = None
            result = self._not_ready(str(exc), control=str(exc) != 'control_not_ready')
        except (sqlite3.Error, OSError):
            self._disconnect()
            result = self._not_ready('control_not_ready')
        except Exception:
            self._disconnect()
            result = self._not_ready('health_check_failed')
        # Publish completion only after all connection work has finished. A new
        # worker and shutdown cannot race with this worker's final connection IO.
        with self._lock:
            if self._closed:
                self._disconnect()
            future.set_result(result)

    def _start_or_join(self):
        with self._lock:
            if self._closed:
                return None
            if self._future is None or self._future.done():
                future = Future()
                self._future = future
                try:
                    threading.Thread(target=self._worker, args=(future,),
                                     name='aml-health-audit', daemon=True).start()
                except RuntimeError:
                    future.set_result(self._not_ready('health_check_failed'))
            return self._future

    async def check(self):
        future = self._start_or_join()
        if future is None:
            return self._not_ready('health_check_stopped')
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.wait_seconds
        while not future.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                # No wrap_future callbacks retain timed-out/cancelled clients
                # until a long audit finishes. Only active requests keep a ref.
                return self._not_ready('health_check_in_progress')
            await asyncio.sleep(min(0.01, remaining))
        return dict(future.result())

    def close(self):
        with self._lock:
            self._closed = True
            if self._future is None or self._future.done():
                self._disconnect()
        if hasattr(self.coordinator, '_read_readiness'):
            self.coordinator._read_readiness.close()


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
    if resolved.formal and provider is not None:
        from app.embedding_accounting import AccountedEmbeddingProvider
        if not isinstance(provider, AccountedEmbeddingProvider):
            provider = AccountedEmbeddingProvider(provider,
                resolved.db_path.with_name(resolved.db_path.name + '.embedding-usage.sqlite'),
                scope_identity=profile.config_hash)
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
        coordinator.pooled_control_reads = 'dense' in profile.required_routes
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
    # Legacy coordinator-free Add closes its writer after each call. Retaining
    # only a reader there delays SQLite's last-writer WAL checkpoint and changes
    # the existing offline-copy byte identity contract. Formal profiles retain
    # their WriteRuntime handle and reuse the independent read pool normally.
    store._reads.reuse = coordinator is not None
    health_readiness = _HealthReadiness(coordinator, resolved.db_path, profile) if coordinator is not None else None
    @asynccontextmanager
    async def lifespan(application):
        try:
            yield
        finally:
            try:
                if health_readiness is not None:
                    health_readiness.close()
            finally:
                store._reads.dispose()
    app = FastAPI(title="AML Memory", version="0.3.0", lifespan=lifespan)
    app.state.store = store
    app.state.ingest = ingest
    app.state.retrieval = retrieval
    app.state.profile = profile
    app.state.coordinator = coordinator
    app.add_middleware(TransportGuard, settings=resolved)
    app.state.health_readiness = health_readiness

    @app.get("/health")
    async def health():
        if health_readiness is not None:
            return await health_readiness.check()
        return {"status": "ok"}

    @app.post("/add", response_model=AddResponse)
    def add_messages(body: AddRequest) -> AddResponse:
        diagnostic.stage('add')
        try:
            return ingest.add(body)
        except RequestPayloadConflict as exc:
            diagnostic.failure(exc, 'request_payload_conflict')
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except NotReady as exc:
            diagnostic.failure(exc, str(exc))
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"}) from exc
        except sqlite3.Error as exc:
            diagnostic.failure(exc, 'control_not_ready')
            if coordinator is None:
                raise
            raise HTTPException(status_code=503, detail="control_not_ready", headers={"Retry-After": "1"}) from exc
        except Exception as exc:
            diagnostic.failure(exc, 'preparation_failed')
            if coordinator is None:
                raise
            raise HTTPException(status_code=503, detail="preparation_failed", headers={"Retry-After": "1"}) from None

    @app.post("/search", response_model=SearchResponse)
    def search_messages(body: SearchRequest, response: Response) -> SearchResponse:
        diagnostic.stage('search')
        diagnostic.emit('input_shape', query_chars=len(body.query),
                        query_utf8_bytes=len(body.query.encode('utf-8', errors='replace')),
                        option_count=len(body.options or []))
        try:
            result, trace = retrieval.search_with_trace(body)
        except InvalidSearchInput as exc:
            diagnostic.failure(exc, str(exc))
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LifecycleSnapshotBusy as exc:
            diagnostic.failure(exc, 'lifecycle snapshot busy')
            raise HTTPException(status_code=503,detail="lifecycle snapshot busy",headers={"Retry-After":"1"}) from exc
        except NotReady as exc:
            diagnostic.failure(exc, str(exc))
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"}) from exc
        except sqlite3.Error as exc:
            diagnostic.failure(exc, 'control_not_ready')
            if coordinator is None:
                raise
            raise HTTPException(status_code=503, detail="control_not_ready", headers={"Retry-After": "1"}) from exc
        except Exception as exc:
            diagnostic.failure(exc, 'search_failed')
            if coordinator is None:
                raise
            raise HTTPException(status_code=503, detail="search_failed", headers={"Retry-After": "1"}) from None
        diagnostic.emit('input_shape', returned_items=len(result.data),
                        output_characters=trace.get('output_characters'),
                        active_rows=trace.get('active_rows'),
                        snapshot_attempts=trace.get('snapshot_attempts'),
                        packed_record_tokens=(trace.get('packing') or {}).get('packed_record_tokens'))
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

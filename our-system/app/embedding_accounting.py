"""Embedding dispatch accounting and bounded, process-local retry reuse.

Parent wiring (the ledger must be a DIFFERENT file from the application DB)::

    accounted = AccountedEmbeddingProvider(raw_provider, ledger_path,
                                          scope_identity="deployment/profile-v1")
    query_provider = accounted.for_query()
    # Inject accounted INSIDE LeasedEmbeddingProvider, not around it.
    with embedding_scope(user_id, original_request_id + payload_hash, "add"):
        ingest_with_existing_lease_handling()
    # A new UUID for each HTTP Search; scope encloses all three snapshot retries.
    with embedding_scope(user_id, uuid.uuid4().hex, "search"):
        search_with_existing_authorization_and_retries()

Keep the wrapper alive across Add retries. Cache entries contain only vector
values and hashes, have finite TTL/entry/value bounds, and disappear on restart.
Query reuse additionally requires the SAME scope instance. Concurrent misses can
both dispatch; this is not exactly-once delivery or persistent paid-result reuse.
ContextVars propagate through normal async context copying; a custom executor
must explicitly copy_context(). Missing context disables caching, not accounting.

The guarded provider contract calls before_dispatch immediately before EACH HTTP
attempt (including retries). A provider with only embed is counted once per call.
Guarded calls are split at config.batch_size so a second callback in one batch
means a failed attempt was retried. No transaction/lock spans provider execution.
A committed dispatch is a send intent awaiting final authorization: a crash immediately afterward
cannot prove delivery. Pending dispatches and missing/partial usage remain unknown.
Providers expose aggregate batch usage, so incomplete usage conservatively marks
ALL that batch's attempts unknown; known tokens are still counted exactly once.
Neither text, vectors, credentials nor exception bodies are persisted.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Callable, Iterator, Sequence
import uuid

from .embeddings import (EmbeddingBatch, EmbeddingFailure, normalize_vector,
                         provider_version)


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Scope:
    user_hash: str
    request_hash: str
    phase_hash: str
    phase: str
    query_nonce: str


_SCOPE: ContextVar[_Scope | None] = ContextVar("embedding_accounting_scope", default=None)


@contextmanager
def embedding_scope(user_id: str, request_id: str, phase: str) -> Iterator[None]:
    """Authorize retry-key isolation, not access: callers must still supply guards.

    Add's request_id must bind the full payload, not merely a client-supplied ID.
    Search's request_id must be a fresh HTTP-request UUID. Arbitrary phase labels
    are hashed; only the fixed add/search/query/document labels are stored clear.
    """
    if any(not isinstance(value, str) or not value for value in (user_id, request_id, phase)):
        raise ValueError("embedding scope requires nonempty string identities")
    token = _SCOPE.set(_Scope(_hash(user_id), _hash(request_id), _hash(phase),
                             phase if phase in {"add", "search", "query", "document"} else "other",
                             uuid.uuid4().hex))
    try:
        yield
    finally:
        _SCOPE.reset(token)


@dataclass(frozen=True)
class _Cached:
    expires_at: float
    vectors: tuple[tuple[float, ...], ...]
    values: int


class _Shared:
    def __init__(self, ledger_path, max_cache_entries, cache_ttl_seconds, max_cache_values):
        self.path = Path(ledger_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_cache_entries
        self.ttl = cache_ttl_seconds
        self.max_values = max_cache_values
        self.cache: OrderedDict[str, _Cached] = OrderedDict()
        self.values = 0
        self.lock = threading.Lock()
        self.ledger_lock = threading.Lock()
        with self.connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript("""
                CREATE TABLE IF NOT EXISTS embedding_calls (
                    call_id TEXT PRIMARY KEY, started_at REAL NOT NULL, finished_at REAL,
                    user_hash TEXT, request_hash TEXT, phase_hash TEXT, phase TEXT NOT NULL,
                    scope_identity_hash TEXT NOT NULL, provider_hash TEXT NOT NULL,
                    version_hash TEXT NOT NULL, role TEXT NOT NULL, input_hash TEXT NOT NULL,
                    input_count INTEGER NOT NULL, status TEXT NOT NULL,
                    reported_input_tokens INTEGER, usage_complete INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS embedding_dispatches (
                    call_id TEXT NOT NULL REFERENCES embedding_calls(call_id),
                    attempt INTEGER NOT NULL, dispatched_at REAL NOT NULL, finished_at REAL,
                    status TEXT NOT NULL, PRIMARY KEY(call_id, attempt)
                );
            """)

    @contextmanager
    def connection(self):
        # Serialize only short ledger transactions within this process. Without
        # this gate, concurrent writers may starve until the old 5s timeout.
        # Provider I/O and permission callbacks always execute outside this gate.
        if not self.ledger_lock.acquire(timeout=30.0):
            raise sqlite3.OperationalError('embedding ledger admission timeout')
        db = None
        try:
            db = sqlite3.connect(self.path, timeout=30.0)
            with db:
                yield db
        finally:
            if db is not None:
                db.close()
            self.ledger_lock.release()

    def _expire(self, now):
        for key in [key for key, value in self.cache.items() if value.expires_at <= now]:
            self.values -= self.cache.pop(key).values

    def get(self, key):
        with self.lock:
            self._expire(time.monotonic())
            entry = self.cache.get(key)
            if entry is not None:
                self.cache.move_to_end(key)
            return entry

    def put(self, key, vectors):
        values = sum(len(vector) for vector in vectors)
        if not self.max_entries or values > self.max_values:
            return
        with self.lock:
            now = time.monotonic()
            self._expire(now)
            previous = self.cache.pop(key, None)
            if previous is not None:
                self.values -= previous.values
            self.cache[key] = _Cached(now + self.ttl, vectors, values)
            self.values += values
            while len(self.cache) > self.max_entries or self.values > self.max_values:
                _, evicted = self.cache.popitem(last=False)
                self.values -= evicted.values

    def start(self, metadata):
        call_id = uuid.uuid4().hex
        with self.connection() as db:
            db.execute("""INSERT INTO embedding_calls
                (call_id, started_at, user_hash, request_hash, phase_hash, phase,
                 scope_identity_hash, provider_hash, version_hash, role, input_hash,
                 input_count, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'started')""",
                       (call_id, time.time(), *metadata))
        return call_id

    def dispatch(self, call_id, attempt):
        now = time.time()
        with self.connection() as db:
            # There is only one provider batch per call, so this is an HTTP retry.
            db.execute("""UPDATE embedding_dispatches SET status='failed', finished_at=?
                          WHERE call_id=? AND status='dispatched'""", (now, call_id))
            db.execute("INSERT INTO embedding_dispatches VALUES (?, ?, ?, NULL, 'dispatched')",
                       (call_id, attempt, now))

    def finish(self, call_id, status, usage, dispatches):
        tokens = usage.get("reported_input_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 0 <= tokens <= 2**63 - 1:
            tokens = None
        requests = usage.get("requests", dispatches)
        complete = (not dispatches or (tokens is not None and usage.get("usage_complete") is True
                    and isinstance(requests, int) and not isinstance(requests, bool)
                    and requests == dispatches))
        now = time.time()
        with self.connection() as db:
            db.execute("""UPDATE embedding_calls SET finished_at=?, status=?,
                          reported_input_tokens=?, usage_complete=? WHERE call_id=?""",
                       (now, status, tokens if dispatches else 0, int(complete), call_id))
            db.execute("""UPDATE embedding_dispatches SET finished_at=?, status=?
                          WHERE call_id=? AND status='dispatched'""",
                       (now, "completed" if status == "completed" else "failed", call_id))

    def cancel(self, call_id, attempt):
        with self.connection() as db:
            db.execute("UPDATE embedding_dispatches SET status='cancelled',finished_at=? WHERE call_id=? AND attempt=?",
                       (time.time(), call_id, attempt))


def _combined_usage(usages: list[dict]) -> dict:
    """Preserve single-batch provider usage; aggregate the existing HTTP fields."""
    if len(usages) == 1:
        return dict(usages[0])
    result = dict(usages[0])
    for key in ("requests", "input_characters_sent", "reported_input_tokens"):
        values = [usage.get(key) for usage in usages]
        result[key] = sum(v for v in values if isinstance(v, int) and not isinstance(v, bool) and v >= 0)
    result["usage_complete"] = all(usage.get("usage_complete") is True for usage in usages)
    result["cache_hit"] = all(usage.get("cache_hit") is True for usage in usages)
    result["cache_hits"] = sum(usage.get("cache_hit") is True for usage in usages)
    for key in ("request_ids", "response_models"):
        if any(key in usage for usage in usages):
            result[key] = [value for usage in usages for value in usage.get(key, [])]
    if "model_identity_source" in result:
        result["model_identity_source"] = ("response" if result["requests"] and
            len(result.get("response_models", [])) == result["requests"] else "request_only_or_partial")
    return result


class AccountedEmbeddingProvider:
    """Transparent provider identity plus a shared ledger and bounded vector cache.

    Limits apply jointly to the document and for_query() views. max_cache_values
    bounds stored scalar values as well as entry count; oversized batches are not
    cached. Zero entry/value capacity disables caching. TTL never slides on hits.
    """

    def __init__(self, provider, ledger_path, scope_identity: str = "default", *,
                 max_cache_entries: int = 128, cache_ttl_seconds: float = 120.0,
                 max_cache_values: int = 262144):
        for limit in (max_cache_entries, max_cache_values):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError("cache capacity must be a nonnegative integer")
        if (isinstance(cache_ttl_seconds, bool) or not isinstance(cache_ttl_seconds, (int, float))
                or not math.isfinite(cache_ttl_seconds) or cache_ttl_seconds <= 0):
            raise ValueError("cache TTL must be finite and positive")
        if not isinstance(scope_identity, str) or not scope_identity:
            raise ValueError("scope_identity must be a nonempty string")
        self.provider = provider
        self._scope_identity_hash = _hash(scope_identity)
        self._role = "document"
        self._shared = _Shared(ledger_path, max_cache_entries, cache_ttl_seconds, max_cache_values)

    @property
    def config(self):
        return self.provider.config

    @property
    def implementation_kind(self):
        return self.provider.implementation_kind

    @property
    def version(self):
        return getattr(self.provider, "version", provider_version(self.provider))

    @property
    def ledger_path(self) -> Path:
        return self._shared.path

    def for_query(self):
        if self._role == "query":
            return self
        view = object.__new__(type(self))
        factory = getattr(self.provider, "for_query", None)
        view.provider = factory() if callable(factory) else self.provider
        view._scope_identity_hash = self._scope_identity_hash
        view._role = "query"
        view._shared = self._shared
        # Delegate to the provider's own view; retain its shared dispatch rate gate.
        return view

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        return self.embed_guarded(texts)

    def embed_guarded(self, texts: Sequence[str], *, before_dispatch: Callable | None = None) -> EmbeddingBatch:
        if (isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence) or not texts
                or any(not isinstance(text, str) or not text.strip() for text in texts)):
            raise EmbeddingFailure("invalid_input")
        texts = tuple(texts)
        # Match native preflight for the ENTIRE input before splitting batches.
        if self.config.provider == "dashscope":
            try:
                lengths = [len(text.encode("utf-8")) for text in texts]
            except UnicodeEncodeError:
                raise EmbeddingFailure("invalid_input_unicode") from None
            if any(length > self.config.max_input_bytes for length in lengths):
                raise EmbeddingFailure("input_too_long")
        guarded = callable(getattr(self.provider, "embed_guarded", None))
        batch_size = self.config.batch_size if guarded else len(texts)
        vectors, usages = [], []
        for start in range(0, len(texts), batch_size):
            try:
                result = self._embed_batch(texts[start:start + batch_size], before_dispatch, guarded)
            except EmbeddingFailure as exc:
                if usages:
                    raise EmbeddingFailure(exc.code, _combined_usage([*usages, exc.usage])) from None
                raise
            vectors.extend(result.vectors)
            usages.append(result.usage)
        return EmbeddingBatch(vectors, _combined_usage(usages))

    def _embed_batch(self, texts, before_dispatch, guarded):
        scope = _SCOPE.get()
        metadata = (scope.user_hash if scope else None, scope.request_hash if scope else None,
                    scope.phase_hash if scope else None, scope.phase if scope else "unscoped",
                    self._scope_identity_hash, provider_version(self.provider), _hash(self.version),
                    self._role, _hash(texts), len(texts))
        cache_key = _hash((metadata, scope.query_nonce if scope and self._role == "query" else None)) if scope else None
        cached = self._shared.get(cache_key) if cache_key is not None else None
        call_id = self._shared.start(metadata)
        dispatches = 0

        def dispatch():
            nonlocal dispatches
            self._shared.dispatch(call_id, dispatches + 1)  # Commit BEFORE provider I/O.
            try:
                if before_dispatch is not None:
                    # Permission may change while the ledger waits for its lock.
                    # This callback executes once, after that wait and immediately
                    # before provider I/O, without duplicating the caller's ledger.
                    before_dispatch()
            except BaseException:
                self._shared.cancel(call_id, dispatches + 1)
                raise
            dispatches += 1

        usage = {}
        try:
            if cached is not None:
                if before_dispatch is not None:
                    before_dispatch()  # Also lets a new Add lease authorize reuse.
                usage = {"requests": 0, "input_characters_sent": 0, "reported_input_tokens": 0,
                         "usage_complete": True, "monetary_cost": None, "cache_hit": True}
                if self.config.provider == "dashscope":
                    usage.update(text_type=self._role, request_ids=[], response_models=[],
                                 model_identity_source="request_only_or_partial")
                self._shared.finish(call_id, "cache_hit", usage, 0)
                return EmbeddingBatch([list(vector) for vector in cached.vectors], usage)
            if guarded:
                result = self.provider.embed_guarded(list(texts), before_dispatch=dispatch)
            else:
                dispatch()
                result = self.provider.embed(list(texts))
            usage = result.usage
            if len(result.vectors) != len(texts):
                raise EmbeddingFailure("response_count_mismatch", usage)
            for vector in result.vectors:
                normalize_vector(vector, self.config.dimensions)
            frozen_vectors = tuple(tuple(float(value) for value in vector) for vector in result.vectors)
        except BaseException as exc:
            # A KeyboardInterrupt also leaves an auditable failure. An abrupt process
            # death leaves 'dispatched' rows, explicitly unknown in summarize_ledger.
            failure_usage = exc.usage if isinstance(exc, EmbeddingFailure) and exc.usage else usage
            self._shared.finish(call_id, "failed", failure_usage, dispatches)
            if isinstance(exc, EmbeddingFailure) and not exc.usage and usage:
                # Local vector validation must not discard supplier usage upstream.
                exc.usage = dict(usage)
            raise
        self._shared.finish(call_id, "completed", usage, dispatches)
        if cache_key is not None:
            self._shared.put(cache_key, frozen_vectors)
        return result


def summarize_ledger(ledger_path, *, price: float | Decimal | None, currency: str,
                     per_tokens: int = 1_000_000) -> dict[str, object]:
    """Read one consistent snapshot; price is explicit currency units/per_tokens.

    price=None means unknown pricing, never a free call. known_estimate covers
    only reported tokens, not unknown usage or an actual supplier invoice. With
    partial aggregate usage, unknown_usage_requests is a conservative upper bound
    (all attempts in incomplete batches), not extra tokens to add to known tokens.
    completed/failed count HTTP attempts; *_batches count wrapper outcomes.
    """
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("currency must be explicit")
    if isinstance(per_tokens, bool) or not isinstance(per_tokens, int) or per_tokens <= 0:
        raise ValueError("per_tokens must be a positive integer")
    parsed_price = None
    if price is not None:
        try:
            parsed_price = Decimal(str(price))
        except InvalidOperation:
            raise ValueError("price must be finite and nonnegative, or None") from None
        if not parsed_price.is_finite() or parsed_price < 0:
            raise ValueError("price must be finite and nonnegative, or None")
    path = Path(ledger_path).resolve()
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        db.execute("BEGIN")
        counts = dict(db.execute("SELECT status, count(*) FROM embedding_dispatches GROUP BY status"))
        outcomes = dict(db.execute("SELECT status, count(*) FROM embedding_calls GROUP BY status"))
        tokens = db.execute("SELECT coalesce(sum(reported_input_tokens), 0) FROM embedding_calls").fetchone()[0]
        unknown = db.execute("""SELECT count(*) FROM embedding_dispatches d JOIN embedding_calls c
                                ON c.call_id=d.call_id WHERE c.usage_complete=0 AND d.status!='cancelled'""").fetchone()[0]
    finally:
        db.close()
    dispatches = sum(counts.values()) - counts.get('cancelled', 0)
    return {"dispatches": dispatches, "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0), "pending": counts.get("dispatched", 0),
            "cancelled_before_send": counts.get('cancelled', 0),
            "completed_batches": outcomes.get("completed", 0),
            "failed_batches": outcomes.get("failed", 0), "pending_batches": outcomes.get("started", 0),
            "cache_hits": outcomes.get("cache_hit", 0), "reported_input_tokens": tokens,
            "usage_complete": unknown == 0, "unknown_usage_requests": unknown,
            "unknown_usage_count_is_upper_bound": True,
            "known_estimate": None if parsed_price is None else float(Decimal(tokens) * parsed_price / per_tokens),
            "price": None if parsed_price is None else float(parsed_price),
            "currency": currency, "per_tokens": per_tokens,
            "unpriced_requests": dispatches if parsed_price is None else 0}

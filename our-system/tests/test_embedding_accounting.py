"""Semantic accounting/retry counterexamples. No real provider/network requests."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from decimal import Decimal
import json
import socket
import sqlite3
import time
import threading
from types import SimpleNamespace

import httpx
import pytest

from app import embedding_accounting as accounting
from app.embedding_accounting import AccountedEmbeddingProvider, embedding_scope, summarize_ledger
from app.embeddings import (EmbeddingBatch, EmbeddingConfig, EmbeddingFailure,
                            HttpEmbeddingProvider, provider_identity, provider_version)
from app.jobs import LeasedEmbeddingProvider


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network is forbidden in embedding accounting tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setenv("ACCOUNTING_OFFLINE_KEY", "offline-nonworking-key")


def config(**changes):
    return replace(EmbeddingConfig(provider="offline", endpoint="https://offline.invalid/embeddings",
                                   model="offline", dimensions=2,
                                   api_key_env="ACCOUNTING_OFFLINE_KEY"), **changes)


class MockProvider:
    implementation_kind = "offline-test"

    def __init__(self):
        self.config = config()
        self.calls = 0
        self.version = "explicit-provider-version-1"

    def embed(self, texts):
        self.calls += 1
        return EmbeddingBatch([[3.0, 4.0] for _ in texts],
                              {"requests": 1, "reported_input_tokens": 7, "usage_complete": True,
                               "provider_extra": "kept"})


def wrapper(tmp_path, provider=None, **options):
    return AccountedEmbeddingProvider(provider or MockProvider(), tmp_path / "embedding.sqlite",
                                      scope_identity="offline-profile", **options)


def summary(provider, **prices):
    return summarize_ledger(provider.ledger_path, **({"price": Decimal("0.5"), "currency": "CNY"} | prices))


def test_ledger_transactions_do_not_serialize_provider_io(tmp_path):
    gate = threading.Barrier(8)
    class ParallelProvider(MockProvider):
        def embed(self, texts):
            gate.wait(timeout=5)
            return super().embed(texts)
    provider = wrapper(tmp_path, ParallelProvider())
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: provider.embed([str(i)]), range(8)))
    assert len(results) == 8 and provider.provider.calls == 8
    assert summary(provider)['dispatches'] == 8


def test_external_ledger_lock_delays_dispatch_until_intent_can_commit(tmp_path):
    provider = wrapper(tmp_path)
    held = sqlite3.connect(provider.ledger_path)
    held.execute('BEGIN IMMEDIATE')
    started = threading.Event()
    def run():
        started.set()
        return provider.embed(['local lock fixture'])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run)
        assert started.wait(timeout=5)
        time.sleep(.05)
        assert provider.provider.calls == 0 and not future.done()
        held.rollback()
        held.close()
        assert future.result(timeout=5).vectors
    assert provider.provider.calls == 1 and summary(provider)['dispatches'] == 1


def test_permission_is_checked_after_ledger_wait_without_duplicate_guard(tmp_path, monkeypatch):
    provider = wrapper(tmp_path)
    allowed = True
    guards = 0
    original = provider._shared.dispatch
    def revoke_during_commit(*args):
        nonlocal allowed
        original(*args)
        allowed = False
    def guard():
        nonlocal guards
        guards += 1
        if not allowed:
            raise PermissionError('revoked during ledger wait')
    monkeypatch.setattr(provider._shared, 'dispatch', revoke_during_commit)
    with pytest.raises(PermissionError):
        provider.embed_guarded(['local evidence'], before_dispatch=guard)
    stats = summary(provider)
    assert guards == 1 and provider.provider.calls == 0
    assert stats['dispatches'] == stats['unknown_usage_requests'] == 0
    assert stats['cancelled_before_send'] == 1


def rows(provider, table):
    assert table in {"embedding_calls", "embedding_dispatches"}
    with sqlite3.connect(provider.ledger_path) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(f"SELECT * FROM {table}")]


def test_add_commit_failure_reuses_success_with_new_lease(tmp_path):
    accounted = wrapper(tmp_path)
    permissions, finished = [], []
    coordinator = SimpleNamespace(
        permit=lambda lease, messages, plan, texts: permissions.append(lease) or len(permissions),
        finish_attempt=lambda *args, **kwargs: finished.append((args, kwargs)))
    heartbeat = SimpleNamespace(check=lambda: None)
    first_lease = LeasedEmbeddingProvider(accounted, coordinator, "lease-1", heartbeat, [], None)
    with pytest.raises(RuntimeError, match="commit failed"):
        with embedding_scope("user", "request+payloadhash", "add"):
            original = first_lease.embed(["document"])
            raise RuntimeError("commit failed")
    second_lease = LeasedEmbeddingProvider(accounted, coordinator, "lease-2", heartbeat, [], None)
    with embedding_scope("user", "request+payloadhash", "add"):
        retry = second_lease.embed(["document"])
    assert retry.vectors == original.vectors
    assert retry.usage["batches"][0]["requests"] == 0
    assert retry.usage["batches"][0]["cache_hit"] is True
    assert accounted.provider.calls == 1
    assert permissions == ["lease-1", "lease-2"] and len(finished) == 2
    assert summary(accounted)["dispatches"] == 1
    assert summary(accounted)["cache_hits"] == 1


def test_query_retries_share_only_one_http_scope_and_delegate_identity(tmp_path):
    accounted = wrapper(tmp_path)
    query = accounted.for_query()
    assert query.for_query() is query
    assert query.config is accounted.config
    assert query.version == accounted.provider.version
    assert provider_identity(query) == provider_identity(accounted.provider)
    assert provider_version(query) == provider_version(accounted.provider)
    with embedding_scope("user", "http-uuid-1", "search"):
        first = query.embed(["question"])
        for _ in range(2):
            retry = accounted.for_query().embed(["question"])
            assert retry.usage["requests"] == 0
            assert retry.vectors == first.vectors
    with embedding_scope("user", "http-uuid-2", "search"):
        query.embed(["question"])
    # Even an incorrectly repeated HTTP UUID cannot reuse a prior query scope.
    with embedding_scope("user", "http-uuid-1", "search"):
        query.embed(["question"])
    assert accounted.provider.calls == 3
    assert summary(accounted)["cache_hits"] == 2


@pytest.mark.parametrize("change", ["user", "request", "phase", "text", "order", "boundary",
                                     "version", "provider", "revision", "implementation", "role"])
def test_isolation_and_exact_batch_identity(tmp_path, change):
    accounted = wrapper(tmp_path)
    texts, user, request, phase = ["a", "bc"], "user", "request", "add"
    with embedding_scope(user, request, phase):
        accounted.embed(texts)
        if change == "role":
            accounted.for_query().embed(texts)
    if change == "user":
        user += "-2"
    elif change == "request":
        request += "-payload2"
    elif change == "phase":
        phase = "search"
    elif change == "text":
        texts = ["a ", "bc"]
    elif change == "order":
        texts = list(reversed(texts))
    elif change == "boundary":
        texts = ["ab", "c"]
    elif change == "version":
        accounted.provider.version = "explicit-provider-version-2"
    elif change in {"provider", "revision"}:
        accounted.provider.config = replace(accounted.config, **{change: "changed"})
    elif change == "implementation":
        accounted.provider.implementation_kind = "different-test-implementation"
    if change != "role":
        with embedding_scope(user, request, phase):
            accounted.embed(texts)
    assert accounted.provider.calls == 2
    assert summary(accounted)["cache_hits"] == 0


def test_scope_identity_is_hashed_and_isolates_wrappers(tmp_path):
    provider = MockProvider()
    a = AccountedEmbeddingProvider(provider, tmp_path / "ledger.sqlite", "deployment-A")
    b = AccountedEmbeddingProvider(provider, a.ledger_path, "deployment-B")
    with embedding_scope("user", "request", "add"):
        a.embed(["text"])
        b.embed(["text"])
    assert provider.calls == 2
    calls = rows(a, "embedding_calls")
    assert calls[0]["scope_identity_hash"] != calls[1]["scope_identity_hash"]
    assert all(len(call["scope_identity_hash"]) == 64 for call in calls)


@pytest.mark.parametrize("guarded", [False, True])
def test_transparent_wrapper_preserves_list_argument_shape(tmp_path, guarded):
    class ListSensitive(MockProvider):
        def embed(self, texts):
            assert isinstance(texts, list) and texts == ["find target"]
            return super().embed(texts)
    class GuardedListSensitive(ListSensitive):
        def embed_guarded(self, texts, *, before_dispatch):
            assert isinstance(texts, list) and texts == ["find target"]
            before_dispatch()
            return self.embed(texts)
    accounted = wrapper(tmp_path, GuardedListSensitive() if guarded else ListSensitive())
    with embedding_scope("user", "request", "add"):
        accounted.embed(["find target"])
        accounted.embed(["find target"])
    assert accounted.provider.calls == 1


def test_expiration_is_finite_and_hits_do_not_extend_ttl(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(accounting, "time", SimpleNamespace(time=time.time, monotonic=lambda: now[0]))
    accounted = wrapper(tmp_path, cache_ttl_seconds=10)
    with embedding_scope("user", "request", "add"):
        accounted.embed(["text"])
        now[0] += 9
        assert accounted.embed(["text"]).usage["cache_hit"]
        now[0] += 1
        accounted.embed(["text"])
    assert accounted.provider.calls == 2


def test_lru_entry_and_scalar_bounds(tmp_path):
    accounted = wrapper(tmp_path, max_cache_entries=2, max_cache_values=4)
    with embedding_scope("user", "request", "add"):
        accounted.embed(["a"])
        accounted.embed(["b"])
        accounted.embed(["a"])  # a is the most recently used entry.
        accounted.embed(["c"])
        accounted.embed(["a"])
        accounted.embed(["b"])  # b was evicted.
        # A batch above the scalar cap must never stay in memory.
        accounted.embed(["a", "b", "c"])
        accounted.embed(["a", "b", "c"])
    assert accounted.provider.calls == 6
    scalar_limited = wrapper(tmp_path / "scalars", max_cache_entries=100, max_cache_values=2)
    with embedding_scope("user", "request", "add"):
        scalar_limited.embed(["a"])
        scalar_limited.embed(["b"])
        scalar_limited.embed(["a"])
    assert scalar_limited.provider.calls == 3


@pytest.mark.parametrize("options", [{"max_cache_entries": 0}, {"max_cache_values": 0}])
def test_zero_capacity_disables_cache(tmp_path, options):
    accounted = wrapper(tmp_path, **options)
    with embedding_scope("user", "request", "add"):
        accounted.embed(["text"])
        accounted.embed(["text"])
    assert accounted.provider.calls == 2


def test_revoked_permission_blocks_cached_vectors_and_does_not_dispatch(tmp_path):
    accounted = wrapper(tmp_path)
    permitted = [True]
    checks = []
    def guard():
        checks.append(permitted[0])
        if not permitted[0]:
            raise PermissionError("private denial text")
    with embedding_scope("user", "request", "add"):
        accounted.embed_guarded(["text"], before_dispatch=guard)
        permitted[0] = False
        with pytest.raises(PermissionError):
            accounted.embed_guarded(["text"], before_dispatch=guard)
        with pytest.raises(PermissionError):
            accounted.embed_guarded(["new text"], before_dispatch=guard)
    assert checks == [True, False, False]
    result = summary(accounted)
    assert accounted.provider.calls == result["dispatches"] == 1
    assert result["cache_hits"] == 0 and result["failed"] == 0
    assert result["failed_batches"] == 2


def test_no_scope_and_nested_scope_restoration(tmp_path):
    accounted = wrapper(tmp_path)
    for _ in range(2):
        accounted.embed(["text"])
    with embedding_scope("outer", "request", "add"):
        accounted.embed(["text"])
        with pytest.raises(RuntimeError):
            with embedding_scope("inner", "request", "add"):
                accounted.embed(["text"])
                raise RuntimeError("exit inner")
        assert accounted.embed(["text"]).usage["cache_hit"]
    accounted.embed(["text"])
    assert accounted.provider.calls == 5
    assert sum(call["phase"] == "unscoped" for call in rows(accounted, "embedding_calls")) == 3


def test_restart_keeps_ledger_but_discards_cache(tmp_path):
    provider = MockProvider()
    first = wrapper(tmp_path, provider)
    with embedding_scope("user", "request", "add"):
        first.embed(["text"])
    del first
    restarted = wrapper(tmp_path, provider)
    with embedding_scope("user", "request", "add"):
        restarted.embed(["text"])
    assert provider.calls == summary(restarted)["dispatches"] == 2
    assert summary(restarted)["reported_input_tokens"] == 14
    assert summary(restarted)["cache_hits"] == 0


@pytest.mark.parametrize("failure", ["timeout", "refusal", "exception", "count", "nan", "zero", "dimension", "bool"])
def test_failures_and_invalid_vectors_never_cache(tmp_path, failure):
    class Failing(MockProvider):
        def embed(self, texts):
            result = super().embed(texts)
            if self.calls != 1:
                return result
            if failure == "timeout":
                raise httpx.ReadTimeout("private error body")
            if failure == "refusal":
                raise EmbeddingFailure("refused-private-body", {"usage_complete": False})
            if failure == "exception":
                raise RuntimeError("private error body")
            result.vectors = {"count": [], "nan": [[float("nan"), 1]], "zero": [[0, 0]],
                              "dimension": [[1]], "bool": [[True, 1]]}[failure]
            return result
    accounted = wrapper(tmp_path, Failing())
    with embedding_scope("user", "request", "add"):
        with pytest.raises((EmbeddingFailure, RuntimeError, httpx.ReadTimeout)) as error:
            accounted.embed(["text"])
        if failure in {"count", "nan", "zero", "dimension", "bool"}:
            assert error.value.usage["reported_input_tokens"] == 7
        second = accounted.embed(["text"])
        third = accounted.embed(["text"])
    assert accounted.provider.calls == 2 and second.vectors == third.vectors
    result = summary(accounted)
    assert result["dispatches"] == 2 and result["failed"] == 1 and result["cache_hits"] == 1
    assert b"private error body" not in accounted.ledger_path.read_bytes()
    assert b"refused-private-body" not in accounted.ledger_path.read_bytes()


def response(request, *, tokens=7, native=False):
    payload = json.loads(request.content)
    texts = payload["input"]["texts"] if native else payload["input"]
    dimension = payload["parameters"]["dimension"] if native else payload["dimensions"]
    data = [{"text_index" if native else "index": i, "embedding": [1.0] * dimension}
            for i in range(len(texts))]
    body = {"model": payload["model"], "usage": {"total_tokens": tokens}}
    body.update({"output": {"embeddings": data}} if native else {"data": data})
    return httpx.Response(200, json=body)


@pytest.mark.parametrize("first_failure", ["503", "timeout"])
def test_each_real_http_retry_is_recorded_before_transport_and_cache_is_not_sent(tmp_path, monkeypatch, first_failure):
    seen = []
    checks = []
    def transport(request):
        # Inspect through a new connection inside the transport: it is committed.
        dispatched = rows(accounted, "embedding_dispatches")
        assert len(dispatched) == len(seen) + 1
        assert dispatched[-1]["status"] == "dispatched"
        seen.append(request)
        if len(seen) == 1:
            if first_failure == "timeout":
                raise httpx.ReadTimeout("offline timeout")
            return httpx.Response(503)
        return response(request)
    monkeypatch.setattr("app.embeddings.time.sleep", lambda seconds: None)
    raw = HttpEmbeddingProvider(config(max_retries=1), transport=httpx.MockTransport(transport))
    accounted = wrapper(tmp_path, raw)
    with embedding_scope("user", "request", "add"):
        initial = accounted.embed_guarded(["text"], before_dispatch=lambda: checks.append(True))
        retry = accounted.embed_guarded(["text"], before_dispatch=lambda: checks.append(True))
    assert initial.usage["requests"] == 2 and retry.usage["requests"] == 0
    assert len(seen) == 2 and len(checks) == 3
    result = summary(accounted)
    assert (result["dispatches"], result["completed"], result["failed"], result["cache_hits"]) == (2, 1, 1, 1)
    assert result["reported_input_tokens"] == 7 and result["known_estimate"] == 0.0000035
    assert result["unknown_usage_requests"] == 2 and result["usage_complete"] is False


def test_guard_failure_during_http_retry_adds_no_dispatch(tmp_path, monkeypatch):
    seen, checks = [], []
    def guard():
        checks.append(True)
        if len(checks) == 2:
            raise PermissionError("lease revoked")
    def transport(request):
        seen.append(request)
        return httpx.Response(503)
    monkeypatch.setattr("app.embeddings.time.sleep", lambda seconds: None)
    raw = HttpEmbeddingProvider(config(max_retries=1), transport=httpx.MockTransport(transport))
    accounted = wrapper(tmp_path, raw)
    with pytest.raises(PermissionError):
        accounted.embed_guarded(["text"], before_dispatch=guard)
    result = summary(accounted)
    assert len(seen) == result["dispatches"] == result["failed"] == 1


def test_ledger_dispatch_commit_failure_prevents_provider_execution(tmp_path):
    accounted = wrapper(tmp_path)
    with sqlite3.connect(accounted.ledger_path) as db:
        db.executescript("""CREATE TRIGGER prevent_dispatch BEFORE INSERT ON embedding_dispatches
                            BEGIN SELECT RAISE(ABORT, 'offline write failure'); END;""")
    with embedding_scope("user", "request", "add"):
        with pytest.raises(sqlite3.DatabaseError, match="offline write failure"):
            accounted.embed(["text"])
    assert accounted.provider.calls == summary(accounted)["dispatches"] == 0
    with sqlite3.connect(accounted.ledger_path) as db:
        db.execute("DROP TRIGGER prevent_dispatch")
    with embedding_scope("user", "request", "add"):
        accounted.embed(["text"])
    assert accounted.provider.calls == 1


def test_retries_exhausted_are_all_failed_unknown_and_uncached(tmp_path, monkeypatch):
    seen = []
    def transport(request):
        seen.append(request)
        return httpx.Response(503)
    monkeypatch.setattr("app.embeddings.time.sleep", lambda seconds: None)
    raw = HttpEmbeddingProvider(config(max_retries=1), transport=httpx.MockTransport(transport))
    accounted = wrapper(tmp_path, raw)
    with embedding_scope("user", "request", "add"):
        for _ in range(2):
            with pytest.raises(EmbeddingFailure):
                accounted.embed(["text"])
    result = summary(accounted)
    assert len(seen) == result["dispatches"] == result["failed"] == result["unknown_usage_requests"] == 4
    assert result["cache_hits"] == result["completed"] == 0


def test_refusal_response_is_not_automatically_retried_or_cached(tmp_path):
    seen = []
    def transport(request):
        seen.append(request)
        return httpx.Response(403, text="private supplier refusal body")
    raw = HttpEmbeddingProvider(config(max_retries=2), transport=httpx.MockTransport(transport))
    accounted = wrapper(tmp_path, raw)
    with embedding_scope("user", "request", "add"):
        with pytest.raises(EmbeddingFailure, match="embedding_http_403"):
            accounted.embed(["text"])
    assert len(seen) == summary(accounted)["dispatches"] == 1
    assert summary(accounted)["cache_hits"] == 0
    assert b"private supplier refusal body" not in accounted.ledger_path.read_bytes()


def test_multiple_batches_preserve_usage_and_reuse_only_finished_batches(tmp_path):
    sent = []
    def transport(request):
        sent.append(json.loads(request.content)["input"])
        if len(sent) == 2:
            return httpx.Response(403)
        return response(request)
    raw = HttpEmbeddingProvider(config(batch_size=1), transport=httpx.MockTransport(transport))
    accounted = wrapper(tmp_path, raw)
    with embedding_scope("user", "request", "add"):
        with pytest.raises(EmbeddingFailure) as error:
            accounted.embed(["a", "b"])
        assert error.value.usage["requests"] == 2
        assert error.value.usage["reported_input_tokens"] == 7
        result = accounted.embed(["a", "b"])
    assert sent == [["a"], ["b"], ["b"]]
    assert result.usage["requests"] == 1 and result.usage["cache_hits"] == 1
    assert len(result.vectors) == 2
    counts = summary(accounted)
    assert counts["dispatches"] == 3 and counts["completed"] == 2 and counts["failed"] == 1
    assert counts["reported_input_tokens"] == 14


def test_native_query_view_preserves_provider_rate_gate_and_document_query_semantics(tmp_path, monkeypatch):
    native = config(provider="dashscope", model="text-embedding-v4", dimensions=64,
                    endpoint="https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
                    min_request_interval_seconds=0.01)
    sent = []
    def transport(request):
        sent.append(json.loads(request.content)["parameters"]["text_type"])
        return response(request, native=True)
    raw = HttpEmbeddingProvider(native, transport=httpx.MockTransport(transport))
    accounted = wrapper(tmp_path, raw)
    query = accounted.for_query()
    # Use the native gate's clock without depending on real scheduler timing.
    now = [100.0]
    monkeypatch.setattr("app.embeddings.time.monotonic", lambda: now[0])
    assert provider_version(query) == provider_version(raw)
    with embedding_scope("user", "http-uuid", "search"):
        accounted.embed(["same text"])
        with pytest.raises(EmbeddingFailure, match="embedding_rate_limited"):
            query.embed(["same text"])
        now[0] += 1
        assert query.embed(["same text"]).usage["text_type"] == "query"
        assert query.embed(["same text"]).usage["requests"] == 0
    assert sent == ["document", "query"]
    assert summary(accounted)["dispatches"] == 2


def test_native_all_input_preflight_happens_before_first_dispatch(tmp_path):
    def forbidden(request):
        pytest.fail("late invalid input must prevent every dispatch")
    native = config(provider="dashscope", model="text-embedding-v4", dimensions=64,
                    endpoint="https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
                    batch_size=1)
    accounted = wrapper(tmp_path, HttpEmbeddingProvider(native, transport=httpx.MockTransport(forbidden)))
    with pytest.raises(EmbeddingFailure, match="input_too_long"):
        accounted.embed(["valid", "x" * 8193])
    assert summary(accounted)["dispatches"] == 0


def test_native_response_identity_survives_mixed_cached_and_sent_batches(tmp_path):
    native = config(provider="dashscope", model="text-embedding-v4", dimensions=64,
                    endpoint="https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
                    batch_size=1)
    sent = []
    def transport(request):
        sent.append(json.loads(request.content)["input"]["texts"])
        body = response(request, native=True).json()
        body["request_id"] = f"offline-request-{len(sent)}"
        return httpx.Response(200, json=body)
    accounted = wrapper(tmp_path, HttpEmbeddingProvider(native, transport=httpx.MockTransport(transport)))
    with embedding_scope("user", "request", "add"):
        original = accounted.embed(["a"])
        assert original.usage["request_ids"] == ["offline-request-1"]
        mixed = accounted.embed(["a", "b"])
        assert mixed.usage["request_ids"] == ["offline-request-2"]
        assert mixed.usage["response_models"] == ["text-embedding-v4"]
        assert mixed.usage["model_identity_source"] == "response"
        assert mixed.usage["text_type"] == "document"
        assert mixed.usage["requests"] == 1 and mixed.usage["reported_input_tokens"] == 7
        full = accounted.embed(["c", "d"])
        assert full.usage["request_ids"] == ["offline-request-3", "offline-request-4"]
        assert full.usage["response_models"] == ["text-embedding-v4"] * 2
        assert full.usage["model_identity_source"] == "response"
        assert full.usage["requests"] == 2 and full.usage["reported_input_tokens"] == 14
    assert summary(accounted)["reported_input_tokens"] == 28


def test_ledger_is_private_and_independent_of_locked_application_database(tmp_path):
    main_db = sqlite3.connect(tmp_path / "main.sqlite")
    try:
        main_db.execute("CREATE TABLE app_data (id INTEGER)")
        main_db.execute("BEGIN EXCLUSIVE")
        accounted = wrapper(tmp_path)
        with embedding_scope("private-user@example.test", "private-request-body", "private-phase-label"):
            first = accounted.embed(["private original text"])
            first.vectors[0][0] = 9000  # Caller mutation must not corrupt cached data.
            second = accounted.embed(["private original text"])
            assert second.vectors == [[3.0, 4.0]]
            second.vectors[0][0] = 8000
            assert accounted.embed(["private original text"]).vectors == [[3.0, 4.0]]
        encoded = accounted.ledger_path.read_bytes()
        for private in (b"private-user", b"private-request", b"private-phase", b"private original",
                        b"offline-nonworking-key", b"provider_extra"):
            assert private not in encoded
        assert all(len(call["user_hash"]) == len(call["request_hash"]) == 64
                   for call in rows(accounted, "embedding_calls"))
        assert summary(accounted)["dispatches"] == 1
    finally:
        main_db.rollback()
        main_db.close()


def test_crash_after_dispatch_is_pending_unknown_and_does_not_create_cache(tmp_path):
    accounted = wrapper(tmp_path)
    # Simulate abrupt death, which cannot run the provider exception handler.
    metadata = (None, None, None, "unscoped", "s" * 64, "p" * 64, "v" * 64, "query", "i" * 64, 1)
    call_id = accounted._shared.start(metadata)
    accounted._shared.dispatch(call_id, 1)
    restarted = wrapper(tmp_path)
    result = summary(restarted)
    assert result["pending"] == result["pending_batches"] == result["unknown_usage_requests"] == 1
    assert result["reported_input_tokens"] == 0 and not result["usage_complete"]
    assert result["known_estimate"] == 0  # Reported portion only; unknown is separate.
    assert summary(restarted, price=None)["known_estimate"] is None


def test_summary_explicit_price_and_missing_usage_are_not_free_calls(tmp_path):
    class MissingUsage(MockProvider):
        def embed(self, texts):
            result = super().embed(texts)
            result.usage = {}
            return result
    accounted = wrapper(tmp_path, MissingUsage())
    accounted.embed(["text"])
    result = summary(accounted, price=None, currency="USD")
    assert result["known_estimate"] is None and result["unpriced_requests"] == 1
    assert result["unknown_usage_requests"] == 1 and result["usage_complete"] is False
    with pytest.raises(TypeError):
        summarize_ledger(accounted.ledger_path)
    for invalid in (-1, float("inf"), float("nan"), True):
        with pytest.raises(ValueError):
            summary(accounted, price=invalid)
    priced = wrapper(tmp_path / "priced")
    priced.embed(["text"])
    assert summary(priced, price=Decimal("0.0005"), per_tokens=1000)["known_estimate"] == 0.0000035
    assert summary(priced)["usage_complete"]


def test_concurrent_contexts_are_isolated_and_query_views_share_cache_capacity(tmp_path):
    accounted = wrapper(tmp_path, max_cache_entries=2)
    def work(user):
        with embedding_scope(user, "request", "add"):
            return accounted.embed(["text"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, ["a", "b", "c", "d"]))
        with embedding_scope("outer", "request", "add"):
            accounted.embed(["text"])
            reused = pool.submit(copy_context().run, accounted.embed, ["text"]).result()
            assert reused.usage["cache_hit"]
    assert len(results) == 4 and summary(accounted)["dispatches"] == 5
    assert len({row["user_hash"] for row in rows(accounted, "embedding_calls")}) == 5
    with embedding_scope("user", "request", "search"):
        accounted.embed(["d1"])
        accounted.embed(["d2"])
        accounted.for_query().embed(["q"])
        assert "cache_hit" not in accounted.embed(["d1"]).usage


@pytest.mark.parametrize("options", [{"max_cache_entries": -1}, {"max_cache_entries": True},
                                     {"max_cache_values": -1}, {"cache_ttl_seconds": 0},
                                     {"cache_ttl_seconds": float("inf")}, {"cache_ttl_seconds": float("nan")}])
def test_unbounded_or_invalid_cache_configuration_is_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        wrapper(tmp_path, **options)

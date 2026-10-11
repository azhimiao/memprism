"""Deterministic vector/transaction tests. None measures real semantic quality."""
from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.embeddings import EmbeddingBatch, EmbeddingConfig, EmbeddingFailure, provider_version
from app.models import SearchRequest
from app.store import SqliteStore


class DeterministicEmbedding:
    implementation_kind = "test_double"

    def __init__(self, *, revision="1", fail_on=None, barrier=None, model="test-vector"):
        self.config = EmbeddingConfig(provider="unit-test", endpoint="https://embedding.test/v1/embeddings",
                                      model=model, dimensions=2, revision=revision)
        self.calls = []
        self.fail_on = fail_on
        self.barrier = barrier

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.barrier:
            self.barrier.wait(timeout=5)
        if self.fail_on and any(self.fail_on in t for t in texts):
            raise EmbeddingFailure("embedding_timeout", {"requests": 1, "usage_complete": False})
        vectors = [[1.0, 0.0] if "apple" in t else [0.0, 1.0] for t in texts]
        return EmbeddingBatch(vectors, {"requests": 1, "implementation": "test_double"})


def payload(request="r1", user="a", contents=None):
    return {"request_id": request, "user_id": user, "session_id": "session",
            "messages": [{"role": "user", "content": t} for t in (contents or ["apple red"])]}


def query(client, user="a", text="apple", top_k=100):
    return client.post("/search", json={"query": text, "user_id": user, "top_k": top_k})


def settings(path, mode="raw_dense", **kwargs):
    return Settings(db_path=path, retrieval_mode=mode, **kwargs)


@pytest.mark.parametrize("mode", ["raw_dense", "raw_hybrid_rrf"])
def test_vectors_persist_and_replays_conflicts_do_not_call_model(tmp_path, mode):
    path = tmp_path / "m.sqlite"
    provider = DeterministicEmbedding()
    with TestClient(create_app(settings(path, mode), embedding_provider=provider)) as client:
        assert client.post("/add", json=payload(contents=["apple red", "banana yellow"])).status_code == 200
        before_calls = len(provider.calls)
        assert client.post("/add", json=payload(contents=["apple red", "banana yellow"])).status_code == 200
        assert client.post("/add", json=payload(contents=["DIFFERENT"])).status_code == 409
        assert len(provider.calls) == before_calls
        first = query(client)
        assert first.headers["X-AML-Actual-Mode"] == mode
        assert first.headers["X-AML-Degraded"] == "false"
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM raw_vectors").fetchone()[0] == 2
        assert db.execute("SELECT status FROM embedding_requests").fetchone()[0] == "ready"
    fresh = DeterministicEmbedding()
    with TestClient(create_app(settings(path, mode), embedding_provider=fresh)) as client:
        assert client.post("/add", json=payload(contents=["apple red", "banana yellow"])).status_code == 200
        assert fresh.calls == []
        assert query(client).json() == first.json()


def test_failed_add_embedding_is_atomic_raw_success_and_replay_stays_failed(tmp_path):
    path = tmp_path / "m.sqlite"
    provider = DeterministicEmbedding(fail_on="banana")
    app = create_app(settings(path), embedding_provider=provider)
    with TestClient(app) as client:
        body = payload(contents=["apple", "banana"])
        assert client.post("/add", json=body).status_code == 200
        assert client.post("/add", json=body).status_code == 200
        assert len(provider.calls) == 1
        result = query(client)
        assert result.headers["X-AML-Actual-Mode"] == "raw_fts"
        assert result.headers["X-AML-Degraded-Reason"] == "incomplete_current_version"
        assert len(provider.calls) == 1  # no query cost on incomplete coverage
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM raw_vectors").fetchone()[0] == 0
        assert db.execute("SELECT status, error_code FROM embedding_requests").fetchone() == ("failed", "embedding_timeout")
        assert db.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 2


def test_query_timeout_uses_full_fts_budget(tmp_path):
    path = tmp_path / "m.sqlite"
    provider = DeterministicEmbedding(fail_on="timeout")
    app = create_app(settings(path, "raw_hybrid_rrf", candidate_limit=4, lexical_quota=2, dense_quota=2), embedding_provider=provider)
    with TestClient(app) as client:
        assert client.post("/add", json=payload(contents=[f"apple {i}" for i in range(4)])).status_code == 200
        result = query(client, text="apple timeout")
        assert result.headers["X-AML-Degraded-Reason"] == "embedding_timeout"
        assert len(result.json()["data"]) == 4
    with TestClient(create_app(settings(path, "raw_fts", candidate_limit=4, lexical_quota=2, dense_quota=2))) as client:
        assert query(client, text="apple timeout").json() == result.json()


@pytest.mark.parametrize("change", ["model", "endpoint", "revision"])
def test_old_version_does_not_block_replay_or_reuse_vectors(tmp_path, change):
    path = tmp_path / "m.sqlite"
    p1 = DeterministicEmbedding()
    with TestClient(create_app(settings(path), embedding_provider=p1)) as client:
        assert client.post("/add", json=payload()).status_code == 200
    p2 = DeterministicEmbedding()
    p2.config = replace(p2.config, **{change: "https://other.test/embeddings" if change == "endpoint" else "other"})
    with TestClient(create_app(settings(path), embedding_provider=p2)) as client:
        assert client.post("/add", json=payload()).status_code == 200
        assert p2.calls == []
        result = query(client)
        assert result.headers["X-AML-Degraded-Reason"] == "incomplete_current_version"
        assert p2.calls == []
        assert client.post("/add", json=payload(request="r2")).status_code == 200
    with sqlite3.connect(path) as db:
        assert {r[0] for r in db.execute("SELECT DISTINCT embedding_version FROM raw_vectors")} == {provider_version(p1), provider_version(p2)}


def test_partial_user_coverage_and_other_user_failure_are_isolated(tmp_path):
    path = tmp_path / "m.sqlite"
    provider = DeterministicEmbedding(fail_on="fail")
    app = create_app(settings(path), embedding_provider=provider)
    with TestClient(app) as client:
        client.post("/add", json=payload(contents=["apple", "banana"]))
        before = query(client).json()
        client.post("/add", json=payload(user="b", contents=["apple fail"]))
        after = query(client)
        assert after.json() == before
        assert after.headers["X-AML-Actual-Mode"] == "raw_dense"
        calls = len(provider.calls)
        assert query(client, user="unknown").json() == {"data": []}
        assert len(provider.calls) == calls
        client.post("/add", json=payload(request="r2", contents=["apple fail"]))
        degraded = query(client)
        assert degraded.headers["X-AML-Actual-Mode"] == "raw_fts"
        assert {r["content"] for r in degraded.json()["data"]} == {"apple", "apple fail"}


def test_rrf_trace_recomputes_scores_and_budgets(tmp_path):
    provider = DeterministicEmbedding()
    app = create_app(settings(tmp_path / "m.sqlite", "raw_hybrid_rrf", candidate_limit=4, lexical_quota=2, dense_quota=2), embedding_provider=provider)
    with TestClient(app) as client:
        client.post("/add", json=payload(contents=["apple red", "apple green", "banana"]))
        result, trace = app.state.retrieval.search_with_trace(SearchRequest(query="apple", user_id="a", top_k=2))
        assert trace["candidates_before_dedup"] <= 4
        assert len(trace["lexical_ids"]) <= 2 and len(trace["dense_ids"]) <= 2
        assert len({r.id for r in result.data}) == len(result.data) <= 2
        for item in trace["fusion"]:
            assert item["score"] == sum(1 / (trace["rrf_k"] + rank) for rank in item["ranks"].values())


def test_whole_message_budget_and_deleted_sources(tmp_path):
    path = tmp_path / "m.sqlite"
    provider = DeterministicEmbedding()
    with TestClient(create_app(settings(path, return_char_budget=4), embedding_provider=provider)) as client:
        client.post("/add", json=payload(contents=["apple long", "tiny"]))
        assert query(client).json() == {"data": []}  # prefix does not skip to the short second hit
    with sqlite3.connect(path) as db:
        db.execute("UPDATE raw_messages SET deletion_state='deleted' WHERE content='apple long'")
    with TestClient(create_app(settings(path), embedding_provider=provider)) as client:
        assert [h["content"] for h in query(client).json()["data"]] == ["tiny"]


def test_database_error_rolls_back_raw_fts_vectors_state_and_ledger(tmp_path, monkeypatch):
    path = tmp_path / "m.sqlite"
    app = create_app(settings(path), embedding_provider=DeterministicEmbedding())
    original = app.state.store._insert_embeddings

    def fail(connection, messages, prepared):
        original(connection, messages, prepared)
        raise sqlite3.OperationalError("injected after vector and status write")

    monkeypatch.setattr(app.state.store, "_insert_embeddings", fail)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/add", json=payload()).status_code == 500
    with sqlite3.connect(path) as db:
        for table in ("raw_messages", "raw_vectors", "embedding_requests", "add_requests", "fts_partitions"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


@pytest.mark.parametrize("conflict", [False, True])
def test_independent_instances_commit_only_winners_vectors(tmp_path, conflict):
    path = tmp_path / "m.sqlite"
    barrier = threading.Barrier(2)
    providers = [DeterministicEmbedding(barrier=barrier), DeterministicEmbedding(barrier=barrier)]
    apps = [create_app(settings(path), embedding_provider=p) for p in providers]
    bodies = [payload(contents=["apple"]), payload(contents=["banana" if conflict else "apple"])]

    def submit(i):
        with TestClient(apps[i]) as client:
            return client.post("/add", json=bodies[i]).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(submit, range(2)))
    assert sorted(codes) == ([200, 409] if conflict else [200, 200])
    assert sum(len(p.calls) for p in providers) == 2  # separately billed requests may race
    with sqlite3.connect(path) as db:
        body = db.execute("SELECT content FROM raw_messages").fetchone()[0]
        vectors = db.execute("SELECT vector_json FROM raw_vectors").fetchall()
        assert len(vectors) == 1
        assert json.loads(vectors[0][0]) == ([1.0, 0.0] if body == "apple" else [0.0, 1.0])
        assert db.execute("SELECT COUNT(*) FROM embedding_requests").fetchone()[0] == 1


def test_missing_configuration_is_visible_even_with_no_user_data(tmp_path):
    with TestClient(create_app(settings(tmp_path / "m.sqlite"))) as client:
        assert query(client).headers["X-AML-Degraded-Reason"] == "missing_configuration"
        assert client.post("/add", json=payload()).status_code == 200
        result = query(client)
        assert result.headers["X-AML-Actual-Mode"] == "raw_fts"
        assert result.json()["data"]


def test_fake_and_http_identity_cannot_share_cached_vectors():
    p = DeterministicEmbedding()
    fake = provider_version(p)
    p.implementation_kind = "http"
    assert provider_version(p) != fake


def test_same_instance_concurrent_replay_embeds_once(tmp_path):
    provider = DeterministicEmbedding()
    app = create_app(settings(tmp_path / "m.sqlite"), embedding_provider=provider)

    def submit(_):
        with TestClient(app) as client:
            return client.post("/add", json=payload()).status_code

    with ThreadPoolExecutor(max_workers=6) as pool:
        assert list(pool.map(submit, range(6))) == [200] * 6
    assert len(provider.calls) == 1


def test_late_failed_attempt_cannot_overwrite_winner_and_cost_is_logged(tmp_path, caplog):
    caplog.set_level("INFO", logger="aml.ingest")
    barrier, winner_committed = threading.Barrier(2), threading.Event()

    class LateFailure(DeterministicEmbedding):
        def embed(self, texts):
            self.calls.append(list(texts))
            barrier.wait(timeout=5)
            assert winner_committed.wait(timeout=5)
            raise EmbeddingFailure("embedding_timeout", {"requests": 1})

    path = tmp_path / "m.sqlite"
    p1, p2 = DeterministicEmbedding(barrier=barrier), LateFailure()
    apps = [create_app(settings(path), embedding_provider=p) for p in (p1, p2)]

    def submit(i):
        with TestClient(apps[i]) as client:
            code = client.post("/add", json=payload()).status_code
        if i == 0:
            winner_committed.set()
        return code

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(submit, (0, 1))) == [200, 200]
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status FROM embedding_requests").fetchall() == [("ready",)]
        assert db.execute("SELECT COUNT(*) FROM raw_vectors").fetchone()[0] == 1
    attempts = [r.message for r in caplog.records if r.message.startswith("embedding_attempt ")]
    assert len(attempts) == 2
    assert "apple red" not in "\n".join(attempts)


def test_coverage_is_rechecked_after_query_model_call(tmp_path):
    path = tmp_path / "m.sqlite"
    raw_app = create_app(settings(path, "raw_fts"))

    class AddWhileQuerying(DeterministicEmbedding):
        def embed(self, texts):
            if list(texts) == ["apple query"]:
                with TestClient(raw_app) as raw_client:
                    assert raw_client.post("/add", json=payload(request="late", contents=["apple late"])).status_code == 200
            return super().embed(texts)

    provider = AddWhileQuerying()
    app = create_app(settings(path), embedding_provider=provider)
    with TestClient(app) as client:
        client.post("/add", json=payload())
        result = query(client, text="apple query")
        assert result.headers["X-AML-Degraded-Reason"] == "incomplete_current_version"
        assert {h["content"] for h in result.json()["data"]} == {"apple red", "apple late"}

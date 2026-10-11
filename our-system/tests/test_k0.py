"""Formal route/failure tests. All vectors are engineering doubles, no quality evidence."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from contextlib import contextmanager, closing

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.embeddings import EmbeddingBatch, EmbeddingConfig, EmbeddingFailure, HttpEmbeddingProvider
from app.jobs import NotReady, SnapshotChanged
from app.models import AddRequest, SearchRequest


@contextmanager
def connected(path):
    with closing(sqlite3.connect(path)) as db:
        with db:
            yield db


class Provider:
    implementation_kind = "engineering_double"

    def __init__(self, failure=None, action=None, revision="1"):
        self.config = EmbeddingConfig(provider="engineering", endpoint="https://invalid.test/embeddings",
                                      dimensions=2, revision=revision)
        self.failure, self.action, self.calls = failure, action, []

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.action:
            self.action()
        if self.failure == "unexpected":
            raise RuntimeError("sensitive-provider-body-do-not-expose")
        if self.failure == "dimensions":
            return EmbeddingBatch([[1.0] for _ in texts])
        if self.failure:
            raise EmbeddingFailure(self.failure, {"requests": 1})
        return EmbeddingBatch([[1.0, 0.0] for _ in texts], {"requests": 1})


def body(request="r", contents=("alpha",), user="u"):
    return {"request_id": request, "user_id": user, "session_id": "s",
            "messages": [{"role": "user", "content": s} for s in contents]}


def app_at(path, provider=None, profile="dense_sync", **kwargs):
    return create_app(Settings(db_path=path, profile_id=profile, **kwargs), embedding_provider=provider)


def add(app, request="r", contents=("alpha",), user="u"):
    with TestClient(app) as client:
        return client.post("/add", json=body(request, contents, user))


def search(app, text="alpha", user="u"):
    with TestClient(app) as client:
        return client.post("/search", json={"query": text, "user_id": user, "top_k": 20})


def state(path):
    with connected(path) as db:
        return {table: db.execute(f"SELECT * FROM {table}").fetchall() for table in (
            "raw_messages", "raw_vectors", "add_requests", "k0_receipts", "k0_watermarks", "k0_users")}


@pytest.mark.parametrize("profile", ["raw_sync", "dense_sync"])
def test_add_immediate_search_receipt_watermark_and_replay(tmp_path, profile):
    path, provider = tmp_path / "m.db", Provider()
    app = app_at(path, provider, profile)
    result = add(app)
    assert result.status_code == 200
    assert search(app).json()["data"][0]["content"] == "alpha"
    before, calls = state(path), len(provider.calls)
    assert add(app).json() == result.json()
    assert state(path) == before and len(provider.calls) == calls
    assert add(app, contents=("different",)).status_code == 409
    assert len(provider.calls) == calls
    with connected(path) as db:
        marks = db.execute("SELECT route,commit_seq,epoch,ready FROM k0_watermarks").fetchall()
        assert set(marks) == {(route, 1, 1, 1) for route in app.state.profile.required_routes}
        assert db.execute("SELECT state FROM k0_jobs").fetchone()[0] == "committed"
    assert add(app, user="other").status_code == 200
    assert len(search(app, user="u").json()["data"]) == 1


@pytest.mark.parametrize("failure", ["embedding_timeout", "dimensions", "unexpected", "embedding_http_500"])
def test_required_add_failure_publishes_nothing(tmp_path, failure):
    path, provider = tmp_path / "m.db", Provider(failure)
    app = app_at(path, provider)
    result = add(app, contents=("alpha", "beta"))
    assert result.status_code == 503 and result.headers["Retry-After"] == "1"
    assert "sensitive" not in result.text
    assert all(not rows for rows in state(path).values())
    with connected(path) as db:
        assert db.execute("SELECT state FROM k0_jobs").fetchone()[0] == "failed"
        assert db.execute("SELECT status FROM k0_attempts").fetchone()[0] == "failed"
    assert add(app, contents=("different",)).status_code == 409
    assert len(provider.calls) == 1
    provider.failure = None
    assert add(app, contents=("alpha", "beta")).status_code == 200


@pytest.mark.parametrize("failure", ["embedding_timeout", "dimensions", "unexpected"])
def test_required_query_failure_not_200_empty_or_fts(tmp_path, failure):
    provider = Provider()
    app = app_at(tmp_path / "m.db", provider)
    assert add(app).status_code == 200
    provider.failure = failure
    result = search(app)
    assert result.status_code == 503 and "sensitive" not in result.text
    assert "data" not in result.json()


def test_control_ready_three_rows_and_atomic_mixed_failure(tmp_path):
    path, provider = tmp_path / "m.db", Provider()
    app = app_at(path, provider)
    assert add(app).status_code == 200
    old = state(path)
    provider.failure = "embedding_timeout"
    assert add(app, "mixed", ('delete "alpha"', "beta")).status_code == 503
    assert state(path) == old
    assert provider.calls[-1] == ["beta"]
    calls = len(provider.calls)
    assert add(app, "erase", ('delete "alpha"',)).status_code == 200
    assert len(provider.calls) == calls
    missing = app_at(path)
    with TestClient(missing) as client:
        health = client.get("/health").json()
        assert health["control_ready"] and not health["data_route_ready"]
        assert client.post("/search", json={"query": "alpha", "user_id": "u", "top_k": 10}).status_code == 503
    before = state(path)
    assert add(missing, "mix2", ('delete "alpha"', "gamma")).status_code == 503
    assert state(path) == before
    assert add(missing, "erase2", ('delete "alpha"',)).status_code == 200
    with connected(path) as db:
        db.execute("DROP TABLE k0_jobs")
    assert add(missing, "erase3", ('delete "alpha"',)).status_code == 503


@pytest.mark.parametrize("operation", ['delete "alpha"', 'do not retrieve "alpha"', 'I retract the claim that alpha'])
def test_controls_not_embedded_required_coverage_matches_survivors(tmp_path, operation):
    path, provider = tmp_path / "m.db", Provider()
    app = app_at(path, provider)
    assert add(app, "base").status_code == 200
    assert add(app, "control", (operation, "beta")).status_code == 200
    assert provider.calls == [["alpha"], ["beta"]]
    with connected(path) as db:
        active = db.execute("SELECT COUNT(*) FROM raw_messages WHERE deletion_state='active'").fetchone()[0]
        assert db.execute("SELECT COUNT(*) FROM raw_vectors").fetchone()[0] == active
        assert db.execute("SELECT COUNT(*) FROM derived_memories").fetchone()[0] == 0
    app.state.coordinator.require_ready("u")


@pytest.mark.parametrize("operation", ['do not retrieve "alpha"', 'I retract the claim that alpha'])
def test_pure_suppress_retract_controls_do_not_create_permanent_dense_gap(tmp_path, operation):
    provider = Provider()
    app = app_at(tmp_path / "m.db", provider)
    assert add(app).status_code == 200
    assert add(app, "control", (operation,)).status_code == 200
    assert provider.calls == [["alpha"]]
    result = search(app)
    assert result.status_code == 200 and result.headers["X-AML-Degraded"] == "false"
    assert result.json()["data"] == []
    assert add(app, "new", ("beta",)).status_code == 200


def test_copy_migration_certifies_raw_and_preserves_source_bytes_and_receipts(tmp_path):
    from app.k0_migrate import migrate
    path, copied = tmp_path / "old.db", tmp_path / "copy.db"
    research = create_app(Settings(db_path=path))
    assert add(research).status_code == 200
    before_bytes, before = path.read_bytes(), state(path)
    result = migrate(path, copied, Settings(db_path=copied, profile_id="raw_sync"))
    assert result["external_calls"] == 0 and result["proofs"][0]["counts"]["raw"] == 1
    assert path.read_bytes() == before_bytes and state(path) == before
    assert add(app_at(copied, profile="raw_sync")).status_code == 200
    assert state(copied)["k0_receipts"] == before["k0_receipts"]
    with pytest.raises(ValueError, match="new target"):
        migrate(path, copied, Settings(db_path=copied, profile_id="raw_sync"))


def test_migration_missing_dense_route_fails_without_mutating_source(tmp_path):
    from app.k0_migrate import migrate
    path, copied = tmp_path / "old.db", tmp_path / "copy.db"
    assert add(create_app(Settings(db_path=path))).status_code == 200
    before_bytes = path.read_bytes()
    with pytest.raises(NotReady, match="incomplete_dense"):
        migrate(path, copied, Settings(db_path=copied, profile_id="dense_sync"), Provider())
    assert before_bytes == path.read_bytes()


def test_in_batch_erasure_and_later_tombstone_do_not_leave_for_embedding(tmp_path):
    provider = Provider()
    app = app_at(tmp_path / "m.db", provider)
    assert add(app, contents=("alpha", 'delete "alpha"', "alpha", "beta")).status_code == 200
    assert provider.calls == [["beta"]]
    assert [h["content"] for h in search(app, "beta").json()["data"]] == ["beta"]
    assert add(app, "reintroduced", ("alpha", "gamma")).status_code == 200
    assert provider.calls[-1] == ["gamma"]


def test_old_raw_receipt_cannot_fake_dense_and_pure_delete_still_works(tmp_path):
    path = tmp_path / "m.db"
    raw = app_at(path, profile="raw_sync")
    assert add(raw).status_code == 200
    with connected(path) as db:
        old = db.execute("SELECT * FROM k0_receipts").fetchall()
    provider = Provider()
    dense = app_at(path, provider)
    for result in (add(dense), add(dense, "new"), search(dense)):
        assert result.status_code == 503
    assert provider.calls == []
    with pytest.raises(NotReady, match="incomplete_dense"):
        dense.state.coordinator.certify_routes("u")
    with connected(path) as db:
        assert db.execute("SELECT * FROM k0_receipts").fetchall() == old
    assert add(dense, "erase", ('delete "alpha"',)).status_code == 200
    assert add(dense).status_code == 200  # empty current routes now complete, original wire preserved
    assert provider.calls == []


def test_explicit_certification_keeps_original_research_proof_without_model_call(tmp_path):
    path, provider = tmp_path / "m.db", Provider()
    research = create_app(Settings(db_path=path, retrieval_mode="raw_dense"), embedding_provider=provider)
    assert add(research).status_code == 200
    before = state(path)
    dense = app_at(path, provider)
    assert add(dense).status_code == 503
    assert len(provider.calls) == 1
    proof = dense.state.coordinator.certify_routes("u")
    assert proof["counts"] == {"raw": 1, "fts": 1, "dense": 1, "active": 1}
    assert add(dense).status_code == 200 and len(provider.calls) == 1
    assert state(path)["k0_receipts"] == before["k0_receipts"]
    with connected(path) as db:
        assert db.execute("SELECT COUNT(*) FROM k0_switch_proofs").fetchone()[0] == 1


@pytest.mark.parametrize("change", ["revision", "dimensions", "model"])
def test_model_change_blocks_data_keeps_proof_and_requires_rebuild(tmp_path, change):
    path, provider = tmp_path / "m.db", Provider()
    app = app_at(path, provider)
    assert add(app).status_code == 200
    before = state(path)
    newer = Provider()
    newer.config = replace(newer.config, **{change: 3 if change == "dimensions" else "changed"})
    switched = app_at(path, newer)
    assert add(switched).status_code == 503
    assert search(switched).status_code == 503
    assert newer.calls == [] and state(path) == before
    assert search(app).status_code == 503  # superseded config, even if old physical vectors remain


def test_foreign_user_gap_does_not_block_ready_user(tmp_path):
    path, provider = tmp_path / "m.db", Provider()
    dense = app_at(path, provider)
    assert add(dense).status_code == 200
    research = create_app(Settings(db_path=path))
    assert add(research, user="other").status_code == 200
    assert search(dense).status_code == 200
    assert search(dense, user="other").status_code == 503


def test_research_same_user_advances_epoch_seq_and_invalidates_formal_proof(tmp_path):
    path, provider = tmp_path / "m.db", Provider()
    dense = app_at(path, provider)
    assert add(dense).status_code == 200
    research = create_app(Settings(db_path=path))
    assert add(research, "research-new", ("beta",)).status_code == 200
    with connected(path) as db:
        assert db.execute("SELECT commit_seq FROM k0_users WHERE user_id='u'").fetchone()[0] == 2
        assert db.execute("SELECT epoch FROM lifecycle_users WHERE user_id='u'").fetchone()[0] == 2
    assert search(dense).status_code == 503
    assert len(provider.calls) == 1


def test_query_coverage_changes_during_external_call_not_fts_fallback(tmp_path):
    path, provider = tmp_path / "m.db", Provider()
    app = app_at(path, provider)
    assert add(app).status_code == 200
    def remove():
        with connected(path) as db:
            db.execute("DELETE FROM raw_vectors")
    provider.action = remove
    assert search(app).status_code == 503


def test_delete_after_last_gate_reselects_before_response(tmp_path, monkeypatch):
    import app.retrieval as retrieval
    app = app_at(tmp_path / "m.db", profile="raw_sync")
    assert add(app).status_code == 200
    original, calls = retrieval.filter_evidence, []
    def gate(*args):
        result = original(*args)
        calls.append(1)
        if len(calls) == 1:
            assert add(app, "erase", ('delete "alpha"',)).status_code == 200
        return result
    monkeypatch.setattr(retrieval, "filter_evidence", gate)
    assert search(app).json()["data"] == []
    assert len(calls) == 2
    assert search(app).json()["data"] == []


def test_continuous_final_epoch_changes_stop_after_three(tmp_path, monkeypatch):
    import app.retrieval as retrieval
    app = app_at(tmp_path / "m.db", profile="raw_sync")
    assert add(app).status_code == 200
    original, calls = retrieval.filter_evidence, []
    def gate(*args):
        result = original(*args)
        calls.append(1)
        assert add(app, "new" + str(len(calls)), ("alpha " + str(len(calls)),)).status_code == 200
        return result
    monkeypatch.setattr(retrieval, "filter_evidence", gate)
    assert search(app).status_code == 503
    assert len(calls) == 3


def test_expired_unclaimed_lease_cannot_publish(tmp_path):
    path, entered, release = tmp_path / "m.db", threading.Event(), threading.Event()
    def block():
        entered.set()
        assert release.wait(5)
    app = app_at(path, Provider(action=block))
    with ThreadPoolExecutor() as pool:
        future = pool.submit(add, app)
        assert entered.wait(3)
        with connected(path) as db:
            db.execute("UPDATE k0_jobs SET expires_at=0")
        release.set()
        assert future.result().status_code == 503
    assert all(not rows for rows in state(path).values())


def test_takeover_fences_returning_worker(tmp_path):
    path, entered, release = tmp_path / "m.db", threading.Event(), threading.Event()
    def block():
        entered.set()
        assert release.wait(5)
    first = app_at(path, Provider(action=block))
    second_provider = Provider()
    second = app_at(path, second_provider)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(add, first)
        assert entered.wait(3)
        with connected(path) as db:
            old_fence = db.execute("SELECT generation FROM k0_jobs").fetchone()[0]
            db.execute("UPDATE k0_jobs SET expires_at=0")
        assert add(second).status_code == 200
        release.set()
        assert future.result().status_code == 503
    with connected(path) as db:
        assert db.execute("SELECT generation,state FROM k0_jobs").fetchone() == (old_fence + 1, "committed")
        assert db.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0] == 1


def test_same_process_wait_bounded_and_conflict_before_dispatch(tmp_path, monkeypatch):
    import app.jobs as jobs
    monkeypatch.setattr(jobs, "COMPETITION_WAIT", 0.2)
    entered, release = threading.Event(), threading.Event()
    def block():
        entered.set()
        assert release.wait(5)
    provider = Provider(action=block)
    app = app_at(tmp_path / "m.db", provider)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(add, app)
        assert entered.wait(3)
        start = time.monotonic()
        second = add(app)
        assert second.status_code == 503 and second.headers["Retry-After"] == "1"
        assert time.monotonic() - start < 1
        assert add(app, contents=("different",)).status_code == 409
        assert len(provider.calls) == 1
        release.set()
        assert future.result().status_code == 200


def test_delete_during_preparation_blocks_stale_publication_and_http_holds_no_lock(tmp_path):
    path, provider = tmp_path / "m.db", Provider()
    app = app_at(path, provider)
    assert add(app).status_code == 200
    def mutate():
        # Succeeds on a separate connection while provider code is running.
        assert add(app, "delete", ('delete "alpha"',)).status_code == 200
    provider.action = mutate
    assert add(app, "new", ("beta",)).status_code == 503
    with connected(path) as db:
        assert db.execute("SELECT COUNT(*) FROM raw_messages WHERE deletion_state='active'").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0] == 2
        assert db.execute("SELECT commit_seq FROM k0_users").fetchone()[0] == 2


def test_config_change_abandons_generation_and_rejects_old_completion(tmp_path):
    path, entered, release = tmp_path / "m.db", threading.Event(), threading.Event()
    def block():
        entered.set()
        assert release.wait(5)
    app = app_at(path, Provider(action=block))
    with ThreadPoolExecutor() as pool:
        future = pool.submit(add, app)
        assert entered.wait(3)
        newer = app_at(path, Provider(revision="2"))
        assert add(newer).status_code == 200
        release.set()
        assert future.result().status_code == 503
    assert len(state(path)["add_requests"]) == 1


def test_renewal_failure_prevents_next_batch(tmp_path, monkeypatch):
    import app.jobs as jobs
    monkeypatch.setattr(jobs, "RENEW_INTERVAL", 0.01)
    provider = Provider(action=lambda: time.sleep(0.05))
    provider.config = replace(provider.config, batch_size=1)
    app = app_at(tmp_path / "m.db", provider)
    def lose(lease):
        raise NotReady("renewal_fault")
    monkeypatch.setattr(app.state.coordinator, "renew", lose)
    assert add(app, contents=("alpha", "beta")).status_code == 503
    assert provider.calls == [["alpha"]]
    assert not state(tmp_path / "m.db")["add_requests"]


def test_expiry_inside_commit_rolls_back_all_publication(tmp_path, monkeypatch):
    path = tmp_path / "m.db"
    app = app_at(path, profile="raw_sync")
    original = app.state.store._insert_fts
    def expire(db, messages):
        original(db, messages)
        db.execute("UPDATE k0_jobs SET expires_at=0")
    monkeypatch.setattr(app.state.store, "_insert_fts", expire)
    assert add(app).status_code == 503
    assert all(not rows for rows in state(path).values())


def test_db_failure_sanitized_atomic_and_control_failure_not_success(tmp_path, monkeypatch):
    path = tmp_path / "m.db"
    app = app_at(path, profile="raw_sync")
    def fail(db, messages):
        raise sqlite3.OperationalError("sensitive-database-detail")
    monkeypatch.setattr(app.state.store, "_insert_fts", fail)
    result = add(app)
    assert result.status_code == 503 and "sensitive" not in result.text
    assert all(not rows for rows in state(path).values())
    assert add(app, "erase", ('delete "alpha"',)).status_code == 503


def test_mocktransport_timeout_missing_key_bad_dimensions_no_network(tmp_path, monkeypatch):
    key_env = "K0_SYNTHETIC_TEST_KEY"
    config = EmbeddingConfig(provider="engineering", endpoint="https://invalid.test/embeddings",
                             dimensions=2, api_key_env=key_env)
    calls = []
    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("synthetic timeout")
    provider = HttpEmbeddingProvider(config, transport=httpx.MockTransport(handler))
    app = app_at(tmp_path / "m.db", provider)
    monkeypatch.delenv(key_env, raising=False)
    assert add(app).status_code == 503 and calls == []
    monkeypatch.setenv(key_env, "synthetic-not-a-credential")
    assert add(app).status_code == 503 and calls == [1]


def test_formal_query_with_missing_config_never_empty_success(tmp_path):
    app = app_at(tmp_path / "m.db")
    assert add(app).status_code == 503
    assert search(app).status_code == 503
    assert add(app, "erase", ('delete "alpha"',)).status_code == 200


def test_mismatched_profile_and_provider_and_unknown_schema_fail_startup(tmp_path):
    path, provider = tmp_path / "m.db", Provider()
    with pytest.raises(ValueError):
        app_at(path, provider, embedding=replace(provider.config, revision="different"))
    app_at(path, profile="raw_sync")
    with connected(path) as db:
        db.execute("UPDATE k0_schema SET version=99")
    with pytest.raises(RuntimeError, match="unsupported K0"):
        app_at(path, profile="raw_sync")


def test_legacy_without_wire_ledger_replays_without_paid_prepare_or_job(tmp_path):
    from contextlib import closing
    path, provider = tmp_path / "m.db", Provider()
    research = create_app(Settings(db_path=path, retrieval_mode="raw_dense"), embedding_provider=provider)
    assert add(research).status_code == 200
    with closing(sqlite3.connect(path)) as db:
        db.execute("DELETE FROM add_requests")
        db.commit()
    dense = app_at(path, provider)
    dense.state.coordinator.certify_routes("u")
    before_calls = len(provider.calls)
    assert add(dense).status_code == 200
    assert len(provider.calls) == before_calls
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT COUNT(*) FROM k0_jobs").fetchone()[0] == 0


def test_legacy_without_ledger_conflict_is_before_paid_prepare(tmp_path):
    from contextlib import closing
    path, provider = tmp_path / "m.db", Provider()
    research = create_app(Settings(db_path=path, retrieval_mode="raw_dense"), embedding_provider=provider)
    assert add(research).status_code == 200
    with closing(sqlite3.connect(path)) as db:
        db.execute("DELETE FROM add_requests")
        db.commit()
    dense = app_at(path, provider)
    dense.state.coordinator.certify_routes("u")
    before_calls = len(provider.calls)
    assert add(dense, contents=("different",)).status_code == 409
    assert len(provider.calls) == before_calls


def test_second_batch_missing_key_does_not_overwrite_completed_attempt(tmp_path, monkeypatch):
    key_env = "K0_SYNTHETIC_BATCH_KEY"
    monkeypatch.setenv(key_env, "synthetic-not-a-credential")
    calls = []
    def handler(request):
        calls.append(json.loads(request.content)["input"])
        monkeypatch.delenv(key_env)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 1]}],
                                        "usage": {"total_tokens": 7}})
    provider = HttpEmbeddingProvider(EmbeddingConfig(provider="engineering", endpoint="https://invalid.test/embeddings",
        dimensions=2, batch_size=1, api_key_env=key_env), transport=httpx.MockTransport(handler))
    path = tmp_path / "m.db"
    app = app_at(path, provider)
    result = add(app, contents=("first", "second"))
    assert result.status_code == 503 and calls == [["first"]]
    with connected(path) as db:
        ledger = db.execute("SELECT status,error_code,usage_json FROM k0_attempts ORDER BY permitted_at").fetchall()
        assert len(ledger) == 1 and ledger[0][:2] == ("ready", None)
        assert json.loads(ledger[0][2])["requests"] == 1
        assert json.loads(ledger[0][2])["reported_input_tokens"] == 7
    assert all(not rows for rows in state(path).values())


def test_migration_pinned_snapshot_rejects_live_wal_commit(tmp_path, monkeypatch):
    import app.k0_migrate as migration
    path, copied = tmp_path / "old.db", tmp_path / "copy.db"
    research = create_app(Settings(db_path=path))
    with closing(sqlite3.connect(path)) as anchor:
        anchor.execute("PRAGMA wal_autocheckpoint=0")
        assert add(research).status_code == 200
        original = migration.file_hash
        injected = []
        def hash_and_mutate(file):
            result = original(file)
            if file == path.resolve() and not injected:
                injected.append(1)
                assert add(research, "during", ("beta",)).status_code == 200
            return result
        monkeypatch.setattr(migration, "file_hash", hash_and_mutate)
        with pytest.raises(NotReady, match="source_changed"):
            migration.migrate(path, copied, Settings(db_path=copied, profile_id="raw_sync"))
        assert injected == [1]
        with connected(path) as db:
            assert db.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 2


def test_migration_manifest_standalone_target_has_all_proofs_and_closed_connections(tmp_path, monkeypatch):
    import app.k0_migrate as migration
    path, copied = tmp_path / "old.db", tmp_path / "copy.db"
    assert add(create_app(Settings(db_path=path))).status_code == 200
    original, captured = migration.sqlite3.connect, []
    thread_id = threading.get_ident()
    def tracked(*args, **kwargs):
        db = original(*args, **kwargs)
        if threading.get_ident() == thread_id:
            captured.append(db)
        return db
    monkeypatch.setattr(migration.sqlite3, "connect", tracked)
    result = migration.migrate(path, copied, Settings(db_path=copied, profile_id="raw_sync"))
    for db in captured:
        with pytest.raises(sqlite3.ProgrammingError):
            db.execute("SELECT 1")
    assert result["target_sha256"] == migration.file_hash(copied)
    with closing(original(copied.as_uri() + "?immutable=1", uri=True)) as immutable:
        assert immutable.execute("SELECT COUNT(*) FROM k0_watermarks WHERE config_hash=?", (result["config_hash"],)).fetchone()[0] == 2
        assert immutable.execute("SELECT COUNT(*) FROM k0_watermarks").fetchone()[0] == 4  # original research proof retained
        assert immutable.execute("SELECT COUNT(*) FROM k0_switch_proofs").fetchone()[0] == 1

"""Independent K1 attacks. Engineering doubles only; no quality/model claims.

Tests assert the contract, so reproduced defects remain ordinary failures.
Only temporary SQLite databases are modified; production source stays read-only.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sqlite3
import threading

import pytest

from app.api import create_app
from app.config import Settings
from app.jobs import NotReady
from app.k0_migrate import file_hash, migrate
from app.models import AddRequest, RawMessage, SearchRequest
from app.segments import digest, insert, partition, split_source, token_count
from app.jobs import SnapshotChanged
from test_k0 import Provider, add, body, connected, search


def span_app(path, profile="span_sync", provider=None):
    return create_app(Settings(db_path=path, profile_id=profile), embedding_provider=provider)


def raw(text):
    return RawMessage(id="raw_independent", user_id="u", session_id="s", request_id="r",
                      message_index=0, role="user", content=text, source_timestamp=None,
                      recorded_at=1)


def natural_boundary_counterexample():
    # A real o200k_base token merges across the chosen natural newline boundary.
    tail = " }\n//\n//"
    prefix = "a " * 509
    prefix += " " * (1200 - len(prefix) - len(tail)) + tail
    assert len(prefix) == 1200 and token_count(prefix) == 512
    assert token_count(prefix[:1198]) == 513
    return prefix + "tail " * 300


def test_natural_boundary_must_recheck_actual_token_hard_limit():
    pieces = split_source(raw(natural_boundary_counterexample()))
    assert all(token_count(piece["content"]) <= 512 for piece in pieces), [
        (p["span_start"], p["span_end"], p["token_count"], p["boundary_status"])
        for p in pieces
    ]


def test_formal_add_cannot_certify_over_token_limit_span(tmp_path):
    app = span_app(tmp_path / "limits.db")
    assert add(app, contents=(natural_boundary_counterexample(),)).status_code == 200
    with connected(app.state.store.db_path) as db:
        over = db.execute("SELECT span_start,span_end,token_count FROM source_spans WHERE token_count>512").fetchall()
    assert not over, over


def test_unexpected_span_cannot_pass_coverage_and_escape_hard_limits(tmp_path):
    app = span_app(tmp_path / "extra.db")
    text = "alpha " * 501
    assert add(app, contents=(text,)).status_code == 200
    with closing(app.state.store._connect()) as db:
        piece = dict(db.execute("SELECT * FROM source_spans ORDER BY span_index LIMIT 1").fetchone())
        piece.update(id="span_injected_noncanonical", content=text, span_start=0,
                     span_end=len(text), span_sha256=digest(text), token_count=token_count(text),
                     span_index=999, previous_id=None, next_id=None)
        with db:
            insert(db, "u", [piece])
    result = search(app)
    assert result.status_code == 503, (result.status_code, [
        (h["id"], len(h["content"])) for h in result.json().get("data", [])
    ])


def test_empty_active_raw_cannot_turn_integrity_loss_into_ready(tmp_path):
    app = span_app(tmp_path / "empty.db")
    assert add(app).status_code == 200
    with connected(app.state.store.db_path) as db:
        db.execute("UPDATE raw_messages SET content='' WHERE user_id='u'")
    result = search(app)
    assert result.status_code == 503, (result.status_code, result.json())


@pytest.mark.parametrize("field,value", [
    ("role", "user"), ("session_id", "forged_session"), ("message_index", 99),
    ("source_timestamp", 9999999999999), ("recorded_at", 9999999999999),
    ("created_at", 9999999999999),
])
def test_final_source_check_rejects_forged_provenance(tmp_path, field, value):
    app = span_app(tmp_path / "identity.db")
    request = body()
    request["messages"][0].update(role="assistant", timestamp=123)
    app.state.ingest.add(AddRequest(**request))
    response, trace = app.state.retrieval.search_with_trace(SearchRequest(query="alpha", user_id="u", top_k=20))
    assert response.data and trace["evidence"][0]["role"] == "assistant"
    snapshot = app.state.store.search_snapshot(user_id="u", match_query='"alpha"',
        query_vector=None, version="unconfigured", dimensions=0, mode="raw_fts",
        candidate_limit=20, lexical_quota=20, dense_quota=0, span=True)
    candidate = snapshot.lexical[0].model_copy(update={field: value})
    stamp = app.state.coordinator.require_ready("u")
    policy = app.state.store.policy_snapshot("u", [candidate])
    with pytest.raises(NotReady):
        app.state.coordinator.final_check("u", stamp, [candidate], policy)


def test_source_role_time_changed_after_gate_cannot_publish_stale_identity(tmp_path, monkeypatch):
    import app.retrieval as retrieval
    app = span_app(tmp_path / "last_identity.db")
    request = body()
    request["messages"][0].update(role="assistant", timestamp=123)
    app.state.ingest.add(AddRequest(**request))
    original = retrieval.filter_evidence
    calls = []

    def gate(*args):
        result = original(*args)
        if not calls:
            with connected(app.state.store.db_path) as db:
                db.execute("UPDATE raw_messages SET role='user',source_timestamp=456 WHERE user_id='u'")
        calls.append(1)
        return result

    monkeypatch.setattr(retrieval, "filter_evidence", gate)
    response, trace = app.state.retrieval.search_with_trace(SearchRequest(query="alpha", user_id="u", top_k=20))
    assert [(h.created_at, e["role"], e["source_timestamp"]) for h, e in zip(response.data, trace["evidence"])] == [
        (456, "user", 456)
    ], (len(calls), trace["evidence"])


@pytest.mark.parametrize("profile", ["span_sync", "span_dense_sync"])
def test_repeated_multispan_retraction_rebinds_complete_generation(tmp_path, profile):
    provider = Provider() if profile == "span_dense_sync" else None
    app = span_app(tmp_path / "retract.db", profile, provider)
    text = "alpha " * 501
    assert add(app, contents=(text,)).status_code == 200
    expected = len(split_source(raw(text)))
    before_calls = len(provider.calls) if provider else 0
    for generation in (2, 3):
        assert add(app, f"retract{generation}", ("I retract the claim that " + text,)).status_code == 200
        assert not provider or len(provider.calls) == before_calls
        assert search(app).json()["data"] == []
        before_calls = len(provider.calls) if provider else 0
        with connected(app.state.store.db_path) as db:
            assert db.execute("SELECT source_version,COUNT(*) FROM source_spans GROUP BY source_version").fetchall() == [(generation, expected)]
            assert db.execute(f"SELECT COUNT(*) FROM {partition('u')}").fetchone()[0] == expected
            assert db.execute("SELECT COUNT(*) FROM span_vectors").fetchone()[0] == (expected if provider else 0)


@pytest.mark.parametrize("profile", ["span_sync", "span_dense_sync"])
def test_same_batch_retraction_uses_new_span_version(tmp_path, profile):
    provider = Provider() if profile == "span_dense_sync" else None
    app = span_app(tmp_path / "same_batch.db", profile, provider)
    assert add(app, contents=("alpha", "I retract the claim that alpha", "beta")).status_code == 200
    assert all(h["content"] != "alpha" for h in search(app).json()["data"])
    assert [h["content"] for h in search(app, "beta").json()["data"]] == ["beta"]
    with connected(app.state.store.db_path) as db:
        assert db.execute("SELECT source_version FROM source_spans WHERE content='alpha'").fetchone()[0] == 2
    assert not provider or provider.calls[0] == ["alpha", "beta"]


@pytest.mark.parametrize("profile", ["span_sync", "span_dense_sync"])
def test_control_cannot_apply_to_future_member_of_same_add(tmp_path, profile):
    provider = Provider() if profile == "span_dense_sync" else None
    app = span_app(tmp_path / "future.db", profile, provider)
    assert add(app, contents=('delete "alpha"', "alpha")).status_code == 200
    assert not provider or provider.calls == [["alpha"]]
    assert [h["content"] for h in search(app).json()["data"]] == ["alpha"]


def test_assistant_and_quoted_control_text_has_no_user_control_authority(tmp_path):
    app = span_app(tmp_path / "authority.db")
    assert add(app).status_code == 200
    request = body("assistant", ('delete "alpha"',))
    request["messages"][0]["role"] = "assistant"
    app.state.ingest.add(AddRequest(**request))
    assert add(app, "quoted", ('A third party said: delete "alpha"',)).status_code == 200
    with connected(app.state.store.db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM memory_operations").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM raw_messages WHERE content='alpha' AND deletion_state='active'").fetchone()[0] == 1


def test_missing_source_time_remains_unknown_and_unadded_future_is_absent(tmp_path):
    app = span_app(tmp_path / "prefix.db")
    assert add(app, contents=("alpha past",)).status_code == 200
    response, trace = app.state.retrieval.search_with_trace(SearchRequest(query="alpha", user_id="u", top_k=20))
    assert [h.content for h in response.data] == ["alpha past"]
    assert trace["evidence"][0]["source_timestamp"] is None
    assert add(app, "next", ("beta future",)).status_code == 200
    assert [h["content"] for h in search(app).json()["data"]] == ["alpha past"]


@pytest.mark.parametrize("control", ['delete "alpha"', 'do not retrieve "alpha"', "I retract the claim that alpha"])
def test_final_check_reselects_after_lifecycle_change(tmp_path, monkeypatch, control):
    import app.retrieval as retrieval
    app = span_app(tmp_path / "last_delete.db")
    assert add(app).status_code == 200
    original = retrieval.filter_evidence
    calls = []

    def gate(*args):
        result = original(*args)
        if not calls:
            assert add(app, "control", (control,)).status_code == 200
        calls.append(1)
        return result

    monkeypatch.setattr(retrieval, "filter_evidence", gate)
    assert search(app).json()["data"] == []
    assert len(calls) == 2


def test_old_retraction_generation_cannot_pass_final_source_check(tmp_path):
    app = span_app(tmp_path / "old_generation.db")
    assert add(app).status_code == 200
    old = app.state.store.search_snapshot(user_id="u", match_query='"alpha"',
        query_vector=None, version="unconfigured", dimensions=0, mode="raw_fts",
        candidate_limit=20, lexical_quota=20, dense_quota=0, span=True).lexical
    assert add(app, "retract", ("I retract the claim that alpha",)).status_code == 200
    stamp = app.state.coordinator.require_ready("u")
    policy = app.state.store.policy_snapshot("u", old)
    with pytest.raises(SnapshotChanged):
        app.state.coordinator.final_check("u", stamp, old, policy)


def test_add_200_messages_publishes_complete_distinct_sources_immediately(tmp_path):
    app = span_app(tmp_path / "batch200.db")
    text = "alpha " * 501
    assert add(app, contents=(text,) * 200).status_code == 200
    with connected(app.state.store.db_path) as db:
        expected = len(split_source(raw(text))) * 200
        assert db.execute("SELECT COUNT(*),COUNT(DISTINCT source_raw_id) FROM source_spans").fetchone() == (expected, 200)
        assert db.execute(f"SELECT COUNT(*) FROM {partition('u')}").fetchone()[0] == expected
        assert db.execute("SELECT COUNT(*) FROM k0_watermarks WHERE ready=1").fetchone()[0] == 3
    assert search(app).status_code == 200


def test_lost_span_vector_during_query_call_returns_not_ready(tmp_path):
    provider = Provider()
    app = span_app(tmp_path / "query_loss.db", "span_dense_sync", provider)
    assert add(app).status_code == 200

    def lose_vector():
        with connected(app.state.store.db_path) as db:
            db.execute("DELETE FROM span_vectors")

    provider.action = lose_vector
    assert search(app).status_code == 503


def test_span_insert_failure_rolls_back_raw_receipt_and_span_publication(tmp_path, monkeypatch):
    import app.segments as segments
    app = span_app(tmp_path / "atomic_insert.db")
    original = segments.insert

    def fail_after_insert(db, user, pieces, prepared=None):
        original(db, user, pieces, prepared)
        raise sqlite3.OperationalError("synthetic failure after spans")

    monkeypatch.setattr(segments, "insert", fail_after_insert)
    assert add(app).status_code == 503
    with connected(app.state.store.db_path) as db:
        for table in ("raw_messages", "source_spans", "span_vectors", "add_requests", "k0_receipts", "k0_users"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_foreign_user_spans_do_not_change_scores_or_erase_other_user(tmp_path):
    app = span_app(tmp_path / "users.db")
    assert add(app, contents=("alpha", "alpha beta")).status_code == 200
    before = search(app).json()
    assert add(app, contents=("alpha " * 600,), user="other").status_code == 200
    assert add(app, "erase", ('delete "alpha"',), user="other").status_code == 200
    assert search(app).json() == before


def test_mixed_delete_failed_new_embedding_rolls_back_all_span_lifecycle(tmp_path):
    provider = Provider()
    app = span_app(tmp_path / "mixed.db", "span_dense_sync", provider)
    assert add(app).status_code == 200
    with connected(app.state.store.db_path) as db:
        before = {t: db.execute(f"SELECT * FROM {t}").fetchall() for t in (
            "raw_messages", "source_spans", "span_vectors", "source_policies", "add_requests", "k0_receipts", "k0_watermarks")}
    provider.failure = "dimensions"
    assert add(app, "mixed", ('delete "alpha"', "beta")).status_code == 503
    with connected(app.state.store.db_path) as db:
        assert {t: db.execute(f"SELECT * FROM {t}").fetchall() for t in before} == before


def test_expired_span_worker_is_fenced_after_takeover(tmp_path):
    path, entered, release = tmp_path / "lease.db", threading.Event(), threading.Event()

    def block():
        entered.set()
        assert release.wait(5)

    first = span_app(path, "span_dense_sync", Provider(action=block))
    second = span_app(path, "span_dense_sync", Provider())
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(add, first, "r", ("alpha " * 501,))
        assert entered.wait(3)
        with connected(path) as db:
            db.execute("UPDATE k0_jobs SET expires_at=0")
        assert add(second, contents=("alpha " * 501,)).status_code == 200
        release.set()
        assert pending.result().status_code == 503
    with connected(path) as db:
        assert db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0] == len(split_source(raw("alpha " * 501)))
        assert db.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0] == 1


def test_offline_wal_copy_preserves_source_receipts_and_no_resurrection(tmp_path):
    old, new = tmp_path / "old.db", tmp_path / "new.db"
    app = span_app(old, "raw_sync")
    with closing(sqlite3.connect(old)) as anchor:
        anchor.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        anchor.execute("PRAGMA wal_autocheckpoint=0")
        assert add(app, contents=("alpha " * 501, "beta")).status_code == 200
        assert add(app, "erase", ('delete "beta"',)).status_code == 200
        wal = old.with_name(old.name + "-wal")
        assert wal.exists() and wal.stat().st_size
        before = (file_hash(old), file_hash(wal))
        tables = ("raw_messages", "source_policies", "add_requests", "k0_receipts", "memory_tombstones")
        with connected(old) as db:
            source_rows = {t: db.execute(f"SELECT * FROM {t} ORDER BY rowid").fetchall() for t in tables}
        manifest = migrate(old, new, Settings(db_path=new, profile_id="span_sync"))
        assert (file_hash(old), file_hash(wal)) == before
        assert manifest["source_sha256"] == before[0] and manifest["source_wal_sha256"] == before[1]
        assert manifest["target_sha256"] == file_hash(new) and manifest["external_calls"] == 0
        with connected(new) as db:
            assert {t: db.execute(f"SELECT * FROM {t} ORDER BY rowid").fetchall() for t in tables} == source_rows
            assert all(r[0] != "beta" for r in db.execute("SELECT content FROM source_spans"))
        copied = span_app(new)
        assert add(copied, contents=("alpha " * 501, "beta")).status_code == 200  # exact old receipt replay
        assert search(copied, "beta").json()["data"] == []


def test_deleted_source_cannot_reappear_after_span_copy_or_replay(tmp_path):
    old, new = tmp_path / "old_span.db", tmp_path / "new_span.db"
    app = span_app(old)
    assert add(app).status_code == 200
    assert add(app, "erase", ('delete "alpha"',)).status_code == 200
    migrate(old, new, Settings(db_path=new, profile_id="span_sync"))
    copied = span_app(new)
    assert add(copied, "again", ("alpha",)).status_code == 200
    with connected(new) as db:
        assert db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0] == 0
        assert db.execute(f"SELECT COUNT(*) FROM {partition('u')}").fetchone()[0] == 0
    assert search(copied).json()["data"] == []

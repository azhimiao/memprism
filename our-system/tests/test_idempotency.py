from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.governance import PassthroughGovernance
from app.ingest import IngestService
from app.models import AddRequest, RawMessage
from app.store import CommitStatus, SqliteStore, count_lexical_rows


def _payload(*, request_id: str = "request-1", user_id: str = "user-1", content: str = "alpha"):
    return {
        "request_id": request_id,
        "user_id": user_id,
        "session_id": "session-1",
        "messages": [{"role": "user", "content": content}],
    }


def test_same_payload_replays_cached_response_without_duplicate_rows(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    with TestClient(create_app(Settings(db_path=db_path))) as client:
        first = client.post("/add", json=_payload())
        replay = client.post("/add", json=_payload())

        assert first.status_code == 200
        assert replay.status_code == 200
        assert replay.json() == first.json()

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 1
        assert count_lexical_rows(connection) == 1
        assert connection.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0] == 1


def test_same_request_id_with_different_payload_returns_409_and_preserves_original(
    tmp_path: Path,
):
    db_path = tmp_path / "memory.sqlite"
    with TestClient(create_app(Settings(db_path=db_path))) as client:
        first = client.post("/add", json=_payload())
        conflict = client.post("/add", json=_payload(content="replacement"))

        assert first.status_code == 200
        assert conflict.status_code == 409
        assert "request_id" in conflict.json()["detail"]

        search = client.post(
            "/search", json={"query": "alpha", "user_id": "user-1", "top_k": 10}
        )
        assert [hit["content"] for hit in search.json()["data"]] == ["alpha"]

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0] == 1


def test_same_request_id_can_be_used_by_different_users(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    with TestClient(create_app(Settings(db_path=db_path))) as client:
        first = client.post("/add", json=_payload(user_id="user-a", content="alpha"))
        second = client.post("/add", json=_payload(user_id="user-b", content="beta"))

        assert first.status_code == 200
        assert second.status_code == 200

        user_a = client.post(
            "/search", json={"query": "alpha", "user_id": "user-a", "top_k": 10}
        )
        user_b = client.post(
            "/search", json={"query": "beta", "user_id": "user-b", "top_k": 10}
        )
        assert [hit["content"] for hit in user_a.json()["data"]] == ["alpha"]
        assert [hit["content"] for hit in user_b.json()["data"]] == ["beta"]

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0] == 2


def test_replay_survives_rebuilding_the_app(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    payload = _payload()

    with TestClient(create_app(Settings(db_path=db_path))) as client:
        first = client.post("/add", json=payload)
    with TestClient(create_app(Settings(db_path=db_path))) as client:
        replay = client.post("/add", json=payload)

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json() == first.json()
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 1


def test_concurrent_duplicate_add_commits_once(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    app = create_app(Settings(db_path=db_path))
    payload = _payload()

    def submit() -> tuple[int, dict]:
        with TestClient(app) as client:
            response = client.post("/add", json=payload)
            return response.status_code, response.json()

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(lambda _index: submit(), range(6)))

    assert [status for status, _body in results] == [200] * 6
    assert {body["request_id"] for _status, body in results} == {"request-1"}
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 1
        assert count_lexical_rows(connection) == 1


def test_raw_and_fts_and_ledger_roll_back_together(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "memory.sqlite"
    store = SqliteStore(db_path)
    store.initialize()
    message = RawMessage(
        id="raw-failure",
        user_id="user-1",
        session_id="session-1",
        request_id="request-1",
        message_index=0,
        role="user",
        content="rollback me",
        source_timestamp=None,
        recorded_at=1,
    )

    def fail_fts(_connection, _messages):
        raise RuntimeError("synthetic FTS failure")

    monkeypatch.setattr(store, "_insert_fts", fail_fts)
    try:
        store.commit_add(
            [message],
            payload_hash="hash-1",
            response={"success": True},
        )
    except RuntimeError as exc:
        assert str(exc) == "synthetic FTS failure"
    else:
        raise AssertionError("commit_add should propagate the FTS failure")

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 0
        assert count_lexical_rows(connection) == 0
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name GLOB 'fts_u_*'"
        ).fetchall() == []
        assert connection.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0] == 0


def test_partition_creation_rolls_back_with_the_add(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "memory.sqlite"
    store = SqliteStore(db_path)
    store.initialize()
    message = RawMessage(
        id="raw-partition-rollback",
        user_id="user-1",
        session_id="session-1",
        request_id="request-1",
        message_index=0,
        role="user",
        content="rollback the new partition",
        source_timestamp=None,
        recorded_at=1,
    )

    def create_partition_then_fail(connection, messages):
        store._ensure_partition(connection, messages[0].user_id)
        raise RuntimeError("synthetic partition failure")

    monkeypatch.setattr(store, "_insert_fts", create_partition_then_fail)
    try:
        store.commit_add(
            [message],
            payload_hash="hash-1",
            response={"success": True},
        )
    except RuntimeError as exc:
        assert str(exc) == "synthetic partition failure"
    else:
        raise AssertionError("commit_add should propagate the partition failure")

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 0
        assert count_lexical_rows(connection) == 0
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name GLOB 'fts_u_*'"
        ).fetchall() == []
        assert connection.execute("SELECT COUNT(*) FROM fts_partitions").fetchone()[0] == 0


def test_commit_add_returns_created_then_replay(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    store = SqliteStore(db_path)
    store.initialize()
    message = RawMessage(
        id="raw-direct",
        user_id="user-1",
        session_id="session-1",
        request_id="request-1",
        message_index=0,
        role="user",
        content="direct commit",
        source_timestamp=None,
        recorded_at=1,
    )

    created = store.commit_add([message], payload_hash="hash-1", response={"success": True})
    replay = store.commit_add(
        [message], payload_hash="hash-1", response={"success": True, "new": "ignored"}
    )

    assert created.status is CommitStatus.CREATED
    assert replay.status is CommitStatus.REPLAY
    assert replay.response == {"success": True}

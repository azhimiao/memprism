from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.index import build_lexical_text
from app.store import count_lexical_rows


def test_existing_phase_a_rows_are_replayed_without_duplicate_write(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    payload = {
        "request_id": "legacy-request",
        "user_id": "legacy-user",
        "session_id": "legacy-session",
        "messages": [
            {
                "role": "user",
                "content": "legacy message",
                "timestamp": 1704067200000,
            }
        ],
    }
    with TestClient(create_app(Settings(db_path=db_path))) as client:
        response = client.post("/add", json=payload)
        assert response.status_code == 200

    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE add_requests")

    with TestClient(create_app(Settings(db_path=db_path))) as client:
        replay = client.post("/add", json=payload)
        assert replay.status_code == 200
        found = client.post(
            "/search",
            json={"query": "legacy", "user_id": "legacy-user", "top_k": 5},
        )
        assert found.status_code == 200
        assert [hit["content"] for hit in found.json()["data"]] == ["legacy message"]

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 1
        assert count_lexical_rows(connection) == 1
        assert connection.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0] == 1


def _insert_shared_fts_fixture(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE raw_messages (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                source_timestamp INTEGER,
                recorded_at INTEGER NOT NULL,
                privacy_level TEXT NOT NULL,
                deletion_state TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE VIRTUAL TABLE raw_message_fts USING fts5(
                message_id UNINDEXED,
                user_id UNINDEXED,
                lexical_text,
                tokenize='unicode61'
            )
            """
        )
        rows = [
            ("legacy-a-apple", "user-a", "apple", 0),
            ("legacy-a-banana", "user-a", "banana", 1),
        ]
        rows.extend(
            (f"legacy-b-{index}", "user-b", "apple", 0) for index in range(100)
        )
        connection.executemany(
            """
            INSERT INTO raw_messages (
                id, user_id, session_id, request_id, message_index,
                role, content, source_timestamp, recorded_at,
                privacy_level, deletion_state
            ) VALUES (?, ?, 'session-a', ?, ?, 'user', ?, NULL, 1, 'normal', 'active')
            """,
            [
                (message_id, user_id, message_id, message_index, content)
                for message_id, user_id, content, message_index in rows
            ],
        )
        connection.executemany(
            """
            INSERT INTO raw_message_fts (message_id, user_id, lexical_text)
            VALUES (?, ?, ?)
            """,
            [
                (message_id, user_id, build_lexical_text(content))
                for message_id, user_id, content, _message_index in rows
            ],
        )
        connection.commit()
    finally:
        connection.close()


def test_shared_fts_migration_scores_match_a_user_only_index(tmp_path: Path):
    legacy_path = tmp_path / "legacy.sqlite"
    control_path = tmp_path / "control.sqlite"
    _insert_shared_fts_fixture(legacy_path)

    with TestClient(create_app(Settings(db_path=legacy_path))) as client:
        migrated = client.post(
            "/search",
            json={"query": "apple banana", "user_id": "user-a", "top_k": 20},
        )
        user_b = client.post(
            "/search",
            json={"query": "apple", "user_id": "user-b", "top_k": 200},
        )
    assert migrated.status_code == 200
    assert user_b.status_code == 200
    assert len(user_b.json()["data"]) == 100
    assert {hit["content"] for hit in user_b.json()["data"]} == {"apple"}

    with TestClient(create_app(Settings(db_path=control_path))) as client:
        created = client.post(
            "/add",
            json={
                "request_id": "control-a",
                "user_id": "user-a",
                "session_id": "session-a",
                "messages": [
                    {"role": "user", "content": "apple"},
                    {"role": "user", "content": "banana"},
                ],
            },
        )
        assert created.status_code == 200
        control = client.post(
            "/search",
            json={"query": "apple banana", "user_id": "user-a", "top_k": 20},
        )
    assert control.status_code == 200
    assert [(hit["content"], hit["score"]) for hit in migrated.json()["data"]] == [
        (hit["content"], hit["score"]) for hit in control.json()["data"]
    ]

    _insert_shared_fts_rows_again(legacy_path)
    create_app(Settings(db_path=legacy_path))
    with sqlite3.connect(legacy_path) as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'raw_message_fts%'"
        ).fetchall() == []
        assert count_lexical_rows(connection) == 102


def _insert_shared_fts_rows_again(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE raw_message_fts USING fts5(
                message_id UNINDEXED,
                user_id UNINDEXED,
                lexical_text,
                tokenize='unicode61'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO raw_message_fts (message_id, user_id, lexical_text)
            VALUES ('legacy-a-apple', 'user-a', 'apple'),
                   ('legacy-a-banana', 'user-a', 'banana'),
                   ('legacy-b-0', 'user-b', 'apple')
            """
        )
        connection.commit()
    finally:
        connection.close()

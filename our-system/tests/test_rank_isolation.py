"""User-local lexical ranking.

SQLite FTS5 bm25() statistics belong to the whole table. These tests pin the
R0 property that another user's writes, deletes, and index rebuilds do not
change this user's ids, order, or scores.

RANK_SCORE_ABS_TOLERANCE is fixed at 0: the same user-local index must return
the same IEEE score. The tests compare snapshots. They do not pick which of
user A's rows should win.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.store import SqliteStore

RANK_SCORE_ABS_TOLERANCE = 0.0
ROOT = Path(__file__).resolve().parents[1]


def _add(
    client: TestClient,
    *,
    request_id: str,
    user_id: str,
    session_id: str,
    contents: list[str],
):
    response = client.post(
        "/add",
        json={
            "request_id": request_id,
            "user_id": user_id,
            "session_id": session_id,
            "messages": [{"role": "user", "content": content} for content in contents],
        },
    )
    assert response.status_code == 200, response.text
    return response


def _search(client: TestClient, *, user_id: str, query: str, top_k: int = 20) -> list[dict]:
    response = client.post(
        "/search",
        json={"query": query, "user_id": user_id, "top_k": top_k},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"data"}
    return body["data"]


def _snapshot(hits: list[dict]) -> list[tuple[str, str, float]]:
    return [(hit["id"], hit["content"], float(hit["score"])) for hit in hits]


def _assert_same_ranking(
    before: list[tuple[str, str, float]],
    after: list[tuple[str, str, float]],
) -> None:
    assert [(item_id, content) for item_id, content, _score in before] == [
        (item_id, content) for item_id, content, _score in after
    ]
    for (_before_id, _before_content, before_score), (_after_id, _after_content, after_score) in zip(
        before, after
    ):
        assert abs(before_score - after_score) <= RANK_SCORE_ABS_TOLERANCE


def _score_for(hits: list[dict], content: str) -> float:
    matches = [float(hit["score"]) for hit in hits if hit["content"] == content]
    assert len(matches) == 1
    return matches[0]


def _flood(
    client: TestClient,
    *,
    user_id: str,
    session_id: str,
    content: str,
    count: int,
    request_prefix: str,
) -> None:
    for index in range(count):
        _add(
            client,
            request_id=f"{request_prefix}-{index}",
            user_id=user_id,
            session_id=session_id,
            contents=[content],
        )


def _search_in_new_process(db_path: Path, *, user_id: str, query: str, top_k: int) -> list[dict]:
    script = """
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings

db_path, user_id, query, top_k = sys.argv[1:]
with TestClient(create_app(Settings(db_path=Path(db_path)))) as client:
    response = client.post(
        "/search",
        json={"query": query, "user_id": user_id, "top_k": int(top_k)},
    )
response.raise_for_status()
sys.stdout.write(response.text)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, str(db_path), user_id, query, str(top_k)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    body = json.loads(completed.stdout)
    assert set(body) == {"data"}
    return body["data"]


def test_other_user_writes_do_not_change_source_order_or_score(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    with TestClient(create_app(Settings(db_path=db_path))) as client:
        first = _add(
            client,
            request_id="a-apple-banana",
            user_id="user-a",
            session_id="session-a",
            contents=["apple", "banana"],
        )
        replay = _add(
            client,
            request_id="a-apple-banana",
            user_id="user-a",
            session_id="session-a",
            contents=["apple", "banana"],
        )
        assert replay.json() == first.json()

        before = _snapshot(_search(client, user_id="user-a", query="apple banana"))
        assert {content for _item_id, content, _score in before} == {"apple", "banana"}
        assert len(before) == 2

        _flood(
            client,
            user_id="user-b",
            session_id="session-b",
            content="apple",
            count=100,
            request_prefix="b-apple",
        )
        after_write = _search(client, user_id="user-a", query="apple banana")
        assert {hit["content"] for hit in after_write} == {"apple", "banana"}
        _assert_same_ranking(before, _snapshot(after_write))

    with TestClient(create_app(Settings(db_path=db_path))) as client:
        reopened = _search(client, user_id="user-a", query="apple banana")
        _assert_same_ranking(before, _snapshot(reopened))

    restarted = _search_in_new_process(
        db_path, user_id="user-a", query="apple banana", top_k=20
    )
    _assert_same_ranking(before, _snapshot(restarted))

    store = SqliteStore(db_path)
    with sqlite3.connect(db_path) as connection:
        user_a_table = connection.execute(
            "SELECT table_name FROM fts_partitions WHERE user_id = ?",
            ("user-a",),
        ).fetchone()[0]
    store.rebuild_user_lexical_index("user-b")
    with TestClient(create_app(Settings(db_path=db_path))) as client:
        after_rebuild = _search(client, user_id="user-a", query="apple banana")
        user_b_hits = _search(client, user_id="user-b", query="apple", top_k=200)
        _assert_same_ranking(before, _snapshot(after_rebuild))
        assert len(user_b_hits) == 100
        assert {hit["content"] for hit in user_b_hits} == {"apple"}

    with sqlite3.connect(db_path) as connection:
        assert (
            connection.execute(
                "SELECT table_name FROM fts_partitions WHERE user_id = ?",
                ("user-a",),
            ).fetchone()[0]
            == user_a_table
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM raw_messages WHERE user_id = ?",
            ("user-b",),
        ).fetchone()[0] == 100

    store.delete_user_messages_for_rank_fixture("user-b")
    store.rebuild_user_lexical_index("user-b")
    with TestClient(create_app(Settings(db_path=db_path))) as client:
        after_delete = _search(client, user_id="user-a", query="apple banana")
        user_b_after_delete = _search(client, user_id="user-b", query="apple", top_k=200)
        _assert_same_ranking(before, _snapshot(after_delete))
        assert user_b_after_delete == []

    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM raw_messages WHERE user_id = ?",
            ("user-a",),
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM raw_messages WHERE user_id = ?",
            ("user-b",),
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM add_requests WHERE user_id = ?",
            ("user-b",),
        ).fetchone()[0] == 100


def test_a_users_own_writes_change_scores_and_other_users_do_not(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    with TestClient(create_app(Settings(db_path=db_path))) as client:
        _add(
            client,
            request_id="a-apple",
            user_id="user-a",
            session_id="session-a",
            contents=["apple"],
        )
        _add(
            client,
            request_id="a-both",
            user_id="user-a",
            session_id="session-a",
            contents=["apple banana"],
        )
        before_own = _search(client, user_id="user-a", query="apple banana", top_k=50)
        _flood(
            client,
            user_id="user-a",
            session_id="session-a2",
            content="apple",
            count=10,
            request_prefix="a-more-apple",
        )
        after_own = _search(client, user_id="user-a", query="apple banana", top_k=50)
        assert len(after_own) == 12
        assert _score_for(after_own, "apple banana") != _score_for(before_own, "apple banana")

        owned = _snapshot(after_own)
        _flood(
            client,
            user_id="user-b",
            session_id="session-b",
            content="apple",
            count=20,
            request_prefix="b-apple",
        )
        after_other = _search(client, user_id="user-a", query="apple banana", top_k=50)
        assert len(after_other) == 12
        _assert_same_ranking(owned, _snapshot(after_other))


def test_cjk_proper_name_and_number_ranks_stay_inside_the_user(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    with TestClient(create_app(Settings(db_path=db_path))) as client:
        _add(
            client,
            request_id="a-mix",
            user_id="用户A",
            session_id="session-a",
            contents=["苹果", "香蕉", "Ada Lovelace wrote notes", "Grace Hopper wrote notes", "badge 42", "badge 99"],
        )
        cjk_before = _snapshot(_search(client, user_id="用户A", query="苹果 香蕉"))
        name_before = _snapshot(_search(client, user_id="用户A", query="Ada Hopper"))
        number_before = _snapshot(_search(client, user_id="用户A", query="42 99"))
        assert {content for _item_id, content, _score in cjk_before} == {"苹果", "香蕉"}
        assert {content for _item_id, content, _score in name_before} == {
            "Ada Lovelace wrote notes",
            "Grace Hopper wrote notes",
        }
        assert {content for _item_id, content, _score in number_before} == {"badge 42", "badge 99"}

        _flood(client, user_id="用户B", session_id="session-b", content="苹果", count=20, request_prefix="b-cjk")
        _flood(
            client,
            user_id="用户B",
            session_id="session-b",
            content="Ada Lovelace",
            count=20,
            request_prefix="b-name",
        )
        _flood(client, user_id="用户B", session_id="session-b", content="badge 42", count=20, request_prefix="b-num")

        _assert_same_ranking(cjk_before, _snapshot(_search(client, user_id="用户A", query="苹果 香蕉")))
        _assert_same_ranking(name_before, _snapshot(_search(client, user_id="用户A", query="Ada Hopper")))
        _assert_same_ranking(number_before, _snapshot(_search(client, user_id="用户A", query="42 99")))
        assert _search(client, user_id="用户A", query="yaktoken") == []


def test_same_session_id_is_not_a_rank_boundary(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    with TestClient(create_app(Settings(db_path=db_path))) as client:
        _add(
            client,
            request_id="a-shared",
            user_id="user-a'quote",
            session_id="shared-session",
            contents=["apple", "banana"],
        )
        before = _snapshot(
            _search(client, user_id="user-a'quote", query="apple banana")
        )
        _flood(
            client,
            user_id='user-b"quote',
            session_id="shared-session",
            content="apple",
            count=20,
            request_prefix="b-shared",
        )
        after = _search(client, user_id="user-a'quote", query="apple banana")
        assert {hit["content"] for hit in after} == {"apple", "banana"}
        _assert_same_ranking(before, _snapshot(after))
        assert _search(client, user_id="user-a'quote", query="onlybmarker") == []

        _add(
            client,
            request_id="b-marker",
            user_id='user-b"quote',
            session_id="shared-session",
            contents=["onlybmarker lives here"],
        )
        assert _search(client, user_id="user-a'quote", query="onlybmarker") == []
        assert [hit["content"] for hit in _search(client, user_id='user-b"quote', query="onlybmarker")] == [
            "onlybmarker lives here"
        ]

    with sqlite3.connect(db_path) as connection:
        partitions = connection.execute(
            "SELECT user_id, table_name FROM fts_partitions"
        ).fetchall()
    assert partitions
    for user_id, table_name in partitions:
        assert user_id not in table_name
        assert "'" not in table_name
        assert '"' not in table_name


def test_same_user_across_sessions_stays_searchable_and_stable(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    with TestClient(create_app(Settings(db_path=db_path))) as client:
        _add(
            client,
            request_id="a-s1",
            user_id="user-a",
            session_id="session-1",
            contents=["north marker"],
        )
        _add(
            client,
            request_id="a-s2",
            user_id="user-a",
            session_id="session-2",
            contents=["south marker"],
        )
        before = _snapshot(_search(client, user_id="user-a", query="marker"))
        assert {content for _item_id, content, _score in before} == {"north marker", "south marker"}

        _flood(
            client,
            user_id="user-b",
            session_id="session-1",
            content="marker",
            count=20,
            request_prefix="b-marker",
        )
        after = _search(client, user_id="user-a", query="marker")
        assert {hit["content"] for hit in after} == {"north marker", "south marker"}
        _assert_same_ranking(before, _snapshot(after))


def test_equal_scores_keep_message_index_then_id_order(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    with TestClient(create_app(Settings(db_path=db_path))) as client:
        _add(
            client,
            request_id="a-tie",
            user_id="user-a",
            session_id="session-a",
            contents=["red apple", "red apple"],
        )
        before = _snapshot(_search(client, user_id="user-a", query="apple"))
        assert len(before) == 2
        assert before[0][2] == before[1][2]
        with sqlite3.connect(db_path) as connection:
            expected_ids = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT id FROM raw_messages
                    WHERE user_id = ?
                    ORDER BY message_index ASC, id ASC
                    """,
                    ("user-a",),
                )
            ]
        assert [item_id for item_id, _content, _score in before] == expected_ids

        _flood(
            client,
            user_id="user-b",
            session_id="session-b",
            content="red apple",
            count=20,
            request_prefix="b-tie",
        )
        after = _search(client, user_id="user-a", query="apple")
        _assert_same_ranking(before, _snapshot(after))
        assert after[0]["score"] == after[1]["score"]

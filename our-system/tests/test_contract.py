from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient


def _add(client: TestClient, **overrides):
    payload = {
        "request_id": "eval:run_abc123:locomo_refined:conv-0:chunk-0",
        "user_id": "eval:run_abc123:locomo:conv-0",
        "session_id": "eval:run_abc123:sample:0",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "memory text",
            }
        ],
    }
    payload.update(overrides)
    return client.post("/add", json=payload)


def _search(client: TestClient, **overrides):
    payload = {
        "query": "memory",
        "user_id": "eval:run_abc123:locomo:conv-0",
        "top_k": 100,
    }
    payload.update(overrides)
    return client.post("/search", json=payload)


def test_health_is_ok(api):
    client, _db_path = api
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_add_accepts_contract_and_echoes_ids(api):
    client, _db_path = api
    response = _add(client)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"success", "request_id", "user_id", "session_id"}
    assert body["success"] is True
    assert body["request_id"] == "eval:run_abc123:locomo_refined:conv-0:chunk-0"
    assert body["user_id"] == "eval:run_abc123:locomo:conv-0"
    assert body["session_id"] == "eval:run_abc123:sample:0"


def test_add_is_searchable_before_200(api):
    client, db_path = api
    response = _add(
        client,
        messages=[
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "Alice lives in Paris",
            },
            {"role": "assistant", "content": "Noted the Paris note."},
        ],
    )
    assert response.status_code == 200

    with sqlite3.connect(db_path) as connection:
        stored = connection.execute(
            "SELECT content FROM raw_messages ORDER BY message_index ASC"
        ).fetchall()
    assert [row[0] for row in stored] == [
        "Alice lives in Paris",
        "Noted the Paris note.",
    ]

    user_hit = _search(client, query="Paris", top_k=10)
    assistant_hit = _search(client, query="Noted", top_k=10)
    assert user_hit.status_code == 200
    assert assistant_hit.status_code == 200
    user_body = user_hit.json()["data"]
    assistant_body = assistant_hit.json()["data"]
    assert any(item["content"] == "Alice lives in Paris" for item in user_body)
    assert any(item["content"] == "Noted the Paris note." for item in assistant_body)
    paris = next(item for item in user_body if item["content"] == "Alice lives in Paris")
    assert paris["created_at"] == 1704067200000
    assert isinstance(paris["id"], str) and paris["id"]
    assert isinstance(paris["score"], (int, float))


def test_search_is_isolated_by_user_id(api):
    client, _db_path = api
    first = _add(
        client,
        request_id="req-a",
        user_id="user-a",
        session_id="sess-a",
        messages=[{"role": "user", "content": "zebratokenunique lives here"}],
    )
    second = _add(
        client,
        request_id="req-b",
        user_id="user-b",
        session_id="sess-b",
        messages=[{"role": "user", "content": "yaktokenunique lives here"}],
    )
    assert first.status_code == 200
    assert second.status_code == 200

    own = _search(client, query="zebratokenunique", user_id="user-a", top_k=10)
    other = _search(client, query="zebratokenunique", user_id="user-b", top_k=10)
    assert own.status_code == 200
    assert other.status_code == 200
    assert [item["content"] for item in own.json()["data"]] == ["zebratokenunique lives here"]
    assert other.json() == {"data": []}

    swapped = _search(client, query="yaktokenunique", user_id="user-a", top_k=10)
    assert swapped.json() == {"data": []}


def test_search_respects_top_k_and_empty_data(api):
    client, _db_path = api
    created = _add(
        client,
        messages=[
            {"role": "user", "content": "paris alpha"},
            {"role": "user", "content": "paris beta"},
            {"role": "user", "content": "paris gamma"},
        ],
    )
    assert created.status_code == 200
    saved = {"paris alpha", "paris beta", "paris gamma"}

    limited = _search(client, query="paris", top_k=2)
    wide = _search(client, query="paris", top_k=100)
    missing_user = _search(client, query="paris", user_id="user-without-memory", top_k=5)
    unrelated = _search(client, query="xylophone", top_k=5)

    assert limited.status_code == 200
    limited_body = limited.json()
    assert set(limited_body) == {"data"}
    assert len(limited_body["data"]) == 2
    assert {item["content"] for item in limited_body["data"]} <= saved
    for item in limited_body["data"]:
        assert set(item) <= {"id", "content", "score", "created_at"}
        assert item["id"]
        assert item["content"] in saved

    assert wide.status_code == 200
    assert len(wide.json()["data"]) == 3
    assert missing_user.json() == {"data": []}
    assert unrelated.json() == {"data": []}


def test_search_returns_evidence_not_answer(api):
    client, _db_path = api
    evidence = "Sam moved to Lyon in 2019."
    created = _add(
        client,
        messages=[{"role": "user", "content": evidence, "timestamp": 1704067200000}],
    )
    assert created.status_code == 200

    response = _search(
        client,
        query="Where did Sam move",
        options=["A. Lyon", "B. Paris"],
        top_k=5,
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"data"}
    assert "abstain" not in body
    assert "answer" not in body
    assert len(body["data"]) == 1
    hit = body["data"][0]
    assert hit["content"] == evidence
    assert "A. Lyon" not in hit["content"]
    assert "B. Paris" not in hit["content"]
    assert "explanation" not in hit


def test_unsupported_add_fields_are_rejected(api):
    client, _db_path = api
    response = _add(client, answer="A. Lyon")
    assert 400 <= response.status_code < 500


def test_search_finds_saved_keyword_evidence(api):
    client, _db_path = api
    english = "the badge number is 42"
    chinese = "爱丽丝住在巴黎"
    created = _add(
        client,
        messages=[
            {"role": "user", "content": english},
            {"role": "user", "content": chinese},
        ],
    )
    assert created.status_code == 200

    number_hit = _search(client, query="42", top_k=5)
    city_hit = _search(client, query="巴黎", top_k=5)
    assert [item["content"] for item in number_hit.json()["data"]] == [english]
    assert [item["content"] for item in city_hit.json()["data"]] == [chinese]
    assert "\n" not in city_hit.json()["data"][0]["content"]


def test_search_includes_every_session_for_the_same_user(api):
    client, _db_path = api
    first = _add(
        client,
        request_id="req-s1",
        session_id="session-1",
        messages=[{"role": "user", "content": "sessionone marker"}],
    )
    second = _add(
        client,
        request_id="req-s2",
        session_id="session-2",
        messages=[{"role": "user", "content": "sessiontwo marker"}],
    )
    assert first.status_code == 200
    assert second.status_code == 200

    response = _search(client, query="marker", top_k=10)
    assert response.status_code == 200
    assert {item["content"] for item in response.json()["data"]} == {
        "sessionone marker",
        "sessiontwo marker",
    }


def test_invalid_add_and_search_payloads_return_4xx(api):
    client, db_path = api
    invalid_adds = [
        {"user_id": "user-1", "session_id": "sess-1", "messages": [{"role": "user", "content": "x"}]},
        {
            "request_id": "req-1",
            "user_id": "user-1",
            "session_id": "sess-1",
            "messages": [{"role": "user", "content": "   "}],
        },
        {
            "request_id": "req-1",
            "user_id": "user-1",
            "session_id": "sess-1",
            "messages": [{"role": "system", "content": "x"}],
        },
        {
            "request_id": "req-1",
            "user_id": "user-1",
            "session_id": "sess-1",
            "messages": [],
        },
        {
            "request_id": "req-1",
            "user_id": "user-1",
            "session_id": "sess-1",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "image later"}]}],
        },
        {
            "data": {
                "request_id": "req-1",
                "user_id": "user-1",
                "session_id": "sess-1",
                "messages": [{"role": "user", "content": "wrapped"}],
            }
        },
    ]
    for payload in invalid_adds:
        response = client.post("/add", json=payload)
        assert 400 <= response.status_code < 500

    invalid_searches = [
        {"query": "paris", "user_id": "user-1"},
        {"query": "paris", "user_id": "user-1", "top_k": 0},
        {"query": "   ", "user_id": "user-1", "top_k": 5},
        {"query": "paris", "top_k": 5},
        {"query": "paris", "user_id": "user-1", "top_k": 1.5},
    ]
    for payload in invalid_searches:
        response = client.post("/search", json=payload)
        assert 400 <= response.status_code < 500

    with sqlite3.connect(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0]
    assert count == 0

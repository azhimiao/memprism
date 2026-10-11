"""HTTP contract/error tests using MockTransport; no network or real embeddings."""
from __future__ import annotations

import json
import math

import httpx
import pytest

from app.embeddings import EmbeddingConfig, EmbeddingFailure, HttpEmbeddingProvider, normalize_vector, prepare_embedding


def config(**overrides):
    args = dict(provider="unit-http-contract", endpoint="https://embedding.test/v1/embeddings",
                model="text-embedding-v4", dimensions=8, batch_size=2)
    args.update(overrides)
    return EmbeddingConfig(**args)


def success(request):
    body = json.loads(request.content)
    return httpx.Response(200, json={"model": body["model"],
        "data": [{"index": i, "embedding": [float(i + 1)] + [0.0] * 7}
                 for i in reversed(range(len(body["input"])))],
        "usage": {"total_tokens": 5}})


@pytest.mark.parametrize('overrides,accepted', [
    ({}, True),
    ({'provider': 'other'}, False),
    ({'endpoint': 'https://embedding.test/v1/embeddings'}, False),
    ({'model': 'text-embedding-v4'}, False),
])
def test_openrouter_exact_catalog_alias_is_route_scoped(monkeypatch, overrides, accepted):
    monkeypatch.setenv('AML_EMBEDDING_API_KEY', 'isolated-test-key')
    cfg = dict(provider='openrouter', endpoint='https://openrouter.ai/api/v1/embeddings',
               model='qwen/qwen3-embedding-8b')
    cfg.update(overrides)
    def handle(request):
        result = success(request).json()
        result['model'] = 'Qwen/Qwen3-Embedding-8B'
        return httpx.Response(200, json=result)
    prepared = prepare_embedding(HttpEmbeddingProvider(config(**cfg), transport=httpx.MockTransport(handle)), ['text'])
    assert prepared.status == ('ready' if accepted else 'failed')
    if not accepted:
        assert prepared.error_code == 'response_model_mismatch' and prepared.vectors == []


def test_http_batch_indices_usage_and_secret_hygiene(monkeypatch, caplog):
    monkeypatch.setenv("AML_EMBEDDING_API_KEY", "test-secret-do-not-log")
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-secret-do-not-log"
        return success(request)

    provider = HttpEmbeddingProvider(config(), transport=httpx.MockTransport(handle))
    result = provider.embed(["first-private-body", "second", "third"])
    assert [len(b["input"]) for b in seen] == [2, 1]
    assert all(b["encoding_format"] == "float" and b["dimensions"] == 8 for b in seen)
    assert len(result.vectors) == 3
    assert result.vectors[0] == [1.0] + [0.0] * 7
    assert result.usage["requests"] == 2
    assert result.usage["reported_input_tokens"] == 10
    assert result.usage["usage_complete"] is True
    assert "test-secret-do-not-log" not in caplog.text
    assert "first-private-body" not in caplog.text


@pytest.mark.parametrize("kind,expected", [
    ("dimensions", "dimension_mismatch"), ("nan", "invalid_vector"),
    ("zero", "zero_vector"), ("indices", "invalid_response_index"),
    ("count", "response_count_mismatch"), ("model", "response_model_mismatch"),
    ("json", "invalid_embedding_response"),
])
def test_invalid_responses_become_explicit_failed_state(monkeypatch, kind, expected):
    monkeypatch.setenv("AML_EMBEDDING_API_KEY", "test-key")

    def handle(request):
        if kind == "json":
            return httpx.Response(200, text="not-json")
        data = success(request).json()
        if kind == "dimensions": data["data"][0]["embedding"] = [1.0]
        if kind == "nan":
            return httpx.Response(200, content=b'{"data":[{"index":0,"embedding":[NaN,0,0,0,0,0,0,0]}]}')
        if kind == "zero": data["data"][0]["embedding"] = [0.0] * 8
        if kind == "indices": data["data"][0]["index"] = True
        if kind == "count": data["data"] = []
        if kind == "model": data["model"] = "different-model"
        return httpx.Response(200, json=data)

    provider = HttpEmbeddingProvider(config(), transport=httpx.MockTransport(handle))
    prepared = prepare_embedding(provider, ["text"])
    assert prepared.status == "failed"
    assert prepared.error_code == expected
    assert prepared.vectors == []
    assert prepared.usage["requests"] == 1


def test_timeout_and_missing_key_never_silently_supply_vectors(monkeypatch):
    monkeypatch.delenv("AML_EMBEDDING_API_KEY", raising=False)
    seen = []

    def timeout(request):
        seen.append(request.url)
        raise httpx.ReadTimeout("secret remote message must not escape", request=request)

    provider = HttpEmbeddingProvider(config(), transport=httpx.MockTransport(timeout))
    assert prepare_embedding(provider, ["text"]).error_code == "missing_api_key"
    assert seen == []
    monkeypatch.setenv("AML_EMBEDDING_API_KEY", "key")
    failed = prepare_embedding(provider, ["text"])
    assert failed.error_code == "embedding_timeout"
    assert failed.usage["requests"] == 1 and failed.usage["usage_complete"] is False


def test_partial_batch_success_is_not_persisted_as_complete(monkeypatch):
    monkeypatch.setenv("AML_EMBEDDING_API_KEY", "key")
    calls = []

    def handle(request):
        calls.append(request)
        return success(request) if len(calls) == 1 else httpx.Response(500, text="sensitive upstream body")

    failed = prepare_embedding(HttpEmbeddingProvider(config(), transport=httpx.MockTransport(handle)), ["a", "b", "c"])
    assert failed.status == "failed" and failed.vectors == []
    assert failed.error_code == "embedding_http_500"
    assert failed.usage["requests"] == 2
    assert failed.usage["reported_input_tokens"] == 5 and not failed.usage["usage_complete"]


def test_retry_budget_counts_each_http_attempt(monkeypatch):
    monkeypatch.setenv("AML_EMBEDDING_API_KEY", "key")
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(429) if len(calls) == 1 else success(request)

    result = HttpEmbeddingProvider(config(max_retries=1), transport=httpx.MockTransport(handle)).embed(["abc"])
    assert result.usage["requests"] == 2
    assert result.usage["input_characters_sent"] == 6
    assert not result.usage["usage_complete"]


@pytest.mark.parametrize("endpoint", ["http://example.test/embeddings", "https://key:secret@example.test/embeddings", "https://example.test/embeddings?key=secret"])
def test_endpoint_never_accepts_inline_credentials_or_plaintext(endpoint):
    with pytest.raises(ValueError):
        config(endpoint=endpoint)


def test_finite_vector_normalization_handles_extreme_scales():
    for vector in ([1e308, 1e308], [1e-308, 1e-308]):
        result = normalize_vector(vector, 2)
        assert math.isclose(sum(x * x for x in result), 1.0)


@pytest.mark.parametrize("phase", ["add", "search"])
def test_huge_json_integer_has_recorded_raw_fallback(tmp_path, monkeypatch, phase):
    import sqlite3
    from fastapi.testclient import TestClient
    from app.api import create_app
    from app.config import Settings
    from test_dense import payload, query

    monkeypatch.setenv("AML_EMBEDDING_API_KEY", "local-test-key")

    def handle(request):
        data = json.loads(request.content)
        if phase == "add" or data["input"] == ["apple probe"]:
            return httpx.Response(200, json={"model": data["model"], "data": [
                {"index": i, "embedding": [10**400] + [0] * 7} for i in range(len(data["input"]))
            ]})
        return success(request)

    provider = HttpEmbeddingProvider(config(), transport=httpx.MockTransport(handle))
    path = tmp_path / "m.sqlite"
    app = create_app(Settings(db_path=path, retrieval_mode="raw_hybrid_rrf", candidate_limit=4,
                              lexical_quota=2, dense_quota=2), embedding_provider=provider)
    with TestClient(app) as client:
        assert client.post("/add", json=payload(contents=[f"apple {i}" for i in range(4)])).status_code == 200
        if phase == "search":
            result = query(client, text="apple probe")
            assert result.status_code == 200
            assert result.headers["X-AML-Degraded-Reason"] == "invalid_vector"
            assert len(result.json()["data"]) == 4
            from app.models import SearchRequest
            _, trace = app.state.retrieval.search_with_trace(SearchRequest(query="apple probe", user_id="a", top_k=4))
            assert trace["query_embedding_usage"]["requests"] == 1
    if phase == "add":
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 4
            assert db.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0] == 1
            assert db.execute("SELECT COUNT(*) FROM raw_vectors").fetchone()[0] == 0
            state, error, usage = db.execute("SELECT status, error_code, usage_json FROM embedding_requests").fetchone()
            assert (state, error) == ("failed", "invalid_vector")
            assert json.loads(usage)["requests"] == 1

"""Native v4 and deployment boundaries. MockTransport only; no supplier calls."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import (EmbeddingConfig, EmbeddingFailure, HttpEmbeddingProvider,
    DashScopeEmbeddingProvider, create_embedding_provider, prepare_embedding,
    provider_identity, provider_version)
from deploy.healthcheck import healthy
from deploy.runtime import (build_app, check_runtime, check_database, configured_provider,
                            PreflightFailure)

ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"


def config(**overrides):
    values = dict(provider="dashscope", endpoint=ENDPOINT, dimensions=64,
                  api_key_env="V4_TEST_KEY")
    values.update(overrides)
    return EmbeddingConfig(**values)


@pytest.fixture(autouse=True)
def environment_credentials(monkeypatch):
    # Random, non-working credentials enter the provider exclusively via env.
    monkeypatch.setenv("V4_TEST_KEY", uuid.uuid4().hex)
    monkeypatch.setenv("AML_SERVICE_API_KEY", uuid.uuid4().hex)
    for name in ("AML_EMBEDDING_REGION", "AML_EMBEDDING_MAX_INPUT_BYTES",
                 "AML_EMBEDDING_MIN_INTERVAL_SECONDS"):
        monkeypatch.delenv(name, raising=False)


def success(request):
    body = json.loads(request.content)
    return httpx.Response(200, json={"request_id": "offline-mock-request",
        "output": {"embeddings": [{"text_index": i,
            "embedding": [i + 1.0, 1.0] + [0.0] * (body["parameters"]["dimension"] - 2)}
            for i in reversed(range(len(body["input"]["texts"]))) ]},
        "usage": {"total_tokens": 7}})


@pytest.mark.parametrize("dimensions", [64, 128, 256, 512, 768, 1024, 1536, 2048])
def test_supported_dimensions(dimensions):
    assert config(dimensions=dimensions).dimensions == dimensions


@pytest.mark.parametrize("changes", [
    {"dimensions": 8}, {"dimensions": True}, {"model": "text-embedding-v3"},
    {"batch_size": True}, {"batch_size": 11}, {"max_retries": True},
    {"max_retries": 3}, {"timeout_seconds": float("nan")}, {"timeout_seconds": True},
    {"region": "us-east-1"}, {"max_input_bytes": 8193}, {"max_input_bytes": True},
    {"api_key_env": "inline-key-value!"}, {"min_request_interval_seconds": -1},
    {"min_request_interval_seconds": float("inf")},
    {"endpoint": "https://embedding.test" + ENDPOINT.split(".com")[1]},
    {"endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"},
    {"endpoint": ENDPOINT + "?credential=x"}, {"endpoint": ENDPOINT + "#fragment"},
    {"endpoint": ENDPOINT.replace("https://", "http://")},
    {"endpoint": ENDPOINT.replace(".com/", ".com:444/")},
    {"endpoint": ENDPOINT.replace(".com/", ".com.evil.test/")},
])
def test_configuration_rejects_invalid_native_identity(changes):
    with pytest.raises(ValueError):
        config(**changes)


@pytest.mark.parametrize("host,region", [
    ("dashscope.aliyuncs.com", "cn-beijing"),
    ("dashscope-intl.aliyuncs.com", "ap-southeast-1"),
    ("dashscope-us.aliyuncs.com", "us-east-1"),
    ("cn-hongkong.dashscope.aliyuncs.com", "cn-hongkong"),
    ("llm-example.cn-beijing.maas.aliyuncs.com", "cn-beijing"),
    ("llm-example.eu-central-1.maas.aliyuncs.com", "eu-central-1"),
])
def test_region_is_bound_to_official_endpoint(host, region):
    assert config(endpoint=ENDPOINT.replace("dashscope.aliyuncs.com", host)).identity["region"] == region


def test_factory_legacy_identity_and_wire_format_stay_compatible():
    legacy = EmbeddingConfig(provider="openai", endpoint="https://embedding.test/v1/embeddings",
                             dimensions=8, model="legacy-model", api_key_env="V4_TEST_KEY")
    expected = {"provider": "openai", "endpoint": legacy.endpoint, "model": "legacy-model",
        "dimensions": 8, "revision": "1", "representation": "whole_raw_text_l2_float64_v1",
        "implementation_kind": "http"}
    seen = []
    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "legacy-model",
            "data": [{"index": 0, "embedding": [1.0] * 8}], "usage": {"total_tokens": 2}})
    provider = create_embedding_provider(legacy, transport=httpx.MockTransport(respond))
    assert type(provider) is HttpEmbeddingProvider
    assert provider.for_query() is provider
    assert provider_identity(provider) == expected
    assert provider_version(provider) == hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    assert provider.embed(["legacy"]).usage["reported_input_tokens"] == 2
    assert seen == [{"model": "legacy-model", "input": ["legacy"], "dimensions": 8, "encoding_format": "float"}]
    assert isinstance(create_embedding_provider(config()), DashScopeEmbeddingProvider)


def test_batches_native_shape_indices_normalization_and_usage():
    seen = []
    def respond(request):
        seen.append(json.loads(request.content))
        return success(request)
    provider = create_embedding_provider(config(), transport=httpx.MockTransport(respond))
    result = provider.embed(["raw\r\n中文 e\u0301 " + str(i) for i in range(23)])
    assert [len(b["input"]["texts"]) for b in seen] == [10, 10, 3]
    assert all(b["parameters"] == {"dimension": 64, "text_type": "document", "output_type": "dense"} for b in seen)
    assert seen[0]["input"]["texts"][0] == "raw\r\n中文 e\u0301 0"
    assert len(result.vectors) == 23
    assert result.vectors[0][0] == pytest.approx(1 / math.sqrt(2))
    assert result.vectors[1][0] == pytest.approx(2 / math.sqrt(5))
    assert result.usage["requests"] == 3 and result.usage["reported_input_tokens"] == 21
    assert result.usage["model_identity_source"] == "request_only_or_partial"
    assert result.usage["response_models"] == [] and result.usage["usage_complete"]
    query = provider.for_query()
    assert provider_version(query) == provider_version(provider)
    query.embed(["query"])
    assert seen[-1]["parameters"]["text_type"] == "query"
    assert provider_version(create_embedding_provider(config(dimensions=128))) != provider_version(provider)
    assert provider_version(create_embedding_provider(config(revision="2"))) != provider_version(provider)


@pytest.mark.parametrize("texts,code", [
    ("single-string", "invalid_input"), ([], "invalid_input"), ([" "], "invalid_input"),
    (["valid", None], "invalid_input"), (["x" * 8193], "input_too_long"),
    (["中文" * 1400], "input_too_long"), (["\ud800"], "invalid_input_unicode"),
    (["short"] * 10 + ["x" * 8193], "input_too_long"),
])
def test_all_inputs_checked_before_first_dispatch(texts, code):
    seen = []
    provider = create_embedding_provider(config(), transport=httpx.MockTransport(lambda r: seen.append(r)))
    with pytest.raises(EmbeddingFailure, match=code):
        provider.embed(texts)
    assert seen == []


def test_exact_byte_boundary_has_no_truncation():
    seen = []
    def respond(request):
        seen.append(json.loads(request.content)["input"]["texts"])
        return success(request)
    provider = create_embedding_provider(config(), transport=httpx.MockTransport(respond))
    provider.embed(["a" * 8192, "😀" * 2048])
    assert seen == [["a" * 8192, "😀" * 2048]]


@pytest.mark.parametrize("kind,code", [
    ("supplier-code", "embedding_provider_error"), ("status", "embedding_provider_error"),
    ("error", "embedding_provider_error"), ("model", "response_model_mismatch"),
    ("count", "response_count_mismatch"), ("duplicate", "invalid_response_index"),
    ("bool-index", "invalid_response_index"), ("missing-index", "invalid_response_index"),
    ("dimension", "dimension_mismatch"), ("zero", "zero_vector"),
    ("boolean", "invalid_vector"), ("infinite", "invalid_vector"),
    ("malformed", "invalid_embedding_response"), ("list-body", "invalid_embedding_response"),
])
def test_invalid_native_responses_fail_without_partial_vectors(kind, code):
    def respond(request):
        data = success(request).json()
        items = data["output"]["embeddings"]
        if kind == "supplier-code": data["code"] = "private-provider-body"
        if kind == "status": data["status_code"] = False
        if kind == "error": data["error"] = {"message": "private-provider-body"}
        if kind == "model": data["model"] = "wrong-model"
        if kind == "count": items.pop()
        if kind == "duplicate": items[1]["text_index"] = items[0]["text_index"]
        if kind == "bool-index": items[0]["text_index"] = True
        if kind == "missing-index": items[0].pop("text_index")
        if kind == "dimension": items[0]["embedding"] = [1.0]
        if kind == "zero": items[0]["embedding"] = [0.0] * 64
        if kind == "boolean": items[0]["embedding"][0] = True
        if kind == "infinite":
            return httpx.Response(200, content=json.dumps(data).replace("2.0", "Infinity").encode())
        if kind == "malformed": return httpx.Response(200, content=b"not-json")
        if kind == "list-body": data = []
        return httpx.Response(200, json=data)
    failed = prepare_embedding(create_embedding_provider(config(), transport=httpx.MockTransport(respond)), ["one", "two"])
    assert failed.status == "failed" and failed.vectors == [] and failed.error_code == code
    assert failed.usage["requests"] == 1 and failed.usage["usage_complete"] is False
    assert "private-provider-body" not in json.dumps(failed.usage)
    if kind not in {"malformed", "list-body"}:
        assert failed.usage["reported_input_tokens"] == 7


@pytest.mark.parametrize("status", [301, 400, 401, 403, 429, 500])
def test_http_failures_default_to_one_dispatch(status):
    seen = []
    def respond(request):
        seen.append(request)
        return httpx.Response(status, text="private response", headers={"Location": "https://evil.test"})
    result = prepare_embedding(create_embedding_provider(config(), transport=httpx.MockTransport(respond)), ["one"])
    assert result.error_code == f"embedding_http_{status}" and len(seen) == 1
    assert result.vectors == []


def test_partial_batch_failure_retains_usage_but_never_vectors():
    seen = []
    def respond(request):
        seen.append(request)
        return success(request) if len(seen) == 1 else httpx.Response(503)
    result = prepare_embedding(create_embedding_provider(config(batch_size=1), transport=httpx.MockTransport(respond)), ["one", "two"])
    assert result.vectors == [] and result.usage["requests"] == 2
    assert result.usage["reported_input_tokens"] == 7 and not result.usage["usage_complete"]


def test_retry_and_guard_count_actual_dispatch(monkeypatch):
    seen, permits = [], []
    monkeypatch.setattr("app.embeddings.time.sleep", lambda delay: None)
    def respond(request):
        seen.append(request)
        return httpx.Response(429) if len(seen) == 1 else success(request)
    provider = create_embedding_provider(config(max_retries=1), transport=httpx.MockTransport(respond))
    result = provider.embed_guarded(["one"], before_dispatch=lambda: permits.append(True))
    assert len(permits) == len(seen) == result.usage["requests"] == 2
    assert not result.usage["usage_complete"]
    def refuse():
        raise RuntimeError("fenced")
    with pytest.raises(RuntimeError, match="fenced"):
        provider.for_query().embed_guarded(["query"], before_dispatch=refuse)
    assert len(seen) == 2


def test_shared_rate_gate_is_fail_fast_before_query_dispatch(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("app.embeddings.time.monotonic", lambda: now[0])
    provider = create_embedding_provider(config(min_request_interval_seconds=0.5), transport=httpx.MockTransport(success))
    provider.embed(["document"])
    with pytest.raises(EmbeddingFailure, match="embedding_rate_limited") as error:
        provider.for_query().embed(["query"])
    assert error.value.usage["requests"] == 0
    assert error.value.usage["model_identity_source"] == "request_only_or_partial"
    now[0] += 0.5
    assert provider.for_query().embed(["query"]).usage["requests"] == 1


def test_timeout_and_key_errors_do_not_leak_credentials(monkeypatch, caplog):
    seen = []
    def timeout(request):
        seen.append(request)
        raise httpx.ReadTimeout("private timeout body", request=request)
    provider = create_embedding_provider(config(), transport=httpx.MockTransport(timeout))
    monkeypatch.delenv("V4_TEST_KEY")
    assert prepare_embedding(provider, ["one"]).error_code == "missing_api_key" and seen == []
    monkeypatch.setenv("V4_TEST_KEY", "\r\n")
    assert prepare_embedding(provider, ["one"]).error_code == "missing_api_key"
    credential = uuid.uuid4().hex
    monkeypatch.setenv("V4_TEST_KEY", credential)
    result = prepare_embedding(provider, ["one"])
    assert result.error_code == "embedding_timeout" and len(seen) == 1
    assert credential not in caplog.text and "private timeout body" not in str(result)


def settings(path, profile="cycle2_dense"):
    return Settings(db_path=path, profile_id=profile, embedding=config(),
        enable_vector=True, retrieval_mode="raw_hybrid_rrf", auth_mode="bearer")


def test_deployment_add_search_query_role_auth_replay_and_isolation(tmp_path, monkeypatch):
    import os
    seen = []
    def respond(request):
        seen.append(json.loads(request.content))
        return success(request)
    application = build_app(settings(tmp_path / "v4.db"), transport=httpx.MockTransport(respond))
    auth = {"Authorization": "Bearer " + os.environ["AML_SERVICE_API_KEY"]}
    body = {"request_id": "v4-add", "user_id": "alice", "session_id": "s",
            "messages": [{"role": "user", "content": "中文 exact evidence alpha"}]}
    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        assert healthy(client.get("/health").json())
        assert client.post("/add", json=body).status_code == 401
        assert client.post("/search", json={"query": "alpha", "user_id": "alice", "top_k": 5}).status_code == 401
        assert client.post("/add", json=body, headers=auth).json()["success"] is True
        assert seen[-1]["parameters"]["text_type"] == "document"
        after_add = len(seen)
        assert client.post("/add", json=body, headers=auth).status_code == 200
        assert len(seen) == after_add
        result = client.post("/search", json={"query": "alpha", "user_id": "alice", "top_k": 5}, headers=auth)
        assert result.status_code == 200 and result.json()["data"][0]["content"] == body["messages"][0]["content"]
        assert result.headers["x-aml-degraded"] == "false"
        assert seen[-1]["parameters"]["text_type"] == "query"
        other = client.post("/search", json={"query": "alpha", "user_id": "bob", "top_k": 5}, headers=auth)
        assert other.json() == {"data": []}
        assert client.get("/docs").status_code == 404
        with sqlite3.connect(tmp_path / "v4.db") as db:
            db.execute("UPDATE k0_watermarks SET ready=0 WHERE user_id='alice'")
        response = client.get("/health")
        assert response.status_code == 200 and not healthy(response.json())
    with pytest.raises(RuntimeError):
        build_app(settings(tmp_path / "v4.db"), transport=httpx.MockTransport(respond))


def test_failed_add_has_no_partial_publication(tmp_path):
    import os
    application = build_app(settings(tmp_path / "v4.db"), transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with TestClient(application) as client:
        result = client.post("/add", headers={"Authorization": "Bearer " + os.environ["AML_SERVICE_API_KEY"]},
            json={"request_id": "fail", "user_id": "u", "session_id": "s",
                  "messages": [{"role": "user", "content": "must not publish"}]})
        assert result.status_code == 503
    with sqlite3.connect(tmp_path / "v4.db") as db:
        assert db.execute("SELECT count(*) FROM raw_messages").fetchone()[0] == 0


def test_old_database_is_refused_without_identity_relabel(tmp_path):
    from app.api import create_app
    from app.profiles import Profile
    old = tmp_path / "old.db"
    create_app(Settings(db_path=old, profile_id="raw_sync"))
    with sqlite3.connect(old) as db:
        before = db.execute("SELECT * FROM k0_config").fetchall()
    with pytest.raises(PreflightFailure, match="identity_mismatch"):
        build_app(settings(old), transport=httpx.MockTransport(success))
    with sqlite3.connect(old) as db:
        assert db.execute("SELECT * FROM k0_config").fetchall() == before
    new = tmp_path / "new.db"
    application = build_app(settings(new), transport=httpx.MockTransport(success))
    assert check_database(new, application.state.profile)["database"] == "same_identity"
    with pytest.raises(PreflightFailure, match="identity_mismatch"):
        changed, provider = configured_provider(replace(settings(new), embedding=config(dimensions=128)))
        check_database(new, Profile.build(changed, provider))


def test_offline_runtime_and_missing_cache_fail_before_network(tmp_path, monkeypatch):
    result = check_runtime()
    assert result["python"] == "3.12.11" and result["packages"]["tiktoken"] == "0.14.0"
    assert result["external_calls"] == 0
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "absent"))
    with pytest.raises(PreflightFailure, match="tokenizer_cache_missing"):
        check_runtime()


@pytest.mark.parametrize("payload", [{"status": "ok"}, {"status": "not_ready", "control_ready": True, "data_route_ready": False},
                                     {"status": "ok", "control_ready": 1, "data_route_ready": 1}, [], None])
def test_health_requires_explicit_route_readiness(payload):
    assert not healthy(payload)


def test_rebuild_new_copy_preserves_lifecycle_receipts_and_rejects_old_token_identity(tmp_path, monkeypatch):
    from app.api import create_app
    from app import segments
    from app.k0_migrate import logical_hash
    from deploy.rebuild import rebuild, inventory, paths
    source, target = tmp_path / "tk12.db", tmp_path / "tk14-v4.db"
    # Simulate the persisted 0.12 identity with the current tokenizer implementation.
    # This is a migration contract test, not a 0.12/0.14 tokenizer equivalence claim.
    with monkeypatch.context() as old_identity:
        old_identity.setattr(segments, "version", lambda name: "0.12.0")
        segments.fragment_hash.cache_clear()
        old = create_app(Settings(db_path=source, profile_id="span_sync"))
        with TestClient(old) as client:
            for request_id, content in (("erase-source", "deleted alpha"),
                ("erase-control", 'delete "deleted alpha"'), ("live", "surviving beta")):
                assert client.post("/add", json={"request_id": request_id, "user_id": "u", "session_id": "s",
                    "messages": [{"role": "user", "content": content}]}).status_code == 200
        old_hash = segments.fragment_hash()
    segments.fragment_hash.cache_clear()
    with sqlite3.connect(source) as db:
        before = logical_hash(db)
        raws = db.execute("SELECT * FROM raw_messages ORDER BY id").fetchall()
        tombstones = db.execute("SELECT * FROM memory_tombstones ORDER BY raw_id").fetchall()
        receipts = db.execute("SELECT * FROM add_requests ORDER BY request_id").fetchall()
        original_ids = db.execute("SELECT id FROM source_spans").fetchall()
    assert inventory(source)["expected_spans"] == 1
    paths(source, target, tmp_path / "manifest.json")
    assert not target.exists()
    cfg, provider = configured_provider(settings(target), transport=httpx.MockTransport(success))
    with pytest.raises(PreflightFailure, match="identity_mismatch"):
        build_app(settings(source), transport=httpx.MockTransport(success))
    result = rebuild(source, target, cfg, provider)
    assert result["status"] == "certified" and result["external_calls"] == 1
    assert all(p["counts"]["dense"] == p["counts"]["active"] for p in result["proofs"])
    with sqlite3.connect(source) as db:
        assert logical_hash(db) == before
    with sqlite3.connect(target) as db:
        assert db.execute("SELECT * FROM raw_messages ORDER BY id").fetchall() == raws
        assert db.execute("SELECT * FROM memory_tombstones ORDER BY raw_id").fetchall() == tombstones
        assert db.execute("SELECT * FROM add_requests ORDER BY request_id").fetchall() == receipts
        assert db.execute("SELECT id FROM source_spans").fetchall() != original_ids
        assert db.execute("SELECT DISTINCT fragment_hash FROM source_spans").fetchall() == [(segments.fragment_hash(),)]
        assert old_hash != segments.fragment_hash()
    rebuilt_app = build_app(cfg, transport=httpx.MockTransport(success))
    assert healthy(TestClient(rebuilt_app).get("/health").json())
    with pytest.raises(PreflightFailure):
        paths(source, target, tmp_path / "manifest.json")


def test_failed_copy_rebuild_keeps_source_and_target_unservable(tmp_path):
    from app.api import create_app
    from app.k0_migrate import logical_hash
    from deploy.rebuild import rebuild
    source, target = tmp_path / "old.db", tmp_path / "failed.db"
    old = create_app(Settings(db_path=source, profile_id="span_sync"))
    with TestClient(old) as client:
        assert client.post("/add", json={"request_id": "r", "user_id": "u", "session_id": "s",
            "messages": [{"role": "user", "content": "unpublished v4"}]}).status_code == 200
    with sqlite3.connect(source) as db:
        before = logical_hash(db)
    cfg, provider = configured_provider(settings(target), transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    audit = {}
    with pytest.raises(RuntimeError, match="rebuild_embedding_failed"):
        rebuild(source, target, cfg, provider, audit=audit)
    assert audit["embedding_usage"][0]["requests"] == 1
    with sqlite3.connect(source) as db:
        assert logical_hash(db) == before
    with pytest.raises(PreflightFailure, match="identity_mismatch"):
        build_app(cfg, transport=httpx.MockTransport(success))


def test_business_retry_and_usage_do_not_claim_response_identity(monkeypatch):
    monkeypatch.setattr("app.embeddings.time.sleep", lambda delay: None)
    seen = []
    def respond(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(200, json={"code": "Throttling.RateQuota", "message": "private"})
        data = success(request).json()
        data["model"] = "text-embedding-v4"
        return httpx.Response(200, json=data)
    provider = create_embedding_provider(config(max_retries=1), transport=httpx.MockTransport(respond))
    result = provider.embed(["one"])
    assert result.usage["requests"] == 2 and result.usage["reported_input_tokens"] == 7
    assert result.usage["model_identity_source"] == "request_only_or_partial"
    assert result.usage["response_models"] == ["text-embedding-v4"]
    assert not result.usage["usage_complete"]


def test_start_and_health_cli_use_real_loopback_without_supplier_calls(tmp_path):
    import os
    import socket
    import subprocess
    import sys
    import time
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("AML_")}
    env.update(AML_PROFILE="cycle2_dense", AML_RETRIEVAL_MODE="raw_hybrid_rrf",
        AML_ENABLE_VECTOR="true", AML_DB_PATH=str(tmp_path / "process.db"),
        AML_AUTH_MODE="bearer", AML_SERVICE_API_KEY=uuid.uuid4().hex,
        AML_EMBEDDING_PROVIDER="dashscope", AML_EMBEDDING_ENDPOINT=ENDPOINT,
        AML_EMBEDDING_DIMENSIONS="64", AML_EMBEDDING_KEY_ENV="V4_TEST_KEY", AML_PORT=str(port),
        PYTHONDONTWRITEBYTECODE="1")
    root = Path(__file__).resolve().parents[1]
    # No Add/Search is authorized here: health and unauthenticated rejection only.
    process = subprocess.Popen([sys.executable, "-m", "deploy.start"], cwd=root, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        with httpx.Client(trust_env=False, timeout=1) as client:
            while True:
                assert process.poll() is None, "startup failed"
                try:
                    response = client.get(f"http://127.0.0.1:{port}/health")
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                assert time.monotonic() < deadline, "startup timed out"
                time.sleep(0.05)
            assert healthy(response.json())
            assert client.post(f"http://127.0.0.1:{port}/search", json={}).status_code == 401
        result = subprocess.run([sys.executable, "-m", "deploy.healthcheck"], cwd=root, env=env,
                                capture_output=True, timeout=10)
        assert result.returncode == 0
    finally:
        process.terminate()
        process.wait(timeout=10)

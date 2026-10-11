"""Offline tests for scripts/v4_live_smoke.py (no live network)."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "v4_live_smoke.py"
PY = Path(sys.executable)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.v4_live_smoke as smoke  # noqa: E402

FAKE_DASH_KEY = "sk-test-SECRET123"
DIM = smoke.PROBE_DIMENSIONS


def _unit_vector(first: float, second: float = 0.0) -> list[float]:
    raw = [first, second] + [0.0] * (DIM - 2)
    scale = max(abs(v) for v in raw) or 1.0
    scaled = [v / scale for v in raw]
    norm = math.sqrt(sum(v * v for v in scaled))
    return [v / norm for v in scaled]


def _mock_response(request: httpx.Request, *, status: int = 200, body: dict | None = None):
    if body is not None:
        return httpx.Response(status, json=body)
    payload = json.loads(request.content)
    texts = payload["input"]["texts"]
    text_type = payload["parameters"].get("text_type", "document")
    if text_type == "query":
        embeddings = [{"text_index": 0, "embedding": _unit_vector(1.0, 0.25)}]
        req_id = "mock-req-q"
        tokens = 5
    else:
        embeddings = []
        for index, text in enumerate(texts):
            if "Alice" in text or "爱丽丝" in text:
                vec_i = _unit_vector(1.0, 0.25)
            elif "stock" in text.lower():
                vec_i = _unit_vector(0.0, 1.0)
            else:
                vec_i = _unit_vector(0.3, 0.3)
            embeddings.append({"text_index": index, "embedding": vec_i})
        req_id = "mock-req"
        tokens = 11
    return httpx.Response(
        status,
        json={
            "request_id": req_id,
            "model": smoke.PROBE_MODEL,
            "output": {"embeddings": embeddings},
            "usage": {"total_tokens": tokens},
        },
    )


def _install_mock_transport(monkeypatch, handler):
    def factory(max_requests: int, records: list):
        inner = httpx.MockTransport(handler)

        class Wrapped(smoke.CountingTransport):
            def __init__(self):
                super().__init__(max_requests, records, inner=inner)

        return Wrapped()

    monkeypatch.setattr(smoke, "probe_transport_factory", factory)


@pytest.fixture
def probe_out(tmp_path):
    return tmp_path / "probe-out"


def test_probe_mock_success(monkeypatch, probe_out):
    monkeypatch.setenv(smoke.API_KEY_ENV, FAKE_DASH_KEY)

    def handler(request):
        return _mock_response(request)

    _install_mock_transport(monkeypatch, handler)
    code = smoke.run_probe(probe_out, max_requests=2, dry_run=False)
    assert code == 0
    raw = (probe_out / "probe.json").read_text(encoding="utf-8")
    assert FAKE_DASH_KEY not in raw
    report = json.loads(raw)
    assert report["requests_sent"] == 2
    assert report["sanity_pass"] is True
    assert "sanity_pass" in report
    assert len(report["document_vectors"]) == 3
    for item in report["document_vectors"]:
        assert "sha256" in item
        assert "embedding" not in item
    assert report["query_vector"] is not None
    assert "sha256" in report["query_vector"]


def test_probe_missing_key_exit_2(monkeypatch, probe_out):
    monkeypatch.delenv(smoke.API_KEY_ENV, raising=False)

    def handler(request):
        raise AssertionError("no HTTP expected")

    _install_mock_transport(monkeypatch, handler)
    code = smoke.run_probe(probe_out, max_requests=2, dry_run=False)
    assert code == 2
    report = json.loads((probe_out / "probe.json").read_text(encoding="utf-8"))
    assert report["requests_sent"] == 0
    assert report["code"] == "missing_embedding_api_key"


def test_probe_http_401_no_body_leak(monkeypatch, probe_out):
    monkeypatch.setenv(smoke.API_KEY_ENV, FAKE_DASH_KEY)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(
            401,
            json={"code": "InvalidApiKey", "message": "SECRETBODY leak"},
            headers={"content-type": "application/json"},
        )

    _install_mock_transport(monkeypatch, handler)
    code = smoke.run_probe(probe_out, max_requests=2, dry_run=False)
    assert code == 1
    raw = (probe_out / "probe.json").read_text(encoding="utf-8")
    assert "SECRETBODY" not in raw
    assert FAKE_DASH_KEY not in raw
    report = json.loads(raw)
    assert report["failure"]["http_status"] == 401
    assert report["failure"]["code"] == "embedding_http_401"


def test_max_requests_blocks_third_call():
    import httpx as hx

    records: list[dict] = []
    inner = hx.MockTransport(lambda req: hx.Response(200, json={"ok": True}))
    transport = smoke.CountingTransport(2, records, inner=inner)
    req = hx.Request("POST", smoke.ENDPOINT, content=b"{}")
    transport.handle_request(req)
    transport.handle_request(req)
    with pytest.raises(smoke.MaxRequestsExceeded):
        transport.handle_request(req)


def test_probe_dry_run_subprocess(probe_out):
    result = subprocess.run(
        [str(PY), str(SCRIPT), "probe", "--out", str(probe_out), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout.strip())
    assert payload["expected_http_requests"] == 2
    assert payload["api_key_env"] == smoke.API_KEY_ENV


def test_build_service_env_public_endpoint(tmp_path):
    work = tmp_path / "v4live-static"
    env = smoke.build_service_env(work, 18001, "svc-only-key", dashscope_key=FAKE_DASH_KEY)
    assert env["AML_COORDINATOR_LOCK_TIMEOUT_SECONDS"] == "30"
    assert env["AML_MAX_CONCURRENT_REQUESTS"] == "32"
    assert env["AML_EMBEDDING_ENDPOINT"] == smoke.ENDPOINT
    assert env[smoke.API_KEY_ENV] == FAKE_DASH_KEY
    assert env["AML_EMBEDDING_KEY_ENV"] == smoke.API_KEY_ENV
    sidecar = work / "env.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    public = {k: v for k, v in env.items() if k not in (smoke.API_KEY_ENV, "AML_SERVICE_API_KEY")}
    sidecar.write_text(json.dumps(public), encoding="utf-8")
    text = sidecar.read_text(encoding="utf-8")
    assert FAKE_DASH_KEY not in text


def test_serve_smoke_budget_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv(smoke.API_KEY_ENV, FAKE_DASH_KEY)
    out = tmp_path / "serve-out"
    estimated = smoke.estimate_embedding_requests()
    code = smoke.run_serve_smoke(PY, out, max_embedding_requests=estimated - 1, dry_run=False)
    assert code == 1
    report = json.loads((out / "serve-smoke.json").read_text(encoding="utf-8"))
    assert report["estimated_embedding_requests"] == estimated
    failed = [s for s in report["steps"] if s["name"] == "preflight embedding budget"]
    assert failed and failed[0]["ok"] is False


def test_serve_smoke_dry_run_subprocess(tmp_path):
    out = tmp_path / "dry"
    result = subprocess.run(
        [
            str(PY),
            str(SCRIPT),
            "serve-smoke",
            "--out",
            str(out),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout.strip())
    assert payload["estimated_embedding_requests_upper_bound"] == smoke.estimate_embedding_requests()
    assert smoke.API_KEY_ENV in payload["env_vars"]

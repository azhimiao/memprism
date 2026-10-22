"""Controlled text-embedding-v4 probe and process HTTP smoke (live endpoints).

Subcommands may perform real supplier HTTP when credentials are present.
Use --dry-run for zero-request planning output only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
)
API_KEY_ENV = "DASHSCOPE_API_KEY"
PROBE_MODEL = "text-embedding-v4"
PROBE_DIMENSIONS = 1024

DOCUMENT_TEXTS = (
    "Alice moved to Paris in 2021.",
    "爱丽丝2021年搬到了巴黎。",
    "The stock market fell sharply today.",
)
QUERY_TEXT = "Where did Alice move?"

SMOKE_USER = "v4live-user"
SMOKE_SESSION = "v4live-session"
CAT_FACT = "I adopted a cat named Miso in March."
SMOKE_MESSAGES = [
    {"role": "user", "content": "Hi, I want to log some life updates."},
    {"role": "assistant", "content": "Sure, tell me what happened."},
    {"role": "user", "content": CAT_FACT},
    {"role": "assistant", "content": "Miso sounds lovely."},
    {"role": "user", "content": "We play with a feather toy every evening."},
    {"role": "assistant", "content": "That sounds like a nice routine."},
]
SEARCH_QUERY = "What is my cat's name?"
MISO_NEEDLE = "Miso"
ERASED_MARKER = "v4live-erase-me"

# Tests monkeypatch this to inject httpx.MockTransport.
probe_transport_factory: Callable[[int, list[dict]], object] | None = None


class MaxRequestsExceeded(RuntimeError):
    pass


class CountingTransport(httpx.BaseTransport):
    """Wraps httpx.HTTPTransport and enforces a hard HTTP dispatch cap."""

    def __init__(self, max_requests: int, records: list[dict], inner=None):
        self._inner = inner if inner is not None else httpx.HTTPTransport()
        self._max = max_requests
        self._records = records

    def handle_request(self, request):
        if len(self._records) >= self._max:
            raise MaxRequestsExceeded(f"max_requests_exceeded:{self._max}")
        started = time.perf_counter()
        response = self._inner.handle_request(request)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        self._records.append(
            {"status_code": response.status_code, "latency_ms": latency_ms}
        )
        return response

    def close(self) -> None:
        self._inner.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def make_probe_transport(max_requests: int, records: list[dict]):
    if probe_transport_factory is not None:
        return probe_transport_factory(max_requests, records)
    return CountingTransport(max_requests, records)


def _deny_secret_env(name: str) -> bool:
    upper = name.upper()
    if upper.startswith("AML_"):
        return True
    return any(
        part in upper
        for part in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PASSWD")
    )


def child_env(extra: dict) -> dict:
    env = {key: value for key, value in os.environ.items() if not _deny_secret_env(key)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    env.update(extra)
    return env


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def redact(text: str, secrets_list: list[str]) -> str:
    for secret in secrets_list:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def redact_obj(value, secrets_list: list[str]):
    if isinstance(value, str):
        return redact(value, secrets_list)
    if isinstance(value, list):
        return [redact_obj(item, secrets_list) for item in value]
    if isinstance(value, dict):
        return {key: redact_obj(item, secrets_list) for key, item in value.items()}
    return value


def vector_sha256(vector: list[float]) -> str:
    payload = json.dumps(vector, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def vector_l2(vector: list[float]) -> float:
    return math.sqrt(math.fsum(v * v for v in vector))


def cosine(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def probe_config():
    from app.embeddings import EmbeddingConfig

    return EmbeddingConfig(
        provider="dashscope",
        endpoint=ENDPOINT,
        dimensions=PROBE_DIMENSIONS,
        model=PROBE_MODEL,
        revision="1",
        api_key_env=API_KEY_ENV,
        batch_size=10,
        timeout_seconds=30.0,
        max_retries=0,
        region="cn-beijing",
    )


def estimate_embedding_requests(*, include_delete_flow: bool = True) -> int:
    batch_size = 10
    message_count = len(SMOKE_MESSAGES)
    document_batches = math.ceil(message_count / batch_size)
    searches = 2  # after add, after restart
    extra_docs = 0
    if include_delete_flow:
        searches += 2
        extra_docs = 1  # delete control ingest message
    return document_batches + searches + extra_docs


def build_service_env(
    work: Path,
    port: int,
    service_key: str,
    *,
    dashscope_key: str | None = None,
    tiktoken_cache: Path | None = None,
) -> dict:
    # deploy.runtime refuses to download the tokenizer; pass a pre-validated cache.
    cache = tiktoken_cache or Path(os.environ.get("TIKTOKEN_CACHE_DIR")
                                   or os.environ.get("DATA_GYM_CACHE_DIR")
                                   or Path(tempfile.gettempdir()) / "data-gym-cache")
    db = work / "smoke.sqlite"
    env = child_env(
        {
            "AML_PROFILE": "cycle2_dense",
            "AML_RETRIEVAL_MODE": "raw_hybrid_rrf",
            "AML_ENABLE_VECTOR": "true",
            "AML_DB_PATH": str(db),
            "AML_AUTH_MODE": "bearer",
            "AML_SERVICE_KEY_ENV": "AML_SERVICE_API_KEY",
            "AML_SERVICE_API_KEY": service_key,
            "AML_EMBEDDING_PROVIDER": "dashscope",
            "AML_EMBEDDING_ENDPOINT": ENDPOINT,
            "AML_EMBEDDING_REGION": "cn-beijing",
            "AML_EMBEDDING_MODEL": PROBE_MODEL,
            "AML_EMBEDDING_DIMENSIONS": str(PROBE_DIMENSIONS),
            "AML_EMBEDDING_REVISION": "1",
            "AML_EMBEDDING_KEY_ENV": API_KEY_ENV,
            "AML_EMBEDDING_BATCH_SIZE": "10",
            "AML_EMBEDDING_TIMEOUT_SECONDS": "30",
            "AML_EMBEDDING_MAX_RETRIES": "0",
            "AML_EMBEDDING_MAX_INPUT_BYTES": "8192",
            "AML_EMBEDDING_MIN_INTERVAL_SECONDS": "0",
            "AML_COORDINATOR_LOCK_TIMEOUT_SECONDS": "30",
            "AML_MAX_CONCURRENT_REQUESTS": "32",
            "AML_MAX_REQUEST_BYTES": "10485760",
            "AML_CANDIDATE_LIMIT": "200",
            "AML_LEXICAL_QUOTA": "100",
            "AML_DENSE_QUOTA": "100",
            "AML_MAX_RETURN_ITEMS": "100",
            "AML_RETURN_CHAR_BUDGET": "32000",
            "AML_RETURN_TOKEN_BUDGET": "117760",
            "AML_PACKING_POLICY": "prefix",
            "AML_ROUTE_POLICY": "none",
            "AML_ENABLE_GOVERNANCE": "false",
            "AML_EXACT_DENSE_ACCELERATION": "false",
            "AML_HOST": "127.0.0.1",
            "AML_PORT": str(port),
            "TIKTOKEN_CACHE_DIR": str(cache),
        }
    )
    key = dashscope_key if dashscope_key is not None else os.environ.get(API_KEY_ENV, "")
    if key:
        env[API_KEY_ENV] = key
    return env


def wait_http(process: subprocess.Popen, port: int, timeout: float = 60.0):
    import httpx

    deadline = time.monotonic() + timeout
    last = None
    with httpx.Client(trust_env=False, timeout=2.0) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return last, "process_exited"
            try:
                response = client.get(f"http://127.0.0.1:{port}/health")
                last = response
                if response.status_code == 200:
                    return response, None
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
    return last, "timeout"


def stop_process(process: subprocess.Popen) -> int | None:
    if process.poll() is None:
        process.terminate()
        try:
            return process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=10)
    return process.returncode


def spawn_deploy_start(py: Path, env: dict, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.Popen(
            [str(py), "-B", "-m", "deploy.start"],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )


def healthy(payload) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("status") == "ok"
        and payload.get("control_ready") is True
        and payload.get("data_route_ready") is True
    )


def search_contents(response) -> list[str]:
    try:
        body = response.json()
    except ValueError:
        return []
    if not isinstance(body, dict):
        return []
    data = body.get("data", [])
    if not isinstance(data, list):
        return []
    return [str(item.get("content", "")) for item in data if isinstance(item, dict)]


def miso_rank(contents: list[str]) -> int | None:
    for index, item in enumerate(contents[:3], start=1):
        if MISO_NEEDLE in item:
            return index
    return None


def http_status_from_failure(code: str) -> int | None:
    prefix = "embedding_http_"
    if code.startswith(prefix):
        try:
            return int(code[len(prefix) :])
        except ValueError:
            return None
    return None


def run_probe(out_dir: Path, max_requests: int, dry_run: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "probe.json"
    api_key = os.environ.get(API_KEY_ENV, "")
    secrets_list = [api_key] if api_key else []

    if dry_run:
        print(
            json.dumps(
                {
                    "cmd": "probe",
                    "dry_run": True,
                    "max_http_requests": max_requests,
                    "expected_http_requests": 2,
                    "api_key_env": API_KEY_ENV,
                    "endpoint": ENDPOINT,
                    "model": PROBE_MODEL,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if not api_key.strip():
        report = {
            "status": "failed",
            "code": "missing_embedding_api_key",
            "requests_sent": 0,
            "http_calls": [],
            "api_key_env": API_KEY_ENV,
        }
        report_path.write_text(
            json.dumps(redact_obj(report, secrets_list), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 2

    from app.embeddings import (
        EmbeddingFailure,
        create_embedding_provider,
        provider_identity,
        provider_version,
    )

    http_records: list[dict] = []
    transport = make_probe_transport(max_requests, http_records)
    config = probe_config()
    provider = create_embedding_provider(config, transport=transport)
    report: dict = {
        "status": "ok",
        "requests_sent": 0,
        "http_calls": [],
        "usage": [],
        "document_vectors": [],
        "query_vector": None,
        "cosine_similarities": {},
        "sanity_pass": None,
        "provider_identity": provider_identity(provider),
        "provider_version": provider_version(provider),
        "response_models": [],
        "failure": None,
    }

    try:
        doc_batch = provider.embed(DOCUMENT_TEXTS)
        report["requests_sent"] = len(http_records)
        report["http_calls"] = list(http_records)
        report["usage"].append(doc_batch.usage)
        models = doc_batch.usage.get("response_models")
        if isinstance(models, list):
            report["response_models"].extend(models)
        for vector in doc_batch.vectors:
            report["document_vectors"].append(
                {
                    "sha256": vector_sha256(vector),
                    "l2_norm": round(vector_l2(vector), 6),
                    "dimensions": len(vector),
                }
            )

        query_provider = provider.for_query()
        query_batch = query_provider.embed([QUERY_TEXT])
        report["requests_sent"] = len(http_records)
        report["http_calls"] = list(http_records)
        report["usage"].append(query_batch.usage)
        q_models = query_batch.usage.get("response_models")
        if isinstance(q_models, list):
            report["response_models"].extend(q_models)
        query_vector = query_batch.vectors[0]
        report["query_vector"] = {
            "sha256": vector_sha256(query_vector),
            "l2_norm": round(vector_l2(query_vector), 6),
            "dimensions": len(query_vector),
        }
        cosines = {
            f"doc{index}": round(cosine(query_vector, doc), 6)
            for index, doc in enumerate(doc_batch.vectors)
        }
        report["cosine_similarities"] = cosines
        report["sanity_pass"] = cosines.get("doc0", 0) > cosines.get(
            "doc2", 0
        ) and cosines.get("doc1", 0) > cosines.get("doc2", 0)
    except EmbeddingFailure as exc:
        report["status"] = "failed"
        report["requests_sent"] = len(http_records)
        report["http_calls"] = list(http_records)
        failure = {"code": exc.code, "exception_type": type(exc).__name__}
        status = http_status_from_failure(exc.code)
        if status is not None:
            failure["http_status"] = status
        request_ids = exc.usage.get("request_ids") if exc.usage else None
        if isinstance(request_ids, list) and request_ids:
            failure["request_id"] = request_ids[-1]
        report["failure"] = failure
    except MaxRequestsExceeded as exc:
        report["status"] = "failed"
        report["requests_sent"] = len(http_records)
        report["http_calls"] = list(http_records)
        report["failure"] = {
            "code": "max_requests_exceeded",
            "exception_type": type(exc).__name__,
        }
        report_path.write_text(
            json.dumps(redact_obj(report, secrets_list), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 1
    except Exception as exc:
        report["status"] = "failed"
        report["requests_sent"] = len(http_records)
        report["http_calls"] = list(http_records)
        report["failure"] = {
            "code": "probe_aborted",
            "exception_type": type(exc).__name__,
        }

    report_path.write_text(
        json.dumps(redact_obj(report, secrets_list), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0 if report["status"] == "ok" else 1


def run_serve_smoke(py: Path, out_dir: Path, max_embedding_requests: int, dry_run: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "serve-smoke.json"
    estimated = estimate_embedding_requests(include_delete_flow=True)
    dash_key = os.environ.get(API_KEY_ENV, "")
    service_key = secrets.token_urlsafe(24)
    secrets_list = [s for s in (dash_key, service_key) if s]

    if dry_run:
        print(
            json.dumps(
                {
                    "cmd": "serve-smoke",
                    "dry_run": True,
                    "estimated_embedding_requests_upper_bound": estimated,
                    "max_embedding_requests": max_embedding_requests,
                    "env_vars": [
                        API_KEY_ENV,
                        "AML_SERVICE_API_KEY",
                        "AML_EMBEDDING_KEY_ENV",
                        "AML_EMBEDDING_ENDPOINT",
                    ],
                    "endpoint": ENDPOINT,
                },
                ensure_ascii=False,
            )
        )
        return 0

    work = Path(tempfile.gettempdir()) / f"aml-v4live-{uuid.uuid4().hex}"
    work.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "scope": "v4_live_serve_smoke",
        "work_dir": str(work),
        "estimated_embedding_requests": estimated,
        "max_embedding_requests": max_embedding_requests,
        "steps": [],
        "usage_summary": [],
        "ok": False,
    }

    def note(name: str, ok: bool, **fields):
        report["steps"].append({"name": name, "ok": ok, **fields})

    if estimated > max_embedding_requests:
        note(
            "preflight embedding budget",
            False,
            reason="estimated_exceeds_max",
            estimated=estimated,
            max_allowed=max_embedding_requests,
        )
        report_path.write_text(
            json.dumps(redact_obj(report, secrets_list), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 1

    if not dash_key.strip():
        note("preflight dashscope key", False, code="missing_embedding_api_key")
        report_path.write_text(
            json.dumps(redact_obj(report, secrets_list), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 2

    port = free_port()
    env = build_service_env(work, port, service_key, dashscope_key=dash_key)
    config_run = subprocess.run(
        [str(py), "-B", "-m", "deploy.runtime", "--configuration"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    config_out = redact(
        (config_run.stdout or "") + (config_run.stderr or ""), secrets_list
    )
    note(
        "deploy.runtime --configuration",
        config_run.returncode == 0,
        exit_code=config_run.returncode,
        output_tail=config_out[-1200:],
    )
    if config_run.returncode != 0:
        report_path.write_text(
            json.dumps(redact_obj(report, secrets_list), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 1

    import httpx

    processes: list[subprocess.Popen] = []
    auth = {"Authorization": f"Bearer {service_key}"}

    def start_server(label: str):
        p = free_port()
        run_env = build_service_env(work, p, service_key, dashscope_key=dash_key)
        log_path = work / f"{label}-{p}.log"
        process = spawn_deploy_start(py, run_env, log_path)
        processes.append(process)
        response, error = wait_http(process, p)
        return {
            "process": process,
            "port": p,
            "env": run_env,
            "log": log_path,
            "error": error,
            "health": response,
        }

    try:
        server = start_server("deploy-start")
        if server["error"]:
            tail = ""
            if server["log"].exists():
                tail = redact(
                    server["log"].read_text(encoding="utf-8", errors="replace")[-800:],
                    secrets_list,
                )
            note(
                "deploy.start startup",
                False,
                error=server["error"],
                log_tail=tail,
            )
            report_path.write_text(
                json.dumps(redact_obj(report, secrets_list), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return 1

        base = f"http://127.0.0.1:{server['port']}"
        with httpx.Client(trust_env=False, timeout=60.0, base_url=base) as client:
            t0 = time.perf_counter()
            health = client.get("/health")
            body = health.json() if health.status_code == 200 else {}
            note(
                "unauthenticated health",
                health.status_code == 200 and healthy(body),
                status=health.status_code,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

            t0 = time.perf_counter()
            denied = client.post(
                "/search",
                json={"query": SEARCH_QUERY, "user_id": SMOKE_USER, "top_k": 5},
            )
            note(
                "unauthenticated search rejected",
                denied.status_code == 401,
                status=denied.status_code,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

            t0 = time.perf_counter()
            added = client.post(
                "/add",
                headers=auth,
                json={
                    "request_id": "v4live-add",
                    "user_id": SMOKE_USER,
                    "session_id": SMOKE_SESSION,
                    "messages": SMOKE_MESSAGES,
                },
            )
            note(
                "authorized add",
                added.status_code == 200,
                status=added.status_code,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

            t0 = time.perf_counter()
            found = client.post(
                "/search",
                headers=auth,
                json={"query": SEARCH_QUERY, "user_id": SMOKE_USER, "top_k": 5},
            )
            contents = search_contents(found)
            rank = miso_rank(contents)
            note(
                "authorized search hits Miso top3",
                found.status_code == 200 and rank is not None,
                status=found.status_code,
                miso_rank=rank,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
            _collect_usage(found, report)

        stop_process(server["process"])

        restarted = start_server("deploy-restart")
        if restarted["error"]:
            note("restart startup", False, error=restarted["error"])
        else:
            base = f"http://127.0.0.1:{restarted['port']}"
            with httpx.Client(trust_env=False, timeout=60.0, base_url=base) as client:
                t0 = time.perf_counter()
                kept = client.post(
                    "/search",
                    headers=auth,
                    json={"query": SEARCH_QUERY, "user_id": SMOKE_USER, "top_k": 5},
                )
                contents = search_contents(kept)
                rank = miso_rank(contents)
                note(
                    "search after restart",
                    kept.status_code == 200 and rank is not None,
                    status=kept.status_code,
                    miso_rank=rank,
                    latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
                _collect_usage(kept, report)

            stop_process(restarted["process"])

            delete_supported = True
            third = start_server("deploy-delete")
            if third["error"]:
                note("delete phase startup", False, error=third["error"])
                delete_supported = False
            else:
                base = f"http://127.0.0.1:{third['port']}"
                with httpx.Client(
                    trust_env=False, timeout=60.0, base_url=base
                ) as client:
                    client.post(
                        "/add",
                        headers=auth,
                        json={
                            "request_id": "v4live-drop",
                            "user_id": SMOKE_USER,
                            "session_id": SMOKE_SESSION,
                            "messages": [
                                {"role": "user", "content": ERASED_MARKER}
                            ],
                        },
                    )
                    erase = client.post(
                        "/add",
                        headers=auth,
                        json={
                            "request_id": "v4live-erase",
                            "user_id": SMOKE_USER,
                            "session_id": SMOKE_SESSION,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f'delete "{ERASED_MARKER}"',
                                }
                            ],
                        },
                    )
                    if erase.status_code != 200 or erase.json().get("success") is not True:
                        delete_supported = False
                        note(
                            "delete control",
                            False,
                            status=erase.status_code,
                            supported=False,
                        )
                    else:
                        after = client.post(
                            "/search",
                            headers=auth,
                            json={
                                "query": ERASED_MARKER,
                                "user_id": SMOKE_USER,
                                "top_k": 5,
                            },
                        )
                        gone = ERASED_MARKER not in search_contents(after)
                        note(
                            "delete control hides row",
                            gone,
                            status=after.status_code,
                            supported=True,
                        )
                        _collect_usage(after, report)
                stop_process(third["process"])

                if delete_supported:
                    fourth = start_server("deploy-after-delete")
                    if fourth["error"]:
                        note("post-delete restart", False, error=fourth["error"])
                    else:
                        base = f"http://127.0.0.1:{fourth['port']}"
                        with httpx.Client(
                            trust_env=False, timeout=60.0, base_url=base
                        ) as client:
                            still = client.post(
                                "/search",
                                headers=auth,
                                json={
                                    "query": ERASED_MARKER,
                                    "user_id": SMOKE_USER,
                                    "top_k": 5,
                                },
                            )
                            note(
                                "deleted row stays gone",
                                still.status_code == 200
                                and ERASED_MARKER not in search_contents(still),
                                status=still.status_code,
                                supported=True,
                            )
                        stop_process(fourth["process"])
                else:
                    note("delete control", False, supported="not_supported")

    except Exception as exc:
        note("serve-smoke aborted", False, error=type(exc).__name__)
        report["abort_error"] = redact(type(exc).__name__, secrets_list)
    finally:
        for process in processes:
            if process.poll() is None:
                stop_process(process)

    failures = [step["name"] for step in report["steps"] if not step.get("ok")]
    report["failed_steps"] = failures
    report["ok"] = not failures
    report_path.write_text(
        json.dumps(redact_obj(report, secrets_list), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logs = []
    for path in sorted(work.glob("*.log")):
        logs.append(
            "===== "
            + path.name
            + " =====\n"
            + redact(path.read_text(encoding="utf-8", errors="replace"), secrets_list)
        )
    (out_dir / "serve-smoke.log").write_text("\n".join(logs), encoding="utf-8")
    return 0 if report["ok"] else 1


def _collect_usage(response, report: dict) -> None:
    try:
        body = response.json()
    except ValueError:
        return
    if not isinstance(body, dict):
        return
    trace = body.get("trace") or body.get("usage")
    if isinstance(trace, dict):
        report["usage_summary"].append(trace)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    probe = sub.add_parser("probe")
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--max-requests", type=int, default=2)
    probe.add_argument("--dry-run", action="store_true")

    serve = sub.add_parser("serve-smoke")
    serve.add_argument("--out", type=Path, required=True)
    serve.add_argument("--max-embedding-requests", type=int, default=8)
    serve.add_argument("--python", type=Path, default=Path(sys.executable))
    serve.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "probe":
        return run_probe(args.out, args.max_requests, args.dry_run)
    return run_serve_smoke(args.python, args.out, args.max_embedding_requests, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

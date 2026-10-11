"""Process-level T02 HTTP smoke. Engineering surrogate only; no supplier calls.

`guarded-start` is deploy.start behind a loopback socket guard.
`serve-surrogate` is the same runtime factory with httpx.MockTransport.
Neither entry loads credentials files or contacts a model endpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost"}
SURVIVOR = "t02survivorbeta"
ERASED = "t02erasedgamma"


def install_guard() -> None:
    if getattr(socket, "_t02_guard", False):
        return
    log_path = os.environ.get("AML_T02_CONNECT_LOG")

    def reject(host: str) -> None:
        if log_path:
            with open(log_path, "a", encoding="ascii") as handle:
                handle.write(host + "\n")
        raise OSError("external_connect_blocked")

    def host_of(address) -> str:
        if isinstance(address, tuple):
            return str(address[0])
        return str(address)

    original_connect = socket.socket.connect

    def connect(self, address):
        host = host_of(address)
        if host not in ALLOWED_HOSTS:
            reject(host)
        return original_connect(self, address)

    original_create = socket.create_connection

    def create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, *args, **kwargs):
        host = host_of(address)
        if host not in ALLOWED_HOSTS:
            reject(host)
        return original_create(address, timeout, source_address, *args, **kwargs)

    socket.socket.connect = connect
    socket.create_connection = create_connection
    socket._t02_guard = True


def mock_embed(request):
    import httpx
    body = json.loads(request.content)
    path = os.environ.get("AML_T02_MOCK_COUNTER")
    if path:
        with open(path, "a", encoding="ascii") as handle:
            handle.write("1\n")
    dimension = int(body["parameters"]["dimension"])
    embeddings = [{"text_index": index, "embedding": [index + 1.0, 1.0] + [0.0] * (dimension - 2)}
                  for index in reversed(range(len(body["input"]["texts"])))]
    return httpx.Response(200, json={"request_id": "offline-mock-request",
                                      "output": {"embeddings": embeddings},
                                      "usage": {"total_tokens": 7}})


def surrogate_app():
    install_guard()
    import httpx
    from deploy.runtime import build_app
    return build_app(transport=httpx.MockTransport(mock_embed))


def guarded_start() -> int:
    install_guard()
    import runpy
    runpy.run_module("deploy.start", run_name="__main__")
    return 0


def serve_surrogate() -> int:
    install_guard()
    import uvicorn
    uvicorn.run(surrogate_app, factory=True, host="127.0.0.1",
                port=int(os.environ["AML_PORT"]), workers=1, access_log=False,
                proxy_headers=False, timeout_keep_alive=5, timeout_graceful_shutdown=45,
                log_level="warning")
    return 0


def _deny_secret_env(name: str) -> bool:
    upper = name.upper()
    if upper.startswith("AML_"):
        return True
    return any(part in upper for part in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PASSWD"))


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


def wait_http(process: subprocess.Popen, port: int, timeout: float = 40.0):
    import httpx
    deadline = time.monotonic() + timeout
    last = None
    with httpx.Client(trust_env=False, timeout=1.0) as client:
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
            return process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=10)
    return process.returncode


def service_env(db: Path, port: int, work: Path, cache: Path, service_key: str, embedding_key: str) -> dict:
    return child_env({
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
        "AML_EMBEDDING_MODEL": "text-embedding-v4",
        "AML_EMBEDDING_DIMENSIONS": "1024",
        "AML_EMBEDDING_REVISION": "1",
        "AML_EMBEDDING_KEY_ENV": "V4_TEST_KEY",
        "V4_TEST_KEY": embedding_key,
        "AML_EMBEDDING_BATCH_SIZE": "10",
        "AML_EMBEDDING_TIMEOUT_SECONDS": "30",
        "AML_EMBEDDING_MAX_RETRIES": "0",
        "AML_EMBEDDING_MAX_INPUT_BYTES": "8192",
        "AML_EMBEDDING_MIN_INTERVAL_SECONDS": "0",
        "AML_HOST": "127.0.0.1",
        "AML_PORT": str(port),
        "AML_T02_CONNECT_LOG": str(work / "connect.log"),
        "AML_T02_MOCK_COUNTER": str(work / "mock-counter.txt"),
        "TIKTOKEN_CACHE_DIR": str(cache),
    })


def spawn(py: Path, args: list[str], env: dict, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.Popen([str(py), "-B", str(Path(__file__).resolve()), *args], cwd=ROOT, env=env,
                                stdout=handle, stderr=subprocess.STDOUT, creationflags=flags)


def healthy(payload) -> bool:
    return (isinstance(payload, dict) and payload.get("status") == "ok"
            and payload.get("control_ready") is True and payload.get("data_route_ready") is True)


def contents(response) -> list[str]:
    try:
        body = response.json()
    except ValueError:
        return []
    return [item.get("content", "") for item in body.get("data", [])] if isinstance(body, dict) else []


def contains(response, needle: str) -> bool:
    return any(needle in item for item in contents(response))


def redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def run_smoke(py: Path, work: Path, evidence: Path, cache: Path) -> int:
    import httpx
    import secrets
    work.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    service_key = secrets.token_hex(16)
    embedding_key = secrets.token_hex(16)
    report: dict = {"scope": "t02_process_http_smoke", "supplier_network_connects": 0,
                    "formal_v4_acceptance": False, "steps": []}
    failures = []
    processes = []

    def note(name: str, ok: bool, **fields):
        report["steps"].append({"name": name, "ok": ok, **fields})
        if not ok:
            failures.append(name)

    def start(kind: str, db: Path):
        port = free_port()
        env = service_env(db, port, work, cache, service_key, embedding_key)
        log_path = work / f"{kind}-{port}.log"
        process = spawn(py, ["guarded-start" if kind == "deploy-start" else "serve-surrogate"], env, log_path)
        processes.append(process)
        response, error = wait_http(process, port)
        if error:
            tail = ""
            if log_path.exists():
                tail = redact(log_path.read_text(encoding="utf-8", errors="replace")[-800:],
                              [service_key, embedding_key])
            note(f"{kind} startup", False, port=port, error=error, log_tail=tail, pid=process.pid)
            return None
        return {"process": process, "port": port, "env": env, "log": log_path, "kind": kind}

    def client_for(item):
        return httpx.Client(trust_env=False, timeout=30.0,
                            base_url=f"http://127.0.0.1:{item['port']}")

    try:
        plain = start("deploy-start", work / "plain.sqlite")
        if plain:
            with client_for(plain) as client:
                health = client.get("/health")
                body = health.json()
                note("deploy.start unauthenticated health",
                     health.status_code == 200 and healthy(body),
                     status=health.status_code, health_status=body.get("status"),
                     control_ready=body.get("control_ready"),
                     data_route_ready=body.get("data_route_ready"))
                add = client.post("/add", json={"request_id": "noauth", "user_id": "u", "session_id": "s",
                                                 "messages": [{"role": "user", "content": SURVIVOR}]})
                search = client.post("/search", json={"query": SURVIVOR, "user_id": "u", "top_k": 5})
                note("deploy.start unauthenticated add/search rejected",
                     add.status_code == 401 and search.status_code == 401,
                     add_status=add.status_code, search_status=search.status_code)
            checked = subprocess.run([str(py), "-B", "-m", "deploy.healthcheck"], cwd=ROOT,
                                     env=plain["env"], capture_output=True, text=True, timeout=20)
            note("deploy.start healthcheck CLI", checked.returncode == 0, exit_code=checked.returncode)
            note("deploy.start stopped", True, exit_code=stop_process(plain["process"]))

        surrogate_db = work / "surrogate.sqlite"
        first = start("surrogate", surrogate_db)
        if first:
            auth = {"Authorization": "Bearer " + service_key}
            with client_for(first) as client:
                health = client.get("/health")
                body = health.json()
                note("surrogate health", health.status_code == 200 and healthy(body),
                     status=health.status_code, health_status=body.get("status"),
                     control_ready=body.get("control_ready"), data_route_ready=body.get("data_route_ready"))
                denied_add = client.post("/add", json={"request_id": "a1", "user_id": "u", "session_id": "s",
                                                        "messages": [{"role": "user", "content": SURVIVOR}]})
                denied_search = client.post("/search", json={"query": SURVIVOR, "user_id": "u", "top_k": 5})
                note("surrogate unauthenticated add/search rejected",
                     denied_add.status_code == 401 and denied_search.status_code == 401,
                     add_status=denied_add.status_code, search_status=denied_search.status_code)
                added_keep = client.post("/add", headers=auth, json={
                    "request_id": "keep", "user_id": "u", "session_id": "s",
                    "messages": [{"role": "user", "content": SURVIVOR}]})
                added_drop = client.post("/add", headers=auth, json={
                    "request_id": "drop", "user_id": "u", "session_id": "s",
                    "messages": [{"role": "user", "content": ERASED}]})
                found = client.post("/search", headers=auth, json={"query": SURVIVOR, "user_id": "u", "top_k": 5})
                found_drop = client.post("/search", headers=auth, json={"query": ERASED, "user_id": "u", "top_k": 5})
                keep_ok = added_keep.status_code == 200 and added_keep.json().get("success") is True
                drop_ok = added_drop.status_code == 200 and added_drop.json().get("success") is True
                note("surrogate authorized add then search",
                     keep_ok and drop_ok and found.status_code == 200 and SURVIVOR in contents(found)
                     and found_drop.status_code == 200 and ERASED in contents(found_drop),
                     keep_status=added_keep.status_code, drop_status=added_drop.status_code,
                     survivor_found=SURVIVOR in contents(found), erased_found=ERASED in contents(found_drop))
            checked = subprocess.run([str(py), "-B", "-m", "deploy.healthcheck"], cwd=ROOT,
                                     env=first["env"], capture_output=True, text=True, timeout=20)
            note("surrogate healthcheck before restart", checked.returncode == 0, exit_code=checked.returncode)
            first_pid = first["process"].pid
            note("surrogate first stop", True, exit_code=stop_process(first["process"]), pid=first_pid)

            second = start("surrogate-restart", surrogate_db)
            if second:
                with client_for(second) as client:
                    health = client.get("/health")
                    kept = client.post("/search", headers=auth, json={"query": SURVIVOR, "user_id": "u", "top_k": 5})
                    still = client.post("/search", headers=auth, json={"query": ERASED, "user_id": "u", "top_k": 5})
                    note("restart retains both rows",
                         health.status_code == 200 and healthy(health.json())
                         and SURVIVOR in contents(kept) and ERASED in contents(still),
                         health_status=health.status_code, survivor_found=SURVIVOR in contents(kept),
                         erased_found=ERASED in contents(still), pid=second["process"].pid,
                         pid_changed=second["process"].pid != first_pid)
                    erased = client.post("/add", headers=auth, json={
                        "request_id": "erase", "user_id": "u", "session_id": "s",
                        "messages": [{"role": "user", "content": f'delete "{ERASED}"'}]})
                    after_keep = client.post("/search", headers=auth, json={"query": SURVIVOR, "user_id": "u", "top_k": 5})
                    after_drop = client.post("/search", headers=auth, json={"query": ERASED, "user_id": "u", "top_k": 5})
                    note("delete control hides erased row",
                         erased.status_code == 200 and SURVIVOR in contents(after_keep)
                         and ERASED not in contents(after_drop),
                         erase_status=erased.status_code, survivor_found=SURVIVOR in contents(after_keep),
                         erased_found=ERASED in contents(after_drop))
                note("surrogate second stop", True, exit_code=stop_process(second["process"]))
                third = start("surrogate-after-delete", surrogate_db)
                if third:
                    with client_for(third) as client:
                        health = client.get("/health")
                        kept = client.post("/search", headers=auth, json={"query": SURVIVOR, "user_id": "u", "top_k": 5})
                        gone = client.post("/search", headers=auth, json={"query": ERASED, "user_id": "u", "top_k": 5})
                        note("deleted row stays gone after restart",
                             health.status_code == 200 and healthy(health.json())
                             and SURVIVOR in contents(kept) and ERASED not in contents(gone),
                             health_status=health.status_code, survivor_found=SURVIVOR in contents(kept),
                             erased_found=ERASED in contents(gone))
                    checked = subprocess.run([str(py), "-B", "-m", "deploy.healthcheck"], cwd=ROOT,
                                             env=third["env"], capture_output=True, text=True, timeout=20)
                    note("surrogate healthcheck after delete restart", checked.returncode == 0,
                         exit_code=checked.returncode)
                    note("surrogate final stop", True, exit_code=stop_process(third["process"]))
    except Exception as exc:
        note("smoke aborted", False, error=type(exc).__name__)
        report["abort_error"] = redact(str(exc), [service_key, embedding_key])[:500]
    finally:
        for process in processes:
            if process.poll() is None:
                stop_process(process)

    connect_log = work / "connect.log"
    connects = connect_log.read_text(encoding="ascii").splitlines() if connect_log.exists() else []
    counter = work / "mock-counter.txt"
    dispatches = len(counter.read_text(encoding="ascii").splitlines()) if counter.exists() else 0
    report["supplier_network_connects"] = len(connects)
    report["blocked_connect_hosts"] = sorted(set(connects))
    report["surrogate_embedding_dispatches"] = dispatches
    report["external_supplier_calls"] = len(connects)
    report["ok"] = not failures and len(connects) == 0
    report["failed_steps"] = failures
    text = json.dumps(report, ensure_ascii=False, indent=2)
    (evidence / "03-http-smoke.json").write_text(text, encoding="utf-8")
    logs = []
    for path in sorted(work.glob("*.log")):
        logs.append("===== " + path.name + " =====\n" + redact(path.read_text(encoding="utf-8", errors="replace"),
                                                                [service_key, embedding_key]))
    (evidence / "03-http-smoke.log").write_text("\n".join(logs), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "failed_steps": failures,
                      "supplier_network_connects": len(connects),
                      "surrogate_embedding_dispatches": dispatches}))
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("guarded-start")
    sub.add_parser("serve-surrogate")
    run = sub.add_parser("run")
    run.add_argument("--python", type=Path, required=True)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--evidence", type=Path, required=True)
    run.add_argument("--tiktoken-cache", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "guarded-start":
        return guarded_start()
    if args.cmd == "serve-surrogate":
        return serve_surrogate()
    return run_smoke(args.python, args.work, args.evidence, args.tiktoken_cache)


if __name__ == "__main__":
    raise SystemExit(main())

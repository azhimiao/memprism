"""Old-identity copy drill: refuse, dry-run, surrogate rebuild, failed copy, rollback.

All databases stay outside the repository. The source copy is opened read-only.
Execute uses deploy.rebuild.main with httpx.MockTransport and a loopback socket
guard. This is not a formal text-embedding-v4 acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
OLD_PYTHON = Path(r"F:/competitions/1AML/our-system/.venv312/Scripts/python.exe")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_guard() -> None:
    import socket
    if getattr(socket, "_t02_guard", False):
        return
    allowed = {"127.0.0.1", "::1", "localhost"}
    log_path = os.environ.get("AML_T02_CONNECT_LOG")

    def reject(host: str) -> None:
        if log_path:
            with open(log_path, "a", encoding="ascii") as handle:
                handle.write(host + "\n")
        raise OSError("external_connect_blocked")

    original = socket.socket.connect

    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if str(host) not in allowed:
            reject(str(host))
        return original(self, address)

    socket.socket.connect = connect
    socket._t02_guard = True


def mock_handler(mode: str):
    def respond(request):
        import httpx
        if mode == "fail":
            return httpx.Response(503, json={"code": "ServiceUnavailable"})
        body = json.loads(request.content)
        dimension = int(body["parameters"]["dimension"])
        embeddings = [{"text_index": index, "embedding": [index + 1.0, 1.0] + [0.0] * (dimension - 2)}
                      for index in reversed(range(len(body["input"]["texts"])))]
        return httpx.Response(200, json={"request_id": "offline-mock-request",
                                          "output": {"embeddings": embeddings},
                                          "usage": {"total_tokens": 7}})
    return respond


def patch_client(mode: str) -> None:
    import httpx
    original = httpx.Client
    handler = mock_handler(mode)

    def wrapped(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original(*args, **kwargs)

    httpx.Client = wrapped


def backup_copy(origin: Path, dest: Path) -> None:
    """Consistent snapshot via the SQLite backup API. Does not write the origin."""
    if dest.exists():
        raise SystemExit("backup_destination_exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(origin.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        source.execute("BEGIN")
        target = sqlite3.connect(dest)
        try:
            source.backup(target)
            target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            target.close()
        source.rollback()
    finally:
        source.close()


def generate_old(path: Path) -> int:
    import tiktoken
    from fastapi.testclient import TestClient
    from app.api import create_app
    from app.config import Settings
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit("generate_target_exists")
    application = create_app(Settings(db_path=path, profile_id="span_sync"))
    with TestClient(application) as client:
        rows = (("erase-source", "deleted alpha"), ("erase-control", 'delete "deleted alpha"'),
                ("live", "surviving beta"))
        for request_id, content in rows:
            response = client.post("/add", json={"request_id": request_id, "user_id": "u", "session_id": "s",
                                                  "messages": [{"role": "user", "content": content}]})
            if response.status_code != 200:
                print(json.dumps({"status": "generate_add_failed", "request_id": request_id,
                                  "http_status": response.status_code}))
                return 1
        live = client.post("/search", json={"query": "surviving", "user_id": "u", "top_k": 5})
        gone = client.post("/search", json={"query": "deleted", "user_id": "u", "top_k": 5})
        survivor = [item["content"] for item in live.json().get("data", [])]
        erased = [item["content"] for item in gone.json().get("data", [])]
        if "surviving beta" not in survivor or any("deleted alpha" in item for item in erased):
            print(json.dumps({"status": "generate_search_mismatch", "survivor_found": "surviving beta" in survivor,
                              "erased_found": bool(erased)}))
            return 1
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if not sidecar.exists():
            continue
        if sidecar.stat().st_size:
            print(json.dumps({"status": "generate_sidecar_not_empty", "name": sidecar.name}))
            return 1
        sidecar.unlink()
    print(json.dumps({"status": "generated", "tiktoken": tiktoken.__version__,
                      "python": sys.version.split()[0], "sha256": sha256_file(path),
                      "bytes": path.stat().st_size, "external_calls": 0}))
    return 0


def rollback_open(path: Path) -> int:
    """Open the original profile on the preserved runtime. Counts only; no row text."""
    import tiktoken
    from fastapi.testclient import TestClient
    from app.api import create_app
    from app.config import Settings
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        user = db.execute("SELECT user_id FROM raw_messages LIMIT 1").fetchone()
        active = db.execute("SELECT COUNT(*) FROM raw_messages WHERE deletion_state='active'").fetchone()[0]
    if user is None:
        print(json.dumps({"status": "empty", "external_calls": 0}))
        return 1
    application = create_app(Settings(db_path=path, profile_id="span_sync"))
    with TestClient(application) as client:
        health = client.get("/health")
        found = client.post("/search", json={"query": "the", "user_id": user[0], "top_k": 5})
    body = health.json() if health.headers.get("content-type", "").startswith("application/json") else {}
    hits = len(found.json().get("data", [])) if found.status_code == 200 else None
    ok = (health.status_code == 200 and body.get("status") == "ok" and body.get("control_ready") is True
          and body.get("data_route_ready") is True and found.status_code == 200 and active > 0)
    print(json.dumps({"status": "ok" if ok else "mismatch", "tiktoken": tiktoken.__version__,
                      "python": sys.version.split()[0], "profile": "span_sync",
                      "health_status": health.status_code, "health_ok": body.get("status") == "ok",
                      "search_status": found.status_code, "hit_count": hits, "active_rows": active,
                      "external_calls": 0}))
    return 0 if ok else 1


def rollback_search(path: Path) -> int:
    import tiktoken
    from fastapi.testclient import TestClient
    from app.api import create_app
    from app.config import Settings
    application = create_app(Settings(db_path=path, profile_id="span_sync"))
    with TestClient(application) as client:
        live = client.post("/search", json={"query": "surviving", "user_id": "u", "top_k": 5})
        gone = client.post("/search", json={"query": "deleted", "user_id": "u", "top_k": 5})
    survivor = [item["content"] for item in live.json().get("data", [])]
    erased = [item["content"] for item in gone.json().get("data", [])]
    ok = live.status_code == 200 and "surviving beta" in survivor and not any("deleted alpha" in item for item in erased)
    print(json.dumps({"status": "ok" if ok else "mismatch", "tiktoken": tiktoken.__version__,
                      "python": sys.version.split()[0], "live_status": live.status_code,
                      "survivor_found": "surviving beta" in survivor, "erased_found": bool(erased),
                      "external_calls": 0}))
    return 0 if ok else 1


def invoke_rebuild(mode: str, source: Path, target: Path, manifest: Path) -> int:
    install_guard()
    patch_client(mode)
    sys.argv = ["deploy.rebuild", "--source", str(source), "--target", str(target),
                "--manifest", str(manifest), "--execute-v4-rebuild"]
    from deploy.rebuild import main as rebuild_main
    return rebuild_main()


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


def runtime_env(db: Path, cache: Path, service_key: str, embedding_key: str) -> dict:
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
        "TIKTOKEN_CACHE_DIR": str(cache),
        "AML_T02_CONNECT_LOG": str(db.parent / "connect.log"),
    })


def run_command(py: Path, args: list[str], env: dict, log_path: Path) -> subprocess.CompletedProcess:
    completed = subprocess.run([str(py), "-B", *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    previous = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    log_path.write_text(previous + f"\n$ {' '.join(args)}\nexit={completed.returncode}\n"
                        + completed.stdout + completed.stderr, encoding="utf-8")
    return completed


def config_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            row = db.execute("SELECT config_hash FROM k0_config WHERE singleton=1").fetchone()
    except sqlite3.Error:
        return None
    return None if row is None else row[0]


def ready_sum(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            row = db.execute("SELECT COALESCE(SUM(ready), 0) FROM k0_watermarks").fetchone()
    except sqlite3.Error:
        return None
    return int(row[0])


def table_fingerprints(path: Path) -> dict:
    """Row counts and hashes. Callers may record names, not row text."""
    fingerprints = {}
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        names = [row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY 1")]
        for name in names:
            digest = hashlib.sha256()
            count = 0
            for row in db.execute("SELECT * FROM " + name):
                count += 1
                digest.update(repr(tuple(row)).encode("utf-8", "surrogatepass"))
            fingerprints[name] = {"rows": count, "sha256_16": digest.hexdigest()[:16]}
    return fingerprints


def logical_sha256(path: Path) -> str | None:
    from app.k0_migrate import logical_hash
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            db.execute("BEGIN")
            return logical_hash(db)
    except sqlite3.Error:
        return None


def sidecar_hashes(path: Path) -> dict:
    return {"sqlite": sha256_file(path), "wal": sha256_file(Path(str(path) + "-wal")),
            "shm": sha256_file(Path(str(path) + "-shm"))}


def run_drill(py: Path, old_python: Path, work: Path, evidence: Path, cache: Path, origin: Path) -> int:
    import secrets
    work.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    service_key = secrets.token_hex(16)
    embedding_key = secrets.token_hex(16)
    report: dict = {"scope": "t02_old_db_rebuild_drill", "formal_v4_acceptance": False,
                    "supplier_network_connects": 0, "steps": [],
                    "source_discovery": "experiments/018-temporal-anchors/isolate-add-01.sqlite; copied with the SQLite backup API because WAL/SHM exist; origin left in place"}
    failures = []
    log_path = evidence / "04-rebuild-drill.log"

    def note(name: str, ok: bool, **fields):
        report["steps"].append({"name": name, "ok": ok, **fields})
        if not ok:
            failures.append(name)

    source = work / "source-readonly.sqlite"
    pristine = work / "pristine.sqlite"
    plan_target = work / "plan-target.sqlite"
    plan_manifest = work / "plan-manifest.json"
    built = work / "rebuilt-v4.sqlite"
    built_manifest = work / "rebuilt-v4-manifest.json"
    failed = work / "failed-v4.sqlite"
    failed_manifest = work / "failed-v4-manifest.json"
    script = str(Path(__file__).resolve())
    env = runtime_env(source, cache, service_key, embedding_key)

    origin_before = sidecar_hashes(origin)
    try:
        backup_copy(origin, source)
    except (sqlite3.Error, OSError) as exc:
        note("backup origin to external copy", False, error=type(exc).__name__)
        report["ok"] = False
        report["failed_steps"] = failures
        (evidence / "04-rebuild-drill.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"ok": False, "failed_steps": failures}))
        return 1
    origin_after_backup = sidecar_hashes(origin)
    try:
        backup_copy(origin, pristine)
    except (sqlite3.Error, OSError) as exc:
        note("backup pristine comparison copy", False, error=type(exc).__name__)
        report["ok"] = False
        report["failed_steps"] = failures
        (evidence / "04-rebuild-drill.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"ok": False, "failed_steps": failures}))
        return 1
    os.chmod(source, stat.S_IREAD)
    sealed = sha256_file(source)
    logical_before = logical_sha256(source)
    note("backup origin to external read-only copy",
         sealed is not None and logical_before is not None
         and origin_after_backup["sqlite"] == origin_before["sqlite"]
         and origin_after_backup["wal"] == origin_before["wal"],
         copy_sha256=sealed, copy_bytes=source.stat().st_size,
         origin_sqlite_unchanged=origin_after_backup["sqlite"] == origin_before["sqlite"],
         origin_wal_unchanged=origin_after_backup["wal"] == origin_before["wal"],
         origin_shm_unchanged=origin_after_backup["shm"] == origin_before["shm"])

    def source_unchanged(label: str) -> None:
        current = sha256_file(source)
        wal_path = Path(str(source) + "-wal")
        wal = sha256_file(wal_path)
        wal_bytes = wal_path.stat().st_size if wal_path.exists() else None
        empty_wal = wal_bytes in (None, 0)
        note(label, current == sealed and empty_wal, sha256=current, wal_sha256=wal, wal_bytes=wal_bytes)

    refused = run_command(py, ["-m", "deploy.runtime", "--configuration"], env, log_path)
    payload = {}
    try:
        payload = json.loads(refused.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        payload = {"unparsed": True}
    note("new runtime refuses old database",
         refused.returncode == 2 and payload.get("code") == "database_identity_mismatch_rebuild_required",
         exit_code=refused.returncode, code=payload.get("code"))
    source_unchanged("source hash unchanged after refusal")

    planned = run_command(py, ["-m", "deploy.rebuild", "--source", str(source), "--target", str(plan_target),
                               "--manifest", str(plan_manifest)], env, log_path)
    plan = {}
    try:
        plan = json.loads(planned.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        plan = {"unparsed": True}
    note("rebuild dry-run",
         planned.returncode == 0 and plan.get("status") == "plan" and plan.get("external_calls") == 0
         and plan.get("target_created") is False and not plan_target.exists() and not plan_manifest.exists(),
         exit_code=planned.returncode, status=plan.get("status"), expected_spans=plan.get("expected_spans"),
         active_sources=plan.get("active_sources"), external_calls=plan.get("external_calls"))
    source_unchanged("source hash unchanged after dry-run")

    executed = run_command(py, [script, "invoke-rebuild", "--mode", "ok", "--source", str(source),
                                "--target", str(built), "--manifest", str(built_manifest)], env, log_path)
    manifest = {}
    if built_manifest.exists():
        manifest = json.loads(built_manifest.read_text(encoding="utf-8"))
        shutil.copyfile(built_manifest, evidence / "04-rebuilt-manifest.json")
    proofs = manifest.get("proofs") or []
    dense_ok = bool(proofs) and all(item.get("counts", {}).get("dense") == item.get("counts", {}).get("active")
                                     for item in proofs)
    note("surrogate execute and certify",
         executed.returncode == 0 and manifest.get("status") == "certified" and dense_ok,
         exit_code=executed.returncode, status=manifest.get("status"),
         reported_embedding_requests=manifest.get("external_calls"), proof_count=len(proofs))
    source_unchanged("source hash unchanged after certified rebuild")

    certified_env = runtime_env(built, cache, service_key, embedding_key)
    certified = run_command(py, ["-m", "deploy.runtime", "--configuration"], certified_env, log_path)
    certified_payload = {}
    try:
        certified_payload = json.loads(certified.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        certified_payload = {"unparsed": True}
    note("certified target passes configuration check",
         certified.returncode == 0 and certified_payload.get("status") == "ok"
         and certified_payload.get("database") == "same_identity"
         and certified_payload.get("external_calls") == 0,
         exit_code=certified.returncode, database=certified_payload.get("database"),
         external_calls=certified_payload.get("external_calls"))

    failed_run = run_command(py, [script, "invoke-rebuild", "--mode", "fail", "--source", str(source),
                                  "--target", str(failed), "--manifest", str(failed_manifest)], env, log_path)
    failed_body = {}
    if failed_manifest.exists():
        failed_body = json.loads(failed_manifest.read_text(encoding="utf-8"))
        shutil.copyfile(failed_manifest, evidence / "04-failed-manifest.json")
    note("injected embedding failure marks target unservable",
         failed_run.returncode == 2 and failed_body.get("target_must_not_be_served") is True
         and config_hash(failed) == "failed_rebuild" and ready_sum(failed) == 0,
         exit_code=failed_run.returncode, manifest_status=failed_body.get("status"),
         config_hash=config_hash(failed), watermark_ready_sum=ready_sum(failed))
    source_unchanged("source hash unchanged after failed rebuild")
    refused_failed = run_command(py, ["-m", "deploy.runtime", "--configuration"],
                                 runtime_env(failed, cache, service_key, embedding_key), log_path)
    note("failed target is refused by new runtime", refused_failed.returncode == 2,
         exit_code=refused_failed.returncode)

    os.chmod(source, stat.S_IWRITE | stat.S_IREAD)
    rolled = run_command(old_python, [script, "rollback-open", "--path", str(source)],
                         child_env({"TIKTOKEN_CACHE_DIR": str(cache)}), log_path)
    rollback_payload = {}
    try:
        rollback_payload = json.loads(rolled.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        rollback_payload = {"unparsed": True}
    logical_after = logical_sha256(source)
    before_tables = table_fingerprints(pristine)
    after_tables = table_fingerprints(source)
    added_tables = sorted(set(after_tables) - set(before_tables))
    removed_tables = sorted(set(before_tables) - set(after_tables))
    changed_tables = sorted(name for name in before_tables if before_tables[name] != after_tables.get(name))
    origin_after = sidecar_hashes(origin)
    prior_rows_unchanged = not changed_tables and not removed_tables
    note("rollback open on tiktoken 0.12 runtime and original profile",
         rolled.returncode == 0 and rollback_payload.get("health_ok") is True
         and rollback_payload.get("search_status") == 200 and prior_rows_unchanged
         and origin_after["sqlite"] == origin_before["sqlite"]
         and origin_after["wal"] == origin_before["wal"],
         exit_code=rolled.returncode, tiktoken=rollback_payload.get("tiktoken"),
         health_status=rollback_payload.get("health_status"),
         search_status=rollback_payload.get("search_status"),
         hit_count=rollback_payload.get("hit_count"),
         active_rows=rollback_payload.get("active_rows"),
         prior_tables_unchanged=prior_rows_unchanged,
         added_empty_tables=added_tables,
         changed_tables=changed_tables,
         logical_sha256_unchanged=logical_after == logical_before,
         origin_sqlite_unchanged=origin_after["sqlite"] == origin_before["sqlite"],
         origin_wal_unchanged=origin_after["wal"] == origin_before["wal"])
    if added_tables:
        report["findings"] = [{
            "id": "initialize_adds_empty_cycle2_tables",
            "tables": added_tables,
            "detail": "SqliteStore.initialize on the current app creates empty atom/sentence tables while opening the preserved span_sync database. Existing table row hashes stayed the same. app/ was not modified.",
        }]

    connect_log = work / "connect.log"
    connects = connect_log.read_text(encoding="ascii").splitlines() if connect_log.exists() else []
    report["supplier_network_connects"] = len(connects)
    report["blocked_connect_hosts"] = sorted(set(connects))
    report["source_sha256"] = sealed
    report["origin_sqlite_sha256"] = origin_before["sqlite"]
    report["origin_wal_sha256"] = origin_before["wal"]
    report["origin_sqlite_sha256_after"] = origin_after["sqlite"]
    report["origin_wal_sha256_after"] = origin_after["wal"]
    report["ok"] = not failures and len(connects) == 0
    report["failed_steps"] = failures
    text = json.dumps(report, ensure_ascii=False, indent=2)
    (evidence / "04-rebuild-drill.json").write_text(text, encoding="utf-8")
    redacted = log_path.read_text(encoding="utf-8", errors="replace").replace(service_key, "[redacted]").replace(embedding_key, "[redacted]")
    log_path.write_text(redacted, encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "failed_steps": failures, "supplier_network_connects": len(connects)}))
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    generate = sub.add_parser("generate-old")
    generate.add_argument("--path", type=Path, required=True)
    rollback = sub.add_parser("rollback-search")
    rollback.add_argument("--path", type=Path, required=True)
    opened = sub.add_parser("rollback-open")
    opened.add_argument("--path", type=Path, required=True)
    invoke = sub.add_parser("invoke-rebuild")
    invoke.add_argument("--mode", choices=("ok", "fail"), required=True)
    invoke.add_argument("--source", type=Path, required=True)
    invoke.add_argument("--target", type=Path, required=True)
    invoke.add_argument("--manifest", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--python", type=Path, required=True)
    run.add_argument("--old-python", type=Path, default=OLD_PYTHON)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--evidence", type=Path, required=True)
    run.add_argument("--tiktoken-cache", type=Path, required=True)
    run.add_argument("--origin", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "generate-old":
        return generate_old(args.path)
    if args.cmd == "rollback-search":
        return rollback_search(args.path)
    if args.cmd == "rollback-open":
        return rollback_open(args.path)
    if args.cmd == "invoke-rebuild":
        return invoke_rebuild(args.mode, args.source, args.target, args.manifest)
    return run_drill(args.python, args.old_python, args.work, args.evidence, args.tiktoken_cache, args.origin)


if __name__ == "__main__":
    raise SystemExit(main())

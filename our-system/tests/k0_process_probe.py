"""Standalone process/kill probes. Synthetic embeddings; no network or quality claims.

Run from our-system: .venv/Scripts/python.exe -X utf8 -B -m tests.k0_process_probe
--output artifacts/k0-20261006-01/process-01
Each probe uses a new directory and real independent spawned Python processes.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from contextlib import closing

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.embeddings import EmbeddingBatch, EmbeddingConfig


def payload(text="alpha"):
    return {"user_id": "u", "request_id": "r", "session_id": "s",
            "messages": [{"role": "user", "content": text}]}


def wait_for(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError("marker_timeout:" + path.name)
        time.sleep(0.02)


class ProcessProvider:
    implementation_kind = "process_engineering_double"

    def __init__(self, out, label, block=False):
        self.config = EmbeddingConfig(provider="engineering", endpoint="https://invalid.test/embeddings", dimensions=2)
        self.out, self.label, self.block = out, label, block

    def embed(self, texts):
        # Only count/hash, never raw text. Calls are per-worker to avoid log races.
        with (self.out / (self.label + ".calls.jsonl")).open("a", encoding="utf-8") as f:
            f.write(json.dumps({"count": len(texts), "pid": os.getpid(), "kind": "engineering_double"}) + "\n")
        (self.out / (self.label + ".dispatched")).touch()
        if self.block:
            wait_for(self.out / (self.label + ".release"), timeout=40)
        return EmbeddingBatch([[1.0, 0.0] for _ in texts], {"requests": 1, "monetary_cost": 0})


def worker(args):
    out = args.output
    app = create_app(Settings(db_path=out / "m.db", profile_id="dense_sync"),
                     embedding_provider=ProcessProvider(out, args.label, args.block))
    (out / (args.label + ".ready")).write_text(json.dumps({"worker_pid": os.getpid()}), encoding="utf-8")
    if args.start_gate:
        wait_for(out / "start")
    original = app.state.store.commit_add
    if args.boundary == "before":
        def before(*a, **kw):
            (out / (args.label + ".boundary")).touch()
            wait_for(out / (args.label + ".release"), timeout=40)
            return original(*a, **kw)
        app.state.store.commit_add = before
    elif args.boundary == "inside":
        original_insert = app.state.store._insert_embeddings
        def inside(*a, **kw):
            original_insert(*a, **kw)
            (out / (args.label + ".boundary")).touch()
            wait_for(out / (args.label + ".release"), timeout=40)
        app.state.store._insert_embeddings = inside
    elif args.boundary == "after":
        def after(*a, **kw):
            result = original(*a, **kw)
            (out / (args.label + ".boundary")).touch()
            wait_for(out / (args.label + ".release"), timeout=40)
            return result
        app.state.store.commit_add = after
    started = time.monotonic()
    with TestClient(app) as client:
        response = client.post("/add", json=payload(args.text))
    result = {"status": response.status_code, "body": response.json(),
              "retry_after": response.headers.get("Retry-After"),
              "seconds": time.monotonic() - started, "pid": os.getpid()}
    (out / (args.label + ".result.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"worker": args.label, "status": result["status"], "seconds": result["seconds"]}))
    return 0


def launch(out, label, *flags):
    command = [sys.executable, "-X", "utf8", "-B", "-m", "tests.k0_process_probe", "--worker",
               "--output", str(out), "--label", label, *flags]
    log = (out / (label + ".stdout.txt")).open("w", encoding="utf-8")
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    proc.k0_log = log
    return proc


def finish(proc, out, label):
    code = proc.wait(timeout=15)
    proc.k0_log.close()
    (out / (label + ".exit.txt")).write_text(str(code), encoding="utf-8")
    assert code == 0
    return json.loads((out / (label + ".result.json")).read_text(encoding="utf-8"))


def terminate(proc, out, label):
    # Windows venv python.exe is a redirector: Popen.pid may be its launcher,
    # not the Python interpreter holding the DB. Kill the PID from the worker.
    worker_pid = json.loads((out / (label + ".ready")).read_text(encoding="utf-8"))["worker_pid"]
    if os.name == "nt":
        result = subprocess.run(["taskkill", "/PID", str(worker_pid), "/T", "/F"],
                                capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        (out / (label + ".taskkill.stdout.bin")).write_bytes(result.stdout)
        (out / (label + ".taskkill.stderr.bin")).write_bytes(result.stderr)
        assert result.returncode == 0
    else:
        import signal
        os.kill(worker_pid, signal.SIGKILL)
    code = proc.wait(timeout=10)
    proc.k0_log.close()
    (out / (label + ".kill.json")).write_text(json.dumps({"launcher_pid": proc.pid, "worker_pid": worker_pid, "exit_code": code,
        "mechanism": "TerminateProcess on Windows / SIGKILL on POSIX"}), encoding="utf-8")
    assert code != 0
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, worker_pid)
        if handle:
            exit_code = ctypes.c_ulong()
            try:
                assert ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                assert exit_code.value != 259, "actual worker is still running"
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)


def counts(out):
    with closing(sqlite3.connect(out / "m.db")) as db:
        result = {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in (
            "raw_messages", "raw_vectors", "add_requests", "k0_receipts", "k0_attempts")}
        result["job"] = db.execute("SELECT generation,state FROM k0_jobs").fetchone()
        result["watermarks"] = db.execute("SELECT route,commit_seq,epoch,ready FROM k0_watermarks ORDER BY route").fetchall()
        result["commit_seq"] = db.execute("SELECT commit_seq FROM k0_users").fetchone()
        assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        db.rollback()
        return result


def calls(out):
    return sum(len(p.read_text().splitlines()) for p in out.glob("*.calls.jsonl"))


def expire(out):
    with closing(sqlite3.connect(out / "m.db")) as db:
        db.execute("UPDATE k0_jobs SET expires_at=0")
        db.commit()
    (out / "fault-expiry.json").write_text(json.dumps({"fault": "force durable lease expiry after killed/stalled worker",
        "production_ttl_seconds": 60, "note": "expiry accelerated, not an actual 60s wait"}), encoding="utf-8")


def assert_ready(out):
    app = create_app(Settings(db_path=out / "m.db", profile_id="dense_sync"),
                     embedding_provider=ProcessProvider(out, "immediate-search"))
    with TestClient(app) as client:
        response = client.post("/search", json={"query": "alpha", "user_id": "u", "top_k": 10})
    assert response.status_code == 200
    assert [r["content"] for r in response.json()["data"]] == ["alpha"]


def parent(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    summary, running = {}, []
    try:
        out = root / "same-key"
        out.mkdir()
        first = launch(out, "first", "--block", "--start-gate")
        running.append(first)
        wait_for(out / "first.ready")
        second = launch(out, "second", "--start-gate")
        running.append(second)
        wait_for(out / "second.ready")
        (out / "start").touch()
        # If second wins, first replays; if first wins release it after dispatch.
        deadline = time.monotonic() + 5
        while not (out / "first.dispatched").exists() and not (out / "second.result.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        (out / "first.release").touch()
        assert finish(first, out, "first")["status"] == 200
        assert finish(second, out, "second")["status"] == 200
        assert calls(out) == 1 and counts(out)["add_requests"] == 1
        summary["same_key"] = {"status": "passed", "model_calls": calls(out), "state": counts(out)}

        out = root / "wait-conflict-takeover"
        out.mkdir()
        first = launch(out, "old", "--block")
        running.append(first)
        wait_for(out / "old.dispatched")
        second = launch(out, "contender")
        running.append(second)
        result = finish(second, out, "contender")
        assert result["status"] == 503 and result["retry_after"] == "1" and 4.9 <= result["seconds"] <= 5.8
        conflict = launch(out, "conflict", "--text", "different")
        running.append(conflict)
        assert finish(conflict, out, "conflict")["status"] == 409 and calls(out) == 1
        expire(out)
        takeover = launch(out, "takeover")
        running.append(takeover)
        assert finish(takeover, out, "takeover")["status"] == 200
        (out / "old.release").touch()
        assert finish(first, out, "old")["status"] == 503
        assert counts(out)["job"] == (2, "committed") and counts(out)["add_requests"] == 1
        summary["wait_conflict_takeover"] = {"status": "passed", "wait_seconds": result["seconds"],
                                            "model_calls": calls(out), "state": counts(out)}

        out = root / "renewal-20s"
        out.mkdir()
        renewing = launch(out, "renewing", "--block")
        running.append(renewing)
        wait_for(out / "renewing.dispatched")
        with closing(sqlite3.connect(out / "m.db")) as db:
            before_expiry = db.execute("SELECT expires_at FROM k0_jobs").fetchone()[0]
            db.rollback()
        deadline = time.monotonic() + 23
        while True:
            with closing(sqlite3.connect(out / "m.db")) as db:
                after_expiry = db.execute("SELECT expires_at FROM k0_jobs").fetchone()[0]
                db.rollback()
            if after_expiry > before_expiry:
                break
            assert time.monotonic() < deadline, "real 20s renewal missing"
            time.sleep(0.1)
        (out / "renewing.release").touch()
        assert finish(renewing, out, "renewing")["status"] == 200
        assert 19 <= after_expiry - before_expiry <= 22
        summary["renewal_20s"] = {"status": "passed", "ttl_seconds": 60,
            "before_expiry": before_expiry, "after_expiry": after_expiry,
            "expiry_extension_seconds": after_expiry - before_expiry, "state": counts(out)}

        for boundary in ("before", "inside", "after"):
            out = root / ("kill-" + boundary)
            out.mkdir()
            first = launch(out, "killed", "--boundary", boundary)
            running.append(first)
            wait_for(out / "killed.boundary")
            terminate(first, out, "killed")
            before = counts(out)
            if boundary != "after":
                assert before["raw_messages"] == before["raw_vectors"] == before["add_requests"] == 0
                assert before["watermarks"] == []
                expire(out)
            else:
                assert before["raw_messages"] == before["raw_vectors"] == before["add_requests"] == 1
            recovered = launch(out, "recovered")
            running.append(recovered)
            assert finish(recovered, out, "recovered")["status"] == 200
            after = counts(out)
            assert after["raw_messages"] == after["raw_vectors"] == after["add_requests"] == 1
            assert calls(out) == (1 if boundary == "after" else 2)
            summary["kill_" + boundary] = {"status": "passed", "before": before, "after": after,
                                            "add_model_calls": calls(out)}
            assert_ready(out)
            final_state = counts(out)
            (out / "normal-service-recovery.json").write_text(json.dumps({
                "add_status": 200, "immediate_search_status": 200, "integrity_check": "ok",
                "actual_worker_terminated": True, "wal_deleted": False, "io_retry": False,
                "final_state": final_state}, indent=2), encoding="utf-8")
        (root / "summary.json").write_text(json.dumps({"status": "passed", "external_network_calls": 0,
            "scope": "engineering process/crash correctness only", "probes": summary}, indent=2), encoding="utf-8")
        print(json.dumps({"status": "passed", "probes": len(summary), "artifact": str(root)}))
        return 0
    finally:
        for proc in running:
            if proc.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    proc.kill()
                proc.wait(timeout=10)
            if not proc.k0_log.closed:
                proc.k0_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--label", default="worker")
    parser.add_argument("--text", default="alpha")
    parser.add_argument("--block", action="store_true")
    parser.add_argument("--start-gate", action="store_true")
    parser.add_argument("--boundary", choices=("before", "inside", "after"))
    args = parser.parse_args()
    return worker(args) if args.worker else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())

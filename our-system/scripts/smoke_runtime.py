"""Loopback-only two-process durability/concurrency smoke. No model calls.

Creates a new synthetic-data run directory; never opens the configured service DB.
This bounded check is not a capacity certification or full R2 acceptance.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import sqlite3
import subprocess
import sys
import time
import httpx

ROOT=Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1",0))
        return s.getsockname()[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    db=out/"memory.sqlite"
    key=secrets.token_hex(24)
    children=[]
    clients=[]
    files=[]
    started=time.perf_counter()
    checks={}
    report={"scope":"loopback_two_process_smoke","external_model_calls":0,"full_R2_accepted":False}
    base={**os.environ,"AML_DB_PATH":str(db),"AML_RETRIEVAL_MODE":"raw_fts","AML_ENABLE_VECTOR":"false",
          "AML_ENABLE_GOVERNANCE":"true","AML_CONTEXT_NEIGHBOR_RADIUS":"0",
          "AML_AUTH_MODE":"bearer","AML_SERVICE_KEY_ENV":"AML_SMOKE_LOCAL_KEY","AML_SMOKE_LOCAL_KEY":key,
          "AML_MAX_CONCURRENT_REQUESTS":"16","AML_MAX_REQUEST_BYTES":"10485760","PYTHONDONTWRITEBYTECODE":"1"}

    def start(label):
        port=free_port()
        log=(out/(label+".log")).open("w",encoding="utf-8")
        files.append(log)
        process=subprocess.Popen([sys.executable,"-B","-m","uvicorn","app.api:create_app","--factory",
                                  "--host","127.0.0.1","--port",str(port),"--log-level","warning"],
                                 cwd=ROOT,env=base,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        children.append(process)
        client=httpx.Client(base_url=f"http://127.0.0.1:{port}",headers={"Authorization":"Bearer "+key},
                            timeout=15,trust_env=False)
        clients.append(client)
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            if process.poll() is not None:
                raise RuntimeError("child exited before readiness; inspect synthetic smoke log")
            try:
                if client.get("/health").status_code==200:
                    return process,client
            except httpx.TransportError:
                pass
            time.sleep(.1)
        raise RuntimeError("child did not become ready")

    def add_body(r="original",text="我住在衡阳。"):
        return {"user_id":"smoke-a","session_id":"s","request_id":r,
                "messages":[{"role":"user","content":text,"timestamp":1704067200000}]}

    def search(client,text="衡阳"):
        return client.post("/search",json={"user_id":"smoke-a","query":text,"top_k":10})

    try:
        first,c1=start("first")
        second,c2=start("second")
        checks["unauthenticated_rejected"]=c1.post("/search",json={"user_id":"smoke-a","query":"x","top_k":1},
                                                 headers={"Authorization":"Bearer wrong"}).status_code==401
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            responses=list(pool.map(lambda client:client.post("/add",json=add_body()),[c1,c2]))
        checks["concurrent_duplicate_200"]=all(r.status_code==200 for r in responses)
        with sqlite3.connect(db) as conn:
            checks["one_committed_raw_and_ledger"]=(conn.execute("SELECT count(*) FROM raw_messages").fetchone()[0]==1
                and conn.execute("SELECT count(*) FROM add_requests").fetchone()[0]==1)
        checks["changed_payload_409"]=c2.post("/add",json=add_body(text="different text")).status_code==409
        # Simulate loss of the worker after a successful Add, not an untested
        # assumption about a kill inside every possible transaction boundary.
        first.kill()
        first.wait(timeout=5)
        restarted,c3=start("restarted")
        checks["committed_add_survives_process_kill"]=any(h["content"]=="我住在衡阳。" for h in search(c3).json()["data"])
        checks["post_restart_retry_200"]=c3.post("/add",json=add_body()).status_code==200
        delete=add_body("delete",'删除“我住在衡阳。”')
        checks["delete_200"]=c3.post("/add",json=delete).status_code==200
        checks["other_process_observes_delete"]=search(c2).json()=={"data":[]}
        checks["replay_cannot_resurrect"]=c2.post("/add",json=add_body()).status_code==200 and search(c2).json()=={"data":[]}
        with sqlite3.connect(db) as conn:
            record=conn.execute("SELECT status,erasure_complete,erasure_scope FROM memory_operations").fetchone()
            checks["logical_erasure_receipt"]=record==("completed",1,"live_logical_tables_only")
            checks["no_raw_body"]=all(row[0]=="" for row in conn.execute("SELECT content FROM raw_messages"))
            table="fts_u_"+hashlib.sha256(b"smoke-a").hexdigest()
            checks["no_readable_deleted_fts_terms"]=not any("衡阳".encode() in row[0] for row in conn.execute(f"SELECT block FROM {table}_data"))
        samples=[]
        def exercise(index):
            client=c2 if index%2 else c3
            body={"user_id":f"load-{index}","session_id":"shared-session","request_id":"r",
                  "messages":[{"role":"user","content":f"record {index} item {j} about coral projects"} for j in range(20)]}
            start_time=time.perf_counter()
            added=client.post("/add",json=body)
            found=client.post("/search",json={"user_id":body["user_id"],"query":"coral","top_k":5})
            data=found.json().get("data",[])
            return {"latency_ms":(time.perf_counter()-start_time)*1000,"add_status":added.status_code,
                    "search_status":found.status_code,"scoped":len(data)==5 and all(h["content"].startswith(f"record {index} ") for h in data)}
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            samples=list(pool.map(exercise,range(24)))
        checks["bounded_parallel_user_isolation"]=all(x["add_status"]==200 and x["search_status"]==200 and x["scoped"] for x in samples)
        times=sorted(x["latency_ms"] for x in samples)
        report["bounded_load"]={"users":24,"messages_per_user":20,"client_concurrency":8,"server_processes":2,
                                "pair_latency_p50_ms":times[len(times)//2],
                                "pair_latency_p95_ms":times[int(.95*(len(times)-1))],
                                "failure_count":sum(not x["scoped"] or x["add_status"]!=200 or x["search_status"]!=200 for x in samples),
                                "interpretation":"local Add+Search pair latency, not per-operation SLA or production capacity"}
        report["checks"]=checks
        report["passed"]=all(checks.values())
    except Exception as exc:
        report.update({"checks":checks,"passed":False,"error_type":type(exc).__name__,"error":str(exc)})
    finally:
        for client in clients:
            client.close()
        for process in children:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for handle in files:
            handle.close()
        report["elapsed_seconds"]=time.perf_counter()-started
        report["database_bytes"]=db.stat().st_size if db.exists() else 0
        report["source_hashes"]={str(p.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT/"app").glob("*.py"))}
        (out/"result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="source_hashes"},ensure_ascii=False,indent=2))
    return 0 if report.get("passed") else 1

if __name__=="__main__":
    raise SystemExit(main())

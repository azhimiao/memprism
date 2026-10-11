from __future__ import annotations
import asyncio
import sqlite3
import pytest
from fastapi.testclient import TestClient
from app.api import create_app
from app.config import Settings
from app.transport import TransportGuard
from app.models import EvidenceCandidate
from app.safety import SourcePolicy,PolicySnapshot,filter_evidence

@pytest.mark.parametrize("mode,headers",[
    ("bearer",{"Authorization":"Bearer local-test-key"}),
    ("token",{"Authorization":"Token local-test-key"}),
    ("x-api-key",{"X-API-Key":"local-test-key"}),
])
def test_auth_and_public_health(tmp_path,monkeypatch,mode,headers):
    monkeypatch.setenv("AML_SERVICE_API_KEY","local-test-key")
    path=tmp_path/"m.sqlite"
    with TestClient(create_app(Settings(db_path=path,auth_mode=mode))) as c:
        assert c.get("/health").status_code==200
        request={"query":"anything","user_id":"a","top_k":10}
        assert c.post("/search",json=request).status_code==401
        assert c.post("/search",json=request,headers={"Authorization":"Bearer wrong"}).status_code==401
        assert c.post("/search",json=request,headers=headers).status_code==200

def test_missing_auth_key_rejected_at_startup(tmp_path,monkeypatch):
    monkeypatch.delenv("AML_SERVICE_API_KEY",raising=False)
    with pytest.raises(ValueError,match="not configured"):
        create_app(Settings(db_path=tmp_path/"m.sqlite",auth_mode="bearer"))

def test_body_size_limit_and_slot_release(tmp_path):
    with TestClient(create_app(Settings(db_path=tmp_path/"m.sqlite",max_request_bytes=200))) as c:
        assert c.post("/add",content=b"x"*201).status_code==413
        assert c.post("/search",json={"query":"q","user_id":"a","top_k":2}).status_code==200

def test_chunked_body_limit_and_concurrency_without_content_length(tmp_path):
    async def run():
        entered=asyncio.Event()
        release=asyncio.Event()
        async def downstream(scope,receive,send):
            entered.set()
            await release.wait()
            await send({"type":"http.response.start","status":200,"headers":[]})
            await send({"type":"http.response.body","body":b"ok"})
        guard=TransportGuard(downstream,Settings(db_path=tmp_path/"m",max_request_bytes=5,max_concurrent_requests=1))
        scope={"type":"http","path":"/add","headers":[],"method":"POST"}
        async def receive():
            return {"type":"http.request","body":b"a","more_body":False}
        first=[]
        async def send_first(x): first.append(x)
        task=asyncio.create_task(guard(scope,receive,send_first))
        await entered.wait()
        second=[]
        async def send_second(x): second.append(x)
        await guard(scope,receive,send_second)
        assert second[0]["status"]==503
        release.set()
        await task
        chunks=iter([{"type":"http.request","body":b"aaa","more_body":True},
                     {"type":"http.request","body":b"bbb","more_body":False}])
        async def receive_chunks(): return next(chunks)
        third=[]
        async def send_third(x): third.append(x)
        await guard(scope,receive_chunks,send_third)
        assert third[0]["status"]==413
    asyncio.run(run())

def candidate(identity,user="a"):
    return EvidenceCandidate(id=identity,user_id=user,content="evidence",score=1.25,created_at=10,
                             deletion_state="active",source_raw_id=identity,message_index=0)

def test_gate_stable_subset_permissions_integrity_and_no_mutation():
    hits=[candidate("bad"),candidate("restricted"),candidate("good"),candidate("other","b"),candidate("blocked")]
    sources={
        "bad":SourcePolicy("a","bad",1,content_valid=False),
        "restricted":SourcePolicy("a","restricted",1,disclosure_policy="restricted"),
        "good":SourcePolicy("a","good",1),
        "other":SourcePolicy("b","other",1),
        "blocked":SourcePolicy("a","blocked",1,disclosure_policy="blocked"),
    }
    original=[h.model_dump() for h in hits]
    kept,logs=filter_evidence("a",hits,PolicySnapshot(9,sources))
    assert kept==[hits[2]] and kept[0] is hits[2]
    assert [h.model_dump() for h in hits]==original
    assert [log["reason"] for log in logs]==["source_body_mismatch","restricted_without_permission","eligible","wrong_user","disclosure_blocked"]
    kept,_=filter_evidence("a",hits,PolicySnapshot(9,sources,allowed_restricted_ids=frozenset({"restricted"})))
    assert kept==[hits[1],hits[2]]

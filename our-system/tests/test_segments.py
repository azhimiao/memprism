"""Exact-source and lifecycle contracts, not model-quality measurements."""
from dataclasses import replace
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from app.api import create_app
from app.config import Settings
from app.models import RawMessage,SearchRequest,AddRequest
from app.segments import split_source,token_count,digest,partition
from app.jobs import NotReady
from test_k0 import Provider,add,search,connected,body


def raw(content):
    return RawMessage(id="raw_1",user_id="u",session_id="s",request_id="r",message_index=0,
                      role="assistant",content=content,source_timestamp=1,recorded_at=2)


@pytest.mark.parametrize("text",[
    "e\u0301 👩🏽‍💻 中文\r\n否定：不允许。\n"*200,
    'If account is active, execute only after approval; otherwise never execute.\n'*90,
    '{"steps": ["keep", "all"], "unknown": null}\n'+"0123456789"*1000,
    "1. Preconditions\n2. Run\n3. Check\n"*160,
    "😀"*2400,"\t \r\n"*1500,"短文本\r\n"])
def test_spans_cover_exact_unicode_without_normalization(text):
    pieces=split_source(raw(text))
    covered=set()
    for i,p in enumerate(pieces):
        assert p["content"]==text[p["span_start"]:p["span_end"]]
        assert len(p["content"])<=1800 and token_count(p["content"])<=512
        assert p["source_sha256"]==digest(text) and p["span_sha256"]==digest(p["content"])
        if i:
            assert 0<=pieces[i-1]["span_end"]-p["span_start"]<=160
            assert p["previous_id"]==pieces[i-1]["id"]
        covered.update(range(p["span_start"],p["span_end"]))
    assert covered==set(range(len(text)))
    assert pieces==split_source(raw(text))
    assert pieces[0]["id"]!=split_source(raw(text),2)[0]["id"]
    assert pieces[0]["id"]!=split_source(raw(text).model_copy(update={"user_id":"other"}))[0]["id"]


@pytest.mark.parametrize("profile",["span_sync","span_dense_sync"])
def test_atomic_replay_trace_role_time_and_isolation(tmp_path,profile):
    path=tmp_path/"m.db"
    provider=Provider() if profile=="span_dense_sync" else None
    app=create_app(Settings(db_path=path,profile_id=profile),embedding_provider=provider)
    text="alpha e\u0301 👩🏽‍💻\r\n"*260
    request=body(contents=(text,))
    request["messages"][0].update(role="assistant",timestamp=123)
    app.state.ingest.add(AddRequest(**request))
    response,trace=app.state.retrieval.search_with_trace(SearchRequest(query="alpha",user_id="u",top_k=100))
    assert response.data and trace["fragment_policy"]=="span_exact_v1"
    assert all(h.created_at==123 and h.content in text for h in response.data)
    assert all(e["role"]=="assistant" and e["source_timestamp"]==123 and e["source_raw_id"].startswith("raw_") for e in trace["evidence"])
    with connected(path) as db:
        before=db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0]
    calls=len(provider.calls) if provider else 0
    app.state.ingest.add(AddRequest(**request))
    with connected(path) as db:
        assert db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0]==before
    assert not provider or len(provider.calls)==calls
    assert search(app,user="other").json()["data"]==[]


@pytest.mark.parametrize("control",['delete "alpha"','do not retrieve "alpha"','I retract the claim that alpha'])
def test_lifecycle_versions_no_control_embedding_no_resurrection(tmp_path,control):
    path=tmp_path/"m.db";provider=Provider()
    app=create_app(Settings(db_path=path,profile_id="span_dense_sync"),embedding_provider=provider)
    assert add(app).status_code==200
    assert add(app,"control",(control,)).status_code==200
    assert provider.calls==[["alpha"]]
    assert search(app).status_code==200 and search(app).json()["data"]==[]
    if control.startswith("delete"):
        with connected(path) as db:
            assert db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0]==0
            assert db.execute("SELECT COUNT(*) FROM span_vectors").fetchone()[0]==0
            assert db.execute(f"SELECT COUNT(*) FROM {partition('u')}").fetchone()[0]==0
        assert add(app,"replay",("alpha",)).status_code==200
        assert search(app).json()["data"]==[] and provider.calls==[["alpha"]]


@pytest.mark.parametrize("corruption",["missing_span","missing_fts","offset","content","vector"])
def test_route_readiness_rejects_corruption_before_query_dispatch(tmp_path,corruption):
    path=tmp_path/"m.db";provider=Provider()
    app=create_app(Settings(db_path=path,profile_id="span_dense_sync"),embedding_provider=provider)
    assert add(app).status_code==200
    with connected(path) as db:
        if corruption=="missing_span":db.execute("DELETE FROM source_spans")
        elif corruption=="missing_fts":db.execute(f"DELETE FROM {partition('u')}")
        elif corruption=="offset":db.execute("UPDATE source_spans SET span_start=1")
        elif corruption=="content":db.execute("UPDATE source_spans SET content='fake'")
        else:db.execute("UPDATE span_vectors SET vector_json='[0,0]'")
    assert search(app).status_code==503
    assert provider.calls==[["alpha"]]


def test_concurrent_duplicate_only_publishes_one_span_set(tmp_path):
    app=create_app(Settings(db_path=tmp_path/"m.db",profile_id="span_sync"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies=list(pool.map(lambda _:add(app,contents=("alpha "*500,)),range(4)))
    assert all(r.status_code==200 for r in replies)
    with connected(app.state.store.db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0]==1
        assert db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0]==len(split_source(raw("alpha "*500)))


def test_failed_dense_add_is_atomic(tmp_path):
    path=tmp_path/"m.db";provider=Provider(failure="dimensions")
    app=create_app(Settings(db_path=path,profile_id="span_dense_sync"),embedding_provider=provider)
    assert add(app,contents=("alpha "*500,)).status_code==503
    with connected(path) as db:
        for table in ["raw_messages","source_spans","span_vectors","add_requests","k0_receipts"]:
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]==0


def test_offline_copy_rebuild_preserves_raw_receipts_and_source(tmp_path):
    from app.k0_migrate import migrate,file_hash
    old,new=tmp_path/"old.db",tmp_path/"new.db"
    app=create_app(Settings(db_path=old,profile_id="raw_sync"))
    text="alpha "*500
    assert add(app,contents=(text,)).status_code==200
    before=file_hash(old)
    with connected(old) as db:
        old_receipts=db.execute("SELECT * FROM add_requests").fetchall()
    result=migrate(old,new,Settings(db_path=new,profile_id="span_sync"))
    assert file_hash(old)==before and result["external_calls"]==0
    copied=create_app(Settings(db_path=new,profile_id="span_sync"))
    assert search(copied).status_code==200
    with connected(new) as db:
        assert db.execute("SELECT * FROM add_requests").fetchall()==old_receipts
        assert db.execute("SELECT content FROM raw_messages").fetchone()[0]==text
        assert db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0]>1

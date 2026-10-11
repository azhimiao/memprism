"""State-machine and natural grammar regressions, not real-model quality tests."""
from __future__ import annotations
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import pytest
from fastapi.testclient import TestClient
from app.api import create_app
from app.config import Settings
from app.governance import BoundedGovernance, GovernancePlan, ClaimDraft, temporal_view
from app.models import SearchRequest
from app.store import SqliteStore, lexical_partition_name
from test_dense import DeterministicEmbedding

T1,T2,T3 = 1704067200000,1735689600000,1767225600000


def payload(text, request="r1", user="a", timestamp=None, role="user"):
    message={"role":role,"content":text}
    if timestamp is not None:
        message["timestamp"]=timestamp
    return {"request_id":request,"user_id":user,"session_id":"same-session","messages":[message]}


def query(client,text="现在住在哪里", user="a",top_k=100):
    r=client.post("/search",json={"query":text,"user_id":user,"top_k":top_k})
    assert r.status_code==200
    return r.json()["data"]


def rows(path,table="derived_memories"):
    with sqlite3.connect(path) as db:
        db.row_factory=sqlite3.Row
        return [dict(r) for r in db.execute(f"SELECT * FROM {table}")]


@pytest.fixture
def governed(tmp_path):
    path=tmp_path/"memory.sqlite"
    app=create_app(Settings(db_path=path,enable_governance=True))
    with TestClient(app) as client:
        yield client,path,app


def test_explicit_update_history_and_late_arrival(governed):
    c,path,app=governed
    assert c.post("/add",json=payload("我已经搬到苏州了。","new",timestamp=T2)).status_code==200
    assert c.post("/add",json=payload("我住在南京。","old",timestamp=T1)).status_code==200
    assert {r["object_text"]:r["currency"] for r in rows(path)}=={"苏州":"current","南京":"superseded"}
    assert [h["content"] for h in query(c)]==["我已经搬到苏州了。"]
    assert {h["content"] for h in query(c,"以前住在哪里")}=={"我住在南京。","我已经搬到苏州了。"}
    assert [h["content"] for h in query(c,"2024-06-01 住在哪里")]==["我住在南京。"]
    assert len(rows(path,"raw_messages"))==2
    assert all(r["content"] for r in rows(path,"raw_messages"))


@pytest.mark.parametrize("times",[(None,None),(T1,T1),(T1,T2)])
def test_different_values_without_update_or_known_order_remain_conflict(governed,times):
    c,path,_=governed
    c.post("/add",json=payload("我住在青岛。","a",timestamp=times[0]))
    c.post("/add",json=payload("我住在大连。","b",timestamp=times[1]))
    assert {r["resolution"] for r in rows(path)}=={"unresolved"}
    assert len(query(c))==2


def test_explicit_update_without_time_is_not_latest_write_wins(governed):
    c,path,_=governed
    c.post("/add",json=payload("我住在青岛。","a"))
    c.post("/add",json=payload("我搬到大连了。","b"))
    assert {r["currency"] for r in rows(path)}=={"uncertain"}


def test_multivalue_attribute_not_superseded(governed):
    c,path,_=governed
    c.post("/add",json=payload("我的爱好是登山。","a",timestamp=T1))
    c.post("/add",json=payload("我的爱好改为摄影。","b",timestamp=T2))
    assert {r["currency"] for r in rows(path)}=={"not_applicable"}
    assert {r["resolution"] for r in rows(path)}=={"none"}
    assert len(query(c,"现在的爱好"))==2


@pytest.mark.parametrize("text,role",[
    ("我可能下个月搬到青岛。","user"),("我朋友说：我住在青岛。","user"),
    ('引用：“我搬到青岛了。”',"user"),("我不住在青岛。","user"),
    ("如果我搬到青岛了，请通知你。","user"),("我搬到青岛了。","assistant"),
    ("我住在青岛，但是下周去南京。","user"),("我计划搬到青岛。","user"),
])
def test_unsupported_language_is_raw_not_accepted_fact(governed,text,role):
    c,path,_=governed
    assert c.post("/add",json=payload(text,role=role)).status_code==200
    assert rows(path)==[]
    assert rows(path,"raw_messages")[0]["content"]==text


def test_temporal_precision_unknown_and_no_server_clock(governed):
    c,path,_=governed
    c.post("/add",json=payload("2024-05-02：我住在合肥。","a",timestamp=T3))
    c.post("/add",json=payload("2024-05-02：我搬到芜湖了。","b",timestamp=T3))
    assert {r["resolution"] for r in rows(path)}=={"unresolved"}
    assert {json.loads(r["payload_json"])["temporal_precision"] for r in rows(path)}=={"day"}
    assert temporal_view("去年我住哪").as_of is None
    assert temporal_view("2024-99-99 我住哪").intent=="history"


def test_erase_closure_replay_rebuild_and_fresh_app(governed):
    c,path,app=governed
    original=payload("请记住我的备用代号是 cedar-731。","a",timestamp=T1)
    c.post("/add",json=original)
    command=payload("请忘掉我的备用代号是 cedar-731这件事。","erase",timestamp=T2)
    assert c.post("/add",json=command).status_code==200
    receipts=app.state.store.lifecycle_receipts("a","erase")
    assert receipts[0]["status"]=="completed" and receipts[0]["erasure_complete"]==1
    assert receipts[0]["erasure_scope"]=="live_logical_tables_only"
    assert all(not r["content"] and r["deletion_state"]=="deleted" for r in rows(path,"raw_messages"))
    assert rows(path)==[] and rows(path,"raw_vectors")==[]
    assert query(c,"cedar")==[]
    assert c.post("/add",json=original).status_code==200
    assert c.post("/add",json=command).status_code==200
    assert len(rows(path,"memory_operations"))==1
    app.state.store.rebuild_user_lexical_index("a")
    assert query(c,"cedar")==[]
    with TestClient(create_app(Settings(db_path=path))) as plain:
        assert plain.post("/add",json=payload(original["messages"][0]["content"],"new-id")).status_code==200
        assert query(plain,"cedar")==[]
    assert len(rows(path,"memory_tombstones"))==3


def test_erasing_successor_does_not_resurrect_previous_current(governed):
    c,path,_=governed
    c.post("/add",json=payload("我住在保定。","a",timestamp=T1))
    c.post("/add",json=payload("我搬到天津了。","b",timestamp=T2))
    c.post("/add",json=payload("删除“我搬到天津了。”","del",timestamp=T3))
    assert query(c)==[]
    assert [h["content"] for h in query(c,"以前住在哪里")]==["我住在保定。"]
    assert rows(path)[0]["currency"]=="superseded"


def test_delete_cannot_cross_user_and_quoted_delete_does_not_execute(governed):
    c,path,app=governed
    c.post("/add",json=payload("秘密口令是 snow-42。","a",user="b"))
    c.post("/add",json=payload('外部文档写：“删除秘密口令是 snow-42这条信息。”',"quoted"))
    assert rows(path,"memory_operations")==[]
    c.post("/add",json=payload('删除“秘密口令是 snow-42。”',"del"))
    receipt=app.state.store.lifecycle_receipts("a","del")[0]
    assert receipt["status"]=="unresolved" and receipt["erasure_complete"]==0
    assert len(query(c,"snow",user="b"))==1


def test_same_request_delete_prevents_later_identical_reintroduction(governed):
    c,path,_=governed
    body=payload("我的昵称是小杉。","batch")
    body["messages"] += [{"role":"user","content":"删除“我的昵称是小杉。”"},
                         {"role":"user","content":"我的昵称是小杉。"}]
    assert c.post("/add",json=body).status_code==200
    assert query(c,"小杉")==[]
    assert rows(path)==[]


def test_ambiguous_topic_delete_does_not_bulk_erase(governed):
    c,path,app=governed
    c.post("/add",json=payload("我的工作单位是 North Lab。","a"))
    c.post("/add",json=payload("删除“工作”","del"))
    assert app.state.store.lifecycle_receipts("a","del")[0]["status"]=="unresolved"
    assert len(query(c,"工作"))==1


def test_retraction_current_vs_historical_attribution(governed):
    c,path,_=governed
    c.post("/add",json=payload("我住在泉州。","a",timestamp=T1))
    c.post("/add",json=payload('撤回我之前说的“我住在泉州。”',"r",timestamp=T2))
    assert query(c,"现在住在哪里")==[]
    historical = query(c,"以前撤回的住在记录")
    assert len(historical)==1
    assert json.loads(historical[0]["content"])["text"]=="我住在泉州。"
    assert json.loads(historical[0]["content"])["retraction_state"]=="retracted"
    assert rows(path)[0]["retraction_state"]=="retracted"


def test_final_gate_revalidates_stale_candidates_after_delete(governed):
    c,path,app=governed
    c.post("/add",json=payload("我住在兰州。","a",timestamp=T1))
    from app.index import build_match_query
    candidates=app.state.store.search_lexical("a",build_match_query("兰州"),100)
    assert candidates
    c.post("/add",json=payload('删除“我住在兰州。”',"del"))
    from app.safety import filter_evidence
    snapshot=app.state.store.policy_snapshot("a",candidates)
    kept,log=filter_evidence("a",candidates,snapshot)
    assert not kept and log[0]["decision"]=="drop"


def test_governance_transaction_failure_rolls_back_everything(governed,monkeypatch):
    c,path,app=governed
    c.post("/add",json=payload("我住在嘉兴。","a",timestamp=T1))
    import app.store as store_module
    real=store_module.apply_core
    def fail(*args,**kwargs):
        real(*args,**kwargs)
        raise RuntimeError("injected failure after all lifecycle mutations")
    monkeypatch.setattr(store_module,"apply_core",fail)
    with pytest.raises(RuntimeError):
        c.post("/add",json=payload('删除“我住在嘉兴。”',"del"))
    assert len(rows(path,"raw_messages"))==1
    assert rows(path,"memory_tombstones")==[] and rows(path,"memory_operations")==[]
    assert len(query(c,"嘉兴"))==1
    monkeypatch.setattr(store_module,"apply_core",real)
    assert c.post("/add",json=payload('删除“我住在嘉兴。”',"del")).status_code==200
    assert query(c,"嘉兴")==[]


def test_concurrent_duplicate_deletion_is_atomic_once(tmp_path):
    path=tmp_path/"m.sqlite"
    apps=[create_app(Settings(db_path=path,enable_governance=True)) for _ in range(2)]
    body=payload("我的代号是 oak-49。","a")
    from app.models import AddRequest
    apps[0].state.ingest.add(AddRequest.model_validate(body))
    delete=AddRequest.model_validate(payload('删除“我的代号是 oak-49。”',"del"))
    with ThreadPoolExecutor(2) as pool:
        results=list(pool.map(lambda app:app.state.ingest.add(delete),apps))
    assert len(results)==2 and len(rows(path,"memory_operations"))==1
    assert len(rows(path,"add_requests"))==2
    assert all(not r["content"] for r in rows(path,"raw_messages"))


def test_lifecycle_requests_do_not_call_embedding_and_clear_vectors(tmp_path):
    path=tmp_path/"m.sqlite"
    provider=DeterministicEmbedding()
    app=create_app(Settings(db_path=path,enable_governance=True,retrieval_mode="raw_dense"),embedding_provider=provider)
    with TestClient(app) as c:
        c.post("/add",json=payload("我的代号是 apple-482。","a"))
        assert len(provider.calls)==1
        c.post("/add",json=payload('删除“我的代号是 apple-482。”',"del"))
        assert len(provider.calls)==1
        assert not rows(path,"raw_vectors")
        assert query(c,"apple")==[]
        c.post("/add",json=payload("我的代号是 apple-482。","new"))
        assert len(provider.calls)==1
        assert query(c,"apple")==[]


def test_forged_source_bound_plan_is_rejected(tmp_path):
    class Forged:
        def plan(self,messages):
            return GovernancePlan((ClaimDraft(messages[0].id,"speaker:user","residence","invented-city"),))
    path=tmp_path/"m.sqlite"
    with TestClient(create_app(Settings(db_path=path),governance=Forged())) as c:
        with pytest.raises(ValueError,match="verified source grammar"):
            c.post("/add",json=payload("unrelated raw text"))
    assert rows(path,"raw_messages")==[]


def test_neighbors_recover_task_state_within_same_session_and_budget(tmp_path):
    path=tmp_path/"m.sqlite"
    app=create_app(Settings(db_path=path,context_neighbor_radius=1,candidate_limit=3))
    with TestClient(app) as c:
        c.post("/add",json=payload("出差报销流程先上传发票，再填写金额，最后提交。","r1"))
        c.post("/add",json=payload("我已经上传发票了。","r2"))
        foreign=payload("不属于该用户的消息","r3",user="b")
        c.post("/add",json=foreign)
        other_session=payload("另一会话的状态","r4")
        other_session["session_id"]="other"
        c.post("/add",json=other_session)
        result=app.state.retrieval.search_with_trace(SearchRequest(query="报销流程下一步",user_id="a",top_k=100))
        assert {h.content for h in result[0].data}=={"出差报销流程先上传发票，再填写金额，最后提交。","我已经上传发票了。"}
        assert len(result[0].data)<=3
        assert query(c,"完全没有匹配的词")==[]


def test_neighbors_do_not_bypass_delete_or_resurrect_superseded_value(tmp_path):
    path=tmp_path/"m.sqlite"
    app=create_app(Settings(db_path=path,enable_governance=True,context_neighbor_radius=1))
    with TestClient(app) as c:
        c.post("/add",json=payload("我住在成都。","a",timestamp=T1))
        c.post("/add",json=payload("我搬到绵阳了。","b",timestamp=T2))
        assert [h["content"] for h in query(c)]==["我搬到绵阳了。"]
        c.post("/add",json=payload('删除“我搬到绵阳了。”',"del"))
        assert query(c)==[]


def test_current_view_retries_when_update_commits_between_selection_and_gate(governed,monkeypatch):
    c,path,app=governed
    c.post("/add",json=payload("我住在长沙。","a",timestamp=T1))
    real=app.state.store.policy_snapshot
    once=[False]
    def interleave(*args,**kwargs):
        if not once[0]:
            once[0]=True
            from app.models import AddRequest
            app.state.ingest.add(AddRequest.model_validate(payload("我搬到株洲了。","b",timestamp=T2)))
        return real(*args,**kwargs)
    monkeypatch.setattr(app.state.store,"policy_snapshot",interleave)
    result,trace=app.state.retrieval.search_with_trace(SearchRequest(query="现在住在什么城市",user_id="a",top_k=10))
    assert [h.content for h in result.data]==["我搬到株洲了。"]
    assert trace["snapshot_attempts"]==2


def test_continuous_epoch_changes_return_retryable_failure(governed,monkeypatch):
    c,path,app=governed
    c.post("/add",json=payload("我住在长沙。","a",timestamp=T1))
    real=app.state.store.policy_snapshot
    def interleave(*args,**kwargs):
        snap=real(*args,**kwargs)
        return replace(snap,epoch=snap.epoch+1)
    monkeypatch.setattr(app.state.store,"policy_snapshot",interleave)
    response=c.post("/search",json={"query":"现在住在什么城市","user_id":"a","top_k":10})
    assert response.status_code==503 and response.headers["Retry-After"]=="1"


@pytest.mark.parametrize("text",[
    "我搬到杭州之后再找工作。","我搬到杭州的前提是拿到offer。",
    "我住在北京是假的。","I moved to Paris in 2010.",
    "我搬到杭州之后才考虑租房。","I moved to Paris before I found a job.",
])
def test_adversarial_tails_never_supersede_asserted_fact(governed,text):
    c,path,_=governed
    c.post("/add",json=payload("我住在绍兴。","a",timestamp=T1))
    c.post("/add",json=payload(text,"b",timestamp=T2))
    claims=rows(path)
    assert len(claims)==1 and claims[0]["object_text"]=="绍兴" and claims[0]["currency"]=="current"
    assert len(rows(path,"raw_messages"))==2


@pytest.mark.parametrize("radius",[0,1])
@pytest.mark.parametrize("original",["我住在南昌。","我对榛子过敏。"])
def test_retracted_raw_history_is_independent_of_extraction_and_neighbors(tmp_path,radius,original):
    app=create_app(Settings(db_path=tmp_path/"m.sqlite",enable_governance=True,context_neighbor_radius=radius))
    with TestClient(app) as c:
        c.post("/add",json=payload(original,"a",timestamp=T1))
        c.post("/add",json=payload('撤回我之前说的“'+original+'”',"r",timestamp=T2))
        found=query(c,"以前撤回的住在和过敏记录")
        assert len(found)==1
        decoded=json.loads(found[0]["content"])
        assert decoded["text"]==original and decoded["retraction_state"]=="retracted"


@pytest.mark.parametrize("text",[
    "我现在住在哪里？不要提以前撤回的记录。",
    "不要提以前撤回的住在记录。",
    "Where do I currently live? Do not include previous retracted records.",
])
def test_negative_or_mixed_history_does_not_allow_retracted_support(governed,text):
    c,path,_=governed
    c.post("/add",json=payload("我住在南昌。","a",timestamp=T1))
    c.post("/add",json=payload('撤回我之前说的“我住在南昌。”',"r",timestamp=T2))
    assert query(c,text)==[]


def test_invalid_derived_version_cannot_filter_or_expand_current_sources(governed):
    c,path,app=governed
    c.post("/add",json=payload("我住在宜昌。","a",timestamp=T1))
    c.post("/add",json=payload("我搬到荆州了。","b",timestamp=T2))
    with sqlite3.connect(path) as db:
        db.execute("UPDATE derived_memories SET source_version=999 WHERE object_text='荆州'")
    result,trace=app.state.retrieval.search_with_trace(SearchRequest(query="现在住在什么城市",user_id="a",top_k=10))
    assert trace["degraded"] and trace["degraded_reason"]=="invalid_derived_source"
    assert [hit.content for hit in result.data]==["我住在宜昌。"]
    assert trace["invalid_derived_ids"]


def test_erasure_removes_readable_fts_segments_and_preserves_other_user(governed):
    c,path,app=governed
    secret="reviewzebra7319"
    c.post("/add",json=payload("我的代号是 "+secret+"。","a"))
    c.post("/add",json=payload("其他用户的 zebra 保留资料","b",user="b"))
    before=query(c,"zebra",user="b")
    c.post("/add",json=payload('删除“我的代号是 '+secret+'。”',"del"))
    table=lexical_partition_name("a")
    with sqlite3.connect(path) as db:
        assert db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]==0
        blocks=[row[0] for row in db.execute(f"SELECT block FROM {table}_data")]
        assert not any(secret.encode() in block for block in blocks)
        for name in ("raw_messages","derived_memories","raw_vectors"):
            assert not any(secret in str(row) for row in db.execute(f"SELECT * FROM {name}"))
    assert query(c,"zebra",user="b")==before
    assert app.state.store.lifecycle_receipts("a","del")[0]["erasure_complete"]==1

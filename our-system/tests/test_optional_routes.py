"""Optional E30 route contracts. These are not native quality proof."""
from dataclasses import replace

from app.api import create_app
from app.config import Settings
from app.models import AddRequest,SearchRequest
from app.embeddings import EmbeddingBatch
from test_k0 import Provider,add,body,connected,search


def app_at(path,**kwargs):return create_app(Settings(db_path=path,profile_id="span_sync",**kwargs))


def test_options_keep_original_query_allocate_total_200_and_return_sources_only(tmp_path):
    app=app_at(tmp_path/"m.db",route_policy="options")
    assert add(app,contents=("alpha history says Oslo", "beta history says Rome", "No new claim from options.")).status_code==200
    response,trace=app.state.retrieval.search_with_trace(SearchRequest(user_id="u",query="alpha",top_k=100,
        options=["Oslo","Rome","fabricated Paris","Oslo"]))
    assert response.data and all(h.content in {"alpha history says Oslo","beta history says Rome","No new claim from options."} for h in response.data)
    routes=trace["optional_routes"]
    assert sum(r["quota"] for r in routes["options"])==100 and routes["index_hits"]<=200
    assert len(routes["options"])==3 and trace["query_embedding_usage"]["requests"]==0


def test_options_over_cap_preserve_query_only(tmp_path):
    app=app_at(tmp_path/"m.db",route_policy="options")
    assert add(app).status_code==200
    response,trace=app.state.retrieval.search_with_trace(SearchRequest(user_id="u",query="alpha",top_k=100,options=[str(i) for i in range(9)]))
    assert [h.content for h in response.data]==["alpha"] and trace["optional_routes"]["options"]==[]


def test_source_labels_count_toward_wire_budget_preserve_unknown_timestamp(tmp_path):
    app=app_at(tmp_path/"m.db",source_labels=True,return_char_budget=300)
    request=body(contents=("alpha",));request["messages"][0]["role"]="assistant"
    app.state.ingest.add(AddRequest(**request))
    response,trace=app.state.retrieval.search_with_trace(SearchRequest(user_id="u",query="alpha",top_k=100))
    assert response.data and response.data[0].content.endswith("\nalpha")
    assert '"observed_role":"assistant"' in response.data[0].content and '"source_time_unix_ms":null' in response.data[0].content
    assert trace["output_characters"]==sum(len(h.content) for h in response.data)<=300
    small=app_at(tmp_path/"small.db",source_labels=True,return_char_budget=10)
    assert add(small).status_code==200 and search(small).json()["data"]==[]


def test_claim_projection_current_history_asof_and_late_arrival(tmp_path):
    app=app_at(tmp_path/"m.db",claim_projection=True)
    for request,text,time in [("new","我搬到苏州了。",1735689600000),("old","我住在南京。",1704067200000)]:
        payload=body(request,(text,));payload["messages"][0]["timestamp"]=time
        app.state.ingest.add(AddRequest(**payload))
    assert [h["content"] for h in search(app,"目前南京苏州").json()["data"]]==["我搬到苏州了。"]
    assert {h["content"] for h in search(app,"历史南京苏州").json()["data"]}=={"我住在南京。","我搬到苏州了。"}
    assert [h["content"] for h in search(app,"2024-06-01 南京苏州").json()["data"]]==["我住在南京。"]


def test_optional_windows_trigger_only_seeded_query_intent(tmp_path):
    app=app_at(tmp_path/"m.db",route_policy="witness_window")
    assert add(app,contents=("Alice knows Bob.","He moved to Oslo.")).status_code==200
    response,trace=app.state.retrieval.search_with_trace(SearchRequest(user_id="u",query="When did Bob move?",top_k=100))
    assert any("Oslo" in h.content for h in response.data)
    assert trace["optional_routes"]["windows"]["source_ref_visits"]<=100
    response,trace=app.state.retrieval.search_with_trace(SearchRequest(user_id="u",query="unrelatedxyz",top_k=100))
    assert response.data==[] and trace["optional_routes"]["windows"]["groups"]==[]


def test_named_unknown_floor_filters_cosine_not_rrf_and_is_default_off(tmp_path):
    class Orthogonal(Provider):
        def embed(self,texts):
            self.calls.append(list(texts))
            return EmbeddingBatch([[0.,1.] if t=="unknownxyz" else [1.,0.] for t in texts])
    p=Orthogonal()
    app=create_app(Settings(db_path=tmp_path/"m.db",profile_id="span_dense_sync",unknown_dense_floor=.3),embedding_provider=p)
    assert add(app).status_code==200
    response,trace=app.state.retrieval.search_with_trace(SearchRequest(user_id="u",query="unknownxyz",top_k=100))
    assert response.data==[] and trace["optional_routes"]["unknown_dense_dropped"]==1
    assert app.state.profile.identity["unknown_dense_floor"]==.3
    assert Settings(db_path=tmp_path/"other.db").unknown_dense_floor is None

"""Bounded route/source-group contracts; quality requires separate native A/B."""
from app.queryplan import plan_query,evidence_type_hints,typed_order
from app.support_groups import span_windows,pack_groups
from app.models import SearchRequest
from app.api import create_app
from app.config import Settings
from test_k0 import add,connected


def test_multi_intent_and_options_bounded_without_gold_or_clock():
    p=plan_query("When should I execute the recommended procedure?",["one","one","two"])
    assert {"procedure","personalization","temporal"}<=set(p.intents)
    assert len(p.option_queries)==2 and sum(q for _,q in p.option_queries)==100
    assert plan_query("query",[str(i) for i in range(9)]).option_queries==()
    assert plan_query("query",["😀"*600]).option_queries==()
    assert plan_query("last year").as_of is None
    assert "constraint" in evidence_type_hints("If active, do not execute without approval.")


def test_witness_windows_never_global_recent_or_cross_user(tmp_path):
    app=create_app(Settings(db_path=tmp_path/"m.db",profile_id="span_sync"))
    assert add(app,contents=("Alice knows Bob.","He moved to Oslo.","unrelated fact")).status_code==200
    assert add(app,contents=("foreign private Alice",),user="other").status_code==200
    snapshot=app.state.store.search_snapshot(user_id="u",match_query='"Bob"',query_vector=None,version="unconfigured",
        dimensions=0,mode="raw_fts",candidate_limit=20,lexical_quota=20,dense_quota=0,span=True)
    with app.state.store._connect() as db:
        expanded,trace=span_windows(db,"u",snapshot.lexical)
        empty,_=span_windows(db,"u",[])
    assert empty==[] and all(h.user_id=="u" for h in expanded)
    assert any("Oslo" in h.content for h in expanded)
    assert not any("foreign" in h.content for h in expanded)
    assert trace["source_ref_visits"]<=100 and len(expanded)<=200
    kept,_,_=pack_groups(expanded,trace["groups"],1,1000)
    assert not any(h.id==snapshot.lexical[0].id for h in kept)


def test_group_missing_dependency_not_returned_as_complete(tmp_path):
    app=create_app(Settings(db_path=tmp_path/"m.db",profile_id="span_sync"))
    assert add(app).status_code==200
    hit=app.state.store.search_snapshot(user_id="u",match_query='"alpha"',query_vector=None,version="unconfigured",
        dimensions=0,mode="raw_fts",candidate_limit=20,lexical_quota=20,dense_quota=0,span=True).lexical[0]
    kept,_,trace=pack_groups([hit],[{"seed_id":hit.id,"member_ids":[hit.id,"missing"]}],20,32000)
    assert kept==[] and trace["omitted_groups"][0]["reason"]=="missing_dependency"

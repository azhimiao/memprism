"""A committed update survives erasure without promoting its predecessor."""
from app.api import create_app
from app.config import Settings
from app.models import AddRequest,SearchRequest


def test_deleted_successor_has_hash_only_receipt_and_does_not_resurrect(tmp_path):
    app=create_app(Settings(db_path=tmp_path/'claim.sqlite',profile_id='span_sync',claim_projection=True))
    for request,text,stamp in [('old','我住在南京。',1704067200000),('new','我搬到苏州了。',1735689600000)]:
        app.state.ingest.add(AddRequest(user_id='u',session_id='s',request_id=request,
            messages=[{'role':'user','content':text,'timestamp':stamp}]))
    app.state.ingest.add(AddRequest(user_id='u',session_id='s',request_id='delete',
        messages=[{'role':'user','content':'请忘记“我搬到苏州了。”'}]))
    current=app.state.retrieval.search(SearchRequest(user_id='u',query='目前南京苏州',top_k=10))
    history=app.state.retrieval.search(SearchRequest(user_id='u',query='2024-06-01 南京苏州',top_k=10))
    assert current.data==[]
    assert [h.content for h in history.data]==['我住在南京。']
    with app.state.store._connect() as db:
        proof=db.execute('SELECT * FROM claim_supersessions').fetchone()
        assert proof is not None and '苏州' not in str(dict(proof))

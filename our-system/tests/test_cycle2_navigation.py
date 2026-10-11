from dataclasses import replace
from fastapi.testclient import TestClient
from app.api import create_app
from app.config import Settings


def client(tmp_path,**kw):
    settings=Settings(db_path=tmp_path/'memory.sqlite',profile_id='cycle2_raw',**kw)
    return TestClient(create_app(settings))


def add(c,user='u',request='r',content=None):
    payload={'user_id':user,'session_id':'s','request_id':request,'messages':[
        {'role':'user','content':content or 'I prefer tea. My timetable includes Mondays.', 'timestamp':1704067200000},
        {'role':'assistant','content':'Except on holidays, where the Friday rule applies.'}]}
    return c.post('/add',json=payload)


def search(c,user='u',query='tea'):
    return c.post('/search',json={'user_id':user,'query':query,'top_k':100})


def test_sentence_parent_metadata_and_replay(tmp_path):
    c=client(tmp_path,sentence_index=True,parent_evidence=True,metadata_roles=True,metadata_dates=True)
    assert add(c).status_code==200
    assert add(c).status_code==200
    result=search(c).json()['data']
    assert len(result)==1
    assert '[role=user; source_date=2024-01-01T00:00:00Z]' in result[0]['content']
    assert 'My timetable includes Mondays.' in result[0]['content']
    assert search(c,'other').json()['data']==[]


def test_neighbor_only_same_received_add(tmp_path):
    c=client(tmp_path,parent_evidence=True,context_neighbor_radius=1)
    assert add(c).status_code==200
    assert c.post('/add',json={'user_id':'u','session_id':'s','request_id':'other','messages':[
        {'role':'user','content':'Unrelated notes concerning Sunday.'}]}).status_code==200
    data=search(c).json()['data']
    assert len(data)==2
    assert any('holidays' in h['content'] for h in data)
    assert not any('Sunday' in h['content'] for h in data)


def test_full_received_parent_not_legacy_character_prefix(tmp_path):
    c=client(tmp_path,parent_evidence=True,return_char_budget=1000000,return_token_budget=64000)
    text='orchid procedure\n'+('Rule: retain the complete reference table.\n'*1000)+'Final exception: orchid never runs on Tuesday.'
    assert add(c,content=text).status_code==200
    data=search(c,query='orchid procedure').json()['data']
    assert data[0]['content']==text


def test_sentence_corruption_blocks_formal_route(tmp_path):
    c=client(tmp_path,sentence_index=True)
    assert add(c).status_code==200
    db=c.app.state.store._connect()
    db.execute('DELETE FROM sentence_units')
    db.commit();db.close()
    assert search(c).status_code==503


def test_research_default_does_not_enable_new_mechanisms(tmp_path):
    s=Settings(tmp_path/'m.sqlite')
    assert not s.sentence_index and not s.enable_extractor
    try:
        replace(s,sentence_index=True).ensure_phase_a_defaults()
    except ValueError:
        pass
    else:
        raise AssertionError('new route needs explicit profile')

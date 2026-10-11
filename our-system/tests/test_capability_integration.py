from app.api import create_app
from app.config import Settings
from app.models import AddRequest,SearchRequest
from app.profiles import Profile
import pytest


def add(service,user,rid,text):
    return service.state.ingest.add(AddRequest(user_id=user,session_id='s',request_id=rid,
        messages=[{'role':'user','content':text,'timestamp':1704067200000}]))


@pytest.mark.parametrize('views,selection',[(True,False),(False,True),(True,True)])
def test_actual_search_views_and_selector_respect_scope_delete_and_read_only(tmp_path,views,selection):
    settings=Settings(db_path=tmp_path/'m.db',profile_id='span_sync',capability_views=views,set_selection=selection)
    service=create_app(settings)
    add(service,'u','1','I prefer quiet rooms if available, except when I need an accessible room.')
    add(service,'other','2','I prefer private quiet rooms with secret equipment.')
    with service.state.store._connect() as db:
        before=db.execute('SELECT total_changes()').fetchone()[0]
        seq=db.execute('SELECT commit_seq FROM k0_users WHERE user_id=?',('u',)).fetchone()[0]
    response,trace=service.state.retrieval.search_with_trace(SearchRequest(user_id='u',query='What room do I prefer?',top_k=10))
    assert response.data and all('secret' not in h.content for h in response.data)
    assert response.data[0].content=='I prefer quiet rooms if available, except when I need an accessible room.'
    if views:assert trace['capability_views']['enabled'] and trace['capability_views']['source_valid']=='unknown'
    with service.state.store._connect() as db:
        assert db.execute('SELECT commit_seq FROM k0_users WHERE user_id=?',('u',)).fetchone()[0]==seq
        assert db.execute('SELECT total_changes()').fetchone()[0]==before
    add(service,'u','3','Please forget "I prefer quiet rooms if available, except when I need an accessible room."')
    response,trace=service.state.retrieval.search_with_trace(SearchRequest(user_id='u',query='What room do I prefer?',top_k=10))
    assert not response.data


def test_effect_flags_off_and_profile_identity_has_no_collisions(tmp_path):
    defaults=Settings(db_path=tmp_path/'m.db',profile_id='span_sync')
    assert not defaults.capability_views and not defaults.set_selection and not defaults.exact_dense_acceleration
    identities={Profile.build(defaults,None).config_hash}
    from dataclasses import replace
    for flag in ('capability_views','set_selection','exact_dense_acceleration'):
        config=replace(defaults,**{flag:True});config.ensure_phase_a_defaults()
        identities.add(Profile.build(config,None).config_hash)
    assert len(identities)==4
    with pytest.raises(ValueError):Settings(db_path=tmp_path/'other.db',set_selection=True).ensure_phase_a_defaults()


def test_long_legal_user_id_degrades_optional_navigation_without_search_failure(tmp_path):
    user='u'*513
    service=create_app(Settings(db_path=tmp_path/'long.db',profile_id='span_sync',capability_views=True))
    add(service,user,'1','I prefer quiet rooms.')
    response,trace=service.state.retrieval.search_with_trace(SearchRequest(user_id=user,query='Which room do I prefer?',top_k=10))
    assert len(response.data)==1
    assert trace['capability_views']['reason']=='identifier_outside_navigation_limit'
    assert not trace['capability_views']['ranking']['triggered']


def test_explicit_knowledge_asof_filters_future_sources_but_event_history_keeps_later_reports(tmp_path):
    service=create_app(Settings(db_path=tmp_path/'time.db',profile_id='span_sync',capability_views=True,set_selection=True))
    for rid,stamp,text in [('old',1704067200000,'alpha past fact.'),('new',1735689600000,'alpha future fact.'),
                           ('missing',None,'alpha unknown timestamp fact.'),('late-day',1704103200000,'alpha known later on the same source day.'),
                           ('day-end',1704153599999,'alpha exact source day end.'),('next-day',1704153600000,'alpha NEXT_DAY source.')]:
        service.state.ingest.add(AddRequest(user_id='u',session_id='s',request_id=rid,
            messages=[{'role':'user','content':text,'timestamp':stamp}]))
    response,trace=service.state.retrieval.search_with_trace(SearchRequest(user_id='u',query='As of 2024-01-01 what alpha fact is known?',top_k=10))
    assert len(response.data)==4 and all('future' not in h.content and 'NEXT_DAY' not in h.content for h in response.data)
    assert trace['knowledge_time_gate']['source_day_end_unix_ms']==1704153599999
    assert trace['knowledge_time_gate']['unknown_source_time_ids']
    response,trace=service.state.retrieval.search_with_trace(SearchRequest(user_id='u',query='What alpha event happened on 2024-01-01?',top_k=10))
    assert len(response.data)==6 and not trace['knowledge_time_gate']['active']


def test_accelerated_search_still_detects_corrupt_losing_vector_before_result(tmp_path):
    # Existing provider fixture validates the production contracts, not quality.
    from app.embeddings import EmbeddingConfig,EmbeddingBatch
    class Provider:
        implementation_kind='fixture'
        config=EmbeddingConfig(provider='fixture',endpoint='https://fixture.invalid/embeddings',dimensions=2)
        def embed(self,texts):return EmbeddingBatch([[1.,0.] for _ in texts])
    p=Provider()
    service=create_app(Settings(db_path=tmp_path/'d.db',profile_id='span_dense_sync',embedding=p.config,
        exact_dense_acceleration=True),embedding_provider=p)
    add(service,'u','1','A procedure: first inspect the input.')
    with service.state.store._connect() as db:
        db.execute("UPDATE span_vectors SET vector_json='[0,0]' WHERE user_id='u'")
    from app.jobs import NotReady
    with pytest.raises(NotReady):
        service.state.retrieval.search(SearchRequest(user_id='u',query='procedure',top_k=1))

"""Engineering invariants for 062. Synthetic data, no quality comparisons."""
from contextlib import closing
from dataclasses import replace
import json
import pytest
from fastapi.testclient import TestClient
from app.api import create_app
from app.config import Settings
from app import evidence_catalog as catalog
from app.evidence_shapes import demand
from app.models import SearchRequest
from app.jobs import SnapshotChanged, NotReady


@pytest.fixture
def c(tmp_path):
    settings=Settings(db_path=tmp_path/'memory.sqlite',profile_id='cycle2_raw',
        retrieval_policy='evidence_architecture_v1',return_char_budget=1000000,
        return_token_budget=65536,packing_policy='skip',temporal_source_dates=True)
    with TestClient(create_app(settings)) as client:
        yield client


def add(c,text,req='r',user='u',session='s',timestamp=1704067200000):
    return c.post('/add',json={'user_id':user,'session_id':session,'request_id':req,
        'messages':[{'role':'user','content':text,'timestamp':timestamp}]})


def search(c,query,user='u',k=100):
    response=c.post('/search',json={'user_id':user,'query':query,'top_k':k})
    assert response.status_code==200,response.text[:300]
    return response.json()['data']


def test_add_publishes_directory_atomically_and_replays(c):
    assert add(c,'cobalt garden grows orchids').status_code==200
    assert add(c,'cobalt garden grows orchids').status_code==200
    store=c.app.state.store
    with closing(store._connect()) as db:
        assert catalog.ready(db,'u')
        assert db.execute('SELECT COUNT(*) FROM evidence_062_sources').fetchone()[0]==1
    assert any('orchids' in h['content'] for h in search(c,'cobalt'))
    assert search(c,'cobalt',user='v')==[]


@pytest.mark.parametrize('delimiter',[',','\t'])
def test_observation_header_and_row_are_a_unit(c,delimiter):
    text=delimiter.join(['Input','Output','Unit'])+'\n'+delimiter.join(['cobalt','14','kg'])+'\n'
    assert add(c,text).status_code==200
    hits=search(c,'What output was recorded for cobalt?')
    assert any('Input' in h['content'] for h in hits)
    assert any('cobalt' in h['content'] for h in hits)
    assert len(search(c,'What output was recorded for cobalt?',k=1))<=1


def test_unheaded_rule_preserves_exception(c):
    assert add(c,'A cobalt permit is required.\nYou must obtain approval before entry.\nException: emergency crews may enter directly.').status_code==200
    hits=search(c,'What rule applies to cobalt entry?')
    assert any('Exception' in h['content'] for h in hits)
    assert any('approval' in h['content'] for h in hits)


def test_whole_history_covers_sessions_without_keyword_seed(c):
    for i in range(6):
        assert add(c,f'Phase {i}: project milestone marker{i}',req=str(i),session=f's{i}').status_code==200
    hits=search(c,'Summarize all my history')
    assert all(any(f'marker{i}' in h['content'] for h in hits) for i in range(6))
    assert search(c,'an entirely nonexistent glorpazwex')==[]


def test_background_summary_is_not_current_intent():
    plan=demand('Context: the report contains a summary.\n\nQuestion: What is the invoice number?')
    assert not plan['summary'] and not plan['global']
    assert demand('请总结全部历史')['global']


def test_privacy_replaces_raw_with_relevant_blocks(c):
    assert add(c,'My favorite drink is cobalt tea.\n\nDiagnosis: rare medical condition.').status_code==200
    hits=search(c,'What is my favorite drink?')
    assert any('cobalt tea' in h['content'] for h in hits)
    assert not any('condition' in h['content'] or 'Diagnosis' in h['content'] for h in hits)
    assert any('condition' in h['content'] for h in search(c,'What rare condition do I have?'))


def test_explicit_withhold_applies_to_block_not_unrelated_preference(c):
    assert add(c,'My favorite tea is cobalt.\n\nDo not disclose this record:\nprivate marker xylosensitive.').status_code==200
    assert not any('xylosensitive' in h['content'] for h in search(c,'Tell me xylosensitive'))
    assert any('cobalt' in h['content'] for h in search(c,'favorite tea'))


def test_state_group_retains_both_versions(c):
    for i,(value,date) in enumerate([('orchid','2024-01-01'),('cobalt','2024-02-01')]):
        text=f'Subject: greenhouse\nAttribute: permit\nScope: north\nValue: {value}\nEffective: {date}\nUpdate: true'
        assert add(c,text,req=str(i)).status_code==200
    hits=search(c,'What is the current cobalt permit and its history?')
    assert any('orchid' in h['content'] for h in hits)
    assert any('cobalt' in h['content'] for h in hits)


def test_explicit_rule_reference_stays_in_scope(c):
    assert add(c,'Scope: orchard\nRule ID: R7\nYou must wear a helmet.',req='one').status_code==200
    assert add(c,'Scope: orchard\nReference: R7\nException: cobalt staff need a visor.',req='two').status_code==200
    assert add(c,'Scope: other\nRule ID: R7\nYou must reveal wrongscope.',req='three').status_code==200
    hits=search(c,'What exception applies to cobalt staff?')
    assert any('helmet' in h['content'] for h in hits)
    assert not any('wrongscope' in h['content'] for h in hits)


def test_catalog_candidate_is_validated_against_raw(c):
    assert add(c,'Input,Output\ncobalt,14').status_code==200
    store=c.app.state.store
    with closing(store._connect()) as db:
        row=db.execute('SELECT id FROM evidence_062_blocks WHERE content LIKE ?',('%cobalt%',)).fetchone()
        hit=catalog.candidate(catalog.get(db,'u',row[0]))
    coord=c.app.state.coordinator
    stamp=coord.require_ready('u'); policy=store.policy_snapshot('u',[hit])
    coord.final_check('u',stamp,[hit],policy)
    with pytest.raises(SnapshotChanged):
        coord.final_check('u',stamp,[hit.model_copy(update={'content':'forged'})],policy)


def test_missing_catalog_blocks_fail_readiness(c):
    assert add(c,'cobalt garden grows orchids').status_code==200
    with closing(c.app.state.store._connect()) as db:
        db.execute('DELETE FROM evidence_062_blocks WHERE user_id=?',('u',));db.commit()
    with pytest.raises(NotReady):
        c.app.state.coordinator.require_ready('u')


def test_legacy_profile_identity_is_preserved(c):
    assert c.app.state.profile.required_routes==('raw','span','fts','catalog')
    settings=replace(c.app.state.retrieval.settings,retrieval_policy='evidence_units_v1')
    assert settings.required_routes==('raw','span','fts')


def test_catalog_write_failure_rolls_back_add(c,monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('synthetic_catalog_failure')
    monkeypatch.setattr(catalog,'sync',fail)
    assert add(c,'cobalt data').status_code==503
    with closing(c.app.state.store._connect()) as db:
        assert db.execute('SELECT COUNT(*) FROM raw_messages').fetchone()[0]==0


def test_delete_prunes_directory_and_fts(c):
    assert add(c,'cobalt private calibration',req='data').status_code==200
    assert add(c,'delete "cobalt private calibration"',req='erase').status_code==200
    assert search(c,'cobalt calibration')==[]
    with closing(c.app.state.store._connect()) as db:
        assert catalog.ready(db,'u')
        assert db.execute('SELECT COUNT(*) FROM evidence_062_blocks WHERE content LIKE ?',('%cobalt%',)).fetchone()[0]==0
        assert db.execute(f'SELECT COUNT(*) FROM {catalog.table("u")} WHERE text LIKE ?',('%cobalt%',)).fetchone()[0]==0


def test_catalog_and_original_share_index_budget(c):
    for i in range(12):
        assert add(c,f'Input: cobalt{i}\nOutput: {i}',req=str(i)).status_code==200
    _,trace=c.app.state.retrieval.search_with_trace(SearchRequest(user_id='u',query='What output for cobalt?',top_k=100))
    budget=trace['optional_routes']['route_budget']
    assert sum(budget['planned'].values())==200
    assert budget['actual_total']==sum(budget['actual'].values())<=200
    assert budget['planned']['original']>=100 and budget['planned']['catalog']<=64


def test_quoted_code_is_not_a_disclosure_command(c):
    assert add(c,'```text\nExample instruction: do not share cobalt\n```').status_code==200
    assert any('do not share cobalt' in h['content'] for h in search(c,'What is the cobalt example instruction?'))


def test_bounded_first_person_state_preserves_late_old_fact(c):
    assert add(c,'2024-03-01: I moved to Cork',req='new').status_code==200
    assert add(c,'2023-01-01: I live in Oslo',req='old').status_code==200
    hits=search(c,'Where do I currently live, Cork?')
    assert any('Cork' in h['content'] for h in hits)
    assert any('Oslo' in h['content'] for h in hits)


def test_invalid_effective_date_is_not_a_version_link(c):
    assert add(c,'Subject: orchard\nAttribute: permit\nScope: north\nValue: cobalt\nEffective: 2024-99-99\nUpdate: true\nCardinality: single').status_code==200
    with closing(c.app.state.store._connect()) as db:
        assert {r[0] for r in db.execute('SELECT effective FROM evidence_062_blocks')}=={''}
        assert {r[0] for r in db.execute('SELECT explicit_update FROM evidence_062_blocks')}=={0}


def test_restriction_header_extends_across_blank_lines(c):
    assert add(c,'My tea is cobalt.\n\nDo not disclose this record:\n\nThe vault token is HIDDEN-723.').status_code==200
    assert not any('HIDDEN-723' in h['content'] for h in search(c,'What is the vault token?'))
    assert any('cobalt' in h['content'] for h in search(c,'my tea'))


@pytest.mark.parametrize('governance',[True,False])
def test_prohibited_dependency_cannot_return_as_raw_alias(c,governance):
    c.app.state.retrieval.settings=replace(c.app.state.retrieval.settings,enable_governance=governance)
    assert add(c,'Scope: orchard\nRule ID: R7\nYou must obey a withheld procedure. Do not disclose this record.',req='a').status_code==200
    assert add(c,'Scope: orchard\nReference: R7\nException: cobalt staff need a visor.',req='b').status_code==200
    assert not any('visor' in h['content'] for h in search(c,'What exception applies to cobalt staff?'))


def test_top_one_cannot_return_orphan_table_row(c):
    assert add(c,'Input,Output,Unit\nsaffron,14,kg\n').status_code==200
    hits=search(c,'What rule applies to saffron?',k=1)
    assert not any('saffron' in h['content'] and 'Input' not in h['content'] for h in hits)


@pytest.mark.parametrize('last',['complete','missing','blocked','cycle'])
def test_reference_chain_closure_or_omission(c,last):
    assert add(c,'Scope: orchard\nRule ID: R7\nReference: R8\nYou must wear a helmet.',req='a').status_code==200
    if last!='missing':
        tail={'complete':'Except after dusk, when access is forbidden.',
              'blocked':'Do not disclose this record: dusk forbids access.',
              'cycle':'Reference: R7\nYou must obey R7.'}[last]
        assert add(c,'Scope: orchard\nRule ID: R8\n'+tail,req='b').status_code==200
    assert add(c,'Scope: orchard\nReference: R7\nException: cobalt staff need a visor.',req='c').status_code==200
    hits=search(c,'What exception applies to cobalt staff?')
    if last=='complete':
        assert any('visor' in h['content'] for h in hits)
        assert any('dusk' in h['content'] for h in hits)
    else:
        assert not any('visor' in h['content'] for h in hits)


def test_history_includes_phases_and_final_decision(c):
    assert add(c,'# Orchard project\n\nPhase one: planted apples.\n\nPhase two: trees failed in frost.\n\nFinal decision: switch to pears.').status_code==200
    hits=search(c,'Summarize all my history')
    assert all(any(word in h['content'] for h in hits) for word in ['apples','frost','pears'])


def test_search_uses_publication_proof_not_full_catalog_audit(c,monkeypatch):
    assert add(c,'Input,Output,Unit\n'+'\n'.join(f'probe{i},{i},kg' for i in range(80))).status_code==200
    queries=[];connect=c.app.state.store._connect
    def traced(*args,**kwargs):
        db=connect(*args,**kwargs);db.set_trace_callback(queries.append);return db
    monkeypatch.setattr(c.app.state.store,'_connect',traced)
    search(c,'What rule applies to probe79?',k=1)
    assert not any('SELECT COUNT(*) FROM evidence_062_sources c JOIN' in q for q in queries)


@pytest.mark.parametrize('inside_heading',['','# Example heading\n'])
def test_outer_restriction_covers_code_without_code_releasing_scope(c,inside_heading):
    assert add(c,'Do not disclose this record:\n\n```text\n'+inside_heading+
               'The vault token is HIDDEN-723.\n```\n\nAnother vault token is HIDDEN-724.').status_code==200
    hits=search(c,'What is the vault token?')
    assert not any('HIDDEN-' in h['content'] for h in hits)


@pytest.mark.parametrize('cycle',[False,True])
def test_distinct_rules_in_one_source_are_distinct_dependency_nodes(c,cycle):
    tail='Reference: R7\nYou must obey R7.' if cycle else 'Except after dusk, access is forbidden.'
    assert add(c,'Scope: orchard\nRule ID: R7\nReference: R8\nYou must wear a helmet.\n\n'
               'Scope: orchard\nRule ID: R8\n'+tail,req='a').status_code==200
    assert add(c,'Scope: orchard\nReference: R7\nException: cobalt staff need a visor.',req='b').status_code==200
    hits=search(c,'What exception applies to cobalt staff?')
    if cycle:
        assert not any('visor' in h['content'] for h in hits)
    else:
        assert any('visor' in h['content'] for h in hits)
        assert any('dusk' in h['content'] for h in hits)


@pytest.mark.parametrize('large',[False,True])
def test_one_candidate_budget_keeps_only_complete_table(c,tmp_path,large):
    settings=Settings(db_path=tmp_path/'one.sqlite',profile_id='cycle2_raw',
        retrieval_policy='evidence_architecture_v1',candidate_limit=1,
        lexical_quota=1,dense_quota=1,return_token_budget=65536)
    with TestClient(create_app(settings)) as client:
        rows='\n'.join(f'sample{i:03d},{i},kg' for i in range(100))+'\n' if large else ''
        assert add(client,'Input,Output,Unit\n'+rows+'saffron,314,kg\n').status_code==200
        hits=search(client,'What rule applies to saffron?')
        assert not any('saffron' in h['content'] and 'Input,Output,Unit' not in h['content'] for h in hits)
        if not large:
            assert any('saffron' in h['content'] for h in hits)

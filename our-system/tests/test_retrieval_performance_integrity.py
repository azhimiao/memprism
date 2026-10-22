"""037: exact-output and out-of-band corruption regressions, no model calls."""
import importlib.util
import json
import math
from pathlib import Path
import random
import sqlite3
import subprocess
import sys

import pytest

from app.api import create_app
from app.config import Settings
from app.embeddings import EmbeddingBatch, EmbeddingConfig, EmbeddingFailure, normalize_vector
from app.jobs import NotReady
from app.models import AddRequest, SearchRequest
from app import segments
from app.exact_dense import exact_top_scores


class Provider:
    implementation_kind = 'engineering_double'
    config = EmbeddingConfig(provider='engineering', endpoint='https://invalid.test/embeddings', dimensions=2)

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return EmbeddingBatch([[1., 0.] for _ in texts])


def build(path, dense=True):
    p = Provider() if dense else None
    app = create_app(Settings(db_path=path, profile_id='cycle2_dense' if dense else 'cycle2_raw'), embedding_provider=p)
    return app, p


def add(app, text='alpha content', user='u', rid='r'):
    return app.state.ingest.add(AddRequest(user_id=user, session_id='s', request_id=rid,
        messages=[{'role':'user', 'content':text, 'timestamp':123}]))


def search(app, user='u'):
    return app.state.retrieval.search(SearchRequest(user_id=user, query='alpha', top_k=20))


def mutate(path, sql, args=()):
    with sqlite3.connect(path) as db:
        db.execute(sql, args)


@pytest.mark.parametrize('mutation', [
    "UPDATE raw_messages SET content='altered same ID'",
    "UPDATE source_policies SET content_sha256='altered'",
    "UPDATE source_policies SET source_version=source_version+1",
    "UPDATE source_spans SET content='altered same ID'",
    "UPDATE source_spans SET token_count=token_count+1",
    "UPDATE source_spans SET previous_id='foreign'",
    "UPDATE source_spans SET span_end=span_end-1",
    "UPDATE span_vectors SET content_sha256='altered'",
    "UPDATE span_vectors SET dimensions=3",
    "UPDATE span_vectors SET embedding_version='other'",
    "UPDATE span_vectors SET vector_json='[true,0]'",
    "UPDATE span_vectors SET vector_json='[0,0]'",
    "UPDATE span_vectors SET vector_json='[NaN,0]'",
    "UPDATE span_vectors SET vector_json='[1e400,0]'",
    "UPDATE k0_watermarks SET ready=0",
    "UPDATE k0_watermarks SET commit_seq=commit_seq+1",
    "UPDATE k0_watermarks SET epoch=epoch+1",
    "UPDATE k0_config SET generation=generation+1,config_hash='other'",
])
def test_warm_readiness_rechecks_external_mutation_without_epoch_change(tmp_path, mutation):
    path=tmp_path/'m.db'
    app, provider=build(path)
    add(app)
    assert search(app).data
    before=provider.calls
    with sqlite3.connect(path) as db:
        stamp=db.execute('SELECT * FROM k0_users').fetchall()
    mutate(path, mutation)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT * FROM k0_users').fetchall()==stamp
    with pytest.raises(NotReady): search(app)
    assert provider.calls==before


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'body'])
def test_fts_exact_multiplicity_and_body_are_in_snapshot_digest(tmp_path, mutation):
    path=tmp_path/'m.db'; app,_=build(path,False); add(app)
    assert search(app).data
    table=segments.partition('u')
    sql={'missing':f'DELETE FROM {table}',
         'duplicate':f'INSERT INTO {table} SELECT * FROM {table}',
         'body':f"UPDATE {table} SET lexical_text='forged'"}[mutation]
    mutate(path,sql)
    with pytest.raises(NotReady): search(app)


def test_different_database_same_user_cannot_reuse_valid_counts(tmp_path):
    good,_=build(tmp_path/'good.db'); bad,p=build(tmp_path/'bad.db')
    add(good); add(bad,'alpha other content')
    assert search(good).data and search(bad).data
    mutate(tmp_path/'bad.db',"UPDATE span_vectors SET vector_json='[0,0]'")
    before=p.calls
    assert search(good).data
    with pytest.raises(NotReady): search(bad)
    assert p.calls==before


def test_metadata_update_and_user_isolation_are_current(tmp_path):
    path=tmp_path/'m.db'; app,_=build(path)
    add(app); add(app,'alpha PRIVATE',user='other')
    initial=search(app)
    mutate(path,"UPDATE raw_messages SET source_timestamp=456 WHERE user_id='u'")
    result=search(app)
    assert [r.content for r in result.data]==[r.content for r in initial.data]
    assert all('PRIVATE' not in r.content and r.created_at==456 for r in result.data)


@pytest.mark.parametrize('dense',[False,True])
def test_delete_and_same_content_replay_do_not_resurrect_cached_results(tmp_path,dense):
    app,_=build(tmp_path/'m.db',dense); add(app)
    assert search(app).data
    add(app,'delete "alpha content"',rid='erase')
    assert not search(app).data
    add(app,rid='replay')
    assert not search(app).data


def test_process_restart_does_not_restore_deleted_source(tmp_path):
    path=tmp_path/'m.db'; app,_=build(path,False); add(app)
    assert search(app).data
    add(app,'delete "alpha content"',rid='erase')
    code='''from pathlib import Path
from app.api import create_app
from app.config import Settings
from app.models import SearchRequest
import sys
a=create_app(Settings(db_path=Path(sys.argv[1]),profile_id="cycle2_raw"))
r=a.state.retrieval.search(SearchRequest(user_id="u",query="alpha",top_k=20))
assert not r.data
print("restart_delete_ok")
'''
    result=subprocess.run([sys.executable,'-B','-c',code,str(path)],cwd=Path(__file__).resolve().parents[1],
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='restart_delete_ok'


def test_readiness_cache_bytes_and_entries_bounded_under_user_churn(tmp_path,monkeypatch):
    with segments._counts_lock:
        segments._counts_cache.clear(); segments._counts_cache_bytes=0
    monkeypatch.setattr(segments,'_COUNTS_CACHE_MAX_BYTES',4096)
    app,_=build(tmp_path/'m.db',False)
    for i in range(15):
        add(app,user=f'u{i}')
        assert search(app,user=f'u{i}').data
    assert 0 < segments._counts_cache_bytes <= 4096
    assert len(segments._counts_cache)<15
    # Eviction recomputes the same result and cannot bypass subsequent damage.
    assert search(app,user='u0').data
    mutate(tmp_path/'m.db',"UPDATE raw_messages SET content='forged' WHERE user_id='u0'")
    with pytest.raises(NotReady): search(app,user='u0')


def scalar(vectors,query,dimensions,k,keys):
    q=normalize_vector(query,dimensions)
    scores=[(i,max(-1.,min(1.,math.fsum(a*b for a,b in zip(q,normalize_vector(v,dimensions))))))
            for i,v in enumerate(vectors)]
    return sorted(scores,key=lambda p:(-p[1],*keys[p[0]]))[:k]


@pytest.mark.parametrize('dimensions',[2,17,1024,4096])
def test_default_exact_dense_bitwise_scores_ties_extremes_and_subnormals(dimensions,monkeypatch):
    import app.exact_dense as dense
    # Multiple blocks and the oversized-single-vector fallback are both covered.
    monkeypatch.setattr(dense,'_MAX_BLOCK_COMPONENTS',max(8,dimensions*9))
    rng=random.Random(3700+dimensions)
    vectors=[[rng.uniform(-1,1) for _ in range(dimensions)] for _ in range(39)]
    vectors += [vectors[0],vectors[0], [math.ldexp(v,1000) for v in vectors[1]],
                [math.ldexp(v,-1050) for v in vectors[2]], [5e-324]+[0.]*(dimensions-1),
                [1.]+[0.]*(dimensions-1),[1.,5e-324]+[0.]*(dimensions-2)]
    q=[rng.uniform(-1,1) for _ in range(dimensions)]
    keys=[(len(vectors)-i,str(i)) for i in range(len(vectors))]
    for k in (1,7,len(vectors)):
        expected=scalar(vectors,q,dimensions,k,keys)
        actual=exact_top_scores([json.dumps(v) for v in vectors],q,dimensions,k,keys)
        assert [(i,s.hex()) for i,s in actual]==[(i,s.hex()) for i,s in expected]
    monkeypatch.setattr(dense,'_MAX_BLOCK_COMPONENTS',1)
    assert exact_top_scores([json.dumps(v) for v in vectors],q,dimensions,7,keys)==scalar(vectors,q,dimensions,7,keys)


@pytest.mark.parametrize('bad',['[NaN,0]','[Infinity,0]','[-Infinity,0]','[1e400,0]',
    '[0,0]','[true,0]','["1",0]','[1]','[1,0,0]','null','[1,]'])
def test_corrupt_losing_vector_in_later_block_never_skipped(bad,monkeypatch):
    import app.exact_dense as dense
    monkeypatch.setattr(dense,'_MAX_BLOCK_COMPONENTS',2)
    with pytest.raises(EmbeddingFailure):
        exact_top_scores(['[1,0]',bad],[1,0],2,1,[(0,),(1,)])


def test_default_path_and_numpy_unavailable_fallback_match_scalar(tmp_path,monkeypatch):
    import builtins
    app,_=build(tmp_path/'m.db'); add(app)
    assert not app.state.retrieval.settings.exact_dense_acceleration
    accelerated=search(app).model_dump()
    original=builtins.__import__
    def without_dense(name,*args,**kwargs):
        if name=='app.exact_dense':raise ImportError('simulated optional numpy absence')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',without_dense)
    assert search(app).model_dump()==accelerated


def test_cached_counts_equal_frozen_baseline_on_same_live_snapshot(tmp_path):
    baseline=Path(__file__).resolve().parents[2]/'experiments/037-retrieval-performance/baseline/app/segments.py'
    spec=importlib.util.spec_from_file_location('baseline037_segments',baseline)
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    app,_=build(tmp_path/'m.db')
    add(app,'alpha 中文 e\u0301 👩🏽‍💻\r\n'*200)
    for _ in range(2):
        with app.state.coordinator.transaction(False) as db:
            assert segments.counts(db,'u',app.state.profile)==old.counts(db,'u',app.state.profile)


@pytest.mark.parametrize('left,right',[
    ([('a', 'b')],[('a\", \"b',)]),
    ([(b'alpha',)], [('alpha',)]),
    ([('groupendgroup',)], [('group',),('endgroup',)]),
    ([(None,)], [('None',)]),
    ([('alpha\x00\n中文',)], [('alpha\\x00\\n中文',)]),
])
def test_snapshot_framing_cannot_alias_values_or_row_boundaries(left,right):
    assert segments._counts_key([left])!=segments._counts_key([right])
    assert segments._counts_key([left])==segments._counts_key([left])


@pytest.mark.parametrize('mutation',[
    "UPDATE source_policies SET visibility='hidden' WHERE user_id='u'",
    "UPDATE raw_messages SET privacy_level='private' WHERE user_id='u'",
])
def test_permission_change_after_selection_without_epoch_change_still_fenced(tmp_path,monkeypatch,mutation):
    path=tmp_path/'m.db'; app,_=build(path); add(app)
    assert search(app).data
    original=app.state.coordinator.final_check
    changed=False
    def gate(*args,**kwargs):
        nonlocal changed
        if not changed:
            mutate(path,mutation); changed=True
        return original(*args,**kwargs)
    monkeypatch.setattr(app.state.coordinator,'final_check',gate)
    assert not search(app).data
    assert changed


def test_second_instance_write_and_delete_invalidate_only_matching_user(tmp_path):
    path=tmp_path/'m.db'; first,_=build(path); add(first)
    assert search(first).data
    second,_=build(path)
    add(second,'alpha second instance',rid='new')
    assert {r.content for r in search(first).data}=={'alpha content','alpha second instance'}
    add(second,'delete "alpha content"',rid='erase')
    assert {r.content for r in search(first).data}=={'alpha second instance'}

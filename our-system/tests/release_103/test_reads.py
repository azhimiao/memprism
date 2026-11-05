from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sqlite3
import threading
import time

import pytest
from app.api import create_app
from app.config import Settings
from app.embeddings import EmbeddingBatch, EmbeddingConfig
from app.jobs import NotReady
from app.models import AddRequest, SearchRequest
from app.read_pool import ReadPool


class Provider:
    implementation_kind = 'engineering_double'
    config = EmbeddingConfig(provider='engineering', endpoint='https://invalid.test/embeddings', dimensions=2)
    def embed(self, texts):
        return EmbeddingBatch([[1., 0.] for _ in texts])


def add(app, user='u', request='seed', content='alpha original'):
    return app.state.ingest.add(AddRequest(user_id=user, session_id='s', request_id=request,
        messages=[dict(role='user',content=content)]))


@pytest.fixture
def app(tmp_path):
    a = create_app(Settings(db_path=tmp_path/'memory.sqlite', profile_id='cycle2_dense'),
                   embedding_provider=Provider())
    add(a)
    yield a
    a.state.store._reads.close()
    a.state.coordinator._read_readiness.close()
    with a.state.store._lock:
        a.state.store._writer.close()


def test_pool_reuses_schema_not_results_and_refuses_writes(tmp_path):
    path=tmp_path/'data.sqlite'
    with closing(sqlite3.connect(path)) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE values_now(value)')
        db.execute('INSERT INTO values_now VALUES(1)');db.commit()
    pool=ReadPool(path)
    with pool.snapshot() as first:
        assert first.execute('SELECT value FROM values_now').fetchone()[0]==1
    with closing(sqlite3.connect(path)) as db:
        db.execute('UPDATE values_now SET value=2');db.commit()
    with pool.snapshot() as second:
        assert first is second
        assert second.execute('SELECT value FROM values_now').fetchone()[0]==2
        with pytest.raises(sqlite3.OperationalError):
            second.execute('UPDATE values_now SET value=3')
    assert pool.statistics()['created']==1
    pool.close()
    with pytest.raises(NotReady, match='read_pool_closed'):
        with pool.snapshot(): pass


def test_pool_bound_and_failure_cleanup(tmp_path):
    path=tmp_path/'data.sqlite'
    with closing(sqlite3.connect(path)) as db: db.execute('CREATE TABLE t(x)')
    pool=ReadPool(path,limit=1,timeout=.05)
    with pool.snapshot() as db:
        with pool.snapshot() as nested: assert nested is db
        with ThreadPoolExecutor(1) as threads:
            def attempt():
                with pool.snapshot(): pass
            with pytest.raises(NotReady, match='read_pool_busy'):threads.submit(attempt).result()
    with pytest.raises(ValueError):
        with pool.snapshot(): raise ValueError('synthetic failure')
    assert pool.statistics()['active']==0
    assert pool.statistics()['connections']==0
    with pool.snapshot() as db: assert db.execute('SELECT COUNT(*) FROM t').fetchone()[0]==0
    pool.close()


def test_database_replacement_is_rejected(tmp_path):
    path=tmp_path/'data.sqlite'
    with closing(sqlite3.connect(path)) as db:db.execute('CREATE TABLE t(x)')
    pool=ReadPool(path)
    with pool.snapshot():pass
    # Closing idle handles also makes Windows replacement behavior explicit.
    with pool.condition:
        for db in pool.idle:db.close()
        pool.idle.clear();pool.total=0
    path.rename(tmp_path/'retained.sqlite')
    with closing(sqlite3.connect(path)) as db:db.execute('CREATE TABLE t(x)')
    with pytest.raises(NotReady,match='readiness_database_replaced'):
        with pool.snapshot():pass
    assert pool.statistics()['active']==0
    pool.close()


def test_no_missing_database_is_created(tmp_path):
    path=tmp_path/'missing.sqlite';pool=ReadPool(path)
    with pytest.raises(OSError):
        with pool.snapshot():pass
    assert not path.exists() and pool.statistics()['active']==0
    pool.close()


def test_16_live_read_snapshots_do_not_block_real_add(app,monkeypatch):
    from app import segments
    entered=threading.Barrier(17); release=threading.Event()
    original=segments.search
    def held(db,*args,**kwargs):
        db.execute('SELECT COUNT(*) FROM k0_config').fetchone()
        entered.wait(timeout=15)
        assert release.wait(15)
        return original(db,*args,**kwargs)
    monkeypatch.setattr(segments,'search',held)
    store=app.state.store;profile=app.state.profile
    def search_snapshot():
        return store.search_snapshot(user_id='u',match_query=None,query_vector=[1.,0.],
            version=profile.model_version,dimensions=2,mode='raw_dense',candidate_limit=100,
            lexical_quota=0,dense_quota=100,span=True)
    with ThreadPoolExecutor(17) as threads:
        readers=[threads.submit(search_snapshot) for _ in range(16)]
        try:
            entered.wait(timeout=15)
            writer=threads.submit(add,app,'u','during-search','alpha concurrent')
            writer.result(timeout=5)
            assert store._reads.statistics()['peak_active']>=16
        finally:release.set()
        for f in readers:f.result(timeout=15)
    with closing(sqlite3.connect(store.db_path)) as db:
        assert db.execute('SELECT COUNT(*) FROM add_requests').fetchone()[0]==2
        assert db.execute("SELECT COUNT(*) FROM k0_jobs WHERE state='committed'").fetchone()[0]==2


def test_16_searches_and_16_real_adds_preserve_receipts(app):
    for i in range(16):add(app,user='u'+str(i))
    def search(i):
        response=app.state.retrieval.search(SearchRequest(user_id='u'+str(i),query='alpha',top_k=100))
        assert response is not None
    with ThreadPoolExecutor(32) as threads:
        tasks=[]
        for i in range(16):
            tasks.append(threads.submit(search,i))
            tasks.append(threads.submit(add,app,'u'+str(i),'new','alpha next'))
        for task in tasks:task.result(timeout=40)
    with closing(sqlite3.connect(app.state.store.db_path)) as db:
        assert db.execute('SELECT COUNT(*) FROM add_requests').fetchone()[0]==33
        assert db.execute("SELECT COUNT(*) FROM k0_jobs WHERE state='committed'").fetchone()[0]==33
        assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'


def test_permission_snapshot_sees_external_revocation(app):
    from app.models import EvidenceCandidate
    store=app.state.store
    with closing(sqlite3.connect(store.db_path)) as db:
        row=db.execute('SELECT id,content,message_index FROM raw_messages LIMIT 1').fetchone()
    hit=EvidenceCandidate(id=row[0],source_raw_id=row[0],user_id='u',content=row[1],
                          message_index=row[2],score=1.,created_at=0,deletion_state='active')
    before=store.policy_snapshot('u',[hit])
    with closing(sqlite3.connect(store.db_path)) as db:
        db.execute("UPDATE source_policies SET visibility='hidden' WHERE user_id='u'");db.commit()
    after=store.policy_snapshot('u',[hit])
    assert before!=after


def test_reader_progress_while_writer_gate_is_held(app):
    store=app.state.store
    with store._lock:
        with ThreadPoolExecutor(1) as threads:
            value=threads.submit(store.vector_coverage,'u',app.state.profile.model_version,2,span=True).result(timeout=3)
    assert value[0]>0 and value[0]==value[1]


def test_real_http_16_searches_and_16_adds(tmp_path):
    from fastapi.testclient import TestClient
    a=create_app(Settings(db_path=tmp_path/'http.sqlite',profile_id='cycle2_dense',
                         max_concurrent_requests=32),embedding_provider=Provider())
    for i in range(16):add(a,user='u'+str(i))
    with TestClient(a) as client, ThreadPoolExecutor(32) as threads:
        tasks=[]
        for i in range(16):
            tasks.append(threads.submit(client.post,'/search',json=dict(user_id='u'+str(i),query='alpha',top_k=100)))
            tasks.append(threads.submit(client.post,'/add',json=dict(user_id='u'+str(i),request_id='new',
                session_id='s',messages=[dict(role='user',content='alpha updated')])))
        assert [f.result(timeout=40).status_code for f in tasks]==[200]*32
    with closing(sqlite3.connect(a.state.store.db_path)) as db:
        assert db.execute('SELECT COUNT(*) FROM add_requests').fetchone()[0]==32
        assert db.execute("SELECT COUNT(*) FROM k0_jobs WHERE state='committed'").fetchone()[0]==32
    a.state.coordinator._read_readiness.close()
    with a.state.store._lock:a.state.store._writer.close()

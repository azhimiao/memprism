"""076: real SQLite, adversarial invalidation and concurrent lease progress."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sqlite3
import threading

import pytest

from app import jobs, segments
from app.api import create_app, _HealthReadiness
from app.config import Settings
from app.embeddings import EmbeddingBatch, EmbeddingConfig
from app.jobs import NotReady
from app.models import AddRequest


class Provider:
    implementation_kind = 'engineering_double'
    config = EmbeddingConfig(provider='engineering', endpoint='https://invalid.test/embeddings', dimensions=2)
    def __init__(self): self.calls = 0
    def embed(self, texts):
        self.calls += 1
        return EmbeddingBatch([[1., 0.] for _ in texts])


def add(app, rid='r', user='u', count=1):
    return app.state.ingest.add(AddRequest(user_id=user, session_id='s', request_id=rid,
        messages=[dict(role='user', content='alpha source '+str(i)) for i in range(count)]))


@pytest.fixture
def app(tmp_path):
    a = create_app(Settings(db_path=tmp_path/'memory.db', profile_id='cycle2_dense'), embedding_provider=Provider())
    add(a)
    yield a
    with a.state.store._lock:
        a.state.store._writer.close()


def mutation(app, sql, args=()):
    with closing(sqlite3.connect(app.state.store.db_path)) as db:
        with db: db.execute(sql, args)


@pytest.mark.parametrize('sql', [
    "UPDATE raw_messages SET content='tampered' WHERE user_id='u'",
    "UPDATE span_vectors SET vector_json='[0,0]' WHERE user_id='u'",
    "DELETE FROM source_spans WHERE user_id='u'",
])
def test_external_actual_data_edits_invalidate_cached_proof_before_model(app, sql):
    c = app.state.coordinator
    c.prepare_readiness('u')
    provider = app.state.ingest.embedding_provider.provider
    calls = provider.calls
    mutation(app, sql)
    with pytest.raises(NotReady): c.reserve('u', 'next', 'hash')
    with pytest.raises(NotReady): c.require_ready('u')
    assert provider.calls == calls
    with c.transaction(False) as db:
        assert not db.execute("SELECT 1 FROM k0_jobs WHERE request_id='next'").fetchone()


def test_own_control_writes_keep_summary_but_external_control_write_reaudits(app, monkeypatch):
    c = app.state.coordinator
    c.prepare_readiness('u')
    original = jobs.route_counts_on
    audits = []
    def counted(*args):
        audits.append(args[1]); return original(*args)
    monkeypatch.setattr(jobs, 'route_counts_on', counted)
    lease = c.reserve('v', 'pending', 'h')
    audits.clear()
    for _ in range(5):
        c.renew(lease)
        c.prepare_readiness('u')
    assert audits == []
    mutation(app, "UPDATE k0_jobs SET expires_at=expires_at+1 WHERE user_id='v'")
    c.prepare_readiness('u')
    assert audits == ['u']


def test_same_connection_repeated_sql_and_rollback_never_leave_false_green(app):
    c, writer = app.state.coordinator, app.state.store._writer
    c.prepare_readiness('u')
    with pytest.raises(RuntimeError, match='rollback'):
        with c.transaction() as db:
            for value in ('bad1', 'bad2'):
                before = writer.stamp(db, 'u', c.profile)
                db.execute('UPDATE raw_messages SET content=? WHERE user_id=?', (value, 'u'))
                assert writer.stamp(db, 'u', c.profile) != before
                assert writer.get(db, 'u', c.profile) is None
                # Even an explicitly remembered invalid count must be discarded.
                writer.remember(db, 'u', c.profile, {'active': 999})
            raise RuntimeError('rollback')
    with c.transaction() as db:
        assert writer.get(db, 'u', c.profile) is None
    c.prepare_readiness('u')
    c.require_ready('u')


def test_expensive_read_snapshot_releases_gate_so_renewal_can_finish(app, monkeypatch):
    c = app.state.coordinator
    lease = c.reserve('v', 'pending', 'h')
    with c.transaction() as db: app.state.store._writer.invalidate_all()
    original = jobs.route_counts_on
    entered, release = threading.Event(), threading.Event()
    def slow(db, user, profile):
        if user == 'u':
            entered.set()
            assert release.wait(5)
        return original(db, user, profile)
    monkeypatch.setattr(jobs, 'route_counts_on', slow)
    with ThreadPoolExecutor(max_workers=2) as pool:
        audit = pool.submit(c.prepare_readiness, 'u')
        assert entered.wait(2)
        try: pool.submit(c.renew, lease).result(1)
        finally: release.set()
        audit.result(3)


def test_external_commit_during_snapshot_is_never_accepted(app, monkeypatch):
    c = app.state.coordinator
    with c.transaction(): app.state.store._writer.invalidate_all()
    original = jobs.route_counts_on
    attempts = []
    def race(db, user, profile):
        result = original(db, user, profile)
        attempts.append(1)
        if len(attempts) == 1:
            mutation(app, "UPDATE raw_messages SET content='tampered' WHERE user_id='u'")
        return result
    monkeypatch.setattr(jobs, 'route_counts_on', race)
    c.prepare_readiness('u')
    assert len(attempts) == 2
    with c.transaction() as db:
        with pytest.raises(NotReady): c.ready_on(db, 'u')


def test_cache_eviction_has_no_aba_and_health_handles_more_than_lru(app, monkeypatch):
    writer, c = app.state.store._writer, app.state.coordinator
    monkeypatch.setattr(writer, 'LIMIT', 2)
    for user in ('v', 'w', 'x'): add(app, user=user)
    c.health_ready()
    assert len(writer.summaries) <= 2 and len(writer.users) <= 2
    assert all(isinstance(key, bytes) and len(key) == 32 for key in writer.summaries)
    mutation(app, "UPDATE span_vectors SET vector_json='[0,0]' WHERE user_id='u'")
    with pytest.raises(NotReady): c.health_ready()


def test_new_user_publication_preserves_other_user_proof_unless_schema_changes(app):
    c, w = app.state.coordinator, app.state.store._writer
    add(app, user='v')  # Create its partition before taking the observation.
    c.prepare_readiness('u')
    with c.transaction() as db: old = w.stamp(db, 'u', c.profile)
    add(app, 'second', user='v')
    with c.transaction() as db:
        assert w.stamp(db, 'u', c.profile) == old
        assert w.get(db, 'u', c.profile) is not None
    # Own FTS partitions for a new, different user have a scoped DDL revision.
    add(app, user='brand-new-user')
    with c.transaction() as db:
        assert w.stamp(db, 'u', c.profile) == old
        assert w.get(db, 'u', c.profile) is not None


def test_health_reuses_actual_proof_during_own_lease_commits_but_not_tampering(app, monkeypatch):
    c = app.state.coordinator
    c.health_ready()
    original = jobs.route_counts_on
    calls = []
    def counted(*args): calls.append(args[1]); return original(*args)
    monkeypatch.setattr(jobs, 'route_counts_on', counted)
    lease = c.reserve('v', 'pending', 'h')
    calls.clear()
    for _ in range(5): c.renew(lease); c.health_ready()
    assert calls == []
    mutation(app, "UPDATE raw_messages SET content='tampered' WHERE user_id='u'")
    h = _HealthReadiness(c, app.state.store.db_path, c.profile)
    with pytest.raises(NotReady): h._evaluate()
    h.close()


def test_sixteen_adds_preserve_atomic_receipts_and_current_routes(app):
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda i: add(app, 'parallel', user='user'+str(i), count=8), range(16)))
    assert all(result.success for result in results)
    c = app.state.coordinator
    c.health_ready()
    with c.transaction(False) as db:
        assert db.execute("SELECT COUNT(*) FROM k0_jobs WHERE state='committed'").fetchone()[0] == 17
        assert db.execute('SELECT COUNT(*) FROM add_requests').fetchone()[0] == 17
        assert db.execute('SELECT COUNT(*) FROM raw_messages').fetchone()[0] == 129
        assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'


def test_hash_frames_types_lengths_and_vector_bytes_without_collision():
    def key(row): return segments._counts_key([[row]])
    values = [(None,), ('',), (b'',), (1,), ('1',), (1.,), ('a', 'bc'), ('ab', 'c'),
              ('[1,0]',), ('[1,0 ]',), ('\x00',), ('你好',)]
    assert len({key(row) for row in values}) == len(values)
    assert key(('x'*10000,)) == key(('x'*10000,))


def test_external_fts_deletion_invalidates_cached_actual_proof(app):
    c = app.state.coordinator
    c.prepare_readiness('u')
    with c.transaction(False) as db:
        table = db.execute("SELECT table_name FROM span_fts_partitions WHERE user_id='u'").fetchone()[0]
    assert table.startswith('span_fts_') and table.isidentifier()
    mutation(app, 'DELETE FROM '+table)
    with pytest.raises(NotReady): c.reserve('u','after-delete','hash')


def test_database_identity_change_reopens_connection_and_invalidates_proof(app,tmp_path):
    c, store = app.state.coordinator, app.state.store
    other = tmp_path/'replacement.db'
    # A real different SQLite file, containing an out-of-band content edit.
    with closing(sqlite3.connect(store.db_path)) as source, closing(sqlite3.connect(other)) as dest:
        source.backup(dest)
        with dest: dest.execute("UPDATE raw_messages SET content='tampered' WHERE user_id='u'")
    old_identity = store._writer.identity
    store.db_path = other
    with pytest.raises(NotReady): c.reserve('u','after-replacement','hash')
    assert store._writer.identity != old_identity


def test_publication_fence_rollback_discards_postpublication_summary(app,monkeypatch):
    c = app.state.coordinator
    original = c.fence_on
    def fail_after_writes(db,lease,*,published=False):
        original(db,lease,published=published)
        if published: raise NotReady('lease_lost')
    monkeypatch.setattr(c,'fence_on',fail_after_writes)
    with pytest.raises(NotReady,match='lease_lost'): add(app,'must-rollback')
    c.prepare_readiness('u')
    with c.transaction() as db:
        assert app.state.store._writer.get(db,'u',c.profile)['active'] == 1
        assert db.execute('SELECT COUNT(*) FROM raw_messages').fetchone()[0] == 1
        assert db.execute('SELECT COUNT(*) FROM add_requests').fetchone()[0] == 1

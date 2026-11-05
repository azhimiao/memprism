"""Real SQLite and concurrency checks; synthetic content, no external calls."""
import asyncio
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app.api import _HealthReadiness, create_app
from app.config import Settings
from app.jobs import NotReady


class Coordinator:
    def __init__(self, path):
        self.path = path
        self.calls = 0
        self.control_calls = 0
        self.enter = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.change_during_audit = False

    def control_ready_on(self, db):
        self.control_calls += 1
        if db.execute('SELECT valid FROM control').fetchone()[0] != 1:
            raise NotReady('control_not_ready')

    def ready_on(self, db, user):
        self.calls += 1
        self.enter.set()
        if not self.release.wait(3):
            raise RuntimeError('test_release_timed_out')
        if self.change_during_audit:
            self.change_during_audit = False
            edit(self.path, "UPDATE source SET body='after-snapshot'")
        if db.execute('SELECT body FROM source WHERE user_id=?', (user,)).fetchone()[0] != 'valid':
            raise NotReady('incomplete_fts_route')


def edit(path, statement):
    with closing(sqlite3.connect(path)) as db:
        with db:
            db.execute(statement)


def create(path):
    with closing(sqlite3.connect(path)) as db:
        with db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE raw_messages(user_id TEXT)')
            db.execute("INSERT INTO raw_messages VALUES ('u')")
            db.execute('CREATE TABLE control(valid INTEGER)')
            db.execute('INSERT INTO control VALUES (1)')
            db.execute('CREATE TABLE source(user_id TEXT,body TEXT)')
            db.execute("INSERT INTO source VALUES ('u','valid')")


@pytest.fixture
def rig(tmp_path):
    path = tmp_path/'health.sqlite'
    create(path)
    c = Coordinator(path)
    h = _HealthReadiness(c, path, SimpleNamespace(profile_id='synthetic', config_hash='hash'), wait_seconds=0.3)
    yield h, c, path
    c.release.set()
    future = h._future
    if future is not None:
        future.result(timeout=4)
    h.close()


def check(h):
    return asyncio.run(h.check())


def test_unchanged_database_reuses_full_audit_but_checks_control(rig):
    h, c, _ = rig
    assert check(h)['status'] == 'ok'
    assert check(h)['status'] == 'ok'
    assert c.calls == 1 and c.control_calls == 2


def test_external_edit_without_watermark_updates_invalidates_positive(rig):
    h, c, path = rig
    assert check(h)['status'] == 'ok'
    edit(path, "UPDATE source SET body='tampered'")
    result = check(h)
    assert result['status'] == 'not_ready' and result['reason'] == 'incomplete_fts_route'
    assert c.calls == 2
    edit(path, "UPDATE source SET body='valid'")
    assert check(h)['status'] == 'ok'
    assert c.calls == 3


def test_new_user_commit_invalidates_positive(rig):
    h, c, path = rig
    assert check(h)['status'] == 'ok'
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO raw_messages VALUES ('v')")
        db.execute("INSERT INTO source VALUES ('v','valid')")
    assert check(h)['status'] == 'ok'
    assert c.calls == 3


def test_control_failure_does_not_return_cached_ok(rig):
    h, _, path = rig
    assert check(h)['status'] == 'ok'
    edit(path, 'UPDATE control SET valid=0')
    result = check(h)
    assert result['control_ready'] is False and result['data_route_ready'] is False


def test_one_worker_for_concurrent_requests_and_timeouts_do_not_cancel_it(rig):
    h, c, _ = rig
    h.wait_seconds = 0.03
    c.release.clear()
    async def many():
        return await asyncio.gather(*(h.check() for _ in range(16)))
    results = asyncio.run(many())
    assert all(r['reason'] == 'health_check_in_progress' for r in results)
    assert c.calls == 1 and not h._future.done() and not h._future.cancelled()
    assert check(h)['reason'] == 'health_check_in_progress' and c.calls == 1
    # Timed-out requests must not accumulate completion callbacks on the long
    # running shared Future. Cancelled clients also do not cancel this Future.
    assert not h._future._done_callbacks
    c.release.set()
    assert h._future.result(timeout=3)['status'] == 'ok'
    assert check(h)['status'] == 'ok' and c.calls == 1


def test_cancelled_client_does_not_cancel_shared_audit(rig):
    h, c, _ = rig
    c.release.clear()
    async def cancel():
        task = asyncio.create_task(h.check())
        while not c.enter.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(cancel())
    assert not h._future.cancelled() and c.calls == 1
    c.release.set()
    assert h._future.result(timeout=3)['status'] == 'ok'


def test_commit_during_full_audit_is_not_reported_current(rig):
    h, c, _ = rig
    c.change_during_audit = True
    result = check(h)
    assert result['reason'] == 'health_snapshot_changed' and not result['data_route_ready']
    assert h._validated is None


def test_main_database_replacement_invalidates_old_connection(rig):
    h, c, path = rig
    assert check(h)['status'] == 'ok'
    # Use a fresh DELETE-journal replacement, avoiding test-only WAL identity
    # confusion from renaming an open WAL database's sidecars.
    # Close the observer before replacing the filesystem object: replacing only
    # a WAL main file while its old sidecars remain open is unsupported SQLite
    # administration, and is not how the guarded production bind is migrated.
    h._db.close()
    replacement = path.with_name('replacement.sqlite')
    create(replacement)
    with closing(sqlite3.connect(replacement)) as db:
        with db:
            db.execute('PRAGMA journal_mode=DELETE')
            db.execute("UPDATE source SET body='tampered'")
    replacement.replace(path)
    result = check(h)
    assert not result['data_route_ready'] and c.calls == 2


def test_health_observer_cannot_write(rig):
    h, _, _ = rig
    assert check(h)['status'] == 'ok'
    with pytest.raises(sqlite3.OperationalError):
        h._db.execute('DELETE FROM source')


def test_shutdown_during_audit_waits_to_close_without_spawning_new_worker(rig):
    h, c, _ = rig
    c.release.clear()
    assert check(h)['reason'] == 'health_check_in_progress'
    h.close()
    assert check(h)['reason'] == 'health_check_stopped'
    c.release.set()
    h._future.result(timeout=3)
    assert h._db is None and c.calls == 1


def test_unexpected_errors_do_not_expose_exception_body(rig, monkeypatch):
    h, c, _ = rig
    def bad(db):
        raise RuntimeError('private-input-body-sentinel')
    monkeypatch.setattr(c, 'control_ready_on', bad)
    result = check(h)
    assert result['reason'] == 'health_check_failed'
    assert 'private-input-body-sentinel' not in json.dumps(result)


def test_research_health_keeps_contract(tmp_path):
    app = create_app(Settings(db_path=tmp_path/'research.sqlite'))
    with TestClient(app) as client:
        assert client.get('/health').json() == {'status': 'ok'}


def test_asgi_health_responds_pending_while_unauthenticated_search_remains_responsive(tmp_path, monkeypatch):
    monkeypatch.setenv('AML_SERVICE_KEY', 'synthetic-test-key')
    app = create_app(Settings(db_path=tmp_path/'formal.sqlite', profile_id='dense_sync',
                              auth_mode='bearer', service_key_env='AML_SERVICE_KEY'))
    guard = app.state.health_readiness
    guard.wait_seconds = 0.03
    entered, release = threading.Event(), threading.Event()
    original = guard._evaluate
    def blocked():
        entered.set()
        if not release.wait(3):
            raise RuntimeError('test_timeout')
        return original()
    monkeypatch.setattr(guard, '_evaluate', blocked)
    try:
        with TestClient(app) as client:
            response = client.get('/health')
            assert entered.is_set() and response.json()['reason'] == 'health_check_in_progress'
            assert client.post('/search', json={}).status_code == 401
            release.set()
            guard._future.result(timeout=3)
    finally:
        release.set()

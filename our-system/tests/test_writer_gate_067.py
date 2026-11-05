"""Real SQLite/thread checks for the operational write admission repair."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from app.api import create_app
from app.config import Settings
from app.jobs import NotReady
from app import jobs
from app.diagnostics import _exception_fields


@pytest.fixture
def coordinator(tmp_path):
    app = create_app(Settings(db_path=tmp_path/'writer.db', profile_id='cycle2_raw'))
    return app.state.coordinator


def test_writer_waits_outside_sqlite_but_reader_proceeds(coordinator, monkeypatch):
    entered = threading.Event()
    connected = threading.Event()
    original = coordinator.store._connect
    def connect(*args, **kwargs):
        connected.set()
        return original(*args, **kwargs)
    monkeypatch.setattr(coordinator.store, '_connect', connect)
    def writer():
        entered.set()
        with coordinator.transaction() as db:
            return db.execute('SELECT 1').fetchone()[0]
    with ThreadPoolExecutor(max_workers=1) as pool:
        with coordinator.store._lock:
            future = pool.submit(writer)
            assert entered.wait(1)
            assert not connected.wait(0.08)
            with coordinator.transaction(False) as db:
                assert db.execute('SELECT 1').fetchone()[0] == 1
        assert future.result(timeout=3) == 1


def test_writer_gate_timeout_does_not_steal_or_leak_lock(coordinator):
    def blocked():
        with pytest.raises(NotReady, match='preparation_busy'):
            with coordinator.transaction(timeout_seconds=0.03):
                pytest.fail('unexpected admission')
    with ThreadPoolExecutor(max_workers=1) as pool:
        with coordinator.store._lock:
            pool.submit(blocked).result(timeout=2)
        with coordinator.transaction() as db:
            assert db.execute('SELECT 1').fetchone()[0] == 1


def test_sqlite_error_keeps_only_safe_code_and_releases_gate(coordinator):
    with pytest.raises(NotReady, match='control_not_ready') as caught:
        with coordinator.transaction() as db:
            db.execute('SELECT private_value FROM missing_secret_table')
    assert caught.value.sqlite_errorcode == sqlite3.SQLITE_ERROR
    fields = _exception_fields(caught.value)
    assert fields['sqlite_errorcode'] == sqlite3.SQLITE_ERROR
    assert 'private_value' not in repr(fields) and 'missing_secret_table' not in repr(fields)
    with ThreadPoolExecutor(max_workers=1) as pool:
        def next_writer():
            with coordinator.transaction():
                return True
        assert pool.submit(next_writer).result(timeout=2)


def test_read_only_draft_does_not_reserve_a_writer(coordinator, monkeypatch):
    lease = coordinator.reserve('u', 'r', 'payload')
    statements = []
    original = coordinator.store._connect
    def connect(*args, **kwargs):
        db = original(*args, **kwargs)
        db.set_trace_callback(statements.append)
        return db
    monkeypatch.setattr(coordinator.store, '_connect', connect)
    from app.models import RawMessage
    messages = [RawMessage(id='raw-r', user_id='u', request_id='r', session_id='s', message_index=0,
                           role='user', content='alpha', source_timestamp=None, recorded_at=1)]
    assert coordinator.draft(lease, messages, SimpleNamespace(mutations=[])) == messages
    assert 'BEGIN' in statements
    assert 'BEGIN IMMEDIATE' not in statements


def test_same_user_waiter_does_not_poll_for_writer(coordinator, monkeypatch):
    coordinator.reserve('u', 'first', 'hash-first')
    monkeypatch.setattr(jobs, 'USER_PREPARATION_WAIT', 0.13)
    writes = []
    original = coordinator.transaction
    @contextmanager
    def traced(write=True, **kwargs):
        writes.append(write)
        with original(write, **kwargs) as db:
            yield db
    monkeypatch.setattr(coordinator, 'transaction', traced)
    with pytest.raises(NotReady, match='preparation_busy'):
        coordinator.reserve('u', 'second', 'hash-second')
    assert len(writes) >= 2
    assert all(write is False for write in writes)


def test_preparation_wait_preserves_payload_conflict(coordinator):
    from app.jobs import PayloadConflict
    coordinator.reserve('u', 'first', 'hash-first')
    with pytest.raises(PayloadConflict):
        coordinator.reserve('u', 'first', 'different-payload')

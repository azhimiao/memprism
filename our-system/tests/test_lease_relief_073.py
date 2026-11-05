"""073: actual threads/SQLite and adversarial fences, no external requests."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
import time

import pytest

from app.api import create_app
from app.config import Settings
from app.jobs import NotReady
from app.models import RawMessage
from app.writer_gate import WriterGate


def test_renewal_precedes_queued_writers_and_cancelled_waiters_do_not_block():
    gate = WriterGate()
    order = []
    normal_queued = threading.Event()
    priority_queued = threading.Event()
    def worker(tag, priority, queued):
        queued.set()
        assert gate.acquire(timeout=2, priority=priority)
        try:
            order.append(tag)
        finally:
            gate.release()
    with ThreadPoolExecutor(max_workers=3) as pool:
        with gate:
            normal = pool.submit(worker, 'normal', False, normal_queued)
            assert normal_queued.wait(1)
            # Observe the gate's queue, not a scheduler-dependent fixed sleep.
            deadline = time.monotonic() + 1
            while not gate._normal and time.monotonic() < deadline:
                time.sleep(.001)
            priority = pool.submit(worker, 'renew', True, priority_queued)
            assert priority_queued.wait(1)
            deadline = time.monotonic() + 1
            while not gate._priority and time.monotonic() < deadline:
                time.sleep(.001)
            expired = pool.submit(gate.acquire, True, .02)
            assert expired.result(1) is False
        priority.result(2)
        normal.result(2)
    assert order == ['renew', 'normal']
    with gate:
        with gate:
            assert gate.acquire(blocking=False)
            gate.release()


def test_commit_admission_is_bounded_and_does_not_write_after_timeout(tmp_path):
    app = create_app(Settings(db_path=tmp_path/'c.db', profile_id='cycle2_raw'))
    coordinator = app.state.coordinator
    coordinator.lock_timeout_seconds = .04
    lease = coordinator.reserve('u', 'r', 'hash')
    message = RawMessage(id='raw-r', user_id='u', request_id='r', session_id='s', message_index=0,
                         role='user', content='alpha', source_timestamp=None, recorded_at=1)
    def commit():
        with pytest.raises(NotReady, match='preparation_busy'):
            app.state.store.commit_add([message], payload_hash='hash', response={},
                                       coordinator=coordinator, lease=lease)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with app.state.store._lock:
            pool.submit(commit).result(1)
    with coordinator.transaction(False) as db:
        assert db.execute('SELECT COUNT(*) FROM raw_messages').fetchone()[0] == 0
    coordinator.renew(lease)


def test_renewal_original_sqlite_code_keeps_request_correlation_without_body(tmp_path, monkeypatch, capsys):
    from app import diagnostics, jobs
    coordinator = create_app(Settings(db_path=tmp_path/'d.db', profile_id='cycle2_raw')).state.coordinator
    lease = coordinator.reserve('u', 'r', 'hash')
    def fail(_):
        exc = NotReady('SECRET_DATABASE_TEXT')
        exc.sqlite_errorcode = 5
        raise exc
    monkeypatch.setattr(coordinator, 'renew', fail)
    monkeypatch.setattr(jobs, 'RENEW_INTERVAL', .005)
    token = diagnostics.begin('/add')
    try:
        with jobs.LeaseHeartbeat(coordinator, lease) as heartbeat:
            assert heartbeat.lost.wait(1)
            with pytest.raises(NotReady, match='lease_renewal_failed'):
                heartbeat.check()
    finally:
        diagnostics.end(token)
    text = capsys.readouterr().err
    assert '"sqlite_errorcode":5' in text and '"request_id":' in text
    assert 'SECRET_DATABASE_TEXT' not in text and '"endpoint":"/add"' in text


def test_real_sqlite_lease_survives_fifo_writer_backlog(tmp_path, monkeypatch):
    from app import jobs
    monkeypatch.setattr(jobs, 'LEASE_TTL', .8)
    monkeypatch.setattr(jobs, 'RENEW_INTERVAL', .05)
    coordinator = create_app(Settings(db_path=tmp_path/'backlog.db', profile_id='cycle2_raw')).state.coordinator
    coordinator.lock_timeout_seconds = 3
    lease = coordinator.reserve('u', 'r', 'hash')
    def writer():
        with coordinator.transaction() as db:
            assert db.execute('SELECT 1').fetchone()[0] == 1
            time.sleep(.06)
    with jobs.LeaseHeartbeat(coordinator, lease) as heartbeat:
        with ThreadPoolExecutor(max_workers=18) as pool:
            futures = [pool.submit(writer) for _ in range(18)]
            for future in futures:
                future.result(4)
        heartbeat.check()
        with coordinator.transaction(False) as db:
            coordinator.fence_on(db, lease)

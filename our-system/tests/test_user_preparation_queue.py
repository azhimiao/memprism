"""Cycle2 preparation queue races. Synthetic vectors, real SQLite, no network.

The second request must reach reserve's actual wait point before assertions;
fixed sleeps are never used to infer that a contender has been scheduled.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import socket
import sqlite3
import threading
import time as real_time

import pytest

from app import jobs
from app.config import Settings
from test_k0 import Provider, add, app_at, connected, search

GUARD = 15
FIRST = 'queue alpha evidence'
SECOND = 'queue beta evidence'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Allow Windows' internal socketpair, reject all other IP connects."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_pair = socket.socketpair
    local = threading.local()

    def socketpair(*args, **kwargs):
        local.socketpair = True
        try:
            return original_pair(*args, **kwargs)
        finally:
            local.socketpair = False

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not getattr(local, 'socketpair', False):
            raise AssertionError('network_forbidden_in_queue_test')
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not getattr(local, 'socketpair', False):
            raise AssertionError('network_forbidden_in_queue_test')
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket, 'socketpair', socketpair)
    monkeypatch.setattr(socket.socket, 'connect', connect)
    monkeypatch.setattr(socket.socket, 'connect_ex', connect_ex)


class ControlledProvider(Provider):
    def __init__(self, *, block_text=None, entered=None, release=None, progress=None):
        super().__init__(action=self._on_call)
        self.block_text = block_text
        self.entered = entered if entered is not None else threading.Event()
        self.release = release if release is not None else threading.Event()
        self.progress = progress

    def _on_call(self):
        if self.progress is not None:
            self.progress.set()
        if self.block_text is not None and self.block_text in self.calls[-1]:
            self.entered.set()
            assert self.release.wait(GUARD), 'blocked provider was not released'


class ReserveClock:
    """Only the watched reserve call sees fake time or an explicit wait gate.

    Assign to jobs.time, NOT time.sleep: TestClient/heartbeat/asyncio keep their
    real clocks. No SQL, lease result, or provider decision is mocked.
    """
    def __init__(self, *, fake=False, hold=False, waiting=None, progress=None, release=None):
        self.fake, self.hold = fake, hold
        self.waiting = waiting if waiting is not None else threading.Event()
        self.progress = progress if progress is not None else threading.Event()
        self.release = release if release is not None else threading.Event()
        self.local = threading.local()
        self.elapsed = 0.0
        self.sleeps = []

    def __getattr__(self, name):
        return getattr(real_time, name)

    def monotonic(self):
        if self.fake and getattr(self.local, 'active', False):
            return 1000.0 + self.elapsed
        return real_time.monotonic()

    def sleep(self, seconds):
        if not getattr(self.local, 'active', False):
            return real_time.sleep(seconds)
        self.sleeps.append(seconds)
        self.waiting.set()
        self.progress.set()
        if self.hold:
            assert self.release.wait(GUARD), 'reserve wait gate was not released'
        if self.fake:
            # Avoid roundoff stalls at min(0.05, deadline-now).
            self.elapsed += max(seconds, 0.001)
        else:
            real_time.sleep(seconds)

    def watch(self, coordinator):
        reserve = coordinator.reserve
        def watched(*args, **kwargs):
            self.local.active = True
            try:
                return reserve(*args, **kwargs)
            finally:
                self.local.active = False
        coordinator.reserve = watched


def _pair(tmp_path, *, progress=None, profile='cycle2_dense'):
    path = tmp_path / 'queue.sqlite'
    provider_a = ControlledProvider(block_text=FIRST)
    provider_b = ControlledProvider(progress=progress)
    first = app_at(path, provider_a, profile)
    second = app_at(path, provider_b, profile)
    assert first.state.coordinator is not second.state.coordinator
    assert first.state.profile.config_hash == second.state.profile.config_hash
    return path, first, second, provider_a, provider_b


@contextmanager
def _pool(*gates):
    with ThreadPoolExecutor(max_workers=3) as pool:
        try:
            yield pool
        finally:
            for gate in gates:
                gate.set()


def _install_clock(monkeypatch, app, clock):
    clock.watch(app.state.coordinator)
    monkeypatch.setattr(jobs, 'time', clock)


def _visible(app, expected, *, user='u'):
    response = search(app, 'queue', user=user)
    assert response.status_code == 200, response.json()
    assert {r['content'] for r in response.json()['data']} == set(expected)


def test_cycle2_same_user_different_requests_wait_before_dispatch_and_publish(tmp_path, monkeypatch):
    progress = threading.Event()
    clock = ReserveClock(hold=True, progress=progress)
    path, first, second, pa, pb = _pair(tmp_path, progress=progress)
    _install_clock(monkeypatch, second, clock)
    with _pool(pa.release, clock.release) as pool:
        a = pool.submit(add, first, 'first', (FIRST,))
        assert pa.entered.wait(GUARD)
        b = pool.submit(add, second, 'second', (SECOND,))
        assert progress.wait(GUARD), 'second request never reached wait or provider'
        assert clock.waiting.is_set(), 'second request dispatched instead of waiting for the user lease'
        assert pb.calls == [] and not b.done()
        with connected(path) as db:
            assert db.execute("SELECT request_id FROM k0_jobs WHERE user_id='u' AND state='preparing'").fetchall() == [('first',)]
            assert db.execute('SELECT request_id FROM k0_attempts').fetchall() == [('first',)]
        pa.release.set()
        assert a.result(timeout=GUARD).status_code == 200
        clock.release.set()
        assert b.result(timeout=GUARD).status_code == 200
    assert pa.calls == [[FIRST]] and pb.calls == [[SECOND]]
    with connected(path) as db:
        assert db.execute("SELECT request_id,state FROM k0_jobs ORDER BY request_id").fetchall() == [('first', 'committed'), ('second', 'committed')]
        assert db.execute('SELECT COUNT(*) FROM k0_receipts').fetchone()[0] == 2
    # First post-success Search, no retry/poll/rebuild. Both app instances see both.
    _visible(first, {FIRST, SECOND})
    _visible(second, {FIRST, SECOND})


def test_cycle2_other_user_can_publish_while_first_provider_is_blocked(tmp_path):
    _, first, second, pa, pb = _pair(tmp_path)
    with _pool(pa.release) as pool:
        a = pool.submit(add, first, 'first', (FIRST,), 'u')
        assert pa.entered.wait(GUARD)
        b = pool.submit(add, second, 'second', (SECOND,), 'other')
        assert b.result(timeout=GUARD).status_code == 200
        assert not a.done() and not pa.release.is_set()
        assert pb.calls == [[SECOND]]
        _visible(second, {SECOND}, user='other')
        pa.release.set()
        assert a.result(timeout=GUARD).status_code == 200
    _visible(first, {FIRST}, user='u')


def test_pure_erase_preempts_and_rejects_stale_publication(tmp_path, monkeypatch):
    path, first, second, pa, pb = _pair(tmp_path)
    seed = 'queue seed to erase'
    assert add(second, 'seed', (seed,)).status_code == 200
    before = list(pb.calls)
    clock = ReserveClock(fake=True)
    _install_clock(monkeypatch, second, clock)
    monkeypatch.setattr(jobs, 'USER_PREPARATION_WAIT', 0.15, raising=False)
    with _pool(pa.release) as pool:
        pending = pool.submit(add, first, 'first', (FIRST,))
        assert pa.entered.wait(GUARD)
        erased = add(second, 'erase', (f'delete "{seed}"',))
        assert erased.status_code == 200 and not pending.done()
        assert pb.calls == before and clock.sleeps == []
        pa.release.set()
        stale = pending.result(timeout=GUARD)
        assert stale.status_code == 503
        assert stale.json()['detail'] in {'source_epoch_changed', 'lease_lost'}
    with connected(path) as db:
        assert db.execute("SELECT COUNT(*) FROM raw_messages WHERE deletion_state='active'").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM add_requests WHERE request_id='first'").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM k0_receipts WHERE request_id='first'").fetchone()[0] == 0
        assert db.execute("SELECT state FROM k0_jobs WHERE request_id='first'").fetchone()[0] == 'failed'
    _visible(second, set())


@pytest.mark.parametrize('same_request', [False, True], ids=['user-queue', 'same-request'])
def test_wait_deadline_uses_its_own_constant_without_dispatch(tmp_path, monkeypatch, same_request):
    path, first, second, pa, pb = _pair(tmp_path)
    clock = ReserveClock(fake=True)
    _install_clock(monkeypatch, second, clock)
    monkeypatch.setattr(jobs, 'USER_PREPARATION_WAIT', 0.6 if same_request else 0.15, raising=False)
    monkeypatch.setattr(jobs, 'COMPETITION_WAIT', 0.15 if same_request else 0.6)
    with _pool(pa.release) as pool:
        pending = pool.submit(add, first, 'first', (FIRST,))
        assert pa.entered.wait(GUARD)
        response = add(second, 'first' if same_request else 'second', (FIRST,) if same_request else (SECOND,))
        assert response.status_code == 503 and response.json()['detail'] == 'preparation_busy'
        assert response.headers['Retry-After'] == '1'
        assert 0.15 - 1e-9 <= clock.elapsed < 0.20, 'wrong competition/user deadline selected'
        assert pb.calls == [] and not pending.done()
        with connected(path) as db:
            assert db.execute('SELECT COUNT(*) FROM k0_attempts').fetchone()[0] == 1
            assert db.execute('SELECT COUNT(*) FROM k0_jobs').fetchone()[0] == 1
        pa.release.set()
        assert pending.result(timeout=GUARD).status_code == 200
    _visible(second, {FIRST})


@pytest.mark.parametrize('provenance', ['receipt', 'legacy', 'failed-job'])
def test_other_request_payload_conflict_precedes_user_queue(tmp_path, monkeypatch, provenance):
    path, first, second, pa, pb = _pair(tmp_path)
    if provenance == 'failed-job':
        pb.failure = 'embedding_timeout'
    original = add(second, 'existing', ('queue original payload',))
    assert original.status_code == (503 if provenance == 'failed-job' else 200)
    pb.failure = None
    if provenance == 'legacy':
        with connected(path) as db:
            db.execute("DELETE FROM add_requests WHERE request_id='existing'")
            db.execute("DELETE FROM k0_jobs WHERE request_id='existing'")
    before = list(pb.calls)
    clock = ReserveClock(fake=True)
    _install_clock(monkeypatch, second, clock)
    monkeypatch.setattr(jobs, 'USER_PREPARATION_WAIT', 0.15, raising=False)
    with _pool(pa.release) as pool:
        pending = pool.submit(add, first, 'other-active-request', (FIRST,))
        assert pa.entered.wait(GUARD)
        conflict = add(second, 'existing', ('queue conflicting payload',))
        assert conflict.status_code == 409
        assert 'different Add payload' in conflict.json()['detail']
        assert clock.sleeps == [] and pb.calls == before and not pending.done()
        pa.release.set()
        assert pending.result(timeout=GUARD).status_code == 200


@pytest.mark.parametrize('provenance', ['receipt', 'legacy'])
def test_replay_does_not_wait_for_other_request_or_repeat_provider(tmp_path, monkeypatch, provenance):
    path, first, second, pa, pb = _pair(tmp_path)
    original = add(second, 'existing', ('queue existing replay',))
    assert original.status_code == 200
    if provenance == 'legacy':
        with connected(path) as db:
            db.execute("DELETE FROM add_requests WHERE request_id='existing'")
            db.execute("DELETE FROM k0_jobs WHERE request_id='existing'")
    before = list(pb.calls)
    clock = ReserveClock(fake=True)
    _install_clock(monkeypatch, second, clock)
    monkeypatch.setattr(jobs, 'USER_PREPARATION_WAIT', 0.15, raising=False)
    with _pool(pa.release) as pool:
        pending = pool.submit(add, first, 'other-active-request', (FIRST,))
        assert pa.entered.wait(GUARD)
        replay = add(second, 'existing', ('queue existing replay',))
        assert replay.status_code == 200 and replay.json() == original.json()
        assert clock.sleeps == [] and pb.calls == before and not pending.done()
        pa.release.set()
        assert pending.result(timeout=GUARD).status_code == 200


def test_same_request_deadline_stays_short_when_pure_erase_also_has_a_lease(tmp_path, monkeypatch):
    """Counterexample: a permitted erase can coexist with a normal lease."""
    path, first, eraser, pa, pe = _pair(tmp_path)
    seed = 'queue erase overlap seed'
    assert add(eraser, 'seed', (seed,)).status_code == 200
    third_provider = Provider()
    duplicate_app = app_at(path, third_provider, 'cycle2_dense')
    clock = ReserveClock(fake=True)
    _install_clock(monkeypatch, duplicate_app, clock)
    monkeypatch.setattr(jobs, 'COMPETITION_WAIT', 0.15)
    monkeypatch.setattr(jobs, 'USER_PREPARATION_WAIT', 0.6)
    commit_entered, commit_release = threading.Event(), threading.Event()
    original_commit = eraser.state.store.commit_add
    def gated_commit(*args, **kwargs):
        commit_entered.set()
        assert commit_release.wait(GUARD)
        return original_commit(*args, **kwargs)
    monkeypatch.setattr(eraser.state.store, 'commit_add', gated_commit)
    with _pool(pa.release, commit_release) as pool:
        pending = pool.submit(add, first, 'first', (FIRST,))
        assert pa.entered.wait(GUARD)
        deletion = pool.submit(add, eraser, 'erase', (f'delete "{seed}"',))
        assert commit_entered.wait(GUARD)
        with connected(path) as db:
            assert db.execute("SELECT COUNT(*) FROM k0_jobs WHERE state='preparing' AND user_id='u'").fetchone()[0] == 2
        response = add(duplicate_app, 'first', (FIRST,))
        assert response.status_code == 503 and response.json()['detail'] == 'preparation_busy'
        assert third_provider.calls == []
        commit_release.set()
        assert deletion.result(timeout=GUARD).status_code == 200
        pa.release.set()
        assert pending.result(timeout=GUARD).status_code == 503
    assert 0.15 - 1e-9 <= clock.elapsed < 0.20, 'same-request wait was incorrectly replaced by USER_PREPARATION_WAIT'


def test_user_deadline_bounds_real_external_sqlite_write_lock(tmp_path, monkeypatch):
    """An actual second SQLite writer outlives the entire 0.15s user budget."""
    path, first, second, pa, pb = _pair(tmp_path)
    monkeypatch.setattr(jobs, 'USER_PREPARATION_WAIT', 0.15, raising=False)
    entered = threading.Event()
    elapsed = []
    reserve = second.state.coordinator.reserve
    def timed_reserve(*args, **kwargs):
        started = real_time.monotonic()
        entered.set()
        try:
            return reserve(*args, **kwargs)
        finally:
            elapsed.append(real_time.monotonic() - started)
    monkeypatch.setattr(second.state.coordinator, 'reserve', timed_reserve)
    with _pool(pa.release) as pool:
        pending = pool.submit(add, first, 'first', (FIRST,))
        assert pa.entered.wait(GUARD)
        locker = sqlite3.connect(path, timeout=1)
        try:
            locker.execute('BEGIN IMMEDIATE')
            contender = pool.submit(add, second, 'second', (SECOND,))
            assert entered.wait(GUARD)
            # Intentional external writer hold >0.3s, not a scheduling guess:
            # the reserve entry event has already fired and the lock stays held.
            threading.Event().wait(0.4)
            assert contender.done(), 'reserve exceeded its 0.15s budget while waiting on SQLite'
            response = contender.result(timeout=GUARD)
            assert response.status_code == 503
            assert response.headers['Retry-After'] == '1'
            assert len(elapsed) == 1 and 0.08 <= elapsed[0] < 0.35
            assert pb.calls == []
            assert locker.execute('SELECT COUNT(*) FROM k0_jobs').fetchone()[0] == 1
            assert locker.execute('SELECT COUNT(*) FROM k0_attempts').fetchone()[0] == 1
        finally:
            locker.rollback()
            locker.close()
        pa.release.set()
        assert pending.result(timeout=GUARD).status_code == 200
    assert response.json()['detail'] == 'preparation_busy', (
        f'reserve_seconds={elapsed[0]:.6f}; contender_provider_calls={len(pb.calls)}; '
        f'detail={response.json()["detail"]}'
    )


def test_erase_fences_old_epoch_without_blocking_a_fresh_request(tmp_path, monkeypatch):
    path, first, second, pa, pb = _pair(tmp_path)
    seed = 'queue erase epoch seed'
    assert add(second, 'seed', (seed,)).status_code == 200
    clock = ReserveClock(fake=True)
    _install_clock(monkeypatch, second, clock)
    monkeypatch.setattr(jobs, 'USER_PREPARATION_WAIT', 0.15, raising=False)
    with _pool(pa.release) as pool:
        pending = pool.submit(add, first, 'first', (FIRST,))
        assert pa.entered.wait(GUARD)
        assert add(second, 'erase', (f'delete "{seed}"',)).status_code == 200
        with connected(path) as db:
            old_epoch, old_state = db.execute("SELECT source_epoch,state FROM k0_jobs WHERE request_id='first'").fetchone()
            new_epoch = db.execute("SELECT epoch FROM lifecycle_users WHERE user_id='u'").fetchone()[0]
        assert old_epoch < new_epoch and old_state == 'preparing'
        fresh = add(second, 'after-erase', (SECOND,))
        assert fresh.status_code == 200, fresh.json()
        assert not pending.done() and not pa.release.is_set()
        assert clock.sleeps == [] and pb.calls == [[seed], [SECOND]]
        _visible(second, {SECOND})
        pa.release.set()
        stale = pending.result(timeout=GUARD)
        assert stale.status_code == 503 and stale.json()['detail'] in {'source_epoch_changed', 'lease_lost'}
    with connected(path) as db:
        assert db.execute("SELECT COUNT(*) FROM k0_receipts WHERE request_id='first'").fetchone()[0] == 0
    _visible(second, {SECOND})


def test_non_cycle2_does_not_acquire_the_new_user_queue(tmp_path):
    _, first, second, pa, pb = _pair(tmp_path, profile='dense_sync')
    assert first.state.profile.identity.get('add_preparation_policy') != 'user_lease_v1'
    with _pool(pa.release) as pool:
        a = pool.submit(add, first, 'first', (FIRST,))
        assert pa.entered.wait(GUARD)
        b = pool.submit(add, second, 'second', (SECOND,))
        assert b.result(timeout=GUARD).status_code == 200
        assert pb.calls == [[SECOND]] and not a.done()
        pa.release.set()
        # Historical behavior: another publication fences the older preparation.
        assert a.result(timeout=GUARD).status_code == 503


def test_cycle2_wait_identity_and_effective_default_lock_timeout(tmp_path):
    app = app_at(tmp_path / 'defaults.sqlite', Provider(), 'cycle2_dense')
    assert jobs.USER_PREPARATION_WAIT == 120.0
    assert jobs.COMPETITION_WAIT == 5.0
    assert app.state.profile.identity['add_preparation_policy'] == 'user_lease_v1'
    assert app.state.profile.identity['user_preparation_wait_seconds'] == 120
    assert app.state.profile.identity['wait_seconds'] == 5
    assert Settings(db_path=tmp_path / 'unused.sqlite').coordinator_lock_timeout_seconds == 30
    assert app.state.coordinator.lock_timeout_seconds == 30


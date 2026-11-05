"""Serving controls on one worker-owned connection; no user-data audit.

SQLite loads its schema on first access. Large per-user FTS schemas exceed the
old 250ms budget even without reading message rows. Reuse metadata, never reuse
a readiness result: every completed check starts a new read transaction.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
import os
import shutil
import sqlite3
import threading
import time

from app.jobs import NotReady


class ServingHealth:
    def __init__(self, coordinator, path, profile, wait_seconds=5.25):
        self.coordinator, self.path, self.profile = coordinator, Path(path), profile
        st = self.path.stat()
        self.identity = (st.st_dev, st.st_ino)
        self.wait_seconds = wait_seconds
        self.reserve_bytes = 256 * 1024**2
        self.sql_budget_seconds = 5
        self._lock = threading.Lock()
        self._future = None
        self._closed = False
        self._db = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='aml-controls-093')

    def failure(self, reason, **extra):
        return dict(status='not_ready', control_ready=False, data_route_ready=False,
                    profile_id=self.profile.profile_id, readiness_revision='093',
                    health_scope='serving_controls', reason=reason,
                    whole_database_integrity_verified=False, **extra)

    def _disconnect(self):
        # Only the single executor worker creates, uses and closes this handle.
        if self._db is not None:
            self._db.close()
            self._db = None

    def _evaluate(self):
        started = time.monotonic()
        try:
            if self._closed:
                return self.failure('service_stopped')
            before = self.path.stat()
            if (before.st_dev, before.st_ino) != self.identity:
                self._disconnect()
                return self.failure('readiness_database_replaced')
            if not os.access(self.path, os.W_OK) or not os.access(self.path.parent, os.W_OK):
                return self.failure('storage_not_writable')
            if shutil.disk_usage(self.path.parent).free < self.reserve_bytes:
                return self.failure('storage_reserve_low')
            if self._db is None:
                self._db = sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro',
                    uri=True, timeout=.1, isolation_level=None)
                self._db.row_factory = sqlite3.Row
                self._db.execute('PRAGMA query_only=ON')
            db = self._db
            # This is a fixed, metadata-only SELECT. Frequent Python callbacks
            # need the GIL and can turn a small schema scan into seconds under
            # concurrent evidence work. Keep the wall budget with coarse checks.
            db.set_progress_handler(lambda: int(time.monotonic()-started > self.sql_budget_seconds), 10000)
            try:
                db.execute('BEGIN')
                self.coordinator.control_ready_on(db)
                db.execute('COMMIT')
            finally:
                db.set_progress_handler(None, 0)
                if db.in_transaction:
                    db.execute('ROLLBACK')
            # Small metadata reads may finish before the next VM callback.
            # Check elapsed time too, keeping the same deadline for those reads.
            if time.monotonic()-started > self.sql_budget_seconds:
                return self.failure('service_controls_sql_budget_exceeded')
            after = self.path.stat()
            if (after.st_dev, after.st_ino) != self.identity:
                self._disconnect()
                return self.failure('readiness_database_replaced')
            return dict(status='ok', control_ready=True, data_route_ready=True,
                        profile_id=self.profile.profile_id, config_hash=self.profile.config_hash,
                        readiness_revision='093', health_scope='serving_controls',
                        evidence_validation='per_user_on_add_and_search',
                        whole_database_integrity_verified=False,
                        elapsed_ms=round((time.monotonic()-started)*1000, 3))
        except NotReady as exc:
            return self.failure(str(exc))
        except sqlite3.Error as exc:
            name = getattr(exc, 'sqlite_errorname', None)
            # An interrupted read transaction was rolled back above. Retain its
            # loaded schema, avoiding repeated cold schema loads under pressure.
            # Other SQL errors still replace the connection on the next check.
            if name != 'SQLITE_INTERRUPT':
                self._disconnect()
            return self.failure('service_controls_sql_budget_exceeded' if name=='SQLITE_INTERRUPT'
                                else 'service_controls_database_unavailable', sqlite_error_name=name)
        except OSError:
            self._disconnect()
            return self.failure('service_controls_storage_unavailable')

    def _start_or_join(self):
        with self._lock:
            if self._closed:
                return None
            if self._future is None or self._future.done():
                self._future = self._executor.submit(self._evaluate)
            return self._future

    def evaluate(self):
        future = self._start_or_join()
        if future is None:
            return self.failure('service_stopped')
        try:
            return dict(future.result(timeout=self.wait_seconds))
        except FutureTimeout:
            return self.failure('service_controls_timeout')

    async def check(self):
        future = self._start_or_join()
        if future is None:
            return self.failure('service_stopped')
        try:
            return dict(await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)),
                                               timeout=self.wait_seconds))
        except asyncio.TimeoutError:
            return self.failure('service_controls_timeout')

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Close in its owning thread after any in-flight query finishes.
            self._executor.submit(self._disconnect)
            self._executor.shutdown(wait=False, cancel_futures=False)

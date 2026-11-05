"""Bounded reusable read-only handles; readers never acquire WriterGate.

Every operation opens a fresh WAL read transaction. This caches connections and
SQLite schema only, never results, evidence, authorization or readiness proofs.
"""
from contextlib import contextmanager
import sqlite3
import threading
import time


class ReadPool:
    def __init__(self, path, *, limit=32, timeout=30):
        self.path = path
        self.limit = limit
        self.timeout = timeout
        self.condition = threading.Condition()
        self.idle = []
        self.total = 0
        self.active = 0
        self.max_active = 0
        self.created = 0
        self.identity = None
        self.closed = False
        self.reuse = True
        self.local = threading.local()

    def _identity(self):
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino

    def _failure(self, reason):
        from app.jobs import NotReady
        return NotReady(reason)

    def _take(self):
        deadline = time.monotonic() + self.timeout
        with self.condition:
            while True:
                if self.closed:
                    raise self._failure('read_pool_closed')
                identity = self._identity()
                if self.identity is None:
                    self.identity = identity
                if identity != self.identity:
                    raise self._failure('readiness_database_replaced')
                if self.idle:
                    db = self.idle.pop()
                    break
                if self.total < self.limit:
                    self.total += 1
                    db = None
                    break
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise self._failure('read_pool_busy')
                self.condition.wait(remaining)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        if db is None:
            try:
                db = sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro', uri=True,
                    timeout=self.timeout, isolation_level=None, check_same_thread=False,
                    cached_statements=64)
                db.row_factory = sqlite3.Row
                db.execute('PRAGMA query_only=ON')
                db.execute('PRAGMA foreign_keys=ON')
                db.execute('PRAGMA cache_size=-4096')
                with self.condition:
                    self.created += 1
            except BaseException:
                if db is not None:
                    db.close()
                with self.condition:
                    self.total -= 1
                    self.active -= 1
                    self.condition.notify_all()
                raise
        return db

    @contextmanager
    def snapshot(self):
        nested = getattr(self.local, 'db', None)
        if nested is not None:
            yield nested
            return
        db = self._take()
        keep = False
        self.local.db = db
        try:
            db.execute('BEGIN')
            yield db
            if self._identity() != self.identity:
                raise self._failure('readiness_database_replaced')
            keep = True
        finally:
            self.local.db = None
            try:
                if db.in_transaction:
                    db.execute('ROLLBACK')
            except sqlite3.Error:
                keep = False
            with self.condition:
                self.active -= 1
                if keep and self.reuse and not self.closed:
                    self.idle.append(db)
                else:
                    db.close()
                    self.total -= 1
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.dispose()
            self.condition.notify_all()

    def dispose(self):
        """Release idle handles; a reused application may start another lifespan."""
        with self.condition:
            for db in self.idle:
                db.close()
            self.total -= len(self.idle)
            self.idle.clear()
            self.condition.notify_all()

    def statistics(self):
        with self.condition:
            return dict(limit=self.limit, connections=self.total, active=self.active,
                        peak_active=self.max_active, created=self.created, closed=self.closed)

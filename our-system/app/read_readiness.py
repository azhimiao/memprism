"""Reuse actual-data audits only on the same unchanged read-only connection.

No TTL, source text, vector, result or permission cache. Configuration, epochs
and publication proofs are read again inside every transaction. All external
commits invalidate counts through SQLite data_version on this retained handle.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import sqlite3
import threading


class _ReadShard:
    LIMIT = 256

    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.db = None
        self.identity = None
        self.version = None
        self.counts = OrderedDict()

    def close(self):
        with self.lock:
            if self.db is not None:
                self.db.close()
            self.db, self.identity, self.version = None, None, None
            self.counts.clear()

    def require(self, coordinator, user):
        with self.snapshot(coordinator, user) as (_, result):
            return result

    def _anchor(self, coordinator, user, reader):
        """Transfer a writer's exact proof into a pinned read snapshot.

        Try the writer gate without waiting. Never perform an audit while it is
        held. IMMEDIATE prevents an external writer changing data between the
        writer-local proof and the reader's first SELECT. Busy writers simply
        cause an independent actual-data audit for this shard.
        """
        store = self.store
        # Ordinary publications are short. Prefer their exact proof to another
        # cold whole-history scan. A bounded wait preserves the read-only WAL
        # fallback for a genuinely long/foreign writer.
        if not store._lock.acquire(timeout=min(.15, coordinator.lock_timeout_seconds), priority=True):
            reader.execute('BEGIN')
            return None, None
        writer = None
        started = False
        try:
            writer = store._writer.connection(0)
            if writer.in_transaction:
                reader.execute('BEGIN')
                return None, None
            writer.execute('BEGIN IMMEDIATE')
            started = True
            stamp = store._writer.stamp(writer, user, coordinator.profile)
            counts = store._writer.get(writer, user, coordinator.profile)
            reader.execute('BEGIN')
            coordinator.config_on(reader)
            writer.execute('COMMIT')
            started = False
            return counts, stamp
        except sqlite3.Error as exc:
            if started and writer.in_transaction:
                writer.execute('ROLLBACK')
            if (getattr(exc, 'sqlite_errorcode', 0) or 0) & 255 in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                if not reader.in_transaction:
                    reader.execute('BEGIN')
                return None, None
            raise
        except BaseException:
            if started and writer.in_transaction:
                writer.execute('ROLLBACK')
            raise
        finally:
            store._lock.release()

    def _publish(self, coordinator, user, stamp, counts):
        # A cold, old user becomes reusable across our own control/Add commits,
        # but only if its audited snapshot still matches the writer-local stamp.
        store = self.store
        if not store._lock.acquire(timeout=min(2., coordinator.lock_timeout_seconds), priority=True):
            return
        writer = None
        started = False
        try:
            writer = store._writer.connection(0)
            if writer.in_transaction:
                return
            writer.execute('BEGIN IMMEDIATE')
            started = True
            coordinator.config_on(writer)
            if store._writer.stamp(writer, user, coordinator.profile) == stamp:
                store._writer.remember(writer, user, coordinator.profile, counts)
            writer.execute('COMMIT')
            started = False
        except sqlite3.Error as exc:
            if started and writer.in_transaction:
                writer.execute('ROLLBACK')
            if (getattr(exc, 'sqlite_errorcode', 0) or 0) & 255 not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise
        except BaseException:
            if started and writer.in_transaction:
                writer.execute('ROLLBACK')
            raise
        finally:
            store._lock.release()

    @contextmanager
    def snapshot(self, coordinator, user):
        from app.jobs import NotReady, route_counts_on
        timeout = max(0, coordinator.lock_timeout_seconds)
        if not self.lock.acquire(timeout=timeout):
            raise NotReady('readiness_audit_busy')
        try:
            stat = self.store.db_path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if self.db is None or self.identity != identity:
                self.close()
                self.db = sqlite3.connect(self.store.db_path.resolve().as_uri()+'?mode=ro', uri=True,
                    timeout=timeout, isolation_level=None, check_same_thread=False, cached_statements=64)
                self.db.row_factory = sqlite3.Row
                self.db.execute('PRAGMA query_only=ON')
                self.db.execute('SELECT COUNT(*) FROM sqlite_schema').fetchone()
                self.identity = identity
            db = self.db
            # Nested instrumentation/tests may request readiness on this thread;
            # preserve the old independent-snapshot behavior in that case.
            if db.in_transaction:
                with coordinator.transaction(False) as other:
                    yield other, coordinator.ready_on(other, user)
                    return
            shared_counts, writer_stamp = self._anchor(coordinator, user, db)
            try:
                coordinator.config_on(db)  # Pin the same SQLite snapshot first.
                version = db.execute('PRAGMA data_version').fetchone()[0]
                if self.version != version:
                    self.counts.clear()
                    self.version = version
                key = (hashlib.sha256(user.encode('utf-8')).digest(), coordinator.profile.config_hash)
                cached = self.counts.get(key)
                counts = (dict(cached) if cached is not None else shared_counts if shared_counts is not None
                          else route_counts_on(db, user, coordinator.profile))
                result = coordinator.ready_on(db, user, _counts=counts)
                # Final evidence checks use this SAME current snapshot. Only
                # unchanged actual-data counts are reused; source authorization
                # and policy comparisons still execute for every candidate.
                yield db, result
                db.execute('COMMIT')
                if shared_counts is None and writer_stamp is not None:
                    self._publish(coordinator, user, writer_stamp, counts)
                after = self.store.db_path.stat()
                if (after.st_dev, after.st_ino) != identity:
                    self.counts.clear()
                    raise NotReady('readiness_database_replaced')
                if db.execute('PRAGMA data_version').fetchone()[0] == version:
                    self.counts[key] = tuple(counts.items())
                    self.counts.move_to_end(key)
                    while len(self.counts) > self.LIMIT:
                        self.counts.popitem(last=False)
                else:
                    self.counts.clear()
                    self.version = None
            finally:
                if db.in_transaction:
                    db.execute('ROLLBACK')
        except sqlite3.Error:
            self.close()
            raise NotReady('control_not_ready') from None
        except OSError:
            self.close()
            raise NotReady('control_not_ready') from None
        finally:
            self.lock.release()


class ReadReadiness:
    """32 bounded independent audit lanes; no global long-running audit lock.

    Same user always reaches the same retained connection. Counts on that
    connection invalidate on every external commit. A trusted writer proof may
    be reused only through the short, snapshot-anchored handoff above. Final
    source/privacy/version checks still read the current transaction.
    """
    SHARDS = 32

    def __init__(self, store):
        self.store = store
        self.shards = [_ReadShard(store) for _ in range(self.SHARDS)]
        self._assign_lock = threading.Lock()
        self._assignments = {}
        self._lane_users = [0] * self.SHARDS

    def close(self):
        for shard in self.shards:
            shard.close()

    def require(self, coordinator, user):
        with self.snapshot(coordinator, user) as (_, result):
            return result

    @contextmanager
    def snapshot(self, coordinator, user):
        digest = hashlib.sha256(user.encode('utf-8')).digest()
        preferred = int.from_bytes(digest[:4], 'little') % self.SHARDS
        # Active calls for one user share one lane. Unrelated users use another
        # idle lane on a hash collision; at most the original 32 handles exist.
        # Only hashed identities of in-flight calls survive in this registry.
        with self._assign_lock:
            assignment = self._assignments.get(digest)
            if assignment is None:
                lane = preferred
                if self._lane_users[lane]:
                    lane = next((i for i,n in enumerate(self._lane_users) if not n), preferred)
                assignment = self._assignments[digest] = [lane, 0]
                self._lane_users[lane] += 1
            assignment[1] += 1
            lane = assignment[0]
        try:
            with self.shards[lane].snapshot(coordinator, user) as result:
                yield result
        finally:
            with self._assign_lock:
                assignment[1] -= 1
                if not assignment[1]:
                    del self._assignments[digest]
                    self._lane_users[lane] -= 1

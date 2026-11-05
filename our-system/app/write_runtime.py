"""Connection-local invalidation for exact, bounded readiness summaries.

All access is under the store writer gate. Other SQLite connections invalidate
through data_version; this connection's writes invalidate through an authorizer
on every prepared statement. No source text, vector or permission is cached.
"""
from collections import OrderedDict
import hashlib
import sqlite3


class WriteRuntime:
    LIMIT = 4096
    CONTROL = frozenset(('k0_jobs', 'k0_attempts', 'k0_config', 'k0_users',
        'k0_receipts', 'k0_watermarks', 'k0_switch_proofs', 'add_requests'))
    SCHEMA = frozenset((sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX, sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER, sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER, sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_DROP_INDEX, sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE, sqlite3.SQLITE_DROP_TEMP_TRIGGER, sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER, sqlite3.SQLITE_DROP_VIEW, sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_CREATE_VTABLE, sqlite3.SQLITE_DROP_VTABLE))

    def __init__(self, store):
        self.store = store
        self.db = None
        self.identity = None
        self.serial = 0
        self.global_serial = 0
        self.users = OrderedDict()
        self.summaries = OrderedDict()
        self.user = None

    @staticmethod
    def key(user):
        return hashlib.sha256(user.encode('utf-8')).digest()

    def _invalidate(self):
        self.serial += 1
        if self.user is None:
            self.global_serial = self.serial
        else:
            if self.user not in self.users and len(self.users) >= self.LIMIT:
                self.users.clear()
                self.summaries.clear()
                self.global_serial = self.serial
            self.users[self.user] = self.serial
            self.users.move_to_end(self.user)

    def invalidate_all(self):
        self.serial += 1
        self.global_serial = self.serial
        self.summaries.clear()

    def _authorize(self, action, table, column, database, trigger):
        if action in self.SCHEMA:
            self._invalidate()
        elif action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE):
            if table not in self.CONTROL:
                self._invalidate()
        return sqlite3.SQLITE_OK

    def connection(self, timeout):
        stat = self.store.db_path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.db is None or self.identity != identity:
            self.close()
            self.db = self.store._connect(timeout=timeout)
            self.db.set_authorizer(self._authorize)
            self.identity = identity
            self._invalidate()
        self.db.execute('PRAGMA busy_timeout='+str(max(0, int(timeout*1000))))
        return self.db

    def stamp(self, db, user, profile):
        if db is not self.db or not db.in_transaction:
            raise RuntimeError('readiness_stamp_requires_writer_transaction')
        stat = self.store.db_path.stat()
        if (stat.st_dev, stat.st_ino) != self.identity:
            raise RuntimeError('readiness_database_replaced')
        key = self.key(user)
        # External DDL changes data_version too. Own trusted per-user FTS DDL
        # is tracked by the authorizer in that user's revision, not by SQLite's
        # database-wide schema_version (which would invalidate unrelated users).
        return (self.identity, db.execute('PRAGMA data_version').fetchone()[0], self.global_serial,
                self.users.get(key, 0), profile.config_hash)

    def get(self, db, user, profile):
        key = self.key(user)
        entry = self.summaries.get(key)
        if entry is not None and entry[0] == self.stamp(db, user, profile):
            self.summaries.move_to_end(key)
            return dict(entry[1])
        return None

    def remember(self, db, user, profile, counts):
        key = self.key(user)
        self.summaries[key] = (self.stamp(db, user, profile), tuple(counts.items()))
        self.summaries.move_to_end(key)
        while len(self.summaries) > self.LIMIT:
            self.summaries.popitem(last=False)

    def close(self):
        if self.db is not None:
            self.db.close()
        self.db = None
        self.identity = None
        self.summaries.clear()
        self.users.clear()
        self.user = None
        self.serial += 1
        self.global_serial = self.serial

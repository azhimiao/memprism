"""SQLite raw-message store with transactional Add idempotency.

Each user_id has its own FTS5 table. bm25() statistics are table-wide, so a
shared lexical table lets one user's inserts change another user's scores.
Partition names are hashes of user_id, not the identifier text.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from app.index import build_lexical_text
from app.embeddings import EmbeddingFailure, PreparedEmbedding, normalize_vector
from app.models import EvidenceCandidate, RawMessage
from app.governance import GovernancePlan, statement_digest, TemporalView
from app.memory_core import initialize_core, apply_core, operation_receipts, valid_derived_source
from app.jobs import initialize_jobs, write_publication_on, embedding_messages_on, NotReady
from app import segments

_PARTITION_NAME = re.compile(r"^fts_u_[0-9a-f]{64}$")


def lexical_partition_name(user_id: str) -> str:
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return f"fts_u_{digest}"


def _checked_partition_name(name: str) -> str:
    if _PARTITION_NAME.fullmatch(name) is None:
        raise ValueError("refusing lexical partition name")
    return name


def _partition_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def count_lexical_rows(connection: sqlite3.Connection) -> int:
    """Count FTS rows across user partitions. Used by persistence tests."""

    ready = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'fts_partitions'
        """
    ).fetchone()
    if ready is None:
        return 0
    total = 0
    for row in connection.execute("SELECT table_name FROM fts_partitions"):
        table_name = _checked_partition_name(row[0])
        total += connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    return total


class CommitStatus(Enum):
    CREATED = "created"
    REPLAY = "replay"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class CommitResult:
    status: CommitStatus
    response: dict[str, object] | None = None


@dataclass(frozen=True)
class RouteSnapshot:
    lexical: list[EvidenceCandidate]
    dense: list[EvidenceCandidate]
    actual_mode: str
    degraded_reason: str | None
    active_count: int
    ready_count: int


class GovernedSelection(list):
    def __init__(self, values, epoch, invalid_derived_ids=()):
        super().__init__(values)
        self.lifecycle_epoch = epoch
        self.invalid_derived_ids = tuple(sorted(set(invalid_derived_ids)))


class MessageStore(Protocol):
    def lookup_add(self, messages: list[RawMessage], *, payload_hash: str, response: dict[str, object]) -> CommitResult | None: ...

    def commit_add(
        self,
        messages: list[RawMessage],
        *,
        payload_hash: str,
        response: dict[str, object],
        embedding: PreparedEmbedding | None = None,
        governance_plan: GovernancePlan | None = None,
    ) -> CommitResult:
        """Persist the request, raw messages, and FTS rows atomically."""

    def search_lexical(self, user_id: str, match_query: str, limit: int) -> list[EvidenceCandidate]:
        """Return matching raw messages for one user, best score first."""

    def vector_coverage(self, user_id: str, version: str, dimensions: int) -> tuple[int, int]: ...
    def has_erased_statements(self, messages: list[RawMessage]) -> bool: ...

    def search_snapshot(
        self, *, user_id: str, match_query: str | None, query_vector: list[float] | None,
        version: str, dimensions: int, mode: str, candidate_limit: int,
        lexical_quota: int, dense_quota: int, degraded_reason: str | None = None,
    ) -> RouteSnapshot: ...


class SqliteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()

    def initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS raw_messages (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        request_id TEXT NOT NULL,
                        message_index INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source_timestamp INTEGER,
                        recorded_at INTEGER NOT NULL,
                        privacy_level TEXT NOT NULL,
                        deletion_state TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_raw_messages_user
                        ON raw_messages(user_id)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fts_partitions (
                        user_id TEXT PRIMARY KEY,
                        table_name TEXT NOT NULL UNIQUE
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS add_requests (
                        user_id TEXT NOT NULL,
                        request_id TEXT NOT NULL,
                        payload_hash TEXT NOT NULL,
                        response_json TEXT NOT NULL,
                        completed_at INTEGER NOT NULL,
                        PRIMARY KEY (user_id, request_id)
                    )
                    """
                )
                self._migrate_shared_fts(connection)
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS raw_vectors (
                        user_id TEXT NOT NULL, raw_id TEXT NOT NULL,
                        embedding_version TEXT NOT NULL, dimensions INTEGER NOT NULL,
                        content_sha256 TEXT NOT NULL, vector_json TEXT NOT NULL,
                        PRIMARY KEY (user_id, raw_id, embedding_version),
                        FOREIGN KEY(raw_id) REFERENCES raw_messages(id) ON DELETE CASCADE
                    )
                """)
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS embedding_requests (
                        user_id TEXT NOT NULL, request_id TEXT NOT NULL,
                        embedding_version TEXT NOT NULL, status TEXT NOT NULL,
                        expected_count INTEGER NOT NULL, error_code TEXT,
                        metadata_json TEXT NOT NULL, usage_json TEXT NOT NULL,
                        PRIMARY KEY(user_id, request_id, embedding_version)
                    )
                """)
                initialize_core(connection)
                initialize_jobs(connection)
                segments.initialize(connection)
                connection.execute("COMMIT")
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

    def lookup_add(
        self, messages: list[RawMessage], *, payload_hash: str, response: dict[str, object]
    ) -> CommitResult | None:
        """Fast replay/conflict check before HTTP; commit_add rechecks under its transaction."""
        user_id, request_id = messages[0].user_id, messages[0].request_id
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT payload_hash, response_json FROM add_requests WHERE user_id=? AND request_id=?",
                    (user_id, request_id),
                ).fetchone()
                if row is not None:
                    if row["payload_hash"] != payload_hash:
                        return CommitResult(CommitStatus.CONFLICT)
                    return CommitResult(CommitStatus.REPLAY, json.loads(row["response_json"]))
                legacy = connection.execute(
                    "SELECT 1 FROM raw_messages WHERE user_id=? AND request_id=? LIMIT 1", (user_id, request_id)
                ).fetchone() is not None
            finally:
                connection.close()
        if legacy:
            # Reconcile the old ledger without adding new derived side effects.
            return self.commit_add(messages, payload_hash=payload_hash, response=response)
        return None

    def commit_add(
        self,
        messages: list[RawMessage],
        *,
        payload_hash: str,
        response: dict[str, object],
        embedding: PreparedEmbedding | None = None,
        governance_plan: GovernancePlan | None = None,
        coordinator=None,
        lease=None,
        embedding_messages: list[RawMessage] | None = None,
        profile=None,
        pure_control: bool = False,
        span_pieces: list[dict] | None = None,
        claim_projection: bool = False,
    ) -> CommitResult:
        if not messages:
            raise ValueError("an Add request must contain at least one message")

        user_id = messages[0].user_id
        request_id = messages[0].request_id
        if any(message.user_id != user_id or message.request_id != request_id for message in messages):
            raise ValueError("all messages in an Add request must share user_id and request_id")

        with self._lock:
            connection = self._connect(timeout=0.1 if coordinator else 30)
            try:
                connection.execute("BEGIN IMMEDIATE")
                if coordinator is not None:
                    coordinator.fence_on(connection, lease)
                    if not pure_control:
                        coordinator.ready_on(connection, user_id)
                    expected = embedding_messages_on(connection, messages, governance_plan)
                    if embedding_messages != expected:
                        raise NotReady("prepared_sources_changed")
                    if profile.identity["profile_version"] == "k1-v1":
                        expected_pieces = [p for m in expected for p in segments.split_source(m)]
                        if span_pieces != expected_pieces:
                            raise NotReady("prepared_spans_changed")
                    if expected and "dense" in profile.required_routes and (
                        embedding is None or embedding.status != "ready" or
                        embedding.version != profile.model_version or embedding.dimensions != profile.dimensions):
                        raise NotReady("required_embedding_not_ready")
                existing = connection.execute(
                    """
                    SELECT payload_hash, response_json
                    FROM add_requests
                    WHERE user_id = ? AND request_id = ?
                    """,
                    (user_id, request_id),
                ).fetchone()
                if existing is not None:
                    if existing["payload_hash"] == payload_hash:
                        connection.execute("COMMIT")
                        return CommitResult(
                            CommitStatus.REPLAY,
                            json.loads(existing["response_json"]),
                        )
                    connection.execute("ROLLBACK")
                    return CommitResult(CommitStatus.CONFLICT)

                legacy_rows = connection.execute(
                    """
                    SELECT session_id, message_index, role, content, source_timestamp
                    FROM raw_messages
                    WHERE user_id = ? AND request_id = ?
                    ORDER BY message_index ASC
                    """,
                    (user_id, request_id),
                ).fetchall()
                if legacy_rows:
                    if _legacy_rows_match(legacy_rows, messages):
                        connection.execute(
                            """
                            INSERT INTO add_requests (
                                user_id, request_id, payload_hash, response_json, completed_at
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                user_id,
                                request_id,
                                payload_hash,
                                json.dumps(response, ensure_ascii=False, sort_keys=True),
                                messages[0].recorded_at,
                            ),
                        )
                        connection.execute("COMMIT")
                        return CommitResult(CommitStatus.REPLAY, response)
                    connection.execute("ROLLBACK")
                    return CommitResult(CommitStatus.CONFLICT)

                connection.executemany(
                    """
                    INSERT INTO raw_messages (
                        id, user_id, session_id, request_id, message_index,
                        role, content, source_timestamp, recorded_at,
                        privacy_level, deletion_state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            message.id,
                            message.user_id,
                            message.session_id,
                            message.request_id,
                            message.message_index,
                            message.role,
                            message.content,
                            message.source_timestamp,
                            message.recorded_at,
                            message.privacy_level,
                            message.deletion_state,
                        )
                        for message in messages
                    ],
                )
                self._insert_fts(connection, messages)
                if embedding is not None and span_pieces is None:
                    self._insert_embeddings(connection, embedding_messages if coordinator else messages, embedding)
                apply_core(connection, messages, governance_plan, self._ensure_partition(connection, user_id),
                           lifecycle_only=coordinator is not None and not claim_projection)
                if span_pieces is not None:
                    segments.insert(connection, user_id, span_pieces, embedding)
                    if embedding is not None:
                        connection.execute("INSERT INTO embedding_requests VALUES (?,?,?,?,?,?,?,?)",(
                            user_id,request_id,embedding.version,embedding.status,len(span_pieces),embedding.error_code,
                            json.dumps(embedding.metadata,sort_keys=True),json.dumps(embedding.usage,sort_keys=True)))
                if span_pieces is not None or connection.execute("SELECT 1 FROM source_spans WHERE user_id=? LIMIT 1",(user_id,)).fetchone():
                    segments.rebind_versions(connection, user_id)
                    segments.prune(connection, user_id)
                write_publication_on(connection, user_id, request_id, profile)
                if coordinator is not None:
                    if not pure_control:
                        coordinator.ready_on(connection, user_id)
                    coordinator.fence_on(connection, lease, published=True)
                    connection.execute("""UPDATE k0_jobs SET state='committed',error_code=NULL
                        WHERE user_id=? AND request_id=?""", (user_id, request_id))
                connection.execute(
                    """
                    INSERT INTO add_requests (
                        user_id, request_id, payload_hash, response_json, completed_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        request_id,
                        payload_hash,
                        json.dumps(response, ensure_ascii=False, sort_keys=True),
                        messages[0].recorded_at,
                    ),
                )
                connection.execute("COMMIT")
                return CommitResult(CommitStatus.CREATED, response)
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

    def _insert_fts(self, connection: sqlite3.Connection, messages: list[RawMessage]) -> None:
        table_name = self._ensure_partition(connection, messages[0].user_id)
        connection.executemany(
            f"INSERT INTO {table_name} (message_id, lexical_text) VALUES (?, ?)",
            [(message.id, build_lexical_text(message.content)) for message in messages],
        )

    def _insert_embeddings(self, connection: sqlite3.Connection, messages: list[RawMessage], prepared: PreparedEmbedding) -> None:
        """Only called after the final ledger check, in the raw/FTS/ledger transaction."""
        if prepared.status not in {"ready", "failed", "unavailable"}:
            raise ValueError("invalid embedding state")
        if prepared.status == "ready":
            if len(prepared.vectors) != len(messages):
                raise ValueError("embedding count mismatch at commit")
            vectors = [normalize_vector(v, prepared.dimensions) for v in prepared.vectors]
            connection.executemany("""
                INSERT INTO raw_vectors(user_id, raw_id, embedding_version, dimensions, content_sha256, vector_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [(message.user_id, message.id, prepared.version, prepared.dimensions,
                    hashlib.sha256(message.content.encode("utf-8")).hexdigest(), json.dumps(vector))
                   for message, vector in zip(messages, vectors)])
        elif prepared.vectors:
            raise ValueError("failed embedding must not commit partial vectors")
        connection.execute("""
            INSERT INTO embedding_requests(user_id, request_id, embedding_version, status, expected_count,
                                           error_code, metadata_json, usage_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (messages[0].user_id, messages[0].request_id, prepared.version, prepared.status, len(messages),
              prepared.error_code, json.dumps(prepared.metadata, sort_keys=True), json.dumps(prepared.usage, sort_keys=True)))

    def _ensure_partition(self, connection: sqlite3.Connection, user_id: str) -> str:
        table_name = _checked_partition_name(lexical_partition_name(user_id))
        row = connection.execute(
            "SELECT table_name FROM fts_partitions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is not None:
            if row["table_name"] != table_name:
                raise RuntimeError("lexical partition registry does not match user_id")
            if not _partition_exists(connection, table_name):
                raise RuntimeError("lexical partition is missing for a registered user")
            return table_name
        if _partition_exists(connection, table_name):
            raise RuntimeError("lexical partition table exists without a registry row")
        connection.execute(
            f"""
            CREATE VIRTUAL TABLE {table_name} USING fts5(
                message_id UNINDEXED,
                lexical_text,
                tokenize='unicode61'
            )
            """
        )
        connection.execute(
            "INSERT INTO fts_partitions (user_id, table_name) VALUES (?, ?)",
            (user_id, table_name),
        )
        return table_name

    def _drop_partition(self, connection: sqlite3.Connection, user_id: str) -> None:
        table_name = _checked_partition_name(lexical_partition_name(user_id))
        row = connection.execute(
            "SELECT table_name FROM fts_partitions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is not None and row["table_name"] != table_name:
            raise RuntimeError("lexical partition registry does not match user_id")
        if _partition_exists(connection, table_name):
            connection.execute(f"DROP TABLE {table_name}")
        connection.execute("DELETE FROM fts_partitions WHERE user_id = ?", (user_id,))

    def _migrate_shared_fts(self, connection: sqlite3.Connection) -> None:
        """Move a pre-partition raw_message_fts into per-user tables.

        Copies the stored lexical text, then drops the shared table. Rows already
        present in a user partition are left in place so a retry after a crash
        does not insert them twice.
        """

        exists = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'raw_message_fts'
            """
        ).fetchone()
        if exists is None:
            return
        rows = connection.execute(
            "SELECT message_id, user_id, lexical_text FROM raw_message_fts"
        ).fetchall()
        grouped: dict[str, list[tuple[str, str]]] = {}
        for row in rows:
            user_id = row["user_id"]
            message_id = row["message_id"]
            if not isinstance(user_id, str) or not user_id:
                raise ValueError("shared FTS row is missing user_id")
            if not isinstance(message_id, str) or not message_id:
                raise ValueError("shared FTS row is missing message_id")
            lexical_text = row["lexical_text"] if isinstance(row["lexical_text"], str) else ""
            grouped.setdefault(user_id, []).append((message_id, lexical_text))
        for user_id, items in grouped.items():
            table_name = self._ensure_partition(connection, user_id)
            present = {
                existing["message_id"]
                for existing in connection.execute(f"SELECT message_id FROM {table_name}")
            }
            pending = [item for item in items if item[0] not in present]
            if pending:
                connection.executemany(
                    f"INSERT INTO {table_name} (message_id, lexical_text) VALUES (?, ?)",
                    pending,
                )
        connection.execute("DROP TABLE raw_message_fts")

    def rebuild_user_lexical_index(self, user_id: str) -> None:
        """Rebuild one user's FTS partition from that user's active raw messages.

        Ranking fixture only. It does not open other users' partitions, write
        tombstones, or implement erasure.
        """

        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id is required")
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._require_partition_registry(connection)
                rows = connection.execute(
                    """
                    SELECT id, content
                    FROM raw_messages
                    WHERE user_id = ? AND deletion_state = 'active'
                    ORDER BY message_index ASC, id ASC
                    """,
                    (user_id,),
                ).fetchall()
                self._drop_partition(connection, user_id)
                if rows:
                    table_name = self._ensure_partition(connection, user_id)
                    connection.executemany(
                        f"INSERT INTO {table_name} (message_id, lexical_text) VALUES (?, ?)",
                        [(row["id"], build_lexical_text(row["content"])) for row in rows],
                    )
                connection.execute("COMMIT")
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

    def delete_user_messages_for_rank_fixture(self, user_id: str) -> None:
        """Delete one user's raw messages and lexical partition.

        Ranking fixture only. add_requests rows stay, and no tombstone,
        suppression, or erasure record is written. Other users are untouched.
        """

        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id is required")
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._require_partition_registry(connection)
                connection.execute("DELETE FROM raw_messages WHERE user_id = ?", (user_id,))
                self._drop_partition(connection, user_id)
                connection.execute("COMMIT")
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

    @staticmethod
    def _require_partition_registry(connection: sqlite3.Connection) -> None:
        row = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'fts_partitions'
            """
        ).fetchone()
        if row is None:
            raise RuntimeError("store is not initialized")

    def search_lexical(
        self, user_id: str, match_query: str, limit: int
    ) -> list[EvidenceCandidate]:
        if limit < 1:
            return []
        with self._lock:
            connection = self._connect()
            try:
                return self._lexical_on(connection, user_id, match_query, limit)
            finally:
                connection.close()

    @staticmethod
    def _lexical_on(connection: sqlite3.Connection, user_id: str, match_query: str | None, limit: int) -> list[EvidenceCandidate]:
        table_name = _checked_partition_name(lexical_partition_name(user_id))
        if not match_query or not _partition_exists(connection, table_name) or limit < 1:
            return []
        rows = connection.execute(f"""
            SELECT m.id, m.user_id, m.content, m.deletion_state, m.message_index,
                   COALESCE(m.source_timestamp, m.recorded_at) AS created_at,
                   bm25({table_name}) AS bm25_score
            FROM {table_name} JOIN raw_messages AS m ON m.id={table_name}.message_id
            WHERE {table_name} MATCH ? AND m.user_id=? AND m.deletion_state='active'
            ORDER BY bm25_score ASC, m.message_index ASC, m.id ASC LIMIT ?
        """, (match_query, user_id, limit)).fetchall()
        return [_candidate_from_row(row) for row in rows]

    def vector_coverage(self, user_id: str, version: str, dimensions: int, *, span=False) -> tuple[int, int]:
        """Cheap preflight to avoid model calls for unknown/incomplete users; search rechecks."""
        with self._lock:
            connection = self._connect()
            try:
                if span:
                    row = connection.execute("""SELECT COUNT(*) AS total,COUNT(v.span_id) AS ready
                        FROM source_spans s JOIN raw_messages m ON m.id=s.source_raw_id AND m.user_id=s.user_id
                        LEFT JOIN span_vectors v ON v.span_id=s.id AND v.user_id=s.user_id
                        AND v.embedding_version=? AND v.dimensions=?
                        WHERE s.user_id=? AND m.deletion_state='active'""",(version,dimensions,user_id)).fetchone()
                    return int(row["total"]),int(row["ready"])
                row = connection.execute("""
                    SELECT COUNT(*) AS total, COUNT(v.raw_id) AS ready
                    FROM raw_messages m LEFT JOIN raw_vectors v
                    ON v.raw_id=m.id AND v.user_id=m.user_id AND v.embedding_version=? AND v.dimensions=?
                    WHERE m.user_id=? AND m.deletion_state='active'
                """, (version, dimensions, user_id)).fetchone()
                return int(row["total"]), int(row["ready"])
            finally:
                connection.close()

    def search_snapshot(
        self, *, user_id: str, match_query: str | None, query_vector: list[float] | None,
        version: str, dimensions: int, mode: str, candidate_limit: int,
        lexical_quota: int, dense_quota: int, degraded_reason: str | None = None,
        span: bool = False,
    ) -> RouteSnapshot:
        """One read snapshot for coverage, raw sources and both candidate routes.

        HTTP is completed before this method. Exact cosine scans only this
        user's active rows; another user's vectors cannot consume the quota.
        """
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN")
                if span:
                    return segments.search(connection,user_id,match_query,query_vector,version,dimensions,
                                           mode,candidate_limit,lexical_quota,dense_quota,degraded_reason)
                if mode == "raw_fts":
                    lexical = self._lexical_on(connection, user_id, match_query, candidate_limit)
                    return RouteSnapshot(lexical, [], "raw_fts", None, 0, 0)
                rows = connection.execute("""
                    SELECT m.id, m.user_id, m.content, m.deletion_state, m.message_index,
                           COALESCE(m.source_timestamp, m.recorded_at) AS created_at,
                           v.vector_json, v.content_sha256
                    FROM raw_messages m LEFT JOIN raw_vectors v
                    ON v.raw_id=m.id AND v.user_id=m.user_id AND v.embedding_version=? AND v.dimensions=?
                    WHERE m.user_id=? AND m.deletion_state='active'
                """, (version, dimensions, user_id)).fetchall()
                ready = sum(row["vector_json"] is not None for row in rows)
                reason = degraded_reason
                if rows and ready != len(rows):
                    reason = reason or "incomplete_current_version"
                if rows and query_vector is None:
                    reason = reason or "query_vector_unavailable"
                dense: list[EvidenceCandidate] = []
                if rows and reason is None:
                    try:
                        query = normalize_vector(query_vector, dimensions)
                        for row in rows:
                            if row["content_sha256"] != hashlib.sha256(row["content"].encode("utf-8")).hexdigest():
                                raise EmbeddingFailure("stale_vector_content")
                            vector = normalize_vector(json.loads(row["vector_json"]), dimensions)
                            score = max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(query, vector))))
                            dense.append(EvidenceCandidate(
                                id=row["id"], user_id=row["user_id"], content=row["content"], score=score,
                                created_at=row["created_at"], deletion_state=row["deletion_state"],
                                source_raw_id=row["id"], message_index=row["message_index"],
                            ))
                    except (EmbeddingFailure, ValueError, TypeError):
                        reason = "invalid_persisted_vector"
                        dense = []
                actual = "raw_fts" if reason else mode
                if actual == "raw_fts":
                    lexical = self._lexical_on(connection, user_id, match_query, candidate_limit)
                    return RouteSnapshot(lexical, [], actual, reason, len(rows), ready)
                dense.sort(key=lambda item: (-item.score, item.message_index, item.id))
                dense = dense[:dense_quota if mode == "raw_hybrid_rrf" else candidate_limit]
                lexical = self._lexical_on(connection, user_id, match_query, lexical_quota) if mode == "raw_hybrid_rrf" else []
                return RouteSnapshot(lexical, dense, actual, None, len(rows), ready)
            finally:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                connection.close()

    def expand_span_windows(self,user_id,candidates,limit=200):
        from app.support_groups import span_windows
        with self._lock:
            db=self._connect()
            try:
                db.execute("BEGIN")
                return span_windows(db,user_id,candidates,limit)
            finally:
                if db.in_transaction:db.rollback()
                db.close()

    def project_span_claims(self,user_id,candidates,view):
        from app.claim_projection import project_claims
        with self._lock:
            db=self._connect()
            try:
                db.execute("BEGIN")
                return project_claims(db,user_id,candidates,view)
            finally:
                if db.in_transaction:db.rollback()
                db.close()

    def context_neighbors(self, user_id: str, candidates: list[EvidenceCandidate], radius: int, limit: int, historical: bool = False) -> list[EvidenceCandidate]:
        """Bounded session-local context, anchored only on a still-eligible hit.

        Source order is committed SQLite row order, not event time or a parsed
        request ID. This optional heuristic is not task-state inference.
        """
        with self._lock:
            db = self._connect()
            try:
                db.execute("BEGIN")
                cache, result, seen = {}, [], set()
                for seed in candidates:
                    source = db.execute("""SELECT m.session_id,p.retraction_state FROM raw_messages m JOIN source_policies p
                        ON m.id=p.raw_id AND m.user_id=p.user_id
                        WHERE m.user_id=? AND m.id=? AND m.content=? AND m.deletion_state='active'
                          AND p.visibility='visible' AND p.disclosure_policy='within_user'
                          AND (p.retraction_state='not_retracted' OR ?=1)""", (user_id,seed.id,seed.content,int(historical))).fetchone()
                    if source is None:
                        continue
                    session = source["session_id"]
                    if session not in cache:
                        cache[session] = db.execute("""SELECT m.id,m.content,m.message_index,m.deletion_state,
                                COALESCE(m.source_timestamp,m.recorded_at) AS created_at
                            FROM raw_messages m WHERE m.user_id=? AND m.session_id=? ORDER BY m.rowid""",
                            (user_id,session)).fetchall()
                    rows = cache[session]
                    position = next((i for i,r in enumerate(rows) if r["id"]==seed.id),None)
                    if position is None:
                        continue
                    nearby = [seed]
                    for offset in range(1,radius+1):
                        for index in (position- offset,position+offset):
                            if 0<=index<len(rows):
                                row=rows[index]
                                if row["deletion_state"] != "active":
                                    continue
                                nearby.append(EvidenceCandidate(
                                    id=row["id"],user_id=user_id,content=row["content"],score=seed.score,
                                    created_at=row["created_at"],deletion_state=row["deletion_state"],
                                    source_raw_id=row["id"],message_index=row["message_index"]))
                    for item in nearby:
                        if item.id not in seen:
                            seen.add(item.id)
                            result.append(item)
                            if len(result)==limit:
                                return result
                return result
            finally:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                db.close()

    def has_erased_statements(self, messages: list[RawMessage]) -> bool:
        with self._lock:
            db = self._connect()
            try:
                return any(db.execute(
                    "SELECT 1 FROM memory_tombstones WHERE user_id=? AND statement_sha256=? LIMIT 1",
                    (message.user_id, statement_digest(message.content))).fetchone() for message in messages)
            finally:
                db.close()

    def lifecycle_receipts(self, user_id: str, request_id: str) -> list[dict]:
        with self._lock:
            db = self._connect()
            try:
                return operation_receipts(db,user_id,request_id)
            finally:
                db.close()

    def policy_snapshot(self, user_id: str, candidates: list[EvidenceCandidate], *, historical: bool = False):
        from app.safety import PolicySnapshot, SourcePolicy
        with self._lock:
            db = self._connect()
            try:
                db.execute("BEGIN")
                row = db.execute("SELECT epoch FROM lifecycle_users WHERE user_id=?", (user_id,)).fetchone()
                epoch = int(row[0]) if row else 0
                policies = {}
                for candidate in candidates:
                    source = db.execute("""SELECT p.*,m.*
                        FROM source_policies p JOIN raw_messages m ON m.id=p.raw_id AND m.user_id=p.user_id
                        WHERE p.user_id=? AND p.raw_id=?""", (user_id,candidate.source_raw_id)).fetchone()
                    if source is not None:
                        expected_content = source["content"]
                        if candidate.representation == "attributed_raw_v1" and source["retraction_state"] == "retracted" and historical:
                            from app.rendering import render_retracted_source
                            expected_content = render_retracted_source(raw_id=source["raw_id"],content=source["content"],
                                                                      role=source["role"],timestamp=source["source_timestamp"])
                        content_valid = (candidate.id == candidate.source_raw_id
                                         and candidate.content == expected_content
                                         and hashlib.sha256(source["content"].encode("utf-8")).hexdigest() == source["content_sha256"]
                                         and (source["retraction_state"] != "retracted" or candidate.representation == "attributed_raw_v1"))
                        if candidate.representation == "span_exact_v1":
                            content_valid = segments.valid_candidate(db,candidate,source) and source["retraction_state"] != "retracted"
                        policies[candidate.source_raw_id] = SourcePolicy(
                            user_id=user_id,raw_id=candidate.source_raw_id,
                            version=int(source["source_version"]),
                            visibility=source["visibility"] if source["deletion_state"] == "active" else "suppressed",
                            disclosure_policy=source["disclosure_policy"] if source["privacy_level"] == "normal" else "restricted",
                            retraction_state=source["retraction_state"], content_valid=content_valid)
                return PolicySnapshot(epoch,policies,allow_historical=historical)
            finally:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                db.close()

    def governed_candidates(self, user_id: str, candidates: list[EvidenceCandidate],
                            view: TemporalView, limit: int) -> list[EvidenceCandidate]:
        """Source expansion within a seeded claim group, never a global recency fallback.

        Returns raw text; only a fully parsed, single-claim source can be excluded
        by a temporal view. No derived summary is asserted as an answer.
        """
        with self._lock:
            db = self._connect()
            try:
                db.execute("BEGIN")
                epoch_row = db.execute("SELECT epoch FROM lifecycle_users WHERE user_id=?", (user_id,)).fetchone()
                selection_epoch = int(epoch_row[0]) if epoch_row else 0
                result, seen = [], set()
                invalid_derived_ids = []
                for seed in candidates:
                    seed_claims = db.execute("""SELECT * FROM derived_memories
                        WHERE user_id=? AND source_raw_id=? AND admission='accepted'""",
                        (user_id,seed.source_raw_id)).fetchall()
                    invalid_seed = [row["id"] for row in seed_claims if not valid_derived_source(db,row)]
                    invalid_derived_ids.extend(invalid_seed)
                    if invalid_seed:
                        seed_claims = []
                    additions = []
                    for keyrow in seed_claims:
                        rows = db.execute("""SELECT d.*,m.id AS raw_id,m.user_id,m.content,m.deletion_state,m.message_index,
                                    COALESCE(m.source_timestamp,m.recorded_at) AS created_at,m.role,m.source_timestamp
                            FROM derived_memories d JOIN raw_messages m ON m.id=d.source_raw_id AND m.user_id=d.user_id
                            WHERE d.user_id=? AND d.claim_key=? AND d.admission='accepted'
                              AND d.visibility='visible' AND m.deletion_state='active'
                            ORDER BY CASE WHEN d.valid_from IS NULL THEN 1 ELSE 0 END,d.valid_from DESC,d.id""",
                            (user_id,keyrow["claim_key"])).fetchall()
                        invalid_group = [row["id"] for row in rows if not valid_derived_source(db,row)]
                        if invalid_group:
                            # Do not apply cached supersession from an invalid
                            # successor. Fall back to the original, gated raw hit.
                            invalid_derived_ids.extend(invalid_group)
                            additions = [seed]
                            break
                        for row in rows:
                            if view.intent == "current" and row["currency"] == "superseded":
                                continue
                            if view.intent == "as_of" and (
                                (row["valid_from"] is not None and row["valid_from"] > view.as_of) or
                                (row["valid_until"] is not None and row["valid_until"] <= view.as_of)):
                                continue
                            content,representation = row["content"],"raw"
                            if row["retraction_state"] == "retracted" and view.intent in {"history","as_of"}:
                                from app.rendering import render_retracted_source
                                content = render_retracted_source(raw_id=row["raw_id"],content=content,role=row["role"],timestamp=row["source_timestamp"])
                                representation = "attributed_raw_v1"
                            additions.append(EvidenceCandidate(
                                id=row["raw_id"],user_id=user_id,content=content,score=seed.score,representation=representation,
                                created_at=row["created_at"],deletion_state=row["deletion_state"],
                                source_raw_id=row["raw_id"],message_index=row["message_index"]))
                    if not seed_claims:
                        additions = [seed]
                    elif view.intent == "all":
                        additions = [seed] + additions
                    for hit in additions:
                        if view.intent in {"history","as_of"} and hit.representation == "raw":
                            source = db.execute("""SELECT m.role,m.source_timestamp,p.retraction_state FROM raw_messages m
                                JOIN source_policies p ON p.raw_id=m.id AND p.user_id=m.user_id
                                WHERE m.id=? AND m.user_id=?""",(hit.source_raw_id,user_id)).fetchone()
                            if source and source["retraction_state"] == "retracted":
                                from app.rendering import render_retracted_source
                                hit=hit.model_copy(update={"representation":"attributed_raw_v1","content":render_retracted_source(
                                    raw_id=hit.source_raw_id,content=hit.content,role=source["role"],timestamp=source["source_timestamp"])})
                        if hit.id not in seen:
                            result.append(hit)
                            seen.add(hit.id)
                            if len(result) == limit:
                                return GovernedSelection(result,selection_epoch,invalid_derived_ids)
                return GovernedSelection(result,selection_epoch,invalid_derived_ids)
            finally:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                db.close()

    def _connect(self, timeout: float = 30) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=timeout)
        connection.row_factory = sqlite3.Row
        connection.isolation_level = None
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA secure_delete = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection


def _candidate_from_row(row: sqlite3.Row) -> EvidenceCandidate:
    score = -float(row["bm25_score"])
    return EvidenceCandidate(
        id=row["id"],
        user_id=row["user_id"],
        content=row["content"],
        score=score,
        created_at=int(row["created_at"]),
        deletion_state=row["deletion_state"],
        source_raw_id=row["id"],
        message_index=int(row["message_index"]),
    )


def _legacy_rows_match(rows: list[sqlite3.Row], messages: list[RawMessage]) -> bool:
    if len(rows) != len(messages):
        return False
    return all(
        row["session_id"] == message.session_id
        and int(row["message_index"]) == message.message_index
        and row["role"] == message.role
        and row["content"] == message.content
        and row["source_timestamp"] == message.source_timestamp
        for row, message in zip(rows, messages)
    )

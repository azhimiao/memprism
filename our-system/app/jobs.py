"""SQLite preparation leases, route proofs and dispatch ledger (no text/secrets).

The active config is persistent, shared by processes. Activation of a different
formal config fences all older workers. HTTP runs outside these transactions.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

from app.embeddings import EmbeddingBatch, EmbeddingFailure, normalize_vector
from app.governance import statement_digest
from app.index import build_lexical_text

LEASE_TTL = 60.0
RENEW_INTERVAL = 20.0
COMPETITION_WAIT = 5.0
USER_PREPARATION_WAIT = 120.0


class NotReady(RuntimeError):
    """Safe public reason; never contains a provider exception/body."""


class PayloadConflict(RuntimeError):
    pass


class SnapshotChanged(NotReady):
    pass


@dataclass(frozen=True)
class Lease:
    user_id: str
    request_id: str
    payload_hash: str
    owner: str
    generation: int
    config_generation: int
    epoch: int


def initialize_jobs(db):
    statements = [
        "CREATE TABLE IF NOT EXISTS k0_schema(version INTEGER PRIMARY KEY)",
        """CREATE TABLE IF NOT EXISTS k0_config (
            singleton INTEGER PRIMARY KEY CHECK(singleton=1), config_hash TEXT NOT NULL,
            generation INTEGER NOT NULL, identity_json TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS k0_jobs (
            user_id TEXT NOT NULL, request_id TEXT NOT NULL, payload_hash TEXT NOT NULL,
            config_hash TEXT NOT NULL, config_generation INTEGER NOT NULL,
            owner TEXT NOT NULL, generation INTEGER NOT NULL, expires_at REAL NOT NULL,
            source_epoch INTEGER NOT NULL, state TEXT NOT NULL, error_code TEXT,
            PRIMARY KEY(user_id,request_id))""",
        """CREATE TABLE IF NOT EXISTS k0_users (
            user_id TEXT PRIMARY KEY, commit_seq INTEGER NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS k0_receipts (
            user_id TEXT NOT NULL, request_id TEXT NOT NULL, config_hash TEXT NOT NULL,
            profile_id TEXT NOT NULL, required_routes_json TEXT NOT NULL,
            model_version TEXT NOT NULL, commit_seq INTEGER NOT NULL, epoch INTEGER NOT NULL,
            PRIMARY KEY(user_id,request_id))""",
        """CREATE TABLE IF NOT EXISTS k0_watermarks (
            user_id TEXT NOT NULL, config_hash TEXT NOT NULL, route TEXT NOT NULL,
            commit_seq INTEGER NOT NULL, epoch INTEGER NOT NULL, ready INTEGER NOT NULL,
            PRIMARY KEY(user_id,config_hash,route))""",
        """CREATE TABLE IF NOT EXISTS k0_switch_proofs (
            proof_id TEXT PRIMARY KEY, config_hash TEXT NOT NULL, user_id TEXT NOT NULL,
            commit_seq INTEGER NOT NULL, epoch INTEGER NOT NULL, counts_json TEXT NOT NULL,
            validated_at REAL NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS k0_attempts (
            attempt_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, request_id TEXT NOT NULL,
            owner TEXT NOT NULL, generation INTEGER NOT NULL, config_hash TEXT NOT NULL,
            model_version TEXT NOT NULL, input_count INTEGER NOT NULL,
            permitted_at REAL NOT NULL, status TEXT NOT NULL, error_code TEXT,
            usage_json TEXT NOT NULL)""",
    ]
    for sql in statements:
        db.execute(sql)
    versions = [r[0] for r in db.execute("SELECT version FROM k0_schema")]
    if versions and versions != [1]:
        raise RuntimeError("unsupported K0 schema")
    db.execute("INSERT OR IGNORE INTO k0_schema VALUES (1)")


def epoch_on(db, user_id):
    row = db.execute("SELECT epoch FROM lifecycle_users WHERE user_id=?", (user_id,)).fetchone()
    return int(row[0]) if row else 0


def seq_on(db, user_id):
    row = db.execute("SELECT commit_seq FROM k0_users WHERE user_id=?", (user_id,)).fetchone()
    return int(row[0]) if row else 0


def embedding_messages_on(db, messages, plan):
    """Mirror the existing ordered exact control grammar, including tombstones.

    No extraction or new deletion semantics. Controls never enter model input.
    Erased earlier batch members and their later reintroductions are excluded.
    """
    user = messages[0].user_id
    forbidden = {r[0] for r in db.execute("SELECT statement_sha256 FROM memory_tombstones WHERE user_id=?", (user,))}
    existing = [(r["id"], r["statement_sha256"]) for r in db.execute(
        "SELECT raw_id AS id,statement_sha256 FROM source_policies WHERE user_id=?", (user,))]
    seen, excluded = list(existing), set()
    mutations = {m.source_raw_id: m for m in plan.mutations}
    for message in messages:
        digest = statement_digest(message.content)
        if digest in forbidden:
            excluded.add(message.id)
        elif message.id in mutations:
            mutation = mutations[message.id]
            targets = [(rid, sha) for rid, sha in seen if (
                rid == mutation.target_id if mutation.target_id else sha == statement_digest(mutation.target_text or ""))]
            excluded.add(message.id)
            if mutation.operation in {"erase", "suppress"}:
                excluded.update(rid for rid, sha in targets)
            if mutation.operation == "erase":
                forbidden.update(sha for rid, sha in targets)
                forbidden.add(digest)  # the control source itself is erased
        seen.append((message.id, digest))
    return [m for m in messages if m.id not in excluded]


def route_counts_on(db, user_id, profile):
    """Validate actual searchable raw units, not just row counts or old receipts."""
    if not isinstance(user_id, str) or not user_id.strip():
        # Empty databases have no user. Do not score per-user routes for a blank id.
        return {route: 0 for route in (*profile.required_routes, "active")}
    from app.store import lexical_partition_name, _partition_exists
    if profile.identity["profile_version"] == "k1-v1":
        from app.segments import counts
        result=counts(db,user_id,profile)
        if 'sentence' in profile.required_routes:
            from app.evidence_navigation import counts as sentence_counts
            raw_count=db.execute("SELECT COUNT(*) FROM raw_messages WHERE user_id=? AND deletion_state='active'",(user_id,)).fetchone()[0]
            result['sentence']=result['active'] if sentence_counts(db,user_id)==raw_count else -1
        if 'atom' in profile.required_routes:
            from app.atomic_memory import counts as atom_counts
            extractor_version = profile.identity.get("atom_extractor_version")
            if not isinstance(extractor_version, str) or not extractor_version.strip():
                raise NotReady("missing_configuration")
            dense_args={'embedding_version':profile.model_version,'dimensions':profile.dimensions} if 'dense' in profile.required_routes else {}
            atom=atom_counts(db,user_id,extractor_version,**dense_args)
            raw_count=db.execute("SELECT COUNT(*) FROM raw_messages WHERE user_id=? AND deletion_state='active'",(user_id,)).fetchone()[0]
            complete=atom['atom']==atom['fts']==raw_count
            if dense_args:complete=complete and atom['dense']==raw_count
            result['atom']=result['active'] if complete else -1
        return result
    rows = db.execute("""SELECT m.*,p.content_sha256 AS policy_hash,p.source_version
        FROM raw_messages m LEFT JOIN source_policies p ON p.user_id=m.user_id AND p.raw_id=m.id
        WHERE m.user_id=? AND m.deletion_state='active'""", (user_id,)).fetchall()
    counts = {"raw": 0, "fts": 0, "dense": 0, "active": len(rows)}
    table = lexical_partition_name(user_id)
    registry = db.execute("SELECT table_name FROM fts_partitions WHERE user_id=?", (user_id,)).fetchone()
    fts = {}
    if registry and registry[0] == table and _partition_exists(db, table):
        for r in db.execute(f"SELECT message_id,lexical_text FROM {table}"):
            fts.setdefault(r[0], []).append(r[1])
    for row in rows:
        sha = hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
        if row["policy_hash"] == sha and row["source_version"] is not None:
            counts["raw"] += 1
        if fts.get(row["id"]) == [build_lexical_text(row["content"])]:
            counts["fts"] += 1
        vector = db.execute("""SELECT * FROM raw_vectors WHERE user_id=? AND raw_id=?
            AND embedding_version=? AND dimensions=?""",
            (user_id, row["id"], profile.model_version, profile.dimensions)).fetchone()
        if vector and vector["content_sha256"] == sha:
            try:
                normalize_vector(json.loads(vector["vector_json"]), profile.dimensions)
                counts["dense"] += 1
            except (EmbeddingFailure, ValueError, TypeError):
                pass
    return counts


def write_publication_on(db, user_id, request_id, profile=None):
    """Called by every Add (including research), inside the publish transaction."""
    db.execute("""INSERT INTO k0_users VALUES (?,1)
        ON CONFLICT(user_id) DO UPDATE SET commit_seq=commit_seq+1""", (user_id,))
    if profile is None:
        return
    epoch, seq = epoch_on(db, user_id), seq_on(db, user_id)
    counts = route_counts_on(db, user_id, profile)
    for route in profile.required_routes:
        ready = counts[route] == counts["active"] and (route != "dense" or profile.dimensions > 0)
        db.execute("INSERT OR REPLACE INTO k0_watermarks VALUES (?,?,?,?,?,?)",
                   (user_id, profile.config_hash, route, seq, epoch, int(ready)))
    db.execute("INSERT INTO k0_receipts VALUES (?,?,?,?,?,?,?,?)",
               (user_id, request_id, profile.config_hash, profile.profile_id,
                json.dumps(profile.required_routes), profile.model_version, seq, epoch))


class JobCoordinator:
    def __init__(self, store, profile, *, lock_timeout_seconds: float = 30.0):
        self.store, self.profile = store, profile
        self.lock_timeout_seconds = lock_timeout_seconds

    @contextmanager
    def transaction(self, write=True, *, timeout_seconds=None):
        db = None
        try:
            timeout = self.lock_timeout_seconds if timeout_seconds is None else min(self.lock_timeout_seconds, timeout_seconds)
            db = self.store._connect(timeout=timeout)
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.execute("COMMIT")
        except sqlite3.Error:
            raise NotReady("control_not_ready") from None
        finally:
            if db is not None and db.in_transaction:
                db.execute("ROLLBACK")
            if db is not None:
                db.close()

    def activate(self):
        with self.transaction() as db:
            old = db.execute("SELECT * FROM k0_config WHERE singleton=1").fetchone()
            if old is not None and old["config_hash"] == self.profile.config_hash:
                return int(old["generation"])
            generation = int(old["generation"]) + 1 if old else 1
            db.execute("INSERT OR REPLACE INTO k0_config VALUES (1,?,?,?)", (
                self.profile.config_hash, generation, json.dumps(self.profile.identity, sort_keys=True)))
            db.execute("""UPDATE k0_jobs SET generation=generation+1,state='abandoned',expires_at=0,
                error_code='config_changed' WHERE state='preparing'""")
            return generation

    def config_on(self, db):
        row = db.execute("SELECT * FROM k0_config WHERE singleton=1").fetchone()
        if row is None or row["config_hash"] != self.profile.config_hash:
            raise NotReady("config_changed")
        return int(row["generation"])

    def control_ready_on(self, db):
        """Profile, required-route configuration, and control schema. No user id."""
        generation = self.config_on(db)
        if "dense" in self.profile.required_routes and self.profile.dimensions == 0:
            raise NotReady("missing_configuration")
        if "atom" in self.profile.required_routes and not str(self.profile.identity.get("atom_extractor_version") or "").strip():
            raise NotReady("missing_configuration")
        required = ["k0_config", "k0_jobs", "k0_attempts", "k0_watermarks", "k0_receipts",
                    "raw_messages", "source_policies"]
        if self.profile.identity.get("profile_version") == "k1-v1":
            required.extend(("source_spans", "span_fts_partitions"))
            if "dense" in self.profile.required_routes:
                required.append("span_vectors")
        if "atom" in self.profile.required_routes:
            required.extend(("atomic_memories", "atomic_extractions", "atomic_fts_partitions"))
        present = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if any(name not in present for name in required):
            raise NotReady("control_not_ready")
        return generation

    def ready_on(self, db, user_id, *, require_proof=True):
        generation = self.config_on(db)
        if "dense" in self.profile.required_routes and self.profile.dimensions == 0:
            raise NotReady("missing_configuration")
        counts = route_counts_on(db, user_id, self.profile)
        epoch, seq = epoch_on(db, user_id), seq_on(db, user_id)
        for route in self.profile.required_routes:
            if counts[route] != counts["active"]:
                raise NotReady("incomplete_" + route + "_route")
            mark = db.execute("SELECT * FROM k0_watermarks WHERE user_id=? AND config_hash=? AND route=?",
                              (user_id, self.profile.config_hash, route)).fetchone()
            if require_proof and counts["active"] and (not mark or not mark["ready"] or
                    mark["epoch"] != epoch or mark["commit_seq"] != seq):
                raise NotReady("route_rebuild_required")
        return (generation, epoch, seq)

    def require_ready(self, user_id):
        with self.transaction(False) as db:
            return self.ready_on(db, user_id)

    def certify_routes(self, user_id):
        """Explicit no-HTTP validation of an already rebuilt COPY, with audit proof.

        Does not build vectors or rewrite old receipts. A missing route fails.
        """
        with self.transaction() as db:
            generation, epoch, seq = self.ready_on(db, user_id, require_proof=False)
            counts = route_counts_on(db, user_id, self.profile)
            proof_id = uuid.uuid4().hex
            for route in self.profile.required_routes:
                db.execute("INSERT OR REPLACE INTO k0_watermarks VALUES (?,?,?,?,?,1)",
                           (user_id, self.profile.config_hash, route, seq, epoch))
            db.execute("INSERT INTO k0_switch_proofs VALUES (?,?,?,?,?,?,?)", (
                proof_id, self.profile.config_hash, user_id, seq, epoch, json.dumps(counts), time.time()))
            return {"proof_id": proof_id, "config_generation": generation, "epoch": epoch,
                    "commit_seq": seq, "counts": counts, "config_hash": self.profile.config_hash}

    @contextmanager
    def _reservation_transaction(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise NotReady("preparation_busy")
        try:
            with self.transaction(timeout_seconds=remaining) as db:
                yield db
        except NotReady as exc:
            cause = exc.__context__
            code = getattr(cause, "sqlite_errorcode", 0) or 0
            if (isinstance(cause, sqlite3.Error) and code & 255 in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
                    and time.monotonic() >= deadline):
                raise NotReady("preparation_busy") from None
            raise

    def reserve(self, user_id, request_id, payload_hash, *, pure_control=False, messages=None, response=None):
        deadline = time.monotonic() + COMPETITION_WAIT
        user_deadline = time.monotonic() + USER_PREPARATION_WAIT
        serialized = self.profile.identity.get("add_preparation_policy") == "user_lease_v1" and not pure_control
        wait_deadline = user_deadline if serialized else deadline
        while True:
            remaining = wait_deadline - time.monotonic()
            if remaining <= 0:
                raise NotReady("preparation_busy")
            with self._reservation_transaction(wait_deadline) as db:
                if time.monotonic() >= wait_deadline:
                    raise NotReady("preparation_busy")
                receipt = db.execute("SELECT * FROM add_requests WHERE user_id=? AND request_id=?",
                                     (user_id, request_id)).fetchone()
                job = db.execute("SELECT * FROM k0_jobs WHERE user_id=? AND request_id=?", (user_id, request_id)).fetchone()
                if (receipt and receipt["payload_hash"] != payload_hash) or (job and job["payload_hash"] != payload_hash):
                    raise PayloadConflict("request_id has already been used with a different Add payload")
                legacy_rows = []
                if not receipt and messages is not None:
                    from app.store import _legacy_rows_match
                    legacy_rows = db.execute("""SELECT session_id,message_index,role,content,source_timestamp
                        FROM raw_messages WHERE user_id=? AND request_id=? ORDER BY message_index""",
                        (user_id, request_id)).fetchall()
                    if legacy_rows and not _legacy_rows_match(legacy_rows, messages):
                        raise PayloadConflict("request_id has already been used with a different Add payload")
                generation = self.config_on(db)
                now = time.time()
                # Competing app instances share this durable lease. Preparing
                # two different Adds for one user wastes work: publication of
                # either invalidates the other's source epoch. Wait BEFORE any
                # model work. Exact erase controls must still preempt a lease.
                other = None
                own_preparing = (job and job["state"] == "preparing" and job["expires_at"] > now
                                 and job["config_generation"] == generation)
                if serialized and not receipt and not legacy_rows and not own_preparing:
                    other = db.execute("""SELECT 1 FROM k0_jobs WHERE user_id=? AND request_id!=?
                        AND state='preparing' AND expires_at>? AND config_generation=?
                        AND config_hash=? AND source_epoch=? LIMIT 1""",
                        (user_id, request_id, now, generation, self.profile.config_hash, epoch_on(db, user_id))).fetchone()
                if other:
                    wait_deadline = user_deadline
                else:
                    if not pure_control:
                        self.ready_on(db, user_id)
                    if receipt:
                        return json.loads(receipt["response_json"])
                    if legacy_rows:
                        # Reconcile only after current-route proof; no embedding.
                        db.execute("INSERT INTO add_requests VALUES (?,?,?,?,?)", (
                            user_id, request_id, payload_hash, json.dumps(response, sort_keys=True, ensure_ascii=False),
                            messages[0].recorded_at))
                        return response
                    now = time.time()
                    if not job or job["state"] != "preparing" or job["expires_at"] <= now or job["config_generation"] != generation:
                        if time.monotonic() >= wait_deadline:
                            raise NotReady("preparation_busy")
                        owner = uuid.uuid4().hex
                        fence = job["generation"] + 1 if job else 1
                        epoch = epoch_on(db, user_id)
                        db.execute("INSERT OR REPLACE INTO k0_jobs VALUES (?,?,?,?,?,?,?,?,?,?,NULL)", (
                            user_id, request_id, payload_hash, self.profile.config_hash, generation,
                            owner, fence, now + LEASE_TTL, epoch, "preparing"))
                        return Lease(user_id, request_id, payload_hash, owner, fence, generation, epoch)
                    wait_deadline = min(wait_deadline, deadline)
            if time.monotonic() >= wait_deadline:
                raise NotReady("preparation_busy")
            time.sleep(min(0.05, max(0, wait_deadline - time.monotonic())))

    def fence_on(self, db, lease, *, published=False):
        generation = self.config_on(db)
        job = db.execute("SELECT * FROM k0_jobs WHERE user_id=? AND request_id=?",
                         (lease.user_id, lease.request_id)).fetchone()
        if job and job["payload_hash"] != lease.payload_hash:
            raise PayloadConflict("request_id has already been used with a different Add payload")
        if (not job or job["state"] != "preparing" or job["owner"] != lease.owner or
            job["generation"] != lease.generation or job["expires_at"] <= time.time() or
            job["config_hash"] != self.profile.config_hash or job["config_generation"] != generation or
            generation != lease.config_generation):
            raise NotReady("lease_lost")
        if epoch_on(db, lease.user_id) != lease.epoch + int(published) or job["source_epoch"] != lease.epoch:
            raise SnapshotChanged("source_epoch_changed")

    def renew(self, lease):
        with self.transaction() as db:
            self.fence_on(db, lease)
            db.execute("UPDATE k0_jobs SET expires_at=? WHERE user_id=? AND request_id=?",
                       (time.time() + LEASE_TTL, lease.user_id, lease.request_id))

    def draft(self, lease, messages, plan):
        with self.transaction() as db:
            self.fence_on(db, lease)
            return embedding_messages_on(db, messages, plan)

    def permit(self, lease, messages, plan, batch):
        with self.transaction() as db:
            self.fence_on(db, lease)
            self.ready_on(db, lease.user_id)
            # Recompute immediately before dispatch, under the fence.
            allowed = embedding_messages_on(db, messages, plan)
            allowed_texts = [m.content for m in allowed]
            if self.profile.identity["profile_version"] == "k1-v1":
                from app.segments import split_source
                allowed_texts = [p["content"] for m in allowed for p in split_source(m)]
            if any(text not in allowed_texts for text in batch):
                raise NotReady("dispatch_source_changed")
            attempt = uuid.uuid4().hex
            db.execute("INSERT INTO k0_attempts VALUES (?,?,?,?,?,?,?,?,?,'dispatched',NULL,'{}')", (
                attempt, lease.user_id, lease.request_id, lease.owner, lease.generation,
                self.profile.config_hash, self.profile.model_version, len(batch), time.time()))
            return attempt

    def permit_derived(self, lease, messages, plan, grant, batch, offset):
        """Authorize one derived embedding batch. permit() stays raw/span text only.

        grant is (extractor_version, config_hash, specs). Each spec is
        (atom_id, prepared_text_hash, ((raw_id, source_version, content_hash), ...)).
        Under the fence, dependencies must still match and the batch must be the
        exact permitted slice. The attempt is recorded only after those checks.
        """
        from app.atomic_memory import source_hash
        from app.memory_core import planned_source_state
        with self.transaction() as db:
            self.fence_on(db, lease)
            self.ready_on(db, lease.user_id)
            if (not isinstance(grant, tuple) or len(grant) != 3 or not isinstance(offset, int) or offset < 0
                    or not messages or any(message.user_id != lease.user_id for message in messages)):
                raise NotReady("dispatch_source_changed")
            extractor_version, config_hash, specs = grant
            expected = self.profile.identity.get("atom_extractor_version")
            if (not isinstance(expected, str) or not expected.strip() or extractor_version != expected
                    or config_hash != expected or not isinstance(specs, tuple)
                    or len({spec[0] for spec in specs}) != len(specs)):
                raise NotReady("dispatch_source_changed")
            if not isinstance(batch, (list, tuple)) or offset + len(batch) > len(specs):
                raise NotReady("dispatch_source_changed")
            window = specs[offset:offset + len(batch)]
            dependencies = []
            for text, spec in zip(batch, window, strict=True):
                if not isinstance(spec, tuple) or len(spec) != 3:
                    raise NotReady("dispatch_source_changed")
                atom_id, text_hash, atom_deps = spec
                if (not isinstance(text, str) or not isinstance(atom_id, str) or not atom_id
                        or source_hash(text) != text_hash or not isinstance(atom_deps, tuple) or not atom_deps):
                    raise NotReady("dispatch_source_changed")
                dependencies.extend(atom_deps)
            allowed = {message.id: message for message in embedding_messages_on(db, messages, plan)}
            planned = planned_source_state(messages, plan)
            for raw_id, source_version, content_hash in dependencies:
                message = allowed.get(raw_id)
                if (not isinstance(raw_id, str) or type(source_version) is not int or source_version < 1
                        or not isinstance(content_hash, str) or message is None
                        or message.deletion_state != "active" or source_hash(message.content) != content_hash
                        or planned.get(raw_id) is None or planned[raw_id][0] != source_version):
                    raise NotReady("dispatch_source_changed")
                row = db.execute("""SELECT m.deletion_state,m.content,p.source_version,p.content_sha256
                    FROM raw_messages m LEFT JOIN source_policies p ON p.user_id=m.user_id AND p.raw_id=m.id
                    WHERE m.user_id=? AND m.id=?""", (lease.user_id, raw_id)).fetchone()
                if row is not None and (row["deletion_state"] != "active" or source_hash(row["content"]) != content_hash
                        or row["source_version"] != source_version or row["content_sha256"] != content_hash):
                    raise NotReady("dispatch_source_changed")
            attempt = uuid.uuid4().hex
            db.execute("INSERT INTO k0_attempts VALUES (?,?,?,?,?,?,?,?,?,'dispatched',NULL,'{}')", (
                attempt, lease.user_id, lease.request_id, lease.owner, lease.generation,
                self.profile.config_hash, self.profile.model_version, len(batch), time.time()))
            return attempt

    def finish_attempt(self, attempt, status, error=None, usage=None):
        with self.transaction() as db:
            db.execute("UPDATE k0_attempts SET status=?,error_code=?,usage_json=? WHERE attempt_id=?",
                       (status, error, json.dumps(usage or {}, sort_keys=True), attempt))

    def fail(self, lease, reason):
        with self.transaction() as db:
            db.execute("""UPDATE k0_jobs SET state='failed',error_code=?,expires_at=0
                WHERE user_id=? AND request_id=? AND owner=? AND generation=? AND state='preparing'""",
                (reason, lease.user_id, lease.request_id, lease.owner, lease.generation))

    def final_check(self, user_id, stamp, candidates, policy):
        with self.transaction(False) as db:
            if self.ready_on(db, user_id) != stamp or policy.epoch != stamp[1]:
                raise SnapshotChanged("search_snapshot_changed")
            for candidate in candidates:
                source = db.execute("""SELECT m.*,p.source_version,p.content_sha256,
                    p.visibility,p.disclosure_policy,p.retraction_state
                    FROM raw_messages m JOIN source_policies p ON p.user_id=m.user_id AND p.raw_id=m.id
                    WHERE m.user_id=? AND m.id=?""", (user_id, candidate.source_raw_id)).fetchone()
                old = policy.sources.get(candidate.source_raw_id)
                from app.segments import valid_candidate, digest
                content_valid = (source is not None and source["content"] == candidate.content and
                                 source["content_sha256"] == digest(candidate.content))
                if candidate.representation == "span_exact_v1":
                    content_valid = valid_candidate(db,candidate,source)
                if (not source or not old or source["source_version"] != old.version or
                    source["deletion_state"] != "active" or not content_valid or
                    source["visibility"] != old.visibility or source["disclosure_policy"] != old.disclosure_policy or
                    source["retraction_state"] != old.retraction_state or source["privacy_level"] != "normal"):
                    raise SnapshotChanged("search_source_changed")


class LeaseHeartbeat:
    def __init__(self, coordinator, lease):
        self.coordinator, self.lease = coordinator, lease
        self.stop, self.lost = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(RENEW_INTERVAL):
            try:
                self.coordinator.renew(self.lease)
            except Exception:
                self.lost.set()
                return

    def __enter__(self):
        self.thread.start()
        return self

    def check(self):
        if self.lost.is_set():
            raise NotReady("lease_renewal_failed")

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=1)


class LeasedEmbeddingProvider:
    """Guard each provider batch and each HTTP dispatch, without a write lock."""
    def __init__(self, provider, coordinator, lease, heartbeat, messages, plan):
        self.provider, self.coordinator, self.lease = provider, coordinator, lease
        self.heartbeat, self.messages, self.plan = heartbeat, messages, plan
        self.config, self.implementation_kind = provider.config, provider.implementation_kind

    def embed(self, texts):
        def authorize(batch, start):
            self.heartbeat.check()
            return self.coordinator.permit(self.lease, self.messages, self.plan, batch)
        return self._embed_batches(texts, authorize)

    def _embed_batches(self, texts, authorize):
        vectors, attempts, usage = [], [], {"batches": []}
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start:start + self.config.batch_size]
            batch_attempts = []
            def permit(batch=batch, start=start):
                attempt = authorize(batch, start)
                batch_attempts.append(attempt)
                attempts.append(attempt)
            try:
                if hasattr(self.provider, "embed_guarded"):
                    result = self.provider.embed_guarded(batch, before_dispatch=permit)
                else:
                    permit()
                    result = self.provider.embed(batch)
                if len(result.vectors) != len(batch):
                    raise EmbeddingFailure("response_count_mismatch", result.usage)
                checked = [normalize_vector(v, self.config.dimensions) for v in result.vectors]
                self.coordinator.finish_attempt(batch_attempts[-1], "ready", usage=result.usage)
                self.heartbeat.check()
                vectors.extend(checked)
                usage["batches"].append(result.usage)
            except EmbeddingFailure as exc:
                if batch_attempts:
                    self.coordinator.finish_attempt(batch_attempts[-1], "failed", exc.code, exc.usage)
                raise
            except NotReady:
                raise
            except Exception:
                if batch_attempts:
                    self.coordinator.finish_attempt(batch_attempts[-1], "failed", "provider_error")
                raise EmbeddingFailure("provider_error") from None
        return EmbeddingBatch(vectors, usage)


class LeasedDerivedEmbeddingProvider(LeasedEmbeddingProvider):
    """Derived atom text uses permit_derived, never the raw/span permit."""

    def __init__(self, provider, coordinator, lease, heartbeat, messages, plan, grant):
        super().__init__(provider, coordinator, lease, heartbeat, messages, plan)
        self.grant = grant

    def embed(self, texts):
        def authorize(batch, start):
            self.heartbeat.check()
            return self.coordinator.permit_derived(self.lease, self.messages, self.plan, self.grant, batch, start)
        return self._embed_batches(texts, authorize)

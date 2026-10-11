"""Transactional governance/lifecycle helpers. No model calls or global user state.

Erasure covers live logical tables only, not storage media, WAL, prior exports or
provider retention. Operation receipts state this scope explicitly.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from typing import Sequence

from app.governance import (
    BoundedGovernance, ClaimDraft, GovernancePlan, POLICY_VERSION,
    claim_key, normalized_statement, statement_digest,
)
from app.models import RawMessage


def initialize_core(db: sqlite3.Connection) -> None:
    statements = [
        """CREATE TABLE IF NOT EXISTS source_policies (
            user_id TEXT NOT NULL, raw_id TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL,
            statement_sha256 TEXT NOT NULL, source_version INTEGER NOT NULL DEFAULT 1,
            visibility TEXT NOT NULL DEFAULT 'visible', disclosure_policy TEXT NOT NULL DEFAULT 'within_user',
            retraction_state TEXT NOT NULL DEFAULT 'not_retracted',
            FOREIGN KEY(raw_id) REFERENCES raw_messages(id) ON DELETE CASCADE)""",
        """CREATE TABLE IF NOT EXISTS lifecycle_users (
            user_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL DEFAULT 0)""",
        """CREATE TABLE IF NOT EXISTS derived_memories (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, source_raw_id TEXT NOT NULL,
            source_version INTEGER NOT NULL, claim_key TEXT NOT NULL, object_text TEXT NOT NULL,
            payload_json TEXT NOT NULL, admission TEXT NOT NULL DEFAULT 'accepted',
            visibility TEXT NOT NULL DEFAULT 'visible', resolution TEXT NOT NULL DEFAULT 'none',
            currency TEXT NOT NULL DEFAULT 'uncertain', retraction_state TEXT NOT NULL DEFAULT 'not_retracted',
            valid_from INTEGER, valid_until INTEGER, superseded_by TEXT,
            FOREIGN KEY(source_raw_id) REFERENCES raw_messages(id) ON DELETE CASCADE)""",
        "CREATE INDEX IF NOT EXISTS derived_by_key ON derived_memories(user_id,claim_key)",
        """CREATE TABLE IF NOT EXISTS claim_supersessions (
            user_id TEXT NOT NULL, predecessor_id TEXT NOT NULL, successor_id TEXT NOT NULL,
            claim_key TEXT NOT NULL, predecessor_payload_sha256 TEXT NOT NULL,
            successor_raw_id TEXT NOT NULL, successor_source_sha256 TEXT NOT NULL,
            valid_until INTEGER NOT NULL,
            PRIMARY KEY(user_id,predecessor_id,successor_id))""",
        """CREATE TABLE IF NOT EXISTS memory_operations (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, request_id TEXT NOT NULL,
            source_raw_id TEXT NOT NULL, operation TEXT NOT NULL, status TEXT NOT NULL,
            target_ids_json TEXT NOT NULL, reason TEXT, lifecycle_epoch INTEGER NOT NULL,
            suppression_complete INTEGER NOT NULL, erasure_complete INTEGER NOT NULL,
            erasure_scope TEXT NOT NULL, policy_version TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS memory_tombstones (
            user_id TEXT NOT NULL, raw_id TEXT NOT NULL, statement_sha256 TEXT NOT NULL,
            epoch INTEGER NOT NULL, PRIMARY KEY(user_id,raw_id))""",
        "CREATE INDEX IF NOT EXISTS tombstones_by_digest ON memory_tombstones(user_id,statement_sha256)",
    ]
    for statement in statements:
        db.execute(statement)
    # Explicit legacy mapping: active != current; normal != public.
    for row in db.execute("""SELECT m.* FROM raw_messages m LEFT JOIN source_policies p ON m.id=p.raw_id
                             WHERE p.raw_id IS NULL""").fetchall():
        _register_source(db, row["user_id"], row["id"], row["content"], row["deletion_state"], row["privacy_level"])


def _register_source(db, user_id, raw_id, content, deletion_state, privacy_level):
    visibility = {"active": "visible", "deleted": "erased", "redacted": "redacted"}.get(deletion_state, "suppressed")
    disclosure = "within_user" if privacy_level == "normal" else "restricted"
    db.execute("""INSERT INTO source_policies
        (user_id,raw_id,content_sha256,statement_sha256,visibility,disclosure_policy)
        VALUES (?,?,?,?,?,?)""",
        (user_id, raw_id, hashlib.sha256(content.encode("utf-8")).hexdigest(),
         statement_digest(content), visibility, disclosure))
    db.execute("INSERT OR IGNORE INTO lifecycle_users(user_id,epoch) VALUES (?,0)", (user_id,))


def _erase(db, user_id, raw_ids, epoch, fts_table):
    for raw_id in sorted(set(raw_ids)):
        policy = db.execute("SELECT * FROM source_policies WHERE user_id=? AND raw_id=?", (user_id, raw_id)).fetchone()
        if policy is None:
            raise ValueError("missing lifecycle source")
        db.execute("""INSERT OR IGNORE INTO memory_tombstones(user_id,raw_id,statement_sha256,epoch)
                      VALUES (?,?,?,?)""", (user_id, raw_id, policy["statement_sha256"], epoch))
        db.execute("DELETE FROM derived_memories WHERE user_id=? AND source_raw_id=?", (user_id, raw_id))
        db.execute("DELETE FROM raw_vectors WHERE user_id=? AND raw_id=?", (user_id, raw_id))
        db.execute(f"DELETE FROM {fts_table} WHERE message_id=?", (raw_id,))
        db.execute("UPDATE raw_messages SET content='',deletion_state='deleted' WHERE user_id=? AND id=?", (user_id, raw_id))
        db.execute("""UPDATE source_policies SET visibility='erased',source_version=source_version+1,
                      content_sha256=? WHERE user_id=? AND raw_id=?""",
                   (hashlib.sha256(b"").hexdigest(), user_id, raw_id))
    # FTS5 DELETE may leave readable terms in existing segment (_data) rows.
    # Rebuild this user's index from its surviving content inside the same txn.
    # WAL, external backups and media are explicitly outside this receipt.
    db.execute(f"INSERT INTO {fts_table}({fts_table}) VALUES('rebuild')")


def apply_core(db: sqlite3.Connection, messages: Sequence[RawMessage], plan: GovernancePlan | None, fts_table: str,
               *, lifecycle_only: bool = False) -> None:
    """Called after raw/index/vector inserts and before ledger commit, on one txn."""
    user_id = messages[0].user_id
    for message in messages:
        _register_source(db, user_id, message.id, message.content, message.deletion_state, message.privacy_level)
    epoch = int(db.execute("SELECT epoch FROM lifecycle_users WHERE user_id=?", (user_id,)).fetchone()[0]) + 1
    db.execute("UPDATE lifecycle_users SET epoch=? WHERE user_id=?", (epoch, user_id))
    # This guard also applies when a process is restarted with governance off.
    for message in messages:
        forbidden = db.execute("SELECT 1 FROM memory_tombstones WHERE user_id=? AND statement_sha256=? LIMIT 1",
                               (user_id, statement_digest(message.content))).fetchone()
        if forbidden:
            _erase(db, user_id, [message.id], epoch, fts_table)
    if plan is None:
        return
    # This first implementation accepts only audited grammar-derived proposals.
    # Arbitrary injected metadata cannot claim semantic support merely by citing an ID.
    expected = BoundedGovernance().plan(messages)
    if lifecycle_only:
        expected = GovernancePlan(mutations=expected.mutations)
    if plan != expected:
        raise ValueError("governance plan does not match the verified source grammar")
    # Process messages in received order; a mutation cannot act on a later message
    # merely because the entire Add was inserted in the same transaction.
    for message in messages:
        if db.execute("SELECT 1 FROM memory_tombstones WHERE user_id=? AND statement_sha256=? LIMIT 1",
                      (user_id,statement_digest(message.content))).fetchone():
            _erase(db,user_id,[message.id],epoch,fts_table)
            continue
        policy = db.execute("SELECT visibility,source_version FROM source_policies WHERE raw_id=? AND user_id=?",
                            (message.id, user_id)).fetchone()
        if policy["visibility"] != "visible":
            continue
        for claim in (c for c in plan.claims if c.source_raw_id == message.id):
            key = claim_key(user_id, claim)
            claim_id = "claim_" + hashlib.sha256((message.id + key + POLICY_VERSION).encode()).hexdigest()
            db.execute("""INSERT INTO derived_memories
                (id,user_id,source_raw_id,source_version,claim_key,object_text,payload_json,valid_from)
                VALUES (?,?,?,?,?,?,?,?)""", (claim_id,user_id,message.id,policy["source_version"],key,
                                               claim.object_text,json.dumps(asdict(claim),ensure_ascii=False),claim.valid_from))
            recompute_key(db, user_id, key)
        for mutation in (m for m in plan.mutations if m.source_raw_id == message.id):
            # Strict scope + original ordering, no substring/keyword delete.
            eligible = db.execute("""SELECT m.id,m.request_id,m.message_index,p.statement_sha256,p.visibility
                FROM raw_messages m JOIN source_policies p ON p.raw_id=m.id AND p.user_id=m.user_id
                WHERE m.user_id=? AND m.id<>?""", (user_id,message.id)).fetchall()
            targets = []
            for row in eligible:
                if row["request_id"] == message.request_id and row["message_index"] >= message.message_index:
                    continue
                match = row["id"] == mutation.target_id if mutation.target_id else row["statement_sha256"] == statement_digest(mutation.target_text or "")
                if match:
                    targets.append(row["id"])
            if mutation.operation == "erase":
                # The request itself may repeat sensitive text. Clear it even if
                # target resolution is unsuccessful; receipt still says unresolved.
                _erase(db, user_id, targets + [message.id], epoch, fts_table)
            else:
                for raw_id in targets:
                    if mutation.operation == "retract":
                        db.execute("UPDATE source_policies SET retraction_state='retracted',source_version=source_version+1 WHERE user_id=? AND raw_id=?",
                                   (user_id,raw_id))
                        db.execute("UPDATE derived_memories SET retraction_state='retracted' WHERE user_id=? AND source_raw_id=?", (user_id,raw_id))
                        # A policy change increments the source version without
                        # changing its body. Revalidate the bound grammar before
                        # rebinding a derived record to that version.
                        refresh_source_bindings(db,user_id,raw_id)
                    elif mutation.operation == "suppress":
                        db.execute("UPDATE source_policies SET visibility='suppressed',source_version=source_version+1 WHERE user_id=? AND raw_id=?", (user_id,raw_id))
                        db.execute("UPDATE raw_messages SET deletion_state='suppressed' WHERE user_id=? AND id=?", (user_id,raw_id))
                        db.execute("UPDATE derived_memories SET visibility='suppressed' WHERE user_id=? AND source_raw_id=?", (user_id,raw_id))
                        db.execute("DELETE FROM raw_vectors WHERE user_id=? AND raw_id=?", (user_id,raw_id))
                        db.execute(f"DELETE FROM {fts_table} WHERE message_id=?", (raw_id,))
                    else:
                        raise ValueError("unknown lifecycle operation")
                # Do not expose a control message as a fresh factual source.
                db.execute("UPDATE source_policies SET visibility='suppressed' WHERE user_id=? AND raw_id=?", (user_id,message.id))
                db.execute("UPDATE raw_messages SET deletion_state='suppressed' WHERE user_id=? AND id=?", (user_id,message.id))
                db.execute(f"DELETE FROM {fts_table} WHERE message_id=?", (message.id,))
                db.execute("DELETE FROM raw_vectors WHERE user_id=? AND raw_id=?", (user_id,message.id))
            operation_id = "op_" + hashlib.sha256((message.id + mutation.operation).encode()).hexdigest()
            db.execute("""INSERT INTO memory_operations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                       (operation_id,user_id,message.request_id,message.id,mutation.operation,
                        "completed" if targets else "unresolved",json.dumps(sorted(targets)),
                        None if targets else "exact_target_not_found",epoch,int(bool(targets)),
                        int(bool(targets) and mutation.operation == "erase"),
                        "live_logical_tables_only" if mutation.operation == "erase" else "not_erasure",POLICY_VERSION))
            for keyrow in db.execute("SELECT DISTINCT claim_key FROM derived_memories WHERE user_id=?", (user_id,)).fetchall():
                recompute_key(db,user_id,keyrow[0])


def recompute_key(db, user_id, key):
    rows = db.execute("""SELECT * FROM derived_memories WHERE user_id=? AND claim_key=?
                         AND visibility='visible' AND retraction_state='not_retracted'""", (user_id,key)).fetchall()
    rows = [row for row in rows if valid_derived_source(db,row)]
    states = {}
    for row in rows:
        claim = json.loads(row["payload_json"])
        successors = [other for other in rows
                      if other["object_text"] != row["object_text"]
                      and claim["single_valued"] and json.loads(other["payload_json"])["single_valued"]
                      and json.loads(other["payload_json"])["explicit_update"]
                      and row["valid_from"] is not None and other["valid_from"] is not None
                      and other["valid_from"] > row["valid_from"] + (86399999 if claim.get("temporal_precision") == "day" else 0)]
        successors.sort(key=lambda r:(r["valid_from"],r["id"]))
        # Never resurrect a superseded value when its successor is later erased.
        until, successor = row["valid_until"], row["superseded_by"]
        if successors and (until is None or successors[0]["valid_from"] < until):
            until, successor = successors[0]["valid_from"], successors[0]["id"]
        if successors:
            first=successors[0]
            source=db.execute('SELECT content_sha256 FROM source_policies WHERE user_id=? AND raw_id=?',
                              (user_id,first['source_raw_id'])).fetchone()
            # Hash-only transition receipt retains no erased source text/value.
            db.execute('INSERT OR IGNORE INTO claim_supersessions VALUES (?,?,?,?,?,?,?,?)',
                (user_id,row['id'],first['id'],key,hashlib.sha256(row['payload_json'].encode()).hexdigest(),
                 first['source_raw_id'],source[0],first['valid_from']))
        states[row["id"]] = (until,successor)
    remaining = [row for row in rows if states[row["id"]][0] is None]
    conflict = (len({row["object_text"] for row in remaining}) > 1
                and all(json.loads(row["payload_json"])["single_valued"] for row in remaining))
    for row in rows:
        until, successor = states[row["id"]]
        currency = "superseded" if until is not None else ("uncertain" if conflict or row["valid_from"] is None else "current")
        resolution = "resolved" if until is not None else ("unresolved" if conflict else "none")
        if not json.loads(row["payload_json"])["single_valued"]:
            currency,resolution = "not_applicable","none"
        db.execute("""UPDATE derived_memories SET currency=?,resolution=?,valid_until=?,superseded_by=?
                      WHERE user_id=? AND id=?""", (currency,resolution,until,successor,user_id,row["id"]))


def operation_receipts(db, user_id, request_id):
    return [dict(row) for row in db.execute(
        "SELECT * FROM memory_operations WHERE user_id=? AND request_id=? ORDER BY id", (user_id,request_id))]


def valid_derived_source(db, derived):
    source = db.execute("""SELECT m.*,p.content_sha256,p.source_version
        FROM raw_messages m JOIN source_policies p ON p.raw_id=m.id AND p.user_id=m.user_id
        WHERE m.id=? AND m.user_id=?""",(derived["source_raw_id"],derived["user_id"])).fetchone()
    if source is None or source["source_version"] != derived["source_version"]:
        return False
    if hashlib.sha256(source["content"].encode("utf-8")).hexdigest() != source["content_sha256"]:
        return False
    try:
        claims=BoundedGovernance().plan([RawMessage.model_validate(dict(source))]).claims
        payload=json.loads(derived["payload_json"])
        return any(asdict(claim)==payload and claim_key(source["user_id"],claim)==derived["claim_key"]
                   and claim.object_text==derived["object_text"] for claim in claims)
    except (ValueError,TypeError,KeyError):
        return False


def refresh_source_bindings(db,user_id,raw_id):
    source=db.execute("""SELECT m.*,p.source_version,p.content_sha256 FROM raw_messages m
        JOIN source_policies p ON p.raw_id=m.id AND p.user_id=m.user_id
        WHERE m.user_id=? AND m.id=?""",(user_id,raw_id)).fetchone()
    if source is None or hashlib.sha256(source["content"].encode("utf-8")).hexdigest()!=source["content_sha256"]:
        return
    claims=BoundedGovernance().plan([RawMessage.model_validate(dict(source))]).claims
    for derived in db.execute("SELECT * FROM derived_memories WHERE user_id=? AND source_raw_id=?",(user_id,raw_id)).fetchall():
        if any(asdict(c)==json.loads(derived["payload_json"]) and claim_key(user_id,c)==derived["claim_key"] for c in claims):
            db.execute("UPDATE derived_memories SET source_version=? WHERE id=? AND user_id=?",(source["source_version"],derived["id"],user_id))

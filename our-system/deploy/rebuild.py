"""Plan, or explicitly rebuild a stopped source DB into a NEW v4 copy.

The default is read-only and makes no embedding calls. --execute-v4-rebuild
permits billable embedding requests; it is not part of startup or health.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from app.config import Settings
from app.embeddings import prepare_embedding
from app.jobs import JobCoordinator, NotReady
from app.k0_migrate import logical_hash, file_hash
from app.profiles import Profile
from app.store import SqliteStore
from app import segments
from deploy.runtime import check_runtime, configured_provider, PreflightFailure


def paths(source, target, manifest):
    source = Path(source).resolve(strict=True)
    target, manifest = Path(target).resolve(), Path(manifest).resolve()
    if (len({source, target, manifest}) != 3 or target.exists() or manifest.exists()
            or Path(str(target) + "-wal").exists() or Path(str(target) + "-shm").exists()
            or str(target) in {str(source) + "-wal", str(source) + "-shm"}
            or str(manifest) in {str(source) + "-wal", str(source) + "-shm",
                                 str(target) + "-wal", str(target) + "-shm"}):
        raise PreflightFailure("new_distinct_target_and_manifest_required")
    return source, target, manifest


def inventory(source):
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        if db.execute("PRAGMA integrity_check").fetchall()[0][0] != "ok":
            raise PreflightFailure("source_integrity_failed")
        # Deliberately require the lifecycle-aware schema; legacy schema-only
        # conversion remains app.k0_migrate's separate offline operation.
        active = db.execute("""SELECT m.*,p.source_version,p.content_sha256 AS policy_hash
            FROM raw_messages m LEFT JOIN source_policies p
            ON p.user_id=m.user_id AND p.raw_id=m.id WHERE m.deletion_state='active'
            ORDER BY m.user_id,m.session_id,m.message_index,m.id""").fetchall()
        pieces = 0
        for raw in active:
            if raw["source_version"] is None or raw["policy_hash"] != segments.digest(raw["content"]):
                raise PreflightFailure("source_lifecycle_identity_invalid")
            if db.execute("SELECT 1 FROM memory_tombstones WHERE user_id=? AND statement_sha256=?",
                (raw["user_id"], db.execute("SELECT statement_sha256 FROM source_policies WHERE user_id=? AND raw_id=?",
                    (raw["user_id"], raw["id"])).fetchone()[0])).fetchone():
                raise PreflightFailure("active_source_has_erasure_tombstone")
            pieces += len(segments.split_source(raw, raw["source_version"]))
        return {"active_sources": len(active), "expected_spans": pieces,
                "source_logical_sha256": logical_hash(db)}


def rebuild(source: Path, target: Path, settings: Settings, provider, *, audit=None) -> dict:
    """Offline copy-only operation. Caller must stop writers and authorize costs."""
    source, target = source.resolve(strict=True), target.resolve()
    if audit is None:
        audit = {}
    if source == target or target.exists():
        raise PreflightFailure("new_distinct_target_required")
    if Path(str(target) + "-wal").exists() or Path(str(target) + "-shm").exists():
        raise PreflightFailure("target_sidecars_exist")
    if settings.fragment_policy != "span_exact_v1" or "dense" not in settings.required_routes:
        raise PreflightFailure("span_dense_rebuild_required")
    # Rebuilding other optional derived indexes/models needs its own migration.
    if settings.enable_extractor or settings.enable_governance or settings.sentence_index or settings.reranker_model:
        raise PreflightFailure("optional_mechanisms_require_separate_migration")
    settings = replace(settings, db_path=target)
    profile = Profile.build(settings, provider)
    source_inventory = inventory(source)
    wal = Path(str(source) + "-wal")
    source_files = (file_hash(source), file_hash(wal) if wal.exists() and wal.stat().st_size else None)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Atomic refusal of an existing target, including concurrent invocations.
    with target.open("xb"):
        pass
    audit["target_created"] = True
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
        original.execute("BEGIN")
        if logical_hash(original) != source_inventory["source_logical_sha256"]:
            raise PreflightFailure("source_changed_during_copy")
        with closing(sqlite3.connect(target)) as copied:
            original.backup(copied)
            if logical_hash(copied) != source_inventory["source_logical_sha256"]:
                raise PreflightFailure("copy_snapshot_mismatch")
        original.rollback()
    store = SqliteStore(target)
    store.initialize()
    def ledger_hash(db):
        return hashlib.sha256(json.dumps({table: [tuple(r) for r in db.execute(
            "SELECT * FROM " + table + " ORDER BY user_id,request_id")]
            for table in ("add_requests", "k0_receipts")}, sort_keys=True).encode()).hexdigest()
    with closing(store._connect()) as db:
        original_receipts = ledger_hash(db)
        db.execute("BEGIN IMMEDIATE")
        # Identifiers are generated from registered user hashes, never arbitrary SQL.
        for row in db.execute("SELECT user_id,table_name FROM span_fts_partitions").fetchall():
            expected = segments.partition(row["user_id"])
            if row["table_name"] != expected:
                raise PreflightFailure("span_partition_identity_invalid")
            db.execute(f"DROP TABLE IF EXISTS {expected}")
        db.execute("DELETE FROM span_fts_partitions")
        db.execute("DELETE FROM span_vectors")
        db.execute("DELETE FROM source_spans")
        db.execute("DELETE FROM raw_vectors")
        db.execute("DELETE FROM k0_watermarks")
        users = [r[0] for r in db.execute("SELECT DISTINCT user_id FROM raw_messages ORDER BY user_id")]
        db.commit()
    batches = audit.setdefault("embedding_usage", [])
    for user in users:
        store.rebuild_user_lexical_index(user)
        with closing(store._connect()) as db:
            raws = db.execute("""SELECT m.*,p.source_version FROM raw_messages m
                JOIN source_policies p ON p.user_id=m.user_id AND p.raw_id=m.id
                WHERE m.user_id=? AND m.deletion_state='active' ORDER BY m.id""", (user,)).fetchall()
        pieces = [piece for raw in raws for piece in segments.split_source(raw, raw["source_version"])]
        for start in range(0, len(pieces), provider.config.batch_size):
            batch = pieces[start:start + provider.config.batch_size]
            prepared = prepare_embedding(provider, [piece["content"] for piece in batch])
            batches.append(prepared.usage)
            if prepared.status != "ready":
                raise NotReady("v4_rebuild_embedding_failed")
            with closing(store._connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                segments.insert(db, user, batch, prepared)
                db.commit()
    # Confirm stopped-source consistency before marking any target routes ready.
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
        original.execute("BEGIN")
        if logical_hash(original) != source_inventory["source_logical_sha256"]:
            raise PreflightFailure("source_changed_during_rebuild")
    current_files = (file_hash(source), file_hash(wal) if wal.exists() and wal.stat().st_size else None)
    if current_files != source_files:
        raise PreflightFailure("source_changed_during_rebuild")
    coordinator = JobCoordinator(store, profile)
    coordinator.activate()
    proofs = [coordinator.certify_routes(user) for user in users]
    with closing(store._connect()) as db:
        if ledger_hash(db) != original_receipts:
            raise PreflightFailure("historical_receipts_changed")
        if [r[0] for r in db.execute("PRAGMA integrity_check")] != ["ok"]:
            raise PreflightFailure("target_integrity_failed")
        if db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] != 0:
            raise PreflightFailure("target_checkpoint_failed")
    return {"status": "certified", **source_inventory, "target_sha256": file_hash(target),
            "profile": profile.identity, "config_hash": profile.config_hash,
            "original_receipt_sha256": original_receipts, "proofs": proofs,
            "embedding_usage": batches, "external_calls": sum(b.get("requests", 0) for b in batches),
            "monetary_cost": None, "migration_mode": "offline_copy_rebuild"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execute-v4-rebuild", action="store_true")
    args = parser.parse_args()
    manifest = None
    audit = {"embedding_usage": []}
    try:
        source, target, manifest = paths(args.source, args.target, args.manifest)
        check_runtime()
        if not args.execute_v4_rebuild:
            print(json.dumps({"status": "plan", **inventory(source), "external_calls": 0,
                              "target_created": False, "manifest_created": False}))
            return 0
        settings, provider = configured_provider(replace(Settings.from_env(), db_path=target))
        result = rebuild(source, target, settings, provider, audit=audit)
    except (ValueError, sqlite3.Error, RuntimeError, OSError):
        if args.execute_v4_rebuild and manifest is not None:
            if audit.get("target_created") and target.exists():
                # Fence a failed copy even if final certification had started.
                try:
                    with closing(sqlite3.connect(target)) as failed:
                        failed.execute("UPDATE k0_config SET config_hash='failed_rebuild'")
                        failed.execute("UPDATE k0_watermarks SET ready=0")
                        failed.commit()
                except sqlite3.Error:
                    pass
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with manifest.open("x", encoding="utf-8") as output:
                json.dump({"status": "failed", "code": "rebuild_not_ready",
                           "external_calls": sum(b.get("requests", 0) for b in audit["embedding_usage"]),
                           **audit, "monetary_cost": None,
                           "target_must_not_be_served": True}, output)
        print(json.dumps({"status": "failed", "code": "rebuild_not_ready"}))
        return 2
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps({"status": result["status"], "external_calls": result["external_calls"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

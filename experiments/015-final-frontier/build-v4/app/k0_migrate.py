"""Copy-only K0 schema/route certification. Never builds vectors or calls HTTP.

Run with an explicit new target; the original DB and receipts are preserved.
Dense certification requires already rebuilt vectors with the exact model hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from contextlib import closing

from app.config import Settings
from app.embeddings import EmbeddingConfig, HttpEmbeddingProvider
from app.jobs import JobCoordinator, NotReady
from app.profiles import Profile
from app.store import SqliteStore


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_hash(db):
    """Hash schema+rows at the connection's pinned snapshot, including WAL data.

    The dump is streamed into SHA256; no source content is saved or printed.
    """
    digest = hashlib.sha256()
    for line in db.iterdump():
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def migrate(source: Path, target: Path, settings: Settings, provider=None):
    source, target = source.resolve(strict=True), target.resolve()
    if source == target or target.exists() or settings.db_path.resolve() != target:
        raise ValueError("migration requires a new target distinct from the source")
    settings.ensure_phase_a_defaults()
    profile = Profile.build(settings, provider)
    # This tool's contract is an offline copy. No online catch-up/switch is
    # attempted; detect changes to the DB and committed WAL while copying.
    def source_hashes():
        wal = Path(str(source) + "-wal")
        return {"db": file_hash(source), "wal": file_hash(wal) if wal.exists() and wal.stat().st_size else None}
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
        original.execute("BEGIN")
        original.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        source_logical_hash = logical_hash(original)
        before = source_hashes()
        # The SQLite backup API captures committed WAL data as one DB snapshot.
        with closing(sqlite3.connect(target)) as copied:
            original.backup(copied)
            copied.commit()
            copied.execute("BEGIN")
            if logical_hash(copied) != source_logical_hash:
                raise NotReady("copy_snapshot_mismatch")
            copied.rollback()
        original.rollback()
    if source_hashes() != before:
        raise NotReady("source_changed_during_copy")
    def require_same_source():
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as current:
            current.execute("BEGIN")
            unchanged = logical_hash(current) == source_logical_hash
            current.rollback()
        if not unchanged or source_hashes() != before:
            raise NotReady("source_changed_during_copy")
    require_same_source()
    store = SqliteStore(target)
    store.initialize()
    if settings.fragment_policy == "span_exact_v1" and "dense" not in profile.required_routes:
        from app.segments import split_source,insert,prune
        with closing(store._connect()) as rebuilt:
            rebuilt.execute("BEGIN IMMEDIATE")
            users = [r[0] for r in rebuilt.execute("SELECT DISTINCT user_id FROM raw_messages")]
            for user in users:
                prune(rebuilt,user)
                raws = rebuilt.execute("""SELECT m.*,p.source_version FROM raw_messages m JOIN source_policies p
                    ON m.id=p.raw_id AND m.user_id=p.user_id WHERE m.user_id=? AND m.deletion_state='active'""",(user,)).fetchall()
                for raw in raws:
                    pieces = split_source(raw,raw["source_version"])
                    missing = [p for p in pieces if not rebuilt.execute("SELECT 1 FROM source_spans WHERE id=? AND user_id=?",(p["id"],user)).fetchone()]
                    insert(rebuilt,user,missing)
            rebuilt.commit()
    coordinator = JobCoordinator(store, profile)
    coordinator.activate()
    with coordinator.transaction(False) as db:
        users = [r[0] for r in db.execute("SELECT DISTINCT user_id FROM raw_messages ORDER BY user_id")]
        receipt_hash = hashlib.sha256(json.dumps([tuple(r) for r in db.execute(
            "SELECT * FROM add_requests ORDER BY user_id,request_id")], ensure_ascii=False).encode()).hexdigest()
    proofs = [coordinator.certify_routes(user) for user in users]
    with closing(store._connect()) as db:
        check = [r[0] for r in db.execute("PRAGMA integrity_check")]
        checkpoint = tuple(db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
        if check != ["ok"] or checkpoint[0] != 0:
            raise NotReady("target_checkpoint_not_ready")
    require_same_source()
    return {"source": str(source), "source_sha256": before["db"], "source_wal_sha256": before["wal"], "target": str(target),
            "target_sha256": file_hash(target), "schema_version": 1,
            "source_logical_sha256": source_logical_hash,
            "integrity_check": "ok", "checkpoint": checkpoint, "migration_mode": "offline_copy",
            "profile": profile.identity, "config_hash": profile.config_hash,
            "original_receipt_hash": receipt_hash, "proofs": proofs,
            "external_calls": 0, "vector_rebuild": "not_performed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", choices=("raw_sync", "dense_sync", "span_sync", "span_dense_sync"), required=True)
    parser.add_argument("--mode", choices=("raw_fts", "raw_dense", "raw_hybrid_rrf"), default="raw_fts")
    parser.add_argument("--provider")
    parser.add_argument("--endpoint")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--model", default="text-embedding-v4")
    parser.add_argument("--revision", default="1")
    args = parser.parse_args()
    if args.manifest.exists() or args.manifest.resolve() in {args.source.resolve(), args.target.resolve()}:
        parser.error("manifest must be a new separate file")
    provider = None
    if args.profile in {"dense_sync", "span_dense_sync"}:
        if not all((args.provider, args.endpoint, args.dimensions)):
            parser.error("dense certification requires explicit provider/endpoint/dimensions metadata")
        provider = HttpEmbeddingProvider(EmbeddingConfig(provider=args.provider, endpoint=args.endpoint,
            dimensions=args.dimensions, model=args.model, revision=args.revision))
    settings = Settings(db_path=args.target, profile_id=args.profile, retrieval_mode=args.mode)
    try:
        result = migrate(args.source, args.target, settings, provider)
    except (NotReady, ValueError, sqlite3.Error):
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({"status": "failed", "reason": "migration_not_ready",
            "source": str(args.source), "target": str(args.target), "external_calls": 0}), encoding="utf-8")
        return 2
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "certified", "users": len(result["proofs"]), "external_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

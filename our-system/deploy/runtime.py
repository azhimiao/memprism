"""Offline runtime/configuration checks and the v4 application factory.

No model calls, database migration or secret values in diagnostic output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import closing
from dataclasses import replace
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

from app.config import Settings
from app.embeddings import create_embedding_provider, provider_identity, provider_version
from app.profiles import Profile

ENCODING_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
ENCODING_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
ROOT = Path(__file__).resolve().parents[1]


class PreflightFailure(RuntimeError):
    pass


def check_runtime() -> dict:
    if sys.version_info[:3] != (3, 12, 11):
        raise PreflightFailure("python_version_mismatch")
    packages = {}
    for line in (ROOT / "requirements-cycle2.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        if ";" in line and sys.platform != "win32":
            continue
        name, expected = line.split(";", 1)[0].strip().split("==")
        try:
            actual = version(name)
        except PackageNotFoundError:
            raise PreflightFailure("missing_runtime_dependency") from None
        if actual != expected:
            raise PreflightFailure("runtime_dependency_version_mismatch")
        packages[name] = actual
    cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR", os.environ.get("DATA_GYM_CACHE_DIR"))
    if cache_dir is None:
        cache_dir = str(Path(tempfile.gettempdir()) / "data-gym-cache")
    asset = Path(cache_dir) / hashlib.sha1(ENCODING_URL.encode()).hexdigest()
    if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest() != ENCODING_SHA256:
        raise PreflightFailure("tokenizer_cache_missing_or_invalid")
    # Cache validation precedes tokenizer initialization, so this never downloads.
    from app.segments import tokenizer, fragment_identity, fragment_hash
    enc = tokenizer()
    sample = "中文 e\u0301 👩🏽‍💻\r\n<|endoftext|>"
    ids = enc.encode(sample, disallowed_special=())
    if enc.decode(ids) != sample:
        raise PreflightFailure("tokenizer_roundtrip_failed")
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("CREATE VIRTUAL TABLE probe USING fts5(text, tokenize='unicode61')")
        db.execute("INSERT INTO probe VALUES ('runtime probe')")
        if db.execute("SELECT count(*) FROM probe WHERE probe MATCH 'runtime'").fetchone()[0] != 1:
            raise PreflightFailure("sqlite_fts5_failed")
    return {"python": sys.version.split()[0], "packages": packages,
            "sqlite": sqlite3.sqlite_version, "fragment_identity": fragment_identity(),
            "fragment_hash": fragment_hash(), "tokenizer_asset_sha256": ENCODING_SHA256,
            "token_probe_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "external_calls": 0}


def configured_provider(settings: Settings, *, transport=None):
    if not settings.formal or "dense" not in settings.required_routes or settings.embedding is None:
        raise PreflightFailure("formal_dense_configuration_required")
    config = replace(settings.embedding,
        region=os.environ.get("AML_EMBEDDING_REGION") or settings.embedding.region,
        max_input_bytes=int(os.environ.get("AML_EMBEDDING_MAX_INPUT_BYTES", "8192")),
        min_request_interval_seconds=float(os.environ.get("AML_EMBEDDING_MIN_INTERVAL_SECONDS", "0")))
    if config.provider != "dashscope":
        raise PreflightFailure("deployment_requires_native_dashscope_v4")
    if settings.auth_mode == "none":
        raise PreflightFailure("service_authentication_required")
    if not os.environ.get(config.api_key_env, "").strip():
        raise PreflightFailure("missing_embedding_api_key")
    if any(ord(c) < 33 or ord(c) > 126 for c in os.environ[config.api_key_env]):
        raise PreflightFailure("invalid_embedding_api_key")
    settings = replace(settings, embedding=config)
    settings.ensure_phase_a_defaults()
    provider = create_embedding_provider(config, transport=transport)
    Profile.build(settings, provider)
    return settings, provider


def check_database(path: Path, profile: Profile) -> dict:
    """Refuse old identities before create_app can initialize/activate the DB."""
    path = path.resolve()
    if not path.exists():
        return {"database": "new", "config_hash": profile.config_hash}
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        if [r[0] for r in db.execute("PRAGMA quick_check")] != ["ok"]:
            raise PreflightFailure("database_integrity_failed")
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='k0_config'").fetchone()
        saved = db.execute("SELECT config_hash FROM k0_config WHERE singleton=1").fetchone() if exists else None
        if saved is None or saved[0] != profile.config_hash:
            raise PreflightFailure("database_identity_mismatch_rebuild_required")
        from app.jobs import JobCoordinator
        from app.store import SqliteStore
        coordinator = JobCoordinator(SqliteStore(path), profile)
        coordinator.control_ready_on(db)
        users = [r[0] for r in db.execute("SELECT DISTINCT user_id FROM raw_messages")]
        for user in users:
            coordinator.ready_on(db, user)
        return {"database": "same_identity", "config_hash": profile.config_hash}


def build_app(settings: Settings | None = None, *, transport=None, verify_runtime=True):
    if verify_runtime:
        check_runtime()
    settings, provider = configured_provider(settings or Settings.from_env(), transport=transport)
    profile = Profile.build(settings, provider)
    check_database(settings.db_path, profile)
    from app.api import create_app
    application = create_app(settings, embedding_provider=provider)
    application.state.retrieval.embedding_provider = provider.for_query()
    # Recheck all required routes without external embedding requests.
    coordinator = application.state.coordinator
    with coordinator.transaction(False) as db:
        coordinator.control_ready_on(db)
        users = [r[0] for r in db.execute("SELECT DISTINCT user_id FROM raw_messages")]
        for user in users:
            coordinator.ready_on(db, user)
    # OpenAPI/debug pages are outside the protected Add/Search contract.
    application.openapi_url = None
    application.router.routes = [r for r in application.router.routes
                                 if getattr(r, "path", "") in {"/health", "/add", "/search"}]
    return application


def create_app():
    """Uvicorn --factory entry; parent API can independently use the provider factory."""
    try:
        return build_app()
    except (ValueError, sqlite3.Error, RuntimeError, OSError):
        raise PreflightFailure("deployment_preflight_failed") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", action="store_true",
                        help="also validate environment and DB identity read-only; never embed")
    args = parser.parse_args()
    try:
        result = check_runtime()
        if args.configuration:
            settings, provider = configured_provider(Settings.from_env())
            result.update(check_database(settings.db_path, Profile.build(settings, provider)))
            result.update(embedding_identity=provider_identity(provider),
                          embedding_version=provider_version(provider))
    except (ValueError, sqlite3.Error, RuntimeError, OSError) as exc:
        code = str(exc) if isinstance(exc, PreflightFailure) else "preflight_failed"
        print(json.dumps({"status": "failed", "code": code, "external_calls": 0}))
        return 2
    print(json.dumps({"status": "ok", **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

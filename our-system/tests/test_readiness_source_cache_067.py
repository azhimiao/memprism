"""067: exact snapshot validation reuse; temporary DBs, no network/model calls."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import importlib.util
from pathlib import Path
import re
import sqlite3

import pytest

from app import segments, vector_checks
from app.api import create_app
from app.config import Settings
from app.embeddings import EmbeddingBatch, EmbeddingConfig
from app.jobs import NotReady
from app.models import AddRequest, SearchRequest


class Provider:
    implementation_kind = "engineering_double"
    config = EmbeddingConfig(provider="engineering", endpoint="https://invalid.test/embeddings", dimensions=2)

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return EmbeddingBatch([[1., 0.] for _ in texts])


def clear_caches(*, full_only=False):
    with segments._counts_lock:
        segments._counts_cache.clear()
        segments._counts_cache_bytes = 0
        if not full_only:
            segments._source_counts_cache.clear()
            segments._source_counts_cache_bytes = 0


@pytest.fixture(autouse=True)
def isolated_caches():
    clear_caches()
    yield
    clear_caches()


@pytest.fixture(scope="module")
def baseline():
    path = Path(__file__).resolve().parents[2] / "experiments/067-add-write-repair/backup-062-app/segments.py"
    spec = importlib.util.spec_from_file_location("segments062_source_cache_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add(app, text="alpha PRIVATE_SOURCE_SENTINEL", rid="r", user="u"):
    return app.state.ingest.add(AddRequest(user_id=user, session_id="s", request_id=rid,
        messages=[{"role": "user", "content": text, "timestamp": 123}]))


def build(path, *, text="alpha PRIVATE_SOURCE_SENTINEL"):
    provider = Provider()
    app = create_app(Settings(db_path=path, profile_id="cycle2_dense"), embedding_provider=provider)
    add(app, text)
    return app, provider


def counts(app, *, user="u", db=None):
    if db is not None:
        return segments.counts(db, user, app.state.profile)
    with closing(app.state.store._connect()) as connection:
        connection.execute("BEGIN")
        return segments.counts(connection, user, app.state.profile)


def search(app):
    return app.state.retrieval.search(SearchRequest(user_id="u", query="alpha", top_k=20))


def mutate(path, sql, args=()):
    with sqlite3.connect(path) as db:
        db.execute(sql, args)


def assert_baseline(app, baseline):
    with closing(app.state.store._connect()) as db:
        db.execute("BEGIN")
        actual = counts(app, db=db)
        assert actual == baseline.counts(db, "u", app.state.profile)
        return actual


def test_new_add_revalidates_only_new_source_and_full_cache_eviction_reuses_history(tmp_path, monkeypatch):
    app, _ = build(tmp_path / "m.db")
    add(app, "alpha second history", "r2")
    add(app, "alpha third history", "r3")
    clear_caches()
    calls = Counter()
    raw_ids = []
    original_split = segments.split_source
    original_lexical = segments.build_lexical_text
    original_vector = vector_checks.valid_vector_json
    original_source = segments._source_counts

    def source(*args):
        calls["source"] += 1
        return original_source(*args)

    def split(raw, *args):
        calls["split"] += 1
        raw_ids.append(raw["id"] if hasattr(raw, "keys") else raw.id)
        return original_split(raw, *args)

    def lexical(text):
        calls["lexical"] += 1
        return original_lexical(text)

    def vector(*args):
        calls["vector"] += 1
        return original_vector(*args)

    monkeypatch.setattr(segments, "split_source", split)
    monkeypatch.setattr(segments, "build_lexical_text", lexical)
    monkeypatch.setattr(segments, "_source_counts", source)
    monkeypatch.setattr(vector_checks, "valid_vector_json", vector)
    assert counts(app) == dict.fromkeys(("active", "raw", "span", "fts", "dense"), 3)
    assert calls == {"source": 3, "split": 3, "lexical": 3, "vector": 3}
    history = set(raw_ids)
    calls.clear()
    raw_ids.clear()
    clear_caches(full_only=True)
    assert counts(app)["active"] == 3
    assert not calls  # Even a full-user miss performs no repeated validation.
    add(app, "alpha NEW_SOURCE_ONLY", "r4")
    assert counts(app) == dict.fromkeys(("active", "raw", "span", "fts", "dense"), 4)
    assert history.isdisjoint(raw_ids)
    # Ingest/permit also split the new source. Count readiness separately so
    # this assertion does not depend on the independently maintained writer.
    assert calls["source"] == 1
    assert calls["split"] >= 2
    assert calls["lexical"] >= 2
    assert calls["vector"] == 1
    before = calls.copy()
    clear_caches(full_only=True)
    assert counts(app)["active"] == 4
    assert calls == before


@pytest.mark.parametrize("mutation", [
    "UPDATE raw_messages SET content='forged'",
    "UPDATE raw_messages SET content=''",
    "UPDATE raw_messages SET deletion_state='deleted'",
    "UPDATE source_policies SET source_version=source_version+1",
    "UPDATE source_policies SET content_sha256='forged'",
    "DELETE FROM source_policies",
    "DELETE FROM source_spans",
    "UPDATE source_spans SET source_raw_id='foreign_raw'",
    "UPDATE source_spans SET id='forged_span'",
    "UPDATE source_spans SET source_version=source_version+1",
    "UPDATE source_spans SET source_sha256='forged'",
    "UPDATE source_spans SET content='forged'",
    "UPDATE source_spans SET token_count=token_count+1",
    "UPDATE source_spans SET span_start=1",
    "UPDATE source_spans SET span_end=span_end+1",
    "UPDATE source_spans SET span_index=99",
    "UPDATE source_spans SET boundary_status='forged'",
    "UPDATE source_spans SET previous_id='forged'",
    "UPDATE source_spans SET next_id='forged'",
    "UPDATE source_spans SET fragment_hash='forged'",
    "UPDATE source_spans SET span_sha256='forged'",
    "UPDATE span_vectors SET vector_json='[0,0]'",
    "UPDATE span_vectors SET vector_json='[true,0]'",
    "UPDATE span_vectors SET vector_json='[NaN,0]'",
    "UPDATE span_vectors SET content_sha256='forged'",
    "UPDATE span_vectors SET dimensions=3",
    "UPDATE span_vectors SET embedding_version='foreign'",
    "DELETE FROM span_vectors",
    "UPDATE span_fts_partitions SET table_name='foreign'",
    "DELETE FROM span_fts_partitions",
    "DELETE FROM {fts}",
    "DROP TABLE {fts}",
    "UPDATE {fts} SET lexical_text='forged'",
    "INSERT INTO {fts} SELECT * FROM {fts}",
    "INSERT INTO {fts} VALUES ('orphan_fts','alpha')",
    "INSERT INTO source_spans SELECT 'extra_span',user_id,source_raw_id,source_version,source_sha256,"
    "span_start,span_end,span_sha256,fragment_hash,content,span_index,boundary_status,token_count,"
    "previous_id,next_id FROM source_spans",
])
def test_independent_connection_corruption_without_epoch_change_fails_closed(tmp_path, baseline, mutation):
    path = tmp_path / "m.db"
    app, provider = build(path)
    assert search(app).data
    assert_baseline(app, baseline)
    with sqlite3.connect(path) as db:
        stamp = db.execute("SELECT * FROM k0_users").fetchall()
    mutate(path, mutation.format(fts=segments.partition("u")))
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT * FROM k0_users").fetchall() == stamp
    actual = assert_baseline(app, baseline)
    assert any(actual[route] != actual["active"] for route in ("raw", "span", "fts", "dense"))
    before = provider.calls
    with pytest.raises(NotReady):
        search(app)
    assert provider.calls == before


def test_changed_source_metadata_invalidates_only_its_cache_and_returns_current_metadata(tmp_path, monkeypatch, baseline):
    path = tmp_path / "m.db"
    app, _ = build(path)
    add(app, "alpha unchanged", "r2")
    counts(app)
    validated = []
    original = segments.split_source

    def split(raw, *args):
        validated.append(raw["request_id"])
        return original(raw, *args)

    monkeypatch.setattr(segments, "split_source", split)
    mutate(path, "UPDATE raw_messages SET source_timestamp=456,session_id='s-current' WHERE request_id='r'")
    assert assert_baseline(app, baseline)["active"] == 2
    assert validated == ["r"]
    result = search(app)
    assert any(hit.created_at == 456 for hit in result.data)


def test_caller_wal_snapshot_is_used_even_when_newer_snapshot_is_cached(tmp_path, baseline):
    path = tmp_path / "m.db"
    app, _ = build(path)
    with closing(app.state.store._connect()) as old:
        old.execute("BEGIN")
        good = counts(app, db=old)
        mutate(path, "UPDATE span_vectors SET vector_json='[0,0]'")
        assert assert_baseline(app, baseline)["dense"] == 0
        clear_caches(full_only=True)
        assert counts(app, db=old) == good
    clear_caches(full_only=True)
    assert counts(app)["dense"] == 0


def test_cached_sources_do_not_cross_user_database_or_embedding_profile(tmp_path, baseline):
    good, _ = build(tmp_path / "good.db")
    bad, _ = build(tmp_path / "bad.db")
    counts(good)
    mutate(tmp_path / "bad.db", "UPDATE span_vectors SET vector_json='[0,0]'")
    assert counts(bad)["dense"] == 0
    assert counts(good)["dense"] == 1
    add(good, "alpha another user", user="other")
    mutate(tmp_path / "good.db", "UPDATE span_vectors SET vector_json='[0,0]' WHERE user_id='other'")
    assert counts(good, user="other")["dense"] == 0
    assert counts(good)["dense"] == 1
    from dataclasses import replace
    with closing(good.state.store._connect()) as db:
        db.execute("BEGIN")
        profile = replace(good.state.profile, model_version="different-embedding")
        assert segments.counts(db, "u", profile) == baseline.counts(db, "u", profile)
        assert segments.counts(db, "u", profile)["dense"] == 0


def test_deleted_source_replay_and_copied_spans_cannot_revive_cached_validation(tmp_path, baseline):
    path = tmp_path / "m.db"
    app, _ = build(path, text="alpha")
    assert search(app).data
    counts(app)
    with closing(app.state.store._connect()) as db:
        saved = dict(db.execute("SELECT * FROM source_spans").fetchone())
        raw_id = saved["source_raw_id"]
    add(app, 'delete "alpha"', "erase")
    add(app, "alpha", "replay")
    assert not search(app).data
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT deletion_state FROM raw_messages WHERE id=?", (raw_id,)).fetchone()[0] != "active"
        db.execute(f"INSERT INTO source_spans({','.join(saved)}) VALUES ({','.join('?' for _ in saved)})",
                   tuple(saved.values()))
        db.execute(f"INSERT INTO {segments.partition('u')} VALUES (?,?)", (saved["id"], "alpha"))
    assert assert_baseline(app, baseline)["span"] == -1
    with pytest.raises(NotReady):
        search(app)


def test_multispan_unicode_results_equal_062_after_full_cache_eviction(tmp_path, baseline):
    app, _ = build(tmp_path / "m.db", text="alpha 中文 e\u0301 👩🏽‍💻\r\n" * 240)
    expected = assert_baseline(app, baseline)
    assert expected["active"] > 1
    for _ in range(2):
        clear_caches(full_only=True)
        assert assert_baseline(app, baseline) == expected


def test_orphan_fts_changes_are_checked_even_when_all_source_summaries_hit(tmp_path, monkeypatch, baseline):
    path = tmp_path / "m.db"
    app, _ = build(path)
    counts(app)
    original = segments.split_source
    calls = []

    def split(raw, *args):
        calls.append(raw["id"])
        return original(raw, *args)

    monkeypatch.setattr(segments, "split_source", split)
    mutate(path, f"INSERT INTO {segments.partition('u')} VALUES ('orphan','alpha')")
    assert assert_baseline(app, baseline)["fts"] == -1
    assert not calls  # The global exact set check still rejects this snapshot.
    mutate(path, f"DELETE FROM {segments.partition('u')} WHERE span_id='orphan'")
    clear_caches(full_only=True)
    assert counts(app)["fts"] == 1
    assert not calls


def test_corruption_repair_and_recorruption_do_not_reuse_other_snapshot(tmp_path, baseline):
    path = tmp_path / "m.db"
    app, _ = build(path)
    assert counts(app)["dense"] == 1
    mutate(path, "UPDATE span_vectors SET vector_json='[0,0]'")
    assert assert_baseline(app, baseline)["dense"] == 0
    mutate(path, "UPDATE span_vectors SET vector_json='[1.0, 0.0]'")
    clear_caches(full_only=True)
    assert assert_baseline(app, baseline)["dense"] == 1
    mutate(path, "UPDATE span_vectors SET vector_json='[0,0]'")
    clear_caches(full_only=True)
    assert counts(app)["dense"] == 0


@pytest.mark.parametrize("budget,limit", [(2400, 4096), (8 * 1024 * 1024, 2), (1, 4096)])
def test_source_cache_is_bounded_contains_only_summaries_and_eviction_revalidates(tmp_path, monkeypatch, budget, limit):
    monkeypatch.setattr(segments, "_SOURCE_COUNTS_CACHE_MAX_BYTES", budget)
    monkeypatch.setattr(segments, "_SOURCE_COUNTS_CACHE_LIMIT", limit)
    app, _ = build(tmp_path / "m.db")
    for i in range(5):
        add(app, f"alpha PRIVATE_SOURCE_SENTINEL_{i}", f"r{i}")
    assert 0 <= segments._source_counts_cache_bytes <= budget
    assert len(segments._source_counts_cache) <= limit
    assert len(segments._source_counts_cache) < 6
    assert segments._source_counts_cache_bytes == sum(entry[1] for entry in segments._source_counts_cache.values())

    def check(value):
        if isinstance(value, tuple):
            for member in value:
                check(member)
        elif isinstance(value, str):
            assert re.fullmatch(r"span_[0-9a-f]{64}", value)
        else:
            assert isinstance(value, (int, bytes))

    for key, entry in segments._source_counts_cache.items():
        assert isinstance(key, bytes) and len(key) == 32
        check(entry)
    clear_caches(full_only=True)
    assert counts(app)["active"] == 6
    mutate(tmp_path / "m.db", "UPDATE source_spans SET token_count=-1")
    assert counts(app)["span"] == 0


def test_concurrent_source_cache_misses_keep_byte_accounting_exact(tmp_path):
    app, _ = build(tmp_path / "m.db")
    add(app, "alpha second", "r2")
    clear_caches()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: counts(app), range(12)))
    assert all(result == results[0] for result in results)
    assert results[0]["active"] == 2
    with segments._counts_lock:
        assert segments._source_counts_cache_bytes == sum(entry[1] for entry in segments._source_counts_cache.values())
        assert len(segments._source_counts_cache) == 2

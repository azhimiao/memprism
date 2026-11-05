"""Versioned, exact Unicode source spans. No extraction or normalization.

All offsets count Python Unicode code points. Token limits use the pinned
local tokenizer, never a provider's inferred count. Adjacency is metadata.
"""
from __future__ import annotations

from functools import lru_cache
from collections import OrderedDict
from importlib.metadata import version
import hashlib
import json
import math
import re
import sys
import threading

import tiktoken

from app.index import build_lexical_text
from app.models import EvidenceCandidate
from app.embeddings import normalize_vector, EmbeddingFailure


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fragment_identity():
    return {"policy": "span_exact_v1", "offset_unit": "unicode_codepoint",
            "tokenizer": "tiktoken", "tokenizer_version": version("tiktoken"),
            "encoding": "o200k_base", "hard_char_limit": 1800,
            "target_char_limit": 1200, "hard_token_limit": 512, "overlap": 160}


@lru_cache(maxsize=1)
def fragment_hash():
    return digest(json.dumps(fragment_identity(), sort_keys=True, separators=(",", ":")))


@lru_cache(maxsize=1)
def tokenizer():
    return tiktoken.get_encoding("o200k_base")


def token_count(text):
    return len(tokenizer().encode(text, disallowed_special=()))


_boundary_cache = OrderedDict()


def boundaries(content):
    if not content:
        return ()
    key = digest(content)
    cached = _boundary_cache.get(key)
    if cached is not None:
        return cached
    result, start = [], 0
    while start < len(content):
        remaining = content[start:]
        if len(remaining) <= 1800 and token_count(remaining) <= 512:
            end, status = len(content), "source_end"
        else:
            end = min(start + 1200, len(content))
            # Reducing the actual substring guarantees the hard bound even
            # when Unicode tokenization does not have monotone prefix counts.
            while end > start + 1 and token_count(content[start:end]) > 512:
                end -= max(1, (end - start) // 16)
            cuts = [m.end() + start for m in re.finditer(r"\n|[.!?。！？](?:\s|$)", content[start:end])]
            natural = [cut for cut in cuts if cut >= start + (end - start) // 2 and token_count(content[start:cut]) <= 512]
            if natural:
                end, status = natural[-1], "natural_boundary"
            else:
                status = "hard_split_requires_context"
        result.append((start, end, status))
        if end == len(content):
            break
        start = max(start + 1, end - min(160, (end - start) // 4))
    result = tuple(result)
    # The cache holds hashes/offsets only; source text cannot survive erasure.
    _boundary_cache[key] = result
    if len(_boundary_cache) > 4096:
        _boundary_cache.popitem(last=False)
    return result


def split_source(raw, source_version=1):
    def field(name):
        return raw[name] if hasattr(raw, "keys") else getattr(raw, name)
    content, user, raw_id = field("content"), field("user_id"), field("id")
    source_hash, policy_hash = digest(content), fragment_hash()
    pieces = []
    for index, (start, end, status) in enumerate(boundaries(content)):
        text = content[start:end]
        span_hash = digest(text)
        identity = [user, raw_id, source_version, source_hash, start, end, span_hash, policy_hash]
        span_id = "span_" + digest(json.dumps(identity, ensure_ascii=False, separators=(",", ":")))
        pieces.append({"id": span_id, "user_id": user, "source_raw_id": raw_id,
                       "source_version": source_version, "source_sha256": source_hash,
                       "span_start": start, "span_end": end, "span_sha256": span_hash,
                       "fragment_hash": policy_hash, "content": text, "span_index": index,
                       "boundary_status": status, "token_count": token_count(text)})
    for i, piece in enumerate(pieces):
        piece["previous_id"] = pieces[i-1]["id"] if i else None
        piece["next_id"] = pieces[i+1]["id"] if i+1 < len(pieces) else None
    return pieces


def initialize(db):
    db.execute("""CREATE TABLE IF NOT EXISTS source_spans(
        id TEXT PRIMARY KEY,user_id TEXT NOT NULL,source_raw_id TEXT NOT NULL,
        source_version INTEGER NOT NULL,source_sha256 TEXT NOT NULL,
        span_start INTEGER NOT NULL,span_end INTEGER NOT NULL,span_sha256 TEXT NOT NULL,
        fragment_hash TEXT NOT NULL,content TEXT NOT NULL,span_index INTEGER NOT NULL,
        boundary_status TEXT NOT NULL,token_count INTEGER NOT NULL,previous_id TEXT,next_id TEXT,
        FOREIGN KEY(source_raw_id) REFERENCES raw_messages(id) ON DELETE CASCADE)""")
    db.execute("CREATE INDEX IF NOT EXISTS spans_user_source ON source_spans(user_id,source_raw_id)")
    db.execute("CREATE TABLE IF NOT EXISTS span_fts_partitions(user_id TEXT PRIMARY KEY,table_name TEXT NOT NULL UNIQUE)")
    db.execute("""CREATE TABLE IF NOT EXISTS span_vectors(user_id TEXT NOT NULL,span_id TEXT NOT NULL,
        embedding_version TEXT NOT NULL,dimensions INTEGER NOT NULL,content_sha256 TEXT NOT NULL,
        vector_json TEXT NOT NULL,PRIMARY KEY(user_id,span_id,embedding_version),
        FOREIGN KEY(span_id) REFERENCES source_spans(id) ON DELETE CASCADE)""")


def partition(user_id):
    return "span_fts_" + digest(user_id)


def ensure_partition(db, user_id):
    table = partition(user_id)
    db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5(span_id UNINDEXED,lexical_text,tokenize='unicode61')")
    db.execute("INSERT OR IGNORE INTO span_fts_partitions VALUES (?,?)", (user_id, table))
    return table


def insert(db, user_id, pieces, prepared=None):
    table = ensure_partition(db, user_id)
    for piece in pieces:
        keys = list(piece)
        db.execute(f"INSERT INTO source_spans({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})", tuple(piece.values()))
        db.execute(f"INSERT INTO {table} VALUES (?,?)", (piece["id"], build_lexical_text(piece["content"])))
    if prepared is not None:
        if prepared.status != "ready" or len(prepared.vectors) != len(pieces):
            raise ValueError("span embedding incomplete")
        for piece, vector in zip(pieces, prepared.vectors):
            vector = normalize_vector(vector, prepared.dimensions)
            db.execute("INSERT INTO span_vectors VALUES (?,?,?,?,?,?)", (user_id,piece["id"],prepared.version,
                prepared.dimensions,piece["span_sha256"],json.dumps(vector)))


def prune(db, user_id):
    """Lifecycle erasure removes source text, lexical derivatives and vectors."""
    table = ensure_partition(db, user_id)
    bad = db.execute("""SELECT s.id FROM source_spans s LEFT JOIN raw_messages m
        ON m.id=s.source_raw_id AND m.user_id=s.user_id LEFT JOIN source_policies p
        ON p.raw_id=m.id AND p.user_id=m.user_id WHERE s.user_id=? AND
        (m.id IS NULL OR m.deletion_state!='active' OR p.source_version!=s.source_version
         OR p.content_sha256!=s.source_sha256 OR p.source_version IS NULL)""", (user_id,)).fetchall()
    for row in bad:
        db.execute(f"DELETE FROM {table} WHERE span_id=?", (row[0],))
        db.execute("DELETE FROM span_vectors WHERE user_id=? AND span_id=?", (user_id,row[0]))
        db.execute("DELETE FROM source_spans WHERE user_id=? AND id=?", (user_id,row[0]))


def rebind_versions(db, user_id):
    """Retractions change provenance versions, not source text or vectors.

    Rebind only a complete existing span set with identical source hashes.
    Erased/suppressed sources are handled by prune and never reconstructed.
    """
    raws = db.execute("""SELECT DISTINCT m.*,p.source_version FROM raw_messages m JOIN source_policies p
        ON p.raw_id=m.id AND p.user_id=m.user_id JOIN source_spans s ON s.source_raw_id=m.id AND s.user_id=m.user_id
        WHERE m.user_id=? AND m.deletion_state='active' AND s.source_version!=p.source_version
        AND s.source_sha256=p.content_sha256""",(user_id,)).fetchall()
    for raw in raws:
        new = split_source(raw,raw["source_version"])
        old = db.execute("SELECT * FROM source_spans WHERE user_id=? AND source_raw_id=? ORDER BY span_index",
                         (user_id,raw["id"])).fetchall()
        if len(old) != len(new) or any((o["span_start"],o["span_end"],o["span_sha256"],o["source_sha256"],o["fragment_hash"]) !=
                (p["span_start"],p["span_end"],p["span_sha256"],p["source_sha256"],p["fragment_hash"]) for o,p in zip(old,new)):
            continue
        insert(db,user_id,new)
        for o,p in zip(old,new):
            db.execute("""INSERT INTO span_vectors SELECT user_id,?,embedding_version,dimensions,content_sha256,vector_json
                FROM span_vectors WHERE user_id=? AND span_id=?""",(p["id"],user_id,o["id"]))


def valid_candidate(db, candidate, source):
    if not source or candidate.representation != "span_exact_v1":
        return False
    row = db.execute("SELECT * FROM source_spans WHERE id=? AND user_id=?", (candidate.id,candidate.user_id)).fetchone()
    identity = [candidate.user_id,candidate.source_raw_id,candidate.source_version,candidate.source_sha256,
                candidate.span_start,candidate.span_end,candidate.span_sha256,candidate.fragment_hash]
    canonical_id = "span_" + digest(json.dumps(identity,ensure_ascii=False,separators=(",",":")))
    metadata_valid = all(candidate.model_dump()[field] == source[field] for field in
                         ("role","session_id","message_index","source_timestamp","recorded_at"))
    metadata_valid = metadata_valid and candidate.created_at == (source["source_timestamp"] if source["source_timestamp"] is not None else source["recorded_at"])
    return bool(row and row["source_raw_id"] == candidate.source_raw_id and
        candidate.id == canonical_id and metadata_valid and len(candidate.content) <= 1800 and
        token_count(candidate.content) <= 512 and row["token_count"] == token_count(candidate.content) and
        row["source_version"] == source["source_version"] == candidate.source_version and
        row["source_sha256"] == source["content_sha256"] == candidate.source_sha256 == digest(source["content"]) and
        row["fragment_hash"] == fragment_hash() == candidate.fragment_hash and
        row["span_start"] == candidate.span_start and row["span_end"] == candidate.span_end and
        0 <= row["span_start"] < row["span_end"] <= len(source["content"]) and
        row["content"] == candidate.content == source["content"][row["span_start"]:row["span_end"]] and
        row["span_sha256"] == candidate.span_sha256 == digest(candidate.content))


# Cache only five counts and a digest, never source text, vectors or permissions.
# Every lookup fingerprints ALL current values used below in the caller's SQLite
# snapshot. Out-of-band edits without epoch/commit changes therefore miss too.
_counts_cache = OrderedDict()
_counts_lock = threading.Lock()
_COUNTS_CACHE_LIMIT = 128
_COUNTS_CACHE_MAX_BYTES = 128 * 1024
_counts_cache_bytes = 0


def _counts_key(groups):
    h = hashlib.sha256()
    for group in groups:
        h.update(b"group")
        for row in group:
            # sqlite3 scalar values have unambiguous tuple reprs (including
            # escaped strings, bytes, None and number types). Row-length framing
            # prevents separator ambiguity. Formatting runs in C rather than
            # invoking Python/hashlib for every field on every readiness call.
            encoded = repr(tuple(row)).encode("utf-8")
            h.update(len(encoded).to_bytes(8, "big"))
            h.update(encoded)
        h.update(b"endgroup")
    return h.digest()


def counts(db, user_id, profile):
    """Expected spans come from raw sources, so missing rows cannot look ready."""
    global _counts_cache_bytes
    rows = db.execute("""SELECT m.*,p.source_version,p.content_sha256 AS policy_hash
        FROM raw_messages m LEFT JOIN source_policies p ON m.id=p.raw_id AND m.user_id=p.user_id
        WHERE m.user_id=? AND m.deletion_state='active'""", (user_id,)).fetchall()
    table = partition(user_id)
    registry = db.execute("SELECT table_name FROM span_fts_partitions WHERE user_id=?",(user_id,)).fetchone()
    fts = {}
    if registry and registry[0] == table and db.execute("SELECT 1 FROM sqlite_master WHERE name=?",(table,)).fetchone():
        for r in db.execute(f"SELECT span_id,lexical_text FROM {table}"):
            fts.setdefault(r[0], []).append(r[1])
    result = {"active":0,"raw":0,"span":0,"fts":0,"dense":0}
    # One user-local snapshot, not two queries per expected span. Keep the
    # exact set, body, metadata, FTS multiplicity and numeric checks below.
    saved_spans = {r["id"]: dict(r) for r in db.execute(
        "SELECT * FROM source_spans WHERE user_id=?", (user_id,))}
    saved_vectors = {r["span_id"]: r for r in db.execute(
        "SELECT * FROM span_vectors WHERE user_id=? AND embedding_version=? AND dimensions=?",
        (user_id, profile.model_version, profile.dimensions))}
    key = _counts_key((
        [(user_id, profile.model_version, profile.dimensions, fragment_hash())],
        (tuple(r) for r in rows),
        (tuple(r.values()) for r in saved_spans.values()),
        ((span_id, *values) for span_id, values in fts.items()),
        (tuple(r) for r in saved_vectors.values()),
    ))
    with _counts_lock:
        cached = _counts_cache.get(key)
        if cached is not None:
            _counts_cache.move_to_end(key)
            return dict(cached[0])
    expected_ids=set();invalid_raw=False
    for raw in rows:
        expected = split_source(raw, raw["source_version"] or 1)
        result["active"] += len(expected)
        expected_ids.update(p["id"] for p in expected)
        if not expected or raw["policy_hash"] != digest(raw["content"]) or raw["source_version"] is None:
            invalid_raw=True
            continue
        result["raw"] += len(expected)
        for piece in expected:
            saved = saved_spans.get(piece["id"])
            if not saved or saved != piece:
                continue
            result["span"] += 1
            if fts.get(piece["id"]) == [build_lexical_text(piece["content"])]:
                result["fts"] += 1
            v = saved_vectors.get(piece["id"])
            if v and v["content_sha256"] == piece["span_sha256"]:
                from app.vector_checks import valid_vector_json
                if valid_vector_json(v["vector_json"],profile.dimensions):
                    result["dense"] += 1
    actual_ids=set(saved_spans)
    if actual_ids != expected_ids:
        result["span"]=-1
    if set(fts) != expected_ids:
        result["fts"]=-1
    if invalid_raw:
        result["raw"]=-1
    with _counts_lock:
        value = tuple(result.items())
        # Charge all recursively retained objects plus a conservative 256-byte
        # allowance for the mapping node/value wrapper (shared strings charged
        # each time). Both entries and accounted bytes are bounded.
        size = sys.getsizeof(key) + sys.getsizeof(value) + 256 + sum(
            sys.getsizeof(pair) + sum(sys.getsizeof(v) for v in pair) for pair in value)
        old = _counts_cache.pop(key, None)
        if old is not None:
            _counts_cache_bytes -= old[1]
        if size <= _COUNTS_CACHE_MAX_BYTES:
            _counts_cache[key] = (value, size)
            _counts_cache_bytes += size
        while (len(_counts_cache) > _COUNTS_CACHE_LIMIT or
               _counts_cache_bytes > _COUNTS_CACHE_MAX_BYTES):
            _, (_, removed_size) = _counts_cache.popitem(last=False)
            _counts_cache_bytes -= removed_size
    return result


def candidate(row, score):
    return EvidenceCandidate(id=row["id"],user_id=row["user_id"],content=row["content"],score=score,
        created_at=row["source_timestamp"] if row["source_timestamp"] is not None else row["recorded_at"],
        deletion_state=row["deletion_state"],source_raw_id=row["source_raw_id"],message_index=row["message_index"],
        representation="span_exact_v1",span_start=row["span_start"],span_end=row["span_end"],
        source_version=row["source_version"],source_sha256=row["source_sha256"],span_sha256=row["span_sha256"],
        fragment_hash=row["fragment_hash"],role=row["role"],session_id=row["session_id"],
        source_timestamp=row["source_timestamp"],recorded_at=row["recorded_at"],
        boundary_status=row["boundary_status"],previous_id=row["previous_id"],next_id=row["next_id"])


_JOIN = """FROM source_spans s JOIN raw_messages m ON m.id=s.source_raw_id AND m.user_id=s.user_id
    JOIN source_policies p ON p.raw_id=m.id AND p.user_id=m.user_id"""
_COLS = "s.*,m.role,m.session_id,m.message_index,m.source_timestamp,m.recorded_at,m.deletion_state"
_VALID = "m.deletion_state='active' AND s.source_version=p.source_version AND s.source_sha256=p.content_sha256"


def search(db, user_id, match_query, query_vector, version, dimensions, mode, candidate_limit, lexical_quota, dense_quota, reason=None, *, exact_acceleration=False):
    from app.store import RouteSnapshot
    table = partition(user_id)
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE name=?",(table,)).fetchone()
    def lexical(limit):
        if not exists or not match_query:
            return []
        rows = db.execute(f"SELECT {_COLS},bm25({table}) AS rank {_JOIN} JOIN {table} ON {table}.span_id=s.id "
            f"WHERE s.user_id=? AND {_VALID} AND {table} MATCH ? ORDER BY rank,s.span_index,s.id LIMIT ?",(user_id,match_query,limit))
        return [candidate(row,-float(row["rank"])) for row in rows]
    if mode == "raw_fts":
        return RouteSnapshot(lexical(candidate_limit),[],mode,None,0,0)
    rows = db.execute(f"SELECT {_COLS},v.vector_json,v.content_sha256 AS vector_hash {_JOIN} LEFT JOIN span_vectors v "
        "ON v.span_id=s.id AND v.user_id=s.user_id AND v.embedding_version=? AND v.dimensions=? "
        f"WHERE s.user_id=? AND {_VALID}",(version,dimensions,user_id)).fetchall()
    ready = sum(r["vector_json"] is not None for r in rows)
    if rows and ready != len(rows):
        reason = reason or "incomplete_current_version"
    if rows and query_vector is None:
        reason = reason or "query_vector_unavailable"
    dense = []
    if rows and not reason:
        try:
            for row in rows:
                if row["vector_hash"] != digest(row["content"]):
                    raise ValueError("stale_span_vector")
            # Exact scores/order are unchanged. Use bounded vectorized screening
            # by default when the optional numerical dependency is available.
            # Explicit opt-in retains its existing missing-dependency failure.
            try:
                from app.exact_dense import exact_top_scores
            except ImportError:
                if exact_acceleration:
                    raise
                exact_top_scores = None
            if exact_top_scores is not None:
                scores = exact_top_scores([r['vector_json'] for r in rows], query_vector, dimensions,
                    dense_quota if mode == 'raw_hybrid_rrf' else candidate_limit,
                    [(r['message_index'], r['span_start'], r['id']) for r in rows])
                dense = [candidate(rows[i], score) for i, score in scores]
            else:
                query = normalize_vector(query_vector,dimensions)
                for row in rows:
                    vector = normalize_vector(json.loads(row["vector_json"]),dimensions)
                    dense.append(candidate(row,max(-1.,min(1.,math.fsum(a*b for a,b in zip(query,vector))))))
        except (EmbeddingFailure,ValueError,TypeError):
            reason, dense = "invalid_persisted_vector", []
    if reason:
        return RouteSnapshot(lexical(candidate_limit),[],"raw_fts",reason,len(rows),ready)
    dense.sort(key=lambda h:(-h.score,h.message_index,h.span_start,h.id))
    return RouteSnapshot(lexical(lexical_quota) if mode == "raw_hybrid_rrf" else [],
        dense[:dense_quota if mode == "raw_hybrid_rrf" else candidate_limit],mode,None,len(rows),ready)

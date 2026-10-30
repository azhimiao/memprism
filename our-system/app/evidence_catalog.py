"""Rebuildable user-partitioned 062 source directory, published with Add."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from app import segments
from app.evidence_shapes import VERSION, blocks, explicit_fields
from app.index import build_lexical_text
from app.models import EvidenceCandidate

SAFE = "m.deletion_state='active' AND m.privacy_level='normal' AND p.visibility='visible' AND p.disclosure_policy='within_user' AND p.retraction_state='not_retracted'"
SOURCE_JOIN = "FROM raw_messages m JOIN source_policies p ON p.raw_id=m.id AND p.user_id=m.user_id"


def initialize(db):
    db.execute("""CREATE TABLE IF NOT EXISTS evidence_062_proofs (
        user_id TEXT PRIMARY KEY, policy TEXT NOT NULL, epoch INTEGER NOT NULL,
        commit_seq INTEGER NOT NULL, config_hash TEXT NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS evidence_062_sources (
        user_id TEXT NOT NULL, raw_id TEXT NOT NULL, version INTEGER NOT NULL, sha TEXT NOT NULL,
        policy TEXT NOT NULL, session_id TEXT NOT NULL, source_time INTEGER, block_count INTEGER NOT NULL,
        units_json TEXT NOT NULL, sensitive INTEGER NOT NULL, PRIMARY KEY(user_id,raw_id))""")
    db.execute("""CREATE TABLE IF NOT EXISTS evidence_062_blocks (
        id TEXT PRIMARY KEY, user_id TEXT NOT NULL, raw_id TEXT NOT NULL, lo INTEGER NOT NULL, hi INTEGER NOT NULL,
        content TEXT NOT NULL, kind TEXT NOT NULL, sensitive INTEGER NOT NULL, withhold INTEGER NOT NULL,
        state_key TEXT NOT NULL, rule_key TEXT NOT NULL, ref_key TEXT NOT NULL, effective TEXT NOT NULL,
        explicit_update INTEGER NOT NULL)""")
    db.execute("CREATE INDEX IF NOT EXISTS evidence_062_source_idx ON evidence_062_blocks(user_id,raw_id,lo)")
    db.execute("CREATE INDEX IF NOT EXISTS evidence_062_state_idx ON evidence_062_blocks(user_id,state_key)")
    db.execute("CREATE INDEX IF NOT EXISTS evidence_062_rule_idx ON evidence_062_blocks(user_id,rule_key)")
    db.execute("CREATE INDEX IF NOT EXISTS evidence_062_session_idx ON evidence_062_sources(user_id,session_id,source_time,raw_id)")
    db.execute("""CREATE TABLE IF NOT EXISTS evidence_062_partitions (
        user_id TEXT PRIMARY KEY, table_name TEXT NOT NULL UNIQUE)""")
    for target in ('raw_messages','source_policies','evidence_062_sources','evidence_062_blocks'):
        for action in ('INSERT','UPDATE','DELETE'):
            who='OLD' if action=='DELETE' else 'NEW'
            name='evidence_062_invalidate_'+target+'_'+action.lower()
            db.execute(f"CREATE TRIGGER IF NOT EXISTS {name} AFTER {action} ON {target} BEGIN DELETE FROM evidence_062_proofs WHERE user_id={who}.user_id; END")


def table(user):
    return "evidence_062_fts_" + hashlib.sha256(user.encode()).hexdigest()


def ensure_partition(db, user):
    name = table(user)
    db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING fts5(id UNINDEXED, text, tokenize='unicode61')")
    db.execute("INSERT OR IGNORE INTO evidence_062_partitions VALUES (?,?)", (user, name))
    return name


def block_id(user, raw, version, sha, lo, hi):
    return "block_" + segments.digest(json.dumps([VERSION,user,raw,version,sha,lo,hi], ensure_ascii=False, separators=(",", ":")))


def key(parts):
    return json.dumps([p.casefold().strip() for p in parts], ensure_ascii=False, separators=(",", ":"))


def sync(db, user, *, force=False):
    """Inside caller write transaction; no network, original mutation or vectors."""
    name = ensure_partition(db, user)
    eligible = {r[0]:(r[1],r[2]) for r in db.execute(
        f"SELECT m.id,p.source_version,p.content_sha256 {SOURCE_JOIN} WHERE m.user_id=? AND {SAFE}", (user,))}
    existing = {r[0]:(r[1],r[2],r[3]) for r in db.execute(
        "SELECT raw_id,version,sha,policy FROM evidence_062_sources WHERE user_id=?", (user,))}
    changed = set(existing) - eligible.keys()
    changed.update(raw for raw, stamp in eligible.items() if force or existing.get(raw) != (*stamp, VERSION))
    for raw in sorted(changed):
        db.execute(f"DELETE FROM {name} WHERE id IN (SELECT id FROM evidence_062_blocks WHERE user_id=? AND raw_id=?)", (user,raw))
        db.execute("DELETE FROM evidence_062_blocks WHERE user_id=? AND raw_id=?", (user,raw))
        db.execute("DELETE FROM evidence_062_sources WHERE user_id=? AND raw_id=?", (user,raw))
        if raw not in eligible:
            continue
        row = db.execute(f"SELECT m.*,p.source_version,p.content_sha256 {SOURCE_JOIN} WHERE m.user_id=? AND m.id=? AND {SAFE}", (user,raw)).fetchone()
        if segments.digest(row['content']) != row['content_sha256']:
            raise ValueError("catalog_source_hash_mismatch")
        spans, units = blocks(row['content'])
        # Reuse the pre-existing narrow full-sentence grammar only as navigation.
        # First-person roles are not resolved into a cross-session named person.
        from app.governance import parse_claim
        from app.models import RawMessage
        claim = parse_claim(RawMessage(**dict(row))) if len(row['content'])<=7000 else None
        if claim and claim.single_valued:
            units.append((0,len(row['content']),[(0,len(row['content']))],'bounded_claim'))
        records = []
        for span in spans:
            lo, hi = span['lo'], span['hi']
            relevant = [u for u in units if lo < u[1] and hi > u[0]]
            fields = {}
            for unit in relevant:
                candidate_fields = explicit_fields(row['content'][unit[0]:unit[1]])
                if candidate_fields:
                    fields = candidate_fields
                    break
            state = key([fields[k] for k in ('subject','attribute','scope')]) if {'subject','attribute','scope','value'} <= fields.keys() else ''
            if not state and claim and claim.single_valued:
                state = key([row['session_id'],claim.subject,claim.predicate])
            scope = fields.get('scope', '')
            rule = key([row['session_id'],scope,fields['rule id']]) if scope and fields.get('rule id') else ''
            ref = key([row['session_id'],scope,fields['reference']]) if scope and fields.get('reference') else ''
            effective = fields.get('effective', '')
            try:
                date.fromisoformat(effective)
            except ValueError:
                effective = ''
            bid = block_id(user, raw, row['source_version'],row['content_sha256'],lo,hi)
            records.append((bid,user,raw,lo,hi,row['content'][lo:hi],span['kind'],int(span['sensitive']),int(span['withhold']),
                            state,rule,ref,effective,int(bool(effective) and fields.get('cardinality','').casefold() in {'single','true','是'}
                                and fields.get('update','').casefold() in {'true','yes','是','明确更新'})))
        db.executemany("INSERT INTO evidence_062_blocks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", records)
        db.executemany(f"INSERT INTO {name} VALUES (?,?)", [(r[0],build_lexical_text(r[5])) for r in records])
        db.execute("INSERT INTO evidence_062_sources VALUES (?,?,?,?,?,?,?,?,?,?)", (
            user,raw,row['source_version'],row['content_sha256'],VERSION,row['session_id'],row['source_timestamp'],
            len(records),json.dumps(units,ensure_ascii=False),int(any(s['sensitive'] or s['withhold'] for s in spans))))
    return len(changed)


def audit(db, user):
    eligible = db.execute(f"SELECT COUNT(*) {SOURCE_JOIN} WHERE m.user_id=? AND {SAFE}", (user,)).fetchone()[0]
    actual = db.execute(f"""SELECT COUNT(*) FROM evidence_062_sources c JOIN raw_messages m ON m.id=c.raw_id AND m.user_id=c.user_id
        JOIN source_policies p ON p.raw_id=m.id AND p.user_id=m.user_id WHERE c.user_id=? AND {SAFE}
        AND c.version=p.source_version AND c.sha=p.content_sha256 AND c.policy=?
        AND c.block_count=(SELECT COUNT(*) FROM evidence_062_blocks b WHERE b.user_id=c.user_id AND b.raw_id=c.raw_id)""", (user,VERSION)).fetchone()[0]
    total = db.execute("SELECT COUNT(*) FROM evidence_062_sources WHERE user_id=?", (user,)).fetchone()[0]
    if not eligible and not total:
        return True
    name = table(user)
    registered = db.execute("SELECT table_name FROM evidence_062_partitions WHERE user_id=?", (user,)).fetchone()
    if not registered or registered[0] != name:
        return False
    block_count = db.execute("SELECT COUNT(*) FROM evidence_062_blocks WHERE user_id=?", (user,)).fetchone()[0]
    return eligible == actual == total and db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] == block_count


def certify_on(db,user,config_hash):
    if not audit(db,user):
        raise ValueError('catalog_certification_failed')
    epoch=db.execute('SELECT epoch FROM lifecycle_users WHERE user_id=?',(user,)).fetchone()
    seq=db.execute('SELECT commit_seq FROM k0_users WHERE user_id=?',(user,)).fetchone()
    db.execute('INSERT OR REPLACE INTO evidence_062_proofs VALUES (?,?,?,?,?)',
               (user,VERSION,epoch[0] if epoch else 0,seq[0] if seq else 0,config_hash))


def ready(db,user,config_hash=None):
    """Constant-size publication proof; returned blocks still validate Raw."""
    row=db.execute('''SELECT p.policy,p.epoch,p.commit_seq,p.config_hash,
        COALESCE(l.epoch,0),COALESCE(k.commit_seq,0) FROM evidence_062_proofs p
        LEFT JOIN lifecycle_users l ON l.user_id=p.user_id LEFT JOIN k0_users k ON k.user_id=p.user_id
        WHERE p.user_id=?''',(user,)).fetchone()
    if row:
        return row[0]==VERSION and row[1]==row[4] and row[2]==row[5] and (config_hash is None or row[3]==config_hash)
    return db.execute('SELECT 1 FROM raw_messages WHERE user_id=? LIMIT 1',(user,)).fetchone() is None


JOIN = """FROM evidence_062_blocks b JOIN evidence_062_sources c ON c.raw_id=b.raw_id AND c.user_id=b.user_id
    JOIN raw_messages m ON m.id=b.raw_id AND m.user_id=b.user_id
    JOIN source_policies p ON p.raw_id=m.id AND p.user_id=m.user_id"""
COLS = "b.*,c.version,c.sha,c.policy,c.units_json,m.role,m.session_id,m.message_index,m.source_timestamp,m.recorded_at,m.deletion_state"
VALID = SAFE + " AND c.version=p.source_version AND c.sha=p.content_sha256"


def candidate(row, score=0.0):
    return EvidenceCandidate(id=row['id'],user_id=row['user_id'],content=row['content'],score=score,
        created_at=row['source_timestamp'] if row['source_timestamp'] is not None else row['recorded_at'],
        deletion_state=row['deletion_state'],source_raw_id=row['raw_id'],message_index=row['message_index'],
        representation='evidence_block_v1',span_start=row['lo'],span_end=row['hi'],source_version=row['version'],
        source_sha256=row['sha'],span_sha256=segments.digest(row['content']),fragment_hash=VERSION,
        role=row['role'],session_id=row['session_id'],source_timestamp=row['source_timestamp'],recorded_at=row['recorded_at'],
        boundary_status='exact_source_block')


def get(db, user, bid):
    return db.execute(f"SELECT {COLS} {JOIN} WHERE b.user_id=? AND b.id=? AND {VALID}", (user,bid)).fetchone()


def valid_candidate(db, hit, source):
    if not source or hit.representation != 'evidence_block_v1':
        return False
    row = get(db, hit.user_id, hit.id)
    return bool(row and candidate(row,hit.score) == hit and row['policy'] == VERSION
        and hit.id == block_id(hit.user_id,hit.source_raw_id,hit.source_version,hit.source_sha256,hit.span_start,hit.span_end)
        and source['content_sha256'] == hit.source_sha256 == segments.digest(source['content'])
        and 0 <= hit.span_start < hit.span_end <= len(source['content'])
        and hit.content == source['content'][hit.span_start:hit.span_end])


def search(db, user, match, limit):
    if not match or limit <= 0:
        return []
    name = table(user)
    if not db.execute("SELECT 1 FROM evidence_062_partitions WHERE user_id=? AND table_name=?", (user,name)).fetchone():
        return []
    rows = db.execute(f"SELECT id FROM {name} WHERE {name} MATCH ? ORDER BY bm25({name}),id LIMIT ?", (match,min(100,limit))).fetchall()
    return [r for (bid,) in rows if (r := get(db,user,bid)) is not None]


def source_blocks(db, user, raw, *, lo=None, hi=None, limit=65):
    extra = " AND b.lo<? AND b.hi>?" if lo is not None else ""
    params = (user,raw,hi,lo,limit) if lo is not None else (user,raw,limit)
    return db.execute(f"SELECT {COLS} {JOIN} WHERE b.user_id=? AND b.raw_id=? AND {VALID}{extra} ORDER BY b.lo,b.id LIMIT ?", params).fetchall()

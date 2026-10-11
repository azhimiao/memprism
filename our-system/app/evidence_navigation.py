"""Source-bound sentence navigation and whole-message evidence expansion."""
from __future__ import annotations
import hashlib
import re
from app.index import build_lexical_text
from app.models import EvidenceCandidate


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def initialize(db):
    db.execute('''CREATE TABLE IF NOT EXISTS sentence_sources(
        user_id TEXT NOT NULL,raw_id TEXT NOT NULL,source_hash TEXT NOT NULL,
        PRIMARY KEY(user_id,raw_id),FOREIGN KEY(raw_id) REFERENCES raw_messages(id) ON DELETE CASCADE)''')
    db.execute('''CREATE TABLE IF NOT EXISTS sentence_units(
        id TEXT PRIMARY KEY,user_id TEXT NOT NULL,raw_id TEXT NOT NULL,
        start_offset INTEGER NOT NULL,end_offset INTEGER NOT NULL,content TEXT NOT NULL,
        FOREIGN KEY(raw_id) REFERENCES raw_messages(id) ON DELETE CASCADE)''')
    db.execute('CREATE INDEX IF NOT EXISTS sentence_user ON sentence_units(user_id,raw_id)')


def table(user):
    return 'sentence_fts_'+digest(user)


def ensure_table(db,user):
    name=table(user)
    db.execute(f'CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING fts5(id UNINDEXED,lexical_text)')
    return name


def sentence_offsets(text):
    start=0
    for match in re.finditer(r'\n+|[.!?。！？](?:\s+|$)',text):
        end=match.end()
        if text[start:end].strip():
            yield start,end
        start=end
    if text[start:].strip():
        yield start,len(text)


def insert(db,user,messages):
    name=ensure_table(db,user)
    for raw in messages:
        content=raw.content
        db.execute('INSERT INTO sentence_sources VALUES(?,?,?)',(user,raw.id,digest(content)))
        for start,end in sentence_offsets(content):
            sid='sentence_'+digest(f'{user}\0{raw.id}\0{start}\0{end}\0{digest(content)}')
            db.execute('INSERT INTO sentence_units VALUES(?,?,?,?,?,?)',(sid,user,raw.id,start,end,content[start:end]))
            db.execute(f'INSERT INTO {name} VALUES(?,?)',(sid,build_lexical_text(content[start:end])))


def prune(db,user):
    name=ensure_table(db,user)
    bad=db.execute('''SELECT s.raw_id FROM sentence_sources s LEFT JOIN raw_messages m
        ON m.id=s.raw_id AND m.user_id=s.user_id LEFT JOIN source_policies p
        ON p.raw_id=m.id AND p.user_id=m.user_id WHERE s.user_id=? AND
        (m.id IS NULL OR m.deletion_state!='active' OR p.content_sha256!=s.source_hash
         OR p.content_sha256 IS NULL)''',(user,)).fetchall()
    for row in bad:
        ids=db.execute('SELECT id FROM sentence_units WHERE user_id=? AND raw_id=?',(user,row[0])).fetchall()
        db.executemany(f'DELETE FROM {name} WHERE id=?',[(x[0],) for x in ids])
        db.execute('DELETE FROM sentence_units WHERE user_id=? AND raw_id=?',(user,row[0]))
        db.execute('DELETE FROM sentence_sources WHERE user_id=? AND raw_id=?',(user,row[0]))


def counts(db,user):
    active=db.execute("SELECT id,content FROM raw_messages WHERE user_id=? AND deletion_state='active'",(user,)).fetchall()
    name=table(user)
    if not db.execute('SELECT 1 FROM sqlite_master WHERE name=?',(name,)).fetchone():
        return 0
    ready=0
    for raw in active:
        source=db.execute('SELECT source_hash FROM sentence_sources WHERE user_id=? AND raw_id=?',(user,raw['id'])).fetchone()
        units=db.execute('SELECT * FROM sentence_units WHERE user_id=? AND raw_id=? ORDER BY start_offset',(user,raw['id'])).fetchall()
        expected=list(sentence_offsets(raw['content']))
        if not source or source[0]!=digest(raw['content']) or len(units)!=len(expected):
            continue
        if all((u['start_offset'],u['end_offset'])==(a,b) and u['content']==raw['content'][a:b]
               and [x[0] for x in db.execute(f'SELECT lexical_text FROM {name} WHERE id=?',(u['id'],))]==[build_lexical_text(u['content'])]
               for u,(a,b) in zip(units,expected)):
            ready+=1
    return ready


def raw_candidate(row,score=0.0):
    return EvidenceCandidate(id=row['id'],user_id=row['user_id'],content=row['content'],score=score,
        created_at=row['source_timestamp'] if row['source_timestamp'] is not None else row['recorded_at'],
        deletion_state=row['deletion_state'],source_raw_id=row['id'],message_index=row['message_index'],
        role=row['role'],session_id=row['session_id'],source_timestamp=row['source_timestamp'],recorded_at=row['recorded_at'],
        source_version=row['source_version'],source_sha256=row['content_sha256'])


def search(db,user,match,limit):
    if not match or limit<1:return [],[]
    name=table(user)
    if not db.execute('SELECT 1 FROM sqlite_master WHERE name=?',(name,)).fetchone():return [],[]
    rows=db.execute(f'''SELECT s.*,bm25({name}) AS rank FROM {name} JOIN sentence_units s
        ON s.id={name}.id WHERE {name} MATCH ? AND s.user_id=? ORDER BY rank,s.id LIMIT ?''',(match,user,limit)).fetchall()
    results=[];seen=set();trace=[]
    for unit in rows:
        raw=db.execute('''SELECT m.*,p.source_version,p.content_sha256 FROM raw_messages m
            JOIN source_policies p ON p.user_id=m.user_id AND p.raw_id=m.id
            WHERE m.user_id=? AND m.id=? AND m.deletion_state='active' AND p.visibility='visible'
            AND p.disclosure_policy='within_user' AND p.retraction_state='not_retracted' ''',(user,unit['raw_id'])).fetchone()
        if not raw or raw['content'][unit['start_offset']:unit['end_offset']]!=unit['content'] or digest(raw['content'])!=raw['content_sha256']:continue
        trace.append({'sentence_id':unit['id'],'source_id':unit['raw_id'],'offsets':[unit['start_offset'],unit['end_offset']]})
        if raw['id'] not in seen:
            seen.add(raw['id']);results.append(raw_candidate(raw,-float(unit['rank'])))
    return results,trace


def expand(db,user,candidates,*,parents=False,radius=0,limit=200):
    result=[];seen=set();trace=[]
    for hit in candidates:
        if len(result)>=limit:break
        raw=db.execute('''SELECT m.*,p.source_version,p.content_sha256 FROM raw_messages m
            JOIN source_policies p ON p.raw_id=m.id AND p.user_id=m.user_id
            WHERE m.user_id=? AND m.id=? AND m.deletion_state='active' ''',(user,hit.source_raw_id)).fetchone()
        if raw is None:continue
        selected=raw_candidate(raw,hit.score) if parents else hit
        if selected.id not in seen:
            result.append(selected);seen.add(selected.id)
            if parents:trace.append({'source_id':raw['id'],'action':'whole_received_message'})
        if radius:
            # Adjacent messages are from this exact Add, not inferred across Add boundaries.
            neighbors=db.execute('''SELECT m.*,p.source_version,p.content_sha256 FROM raw_messages m
                JOIN source_policies p ON p.raw_id=m.id AND p.user_id=m.user_id WHERE m.user_id=?
                AND m.request_id=? AND m.session_id=? AND m.deletion_state='active'
                AND m.message_index BETWEEN ? AND ? ORDER BY m.message_index''',
                (user,raw['request_id'],raw['session_id'],raw['message_index']-radius,raw['message_index']+radius)).fetchall()
            for row in neighbors:
                if len(result)>=limit:break
                if row['id'] not in seen:
                    result.append(raw_candidate(row,hit.score));seen.add(row['id'])
                    trace.append({'source_id':row['id'],'seed':raw['id'],'action':'same_add_neighbor'})
    return result,trace


def bridge_seeds(candidates,query,seen=()):
    """Only literal named entities in already recalled evidence can seed a hop."""
    values=[];blocked=set(seen)
    for hit in candidates[:8]:
        for match in re.finditer(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}\b',hit.content):
            value=match.group(0)
            if value.casefold() in query.casefold() or value in blocked or value in {'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','January','February','March','April','May','June','July','August','September','October','November','December'}:continue
            values.append((value,hit.source_raw_id));blocked.add(value)
            if len(values)==4:return values
    return values

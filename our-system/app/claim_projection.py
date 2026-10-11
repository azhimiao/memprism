"""Optional finite-grammar current/as-of projection over exact source spans.

No inferred value is returned. Invalid cached successors preserve the original
source candidate. Dates remain source/effective times, not the server clock.
"""
import json
import hashlib
from app.memory_core import valid_derived_source


def erased_transition(db,user_id,row):
    """Verify a committed hash-only transition plus a completed legal erasure.

    This trusts transactional service receipts, not arbitrary whole-DB edits.
    A hidden/rejected/missing derived row alone is never an erasure receipt.
    """
    proof=db.execute('''SELECT * FROM claim_supersessions WHERE user_id=? AND predecessor_id=?
        AND successor_id=?''',(user_id,row['id'],row['superseded_by'])).fetchone()
    if proof is None or proof['claim_key']!=row['claim_key'] or proof['valid_until']!=row['valid_until'] or proof['predecessor_payload_sha256']!=hashlib.sha256(row['payload_json'].encode()).hexdigest():return False
    erased=db.execute('''SELECT m.deletion_state,p.visibility,p.content_sha256 FROM raw_messages m
        JOIN source_policies p ON p.raw_id=m.id AND p.user_id=m.user_id JOIN memory_tombstones t
        ON t.raw_id=m.id AND t.user_id=m.user_id WHERE m.user_id=? AND m.id=?''',
        (user_id,proof['successor_raw_id'])).fetchone()
    if erased is None or tuple(erased)!=('deleted','erased',hashlib.sha256(b'').hexdigest()):return False
    receipts=db.execute("SELECT target_ids_json FROM memory_operations WHERE user_id=? AND operation='erase' AND status='completed' AND erasure_complete=1",(user_id,)).fetchall()
    return any(proof['successor_raw_id'] in json.loads(r[0]) for r in receipts)


def verified_group(db,user_id,key):
    rows=db.execute('SELECT * FROM derived_memories WHERE user_id=? AND claim_key=?',(user_id,key)).fetchall()
    bad=[];payloads={}
    for row in rows:
        source=db.execute('''SELECT m.deletion_state,m.privacy_level,p.visibility,p.disclosure_policy,p.retraction_state
            FROM raw_messages m JOIN source_policies p ON m.id=p.raw_id AND m.user_id=p.user_id
            WHERE m.user_id=? AND m.id=?''',(user_id,row['source_raw_id'])).fetchone()
        if not valid_derived_source(db,row) or row['admission']!='accepted' or row['visibility']!='visible' or row['retraction_state']!='not_retracted' or source is None or tuple(source)!=('active','normal','visible','within_user','not_retracted'):
            bad.append(row['id']);continue
        payloads[row['id']]=json.loads(row['payload_json'])
        if row['valid_from']!=payloads[row['id']]['valid_from']:bad.append(row['id'])
    if bad:return {},bad
    states={}
    for row in rows:
        payload=payloads[row['id']]
        successors=[other for other in rows if other['object_text']!=row['object_text']
            and payload['single_valued'] and payloads[other['id']]['single_valued']
            and payloads[other['id']]['explicit_update'] and row['valid_from'] is not None
            and other['valid_from'] is not None
            and other['valid_from']>row['valid_from']+(86399999 if payload.get('temporal_precision')=='day' else 0)]
        successors.sort(key=lambda r:(r['valid_from'],r['id']))
        until,successor=(successors[0]['valid_from'],successors[0]['id']) if successors else (None,None)
        if row['valid_until'] is not None and (until is None or row['valid_until']<until) and erased_transition(db,user_id,row):
            until,successor=row['valid_until'],row['superseded_by']
        # A removed/hidden/rejected successor cannot authorize omission.
        if (row['valid_until'],row['superseded_by'])!=(until,successor):bad.append(row['id'])
        states[row['id']]={'valid_from':payload['valid_from'],'valid_until':until,'superseded_by':successor}
    remaining=[r for r in rows if states[r['id']]['valid_until'] is None]
    conflict=len({r['object_text'] for r in remaining})>1 and all(payloads[r['id']]['single_valued'] for r in remaining)
    for row in rows:
        state=states[row['id']]
        currency='superseded' if state['valid_until'] is not None else ('uncertain' if conflict or state['valid_from'] is None else 'current')
        resolution='resolved' if state['valid_until'] is not None else ('unresolved' if conflict else 'none')
        if not payloads[row['id']]['single_valued']:currency,resolution='not_applicable','none'
        if row['currency']!=currency or row['resolution']!=resolution:bad.append(row['id'])
        state['currency']=currency
    return states,sorted(set(bad))


def project_claims(db,user_id,candidates,view):
    result=[];decisions=[];invalid=[];verified={}
    for hit in candidates:
        claims=db.execute("SELECT * FROM derived_memories WHERE user_id=? AND source_raw_id=? AND admission='accepted'",
                          (user_id,hit.source_raw_id)).fetchall()
        eligible=True
        for claim in claims:
            if claim['claim_key'] not in verified:verified[claim['claim_key']]=verified_group(db,user_id,claim['claim_key'])
            states,bad=verified[claim['claim_key']]
            if bad:
                invalid.extend(bad);continue
            state=states[claim['id']]
            if view.intent=='current' and state['currency']=='superseded':eligible=False
            if view.intent=='as_of' and ((state['valid_from'] is not None and state['valid_from']>view.as_of) or
                (state['valid_until'] is not None and state['valid_until']<=view.as_of)):eligible=False
        if eligible:result.append(hit)
        decisions.append({'candidate_id':hit.id,'claim_ids':[r['id'] for r in claims],
                          'decision':'keep' if eligible else 'omit','view':view.intent})
    return result,{'policy':'verified_grammar_projection_v1','state_proof':'rederive_visible_group_v2',
                   'invalid_group_policy':'retain_raw_as_unresolved_not_current_fact',
                   'decisions':decisions,'invalid_claim_ids':sorted(set(invalid))}

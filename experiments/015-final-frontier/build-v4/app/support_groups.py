"""Bounded source adjacency navigation, with explicit complete-group packing."""
from app import segments


def span_windows(db,user_id,seeds,limit=200,seed_limit=20):
    result=list(seeds[:limit]);seen={h.id for h in result};groups=[];visits=0
    base_count=len(result)
    if base_count>=limit:
        return result,{'groups':[],'source_ref_visits':0,'expanded_spans':0,
                       'max_hops':1,'seed_limit':seed_limit,'disabled_reason':'no_expansion_capacity'}
    for seed in seeds[:seed_limit]:
        if visits+2>100:break
        visits+=2
        row=db.execute(f"SELECT {segments._COLS} {segments._JOIN} WHERE s.user_id=? AND s.id=? AND {segments._VALID}",
                       (user_id,seed.id)).fetchone()
        source=db.execute("""SELECT m.*,m.rowid AS ordinal,p.source_version,p.content_sha256
                FROM raw_messages m JOIN source_policies p ON p.raw_id=m.id AND p.user_id=m.user_id
                WHERE m.user_id=? AND m.id=?""",(user_id,seed.source_raw_id)).fetchone()
        if row is None or not segments.valid_candidate(db,seed,source):continue
        # Only neighbors of a legal retrieved seed; no global last-N fallback.
        related=[]
        for span_id in (row["previous_id"],row["next_id"]):
            if span_id:related.append((span_id,"same_source_adjacency"))
        anchor=source['source_timestamp'] if source['source_timestamp'] is not None else source['recorded_at']
        for comparison,order in (('<','DESC'),('>','ASC')):
            neighbor=db.execute(f"""SELECT id FROM raw_messages WHERE user_id=? AND session_id=?
                AND deletion_state='active' AND (COALESCE(source_timestamp,recorded_at),rowid) {comparison} (?,?)
                ORDER BY COALESCE(source_timestamp,recorded_at) {order},rowid {order} LIMIT 1""",
                (user_id,row['session_id'],anchor,source['ordinal'])).fetchone()
            if neighbor:
                spans=db.execute('SELECT id FROM source_spans WHERE user_id=? AND source_raw_id=? ORDER BY span_index LIMIT 2',
                                 (user_id,neighbor[0])).fetchall()
                related.extend((r[0],'session_neighbor_navigation') for r in spans)
                if not spans:related.append(('unavailable_raw:'+neighbor[0],'missing_span'))
        members=[seed.id];unavailable=[]
        for span_id,witness in dict.fromkeys(related):
            members.append(span_id)
            if visits>=100:
                unavailable.append({'id':span_id,'reason':'visit_budget'});continue
            visits+=1
            neighbor=db.execute(f"SELECT {segments._COLS} {segments._JOIN} WHERE s.user_id=? AND s.id=? AND {segments._VALID} "
                "AND p.visibility='visible' AND p.disclosure_policy='within_user' AND p.retraction_state='not_retracted' AND m.privacy_level='normal'",
                (user_id,span_id)).fetchone()
            if neighbor is None:
                unavailable.append({'id':span_id,'reason':'unavailable_dependency'});continue
            if span_id not in seen:
                # Ranks are assigned after recall; neighboring text cannot gain
                # a cosine score it was never measured to have.
                if len(result)>=limit:
                    unavailable.append({'id':span_id,'reason':'candidate_budget'});continue
                result.append(segments.candidate(neighbor,0.));seen.add(span_id)
        groups.append({"seed_id":seed.id,"member_ids":list(dict.fromkeys(members)),
                       'unavailable_dependencies':unavailable,
                       "witness":"versioned_source/session_adjacency","semantic_entailment":False})
    return result,{"groups":groups,"source_ref_visits":visits,"expanded_spans":len(result)-base_count,
                   "max_hops":1,"seed_limit":seed_limit,'visit_scope':'logical references; not SQL work'}


def pack_groups(candidates,groups,max_items,char_limit,token_limit=None,renderer=None):
    """Admit a complete dependency group or omit it, preserving exact texts."""
    from app.segments import token_count
    registry={h.id:h for h in candidates};edges={};unavailable=set()
    for group in groups:
        ids=list(dict.fromkeys([group['seed_id']]+group['member_ids']))
        for id in ids:edges.setdefault(id,set()).update(ids)
        if group.get('unavailable_dependencies'):unavailable.update(ids)
    kept=[];used=set();processed=set();chars=tokens=0;omitted=[]
    for hit in candidates:
        if hit.id in processed:continue
        component={hit.id};pending=[hit.id]
        while pending:
            current=pending.pop()
            for other in edges.get(current,()):
                if other not in component:component.add(other);pending.append(other)
        processed.update(component)
        if component.intersection(unavailable) or not component<=registry.keys():
            omitted.append({'seed_id':hit.id,'reason':'missing_dependency','member_ids':sorted(component)});continue
        members=[h for h in candidates if h.id in component and h.id not in used]
        texts=[renderer(h) if renderer else h.content for h in members]
        new_chars=sum(len(text) for text in texts);new_tokens=sum(token_count(text) for text in texts)
        if len(kept)+len(members)>max_items or chars+new_chars>char_limit or (token_limit is not None and tokens+new_tokens>token_limit):
            omitted.append({'seed_id':hit.id,'reason':'group_over_budget','member_ids':sorted(component)});continue
        kept.extend(members);used.update(component);chars+=new_chars;tokens+=new_tokens
    return kept,chars,{"policy":"complete_witness_group_v1",'closure':'overlapping_components_v2',
                       "omitted_groups":omitted,"packed_content_tokens":tokens}

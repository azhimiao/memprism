"""062 bounded evidence assembly. Search never writes or executes instructions."""
from __future__ import annotations

import json
import re
from collections import deque
from app import evidence_catalog as catalog
from app.evidence_budget import record_token_cost
from app.evidence_shapes import demand, explicit_fields, SENSITIVE
from app.index import build_match_query

_STOP = set("the a an is are was were my your our his her their it of to for and or in on at with what which when please tell me summarize summary overview all history current latest private sensitive confidential share disclose".split())


def terms(text):
    words = re.findall(r"[A-Za-z0-9_]{2,}|[\u3400-\u9fff]{2,}", text.casefold())
    return set(w for w in words if w not in _STOP) | {w[i:i+2] for w in words if re.search(r"[\u3400-\u9fff]",w) for i in range(len(w)-1)}


def necessary(row, plan):
    if row['withhold']:
        return False
    if not row['sensitive']:
        return True
    # Existing within_user policy provides authorization. This only narrows
    # necessary disclosure; topic matching can never grant new permissions.
    useful = terms(SENSITIVE.sub('',row['content']))
    return bool(useful & terms(plan['focus']))


def _sample_offsets(count, cap):
    if cap <= 0:
        return []
    if count <= cap:
        return range(count)
    return sorted({round(i*(count-1)/(cap-1)) for i in range(cap)}) if cap > 1 else [0]


def history_rows(db, user, seeds, plan, cap=48):
    """Bounded metadata navigation, never a recency fallback for unknown queries."""
    if cap <= 0:
        return []
    if plan['global']:
        count = db.execute("SELECT COUNT(DISTINCT session_id) FROM evidence_062_sources WHERE user_id=?",(user,)).fetchone()[0]
        sessions = [db.execute("SELECT DISTINCT session_id FROM evidence_062_sources WHERE user_id=? ORDER BY session_id LIMIT 1 OFFSET ?",(user,offset)).fetchone()[0]
                    for offset in _sample_offsets(count, min(16,cap))]
    else:
        sessions = list(dict.fromkeys(h.session_id for h in seeds if h.session_id is not None))[:16]
    buckets = []
    for session in sessions:
        bucket=[]
        count = db.execute("SELECT COUNT(*) FROM evidence_062_sources WHERE user_id=? AND session_id=?",(user,session)).fetchone()[0]
        for offset in _sample_offsets(count,3):
            raw = db.execute("SELECT raw_id FROM evidence_062_sources WHERE user_id=? AND session_id=? ORDER BY source_time IS NULL,source_time,raw_id LIMIT 1 OFFSET ?",(user,session,offset)).fetchone()[0]
            total=db.execute('SELECT block_count FROM evidence_062_sources WHERE user_id=? AND raw_id=?',(user,raw)).fetchone()[0]
            for position in _sample_offsets(total,8):
                row=db.execute(f'SELECT {catalog.COLS} {catalog.JOIN} WHERE b.user_id=? AND b.raw_id=? AND {catalog.VALID} ORDER BY b.lo,b.id LIMIT 1 OFFSET ?',
                               (user,raw,position)).fetchone()
                if row:bucket.append(row)
        buckets.append(bucket)
    result=[]
    for i in range(max((len(b) for b in buckets),default=0)):
        for bucket in buckets:
            if i<len(bucket):result.append(bucket[i])
            if len(result)>=cap:return result
    return result


def assemble(db, user, seeds, query, *, limit=200, max_items=100,
             char_limit=1000000, token_limit=65536, renderer=None, recall_limit=64):
    plan = demand(query)
    trace = {'policy':'evidence_architecture_062_v1','summary':plan['summary'],'global_history':plan['global'],
             'state':plan['state'],'rule':plan['rule'],'experience':plan['experience'],
             'navigation_rows':0,'private_blocks_omitted':0,'groups':0,'incomplete_groups':0,
             'explicit_version_links':0,'semantic_sufficiency':'unknown'}
    limit=max(0,min(200,limit)); max_items=max(0,min(100,max_items))
    special = plan['summary'] or plan['state'] or plan['rule'] or plan['experience']
    recall_limit=max(0,min(64,recall_limit))
    rows = catalog.search(db,user,build_match_query(plan['focus'],policy='balanced_v1'),recall_limit//2) if special else []
    if plan['summary']:
        rows += history_rows(db,user,seeds,plan,cap=recall_limit-len(rows))
    trace['index_hits']=len(rows)
    sensitive_sources = set()
    source_cache = {}
    for hit in seeds[:limit]:
        if hit.user_id != user:
            continue
        raw = hit.source_raw_id
        if raw not in source_cache:
            source_cache[raw] = db.execute("SELECT sensitive,units_json FROM evidence_062_sources WHERE user_id=? AND raw_id=?",(user,raw)).fetchone()
        info = source_cache[raw]
        if info and info[0]:
            sensitive_sources.add(raw)
        if special or raw in sensitive_sources or (info and info[1] != '[]'):
            lo, hi = (hit.span_start,hit.span_end) if hit.span_start is not None else (0,len(hit.content))
            rows += catalog.source_blocks(db,user,raw,lo=lo,hi=hi,limit=8)
    # Normal searches preserve their base ranking except for sources requiring
    # minimal disclosure. Relevant blocks replace those larger source spans.
    seen=set(); selected=[]; metadata={}
    for row in rows:
        if row['id'] in seen:
            continue
        seen.add(row['id'])
        if not necessary(row,plan):
            trace['private_blocks_omitted']+=1
            continue
        if not special and row['raw_id'] in sensitive_sources and not (terms(row['content']) & terms(plan['focus'])):
            continue
        selected.append(catalog.candidate(row));metadata[row['id']]=row
    # A complete original may stand alone at a one-candidate budget, provided
    # it has no external dependencies or disclosure restriction. A partial
    # table row still needs its dependency group even when the quota is one.
    if limit==1 and not plan['state']:
        for hit in seeds:
            if hit.user_id!=user or hit.source_raw_id in sensitive_sources:
                continue
            raw=db.execute('SELECT content FROM raw_messages WHERE user_id=? AND id=?',
                           (user,hit.source_raw_id)).fetchone()
            external=db.execute("SELECT 1 FROM evidence_062_blocks WHERE user_id=? AND raw_id=? AND ref_key!='' LIMIT 1",
                                (user,hit.source_raw_id)).fetchone()
            if raw and hit.content==raw[0] and not external:
                trace['complete_original_fallback']=True
                return [hit],[],trace
    # Share the cap with original recall. Never erase dependency knowledge
    # merely because the navigation share rounds down to zero.
    nav_cap=min(96,max(1,limit//2)) if limit else 0
    selected=selected[:nav_cap]
    projected_sources={h.source_raw_id for h in selected}
    base=[h for h in seeds if h.user_id==user and h.source_raw_id not in sensitive_sources | projected_sources]
    base=base[:max(0,limit-len(selected))]
    ordered=[]
    if special or any(r['kind'] != 'text' for r in metadata.values()):
        # Interleave recalled originals with complementary units; do not assume
        # a navigation score is a calibrated support probability.
        for i in range(max(len(selected),len(base))):
            if i<len(selected):ordered.append(selected[i])
            if i<len(base):ordered.append(base[i])
    else:
        ordered=base+selected
    registry={h.id:h for h in ordered}; order=list(registry)
    groups=[]; group_budget=min(16,max_items); state_seen=set()
    queue=deque((h.id,0) for h in selected)
    expanded=set();reference_graph={}
    current_depth=0

    def node(row):
        return ('rule',row['rule_key']) if row['rule_key'] else ('block',row['id'])

    def unavailable(seed,reason):
        trace['incomplete_groups']+=1
        groups.append({'seed_id':seed,'member_ids':[seed],'unavailable_dependencies':[reason]})

    def add_group(seed, supports, kind, complete=True):
        supports=list({r['id']:r for r in supports}.values())
        if not complete or not supports or len(supports)>group_budget:
            unavailable(seed,'incomplete_or_item_budget')
            return
        if any(not necessary(r,plan) for r in supports):
            # A prohibited dependency cannot be silently replaced by Raw.
            unavailable(seed,'disclosure')
            trace['private_blocks_omitted']+=1
            return
        hits=[catalog.candidate(r) for r in supports]
        if sum(len(h.content) for h in hits)>min(char_limit,24000) or sum(record_token_cost(h,renderer(h) if renderer else h.content) for h in hits)>min(token_limit,8192):
            unavailable(seed,'group_content_budget')
            return
        extra=[h for h in hits if h.id not in registry]
        if len(registry)+len(extra)>limit:
            unavailable(seed,'candidate_budget')
            return
        for h in extra:
            registry[h.id]=h;order.append(h.id)
        for row in supports:
            metadata[row['id']]=row
            projected_sources.add(row['raw_id'])
            if row['id'] not in expanded:
                depth=current_depth+int(kind=='explicit_scoped_rule_reference' and node(row)!=node(metadata[seed]))
                queue.append((row['id'],depth))
        groups.append({'seed_id':seed,'member_ids':[r['id'] for r in supports], 'unavailable_dependencies':[], 'witness':kind})

    while queue and len(expanded)<96:
        bid,current_depth=queue.popleft()
        if bid in expanded:continue
        expanded.add(bid)
        hit=registry.get(bid)
        if hit is None:continue
        row=metadata[bid]
        if current_depth>4:
            unavailable(bid,'reference_depth_limit');continue
        units=json.loads(row['units_json'])
        for lo,hi,ranges,kind in units:
            if not row['lo']<hi or not row['hi']>lo:
                continue
            support={row['id']:row}; complete=True
            for start,end in ranges:
                found=catalog.source_blocks(db,user,row['raw_id'],lo=start,hi=end,limit=group_budget+1)
                # whitespace between source lines is permitted; no missing text.
                if not found or len(found)>group_budget:
                    complete=False;break
                original=db.execute('SELECT substr(content,?,?) FROM raw_messages WHERE user_id=? AND id=?',
                                    (start+1,end-start,user,row['raw_id'])).fetchone()
                if original is None or len(original[0]) != end-start:
                    complete=False;break
                cursor=start
                for part in found:
                    if part['lo']>cursor and original[0][cursor-start:part['lo']-start].strip():
                        complete=False;break
                    cursor=max(cursor,part['hi'])
                if original[0][max(0,cursor-start):].strip():complete=False
                support.update((r['id'],r) for r in found)
            add_group(hit.id,list(support.values()),kind,complete)
        if row['ref_key']:
            related=db.execute(f"SELECT {catalog.COLS} {catalog.JOIN} WHERE b.user_id=? AND b.rule_key=? AND {catalog.VALID} ORDER BY b.raw_id,b.lo LIMIT ?",(user,row['ref_key'],group_budget+1)).fetchall()
            reference_graph.setdefault(node(row),set()).update(node(r) for r in related)
            add_group(hit.id,[row]+related,'explicit_scoped_rule_reference',bool(related) and len(related)<group_budget)
        if plan['state'] and row['state_key'] and row['state_key'] not in state_seen:
            state_seen.add(row['state_key'])
            related=db.execute(f"SELECT {catalog.COLS} {catalog.JOIN} WHERE b.user_id=? AND b.state_key=? AND {catalog.VALID} ORDER BY b.effective,b.raw_id,b.lo LIMIT ?",(user,row['state_key'],group_budget+1)).fetchall()
            add_group(hit.id,related,'state_history_conflict_sources',len(related)<=group_budget)
            # Version links are evidence navigation only; older facts stay present.
            versions={(r['raw_id'],r['effective']) for r in related if r['explicit_update'] and r['effective']}
            trace['explicit_version_links']+=max(0,len(versions)-1)
    for bid,_ in queue:
        if bid not in expanded:unavailable(bid,'reference_visit_limit')
    cycle_cache={};visiting=set()
    def cyclic(raw):
        if raw in visiting:return True
        if raw in cycle_cache:return cycle_cache[raw]
        visiting.add(raw)
        result=any(cyclic(other) for other in reference_graph.get(raw,()))
        visiting.remove(raw);cycle_cache[raw]=result
        return result
    for bid,row in metadata.items():
        if bid in registry and cyclic(node(row)):unavailable(bid,'cyclic_reference')
    trace['navigation_rows']=len(selected);trace['groups']=len(groups)
    trace['expanded_blocks']=len(expanded)
    trace['history_coverage']='bounded_source_and_block_samples' if plan['summary'] else 'not_requested'
    return [registry[k] for k in order if k in registry and (registry[k].representation=='evidence_block_v1'
        or registry[k].source_raw_id not in projected_sources)][:limit],groups,trace


def cover_order(candidates, query):
    """Round-robin actual session/source groups only for explicit summaries."""
    if not demand(query)['summary']:
        return candidates
    buckets={}
    for hit in candidates:
        buckets.setdefault((hit.session_id,hit.source_raw_id),[]).append(hit)
    result=[]
    for i in range(max((len(v) for v in buckets.values()),default=0)):
        for values in buckets.values():
            if i<len(values):result.append(values[i])
    return result

"""Budgeted coverage navigation over an already gated, fixed candidate pool.

Lexical coverage is a selection heuristic, never a sufficiency certificate.
Overlapping dependency groups are atomic; original spans remain intact.
"""
import re
from app.segments import token_count

_STOP = frozenset('what which who when where why how is are was were a an the do does did of to and or for in on at my i me should would could you your it that this from with'.split())


def query_terms(query):
    terms = set(re.findall(r'[a-z0-9]+', query.lower())) - _STOP
    for block in re.findall(r'[\u4e00-\u9fff]+', query):
        terms.update(block[i:i+2] for i in range(len(block)-1))
    return frozenset(sorted(terms)[:128])


def select_evidence(candidates, query, max_items, char_limit, token_limit=None, renderer=None, groups=()):
    if len(candidates) > 200 or len({h.id for h in candidates}) != len(candidates):
        raise ValueError('invalid_set_selection_pool')
    registry = {h.id:h for h in candidates}
    edges, unavailable = {}, set()
    for group in groups:
        ids = set([group['seed_id'], *group['member_ids']])
        for sid in ids:
            edges.setdefault(sid, set()).update(ids)
        if group.get('unavailable_dependencies'):
            unavailable.update(ids)
    units, processed, omitted = [], set(), []
    wanted = query_terms(query)
    for rank, hit in enumerate(candidates):
        if hit.id in processed:
            continue
        component, pending = {hit.id}, [hit.id]
        while pending:
            for sid in edges.get(pending.pop(), ()):
                if sid not in component:
                    component.add(sid); pending.append(sid)
        processed.update(component)
        if not component <= registry.keys() or component & unavailable:
            omitted.append({'member_ids':sorted(component), 'reason':'missing_dependency'})
            continue
        members = [h for h in candidates if h.id in component]
        texts = [renderer(h) if renderer else h.content for h in members]
        coverage = set().union(*(query_terms(h.content) & wanted for h in members))
        units.append({'ids':component, 'members':members, 'chars':sum(map(len,texts)),
                      'tokens':sum(token_count(t) for t in texts), 'coverage':coverage, 'rank':rank,
                      'sources':{h.source_raw_id for h in members}})
    selected, covered, sources, chars, tokens, count = set(), set(), set(), 0, 0, 0
    # Rank is the existing calibrated retrieval signal. Protect its first half
    # before a bounded coverage preference can consume the remaining budget.
    protected=[]
    for unit in units:
        if (count+len(unit['members']) > max_items//2 or chars+unit['chars'] > char_limit//2
                or (token_limit is not None and tokens+unit['tokens'] > token_limit//2)):
            break
        selected.update(unit['ids']);covered.update(unit['coverage']);sources.update(unit['sources'])
        chars+=unit['chars'];tokens+=unit['tokens'];count+=len(unit['members']);protected.append(unit)
    protected_ids=[h.id for h in candidates if h.id in selected]
    units=[u for u in units if u not in protected]
    while units:
        feasible = [u for u in units if count+len(u['members']) <= max_items and
                    chars+u['chars'] <= char_limit and (token_limit is None or tokens+u['tokens'] <= token_limit)]
        if not feasible:
            break
        def priority(u):
            bonus=.25*len(u['coverage']-covered)/max(1,len(wanted))
            utility=(1+bonus)/(u['rank']+1)
            return (-utility, u['rank'], sorted(u['ids']))
        unit = min(feasible, key=priority)
        units.remove(unit)
        selected.update(unit['ids']); covered.update(unit['coverage']); sources.update(unit['sources'])
        chars += unit['chars']; tokens += unit['tokens']; count += len(unit['members'])
    for unit in units:
        omitted.append({'member_ids':sorted(unit['ids']), 'reason':'group_over_budget'})
    kept = [h for h in candidates if h.id in selected]
    return kept, chars, {'policy':'protected_rank_bounded_coverage_atomic_set_v2', 'semantic_sufficiency':False,
                         'protected_prefix_ids':protected_ids,'coverage_relative_bonus_cap':.25,
                         'pool_size':len(candidates), 'query_terms':len(wanted), 'covered_terms':len(covered),
                         'packed_content_tokens':tokens, 'omitted_groups':omitted}

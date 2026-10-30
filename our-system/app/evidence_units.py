"""Deterministic, existing-span-only structural recall (061).

Call in the caller's SQLite read snapshot, after policy fusion. Return
``(candidates, groups, trace)``; groups are directed seed -> required_ids.
member_ids also supplies the legacy schema, but legacy undirected pack_groups
must NOT be used for many rows sharing a schema: use the parent's pack_units.
Only Markdown pipe tables and explicitly headed short rule sections qualify.
No synthetic spans, metadata inference, cross-source navigation or LLM calls.
Already-recalled legal complete Raw candidates are retained but not expanded.

Pass actual packing budgets via group_*_limit (including renderer overhead via
renderer). These are per *directed dependency closure*, not a reservation against other
packed evidence: final packing and coordinator.final_check remain mandatory.
An unavailable/over-budget dependency never makes the independent seed hard.
Baseline spans from oversized sources are freshness-checked against saved rows
but not rehashed here; final_check must validate them. Expansion skips such raws.
"""
from __future__ import annotations

import re

from app import segments
from app.evidence_budget import record_token_cost

RAW_CHAR_LIMIT = 128_000
SOURCE_SPAN_LIMIT = 512
MAX_GROUP_ITEMS = 8
MAX_GROUP_CHARS = 12_000
MAX_GROUP_TOKENS = 4096
LOGICAL_REF_LIMIT = 12_000
_SAFE = ("p.visibility='visible' AND p.disclosure_policy='within_user' "
         "AND p.retraction_state='not_retracted' AND m.privacy_level='normal'")
_HEADING = re.compile(r"^\s*(?:#{1,6}\s+\S|(?:Rules?|Policy|Procedure|规则|规定|流程|章程)\s*[:：]|第[^\s]{1,12}[章节条])", re.I)
_UNIT = re.compile(r"^\s*(?:(?:单位|计量单位|金额单位|Units?)\s*[:：]\s*\S|All (?:amounts|values) (?:are )?in\s+\S)", re.I)
_RULE = re.compile(r"\b(?:must|shall|required|rules?|prohibited)\b|必须|应当|禁止|规则|规定", re.I)
_EXCEPTION = re.compile(r"\b(?:except|exception|unless|prerequisite|precondition|only if)\b|例外|除非|前提|仅当|除.*外", re.I)


def _cells(line):
    # Escaped pipes / code are ambiguous: conservatively decline them.
    if "\\|" in line or "`" in line or "|" not in line:
        return []
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _structures(text):
    """Offset-only witnesses; newline boundaries, never reconstructed content."""
    lines = []
    offset = 0
    fence = None
    fenced_offsets = []
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            fenced_offsets.append(offset)
            lines.append((offset, offset + len(line), ""))
        else:
            if fence is not None:
                fenced_offsets.append(offset)
            lines.append((offset, offset + len(line), "" if fence is not None else line.rstrip("\r\n")))
        offset += len(line)
    tables = []
    i = 1
    while i < len(lines):
        cells = _cells(lines[i][2])
        header = _cells(lines[i - 1][2])
        if len(cells) < 2 or len(header) != len(cells) or not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            i += 1
            continue
        end = i + 1
        while end < len(lines) and len(_cells(lines[end][2])) == len(cells):
            # Another header separator ends this table, even without a blank.
            if all(re.fullmatch(r":?-{3,}:?", c) for c in _cells(lines[end][2])):
                end -= 1
                break
            end += 1
        if end > i + 1:
            support = [(lines[i - 1][0], lines[i][1])]
            if i >= 2 and _UNIT.match(lines[i - 2][2]):
                support.append((lines[i - 2][0], lines[i - 2][1]))
            next_separator = _cells(lines[end + 2][2]) if end + 2 < len(lines) else []
            next_header = _cells(lines[end + 1][2]) if end + 1 < len(lines) else []
            unit_starts_next_table = (len(next_separator) >= 2 and len(next_header) == len(next_separator)
                                     and all(re.fullmatch(r":?-{3,}:?", c) for c in next_separator))
            if end < len(lines) and _UNIT.match(lines[end][2]) and not unit_starts_next_table:
                support.append((lines[end][0], lines[end][1]))
            tables.append((lines[i + 1][0], lines[end - 1][1], support, "table_schema"))
        i = max(i + 1, end)
    headings = [i for i, line in enumerate(lines) if _HEADING.match(line[2])]
    rules = []
    for index, start in enumerate(headings):
        end = headings[index + 1] if index + 1 < len(headings) else len(lines)
        lo, hi = lines[start][0], lines[end - 1][1]
        body = text[lo:hi]
        if (hi - lo > 7000 or not _EXCEPTION.search(body) or not _RULE.search(body)
                or any(lo <= pos < hi for pos in fenced_offsets)):
            continue
        if any(lo < t[1] and hi > t[0] for t in tables):
            continue
        rules.append((lo, hi, [(lo, hi)], "rule_section"))
    return tables + rules


def _cover(rows, intervals):
    """Greedy exact interval cover; gaps mean unavailable, not partial success."""
    selected = {}
    for lo, hi in intervals:
        cursor = lo
        while cursor < hi:
            eligible = [r for r in rows if r["span_start"] <= cursor < r["span_end"]]
            if not eligible:
                return None
            row = min(eligible, key=lambda r: (-r["span_end"], r["span_start"], r["id"]))
            selected[row["id"]] = row
            cursor = row["span_end"]
            if len(selected) > MAX_GROUP_ITEMS:
                return None
    return list(selected.values())


def expand_units(db, user_id, seeds, *, limit=200, seed_limit=20, max_extra=50,
                 group_item_limit=8, group_char_limit=12000,
                 group_token_limit=4096, renderer=None):
    """Protect first min(100, limit) base slots; replace only unprotected tails.

    Limits above 200/20/50 are clamped. Nonpositive limits disable that budget.
    Added candidates retain score=0 and all canonical provenance metadata.
    ``required_ids`` excludes the seed; ``optional_ids`` is currently empty.
    ``member_ids`` includes seed and all required support. No recursive recall.
    """
    limit = max(0, min(200, int(limit)))
    seed_limit = max(0, min(20, int(seed_limit)))
    max_extra = max(0, min(50, int(max_extra)))
    item_cap = max(0, min(MAX_GROUP_ITEMS, int(group_item_limit)))
    char_cap = max(0, min(MAX_GROUP_CHARS, int(group_char_limit)))
    token_cap = max(0, min(MAX_GROUP_TOKENS, int(group_token_limit)))
    trace = {"policy": "structural_units_061_v1", "expanded_spans": 0,
             "evicted_ids": [], "rejected_ids": [], "skipped": [],
             "source_ref_visits": 0, "scanned_chars": 0,
             "logical_ref_limit": LOGICAL_REF_LIMIT,
             "raw_char_limit": RAW_CHAR_LIMIT, "source_span_limit": SOURCE_SPAN_LIMIT,
             "visit_scope": "logical row references; not SQLite VM work",
             "optional_navigation": [], "oversized_base_ids": []}
    sources, saved, structures, costs = {}, {}, {}, {}

    def source(raw_id):
        if raw_id not in sources:
            # substr bounds materialized text even for arbitrarily large raws.
            sources[raw_id] = db.execute(f"""SELECT m.id,m.user_id,m.role,m.session_id,
                m.message_index,m.source_timestamp,m.recorded_at,m.deletion_state,
                substr(m.content,1,?) AS content,length(m.content) AS raw_length,
                p.source_version,p.content_sha256
                FROM raw_messages m JOIN source_policies p
                ON p.raw_id=m.id AND p.user_id=m.user_id
                WHERE m.user_id=? AND m.id=? AND m.deletion_state='active' AND {_SAFE}""",
                (RAW_CHAR_LIMIT, user_id, raw_id)).fetchone()
            trace["source_ref_visits"] += 1
        return sources[raw_id]

    def legal(hit):
        if hit.user_id != user_id:
            return False
        src = source(hit.source_raw_id)
        if src is None:
            return False
        if hit.representation == 'raw':
            # Existing parent/sentence routes may return a complete Raw. Check
            # its actual DB text as well as policy/hash/version/metadata, then
            # retain it without attempting span-only structural expansion.
            from app.evidence_navigation import raw_candidate
            if (hit.id != hit.source_raw_id or len(hit.content) != src['raw_length']
                    or segments.digest(hit.content) != src['content_sha256']):
                return False
            full = dict(src, content=hit.content)
            if raw_candidate(full, hit.score) != hit:
                return False
            trace['source_ref_visits'] += 1
            return db.execute('SELECT 1 FROM raw_messages WHERE user_id=? AND id=? AND content=?',
                              (user_id, hit.id, hit.content)).fetchone() is not None
        if hit.representation != 'span_exact_v1':
            return False
        row = db.execute(f"SELECT {segments._COLS} {segments._JOIN} "
                         f"WHERE s.user_id=? AND s.id=? AND {segments._VALID} AND {_SAFE}",
                         (user_id, hit.id)).fetchone()
        trace["source_ref_visits"] += 1
        if row is None or segments.candidate(row, hit.score) != hit:
            return False
        if src["raw_length"] > RAW_CHAR_LIMIT:
            trace["oversized_base_ids"].append(hit.id)
            return True  # Existing base only; caller must still final_check.
        trace["source_ref_visits"] += 1  # valid_candidate's row reference
        return segments.valid_candidate(db, hit, src)

    result = []
    seen = set()
    for hit in seeds[:limit]:
        if hit.id in seen:
            continue
        if not legal(hit):
            trace["rejected_ids"].append(hit.id)
            continue
        result.append(hit)
        seen.add(hit.id)
    base_ids = set(seen)
    protected = {h.id for h in result[:min(100, limit)]}
    # Every structural seed remains protected even if a smaller prefix is used.
    selected_seeds = result[:seed_limit]
    protected.update(h.id for h in selected_seeds)
    groups = []
    for seed in selected_seeds:
        if seed.representation != 'span_exact_v1':
            continue
        src = source(seed.source_raw_id)
        if src["raw_length"] > RAW_CHAR_LIMIT:
            trace["skipped"].append({"seed_id": seed.id, "reason": "raw_scan_budget"})
            continue
        if seed.source_raw_id not in structures:
            structures[seed.source_raw_id] = _structures(src["content"])
            trace["scanned_chars"] += len(src["content"])
        matches = [unit for unit in structures[seed.source_raw_id]
                   if seed.span_end > unit[0] and seed.span_start < unit[1]]
        if len(matches) > 1:
            trace["skipped"].append({"seed_id": seed.id, "reason": "ambiguous_structure"})
            continue
        for lo, hi, intervals, kind in matches:
            if kind == "rule_section" and not _RULE.search(src["content"][max(lo, seed.span_start):min(hi, seed.span_end)]):
                continue
            if seed.source_raw_id not in saved:
                if trace["source_ref_visits"] + SOURCE_SPAN_LIMIT + 1 > LOGICAL_REF_LIMIT:
                    trace["skipped"].append({"seed_id": seed.id, "reason": "visit_budget"})
                    continue
                saved[seed.source_raw_id] = db.execute(
                    f"SELECT {segments._COLS} {segments._JOIN} WHERE s.user_id=? "
                    f"AND s.source_raw_id=? AND {segments._VALID} AND {_SAFE} "
                    "ORDER BY s.span_start,s.id LIMIT ?",
                    (user_id, seed.source_raw_id, SOURCE_SPAN_LIMIT + 1)).fetchall()
                trace["source_ref_visits"] += len(saved[seed.source_raw_id])
            rows = saved[seed.source_raw_id]
            dependencies = _cover(rows, intervals) if len(rows) <= SOURCE_SPAN_LIMIT else None
            if dependencies is None:
                trace["skipped"].append({"seed_id": seed.id, "reason": "unavailable_dependency"})
                continue
            support = [segments.candidate(row, 0.) for row in dependencies if row["id"] != seed.id]
            if trace["source_ref_visits"] + len(support) > LOGICAL_REF_LIMIT:
                trace["skipped"].append({"seed_id": seed.id, "reason": "visit_budget"})
                continue
            trace["source_ref_visits"] += len(support)
            if not all(segments.valid_candidate(db, h, src) for h in support):
                trace["skipped"].append({"seed_id": seed.id, "reason": "invalid_dependency"})
                continue
            members = {seed.id, *(h.id for h in support)}
            if len(members) == 1:
                continue
            # Recalled members keep their measured scores, including the small
            # serialization cost difference from navigation score=0.
            registry = {h.id: h for h in support + result}
            # Directed edges only. A shared header never acquires reverse edges
            # to every row. Recheck all roots because new edges can enlarge an
            # existing ancestor's closure (e.g. overlapping short rule spans).
            edges = {}
            for group in groups:
                edges.setdefault(group["seed_id"], set()).update(group["required_ids"])
            edges.setdefault(seed.id, set()).update(members - {seed.id})
            over_budget = False
            for root in edges:
                closure, pending = {root}, [root]
                while pending:
                    for dependency in edges.get(pending.pop(), ()):
                        if dependency not in closure:
                            closure.add(dependency)
                            pending.append(dependency)
                totals = []
                for id in closure:
                    hit = registry[id]
                    key = (id, hit.score)
                    if key not in costs:
                        text = renderer(hit) if renderer else hit.content
                        costs[key] = (len(text), record_token_cost(hit, text))
                    totals.append(costs[key])
                if (len(closure) > item_cap or sum(t[0] for t in totals) > char_cap or
                        sum(t[1] for t in totals) > token_cap):
                    over_budget = True
                    break
            if over_budget:
                trace["skipped"].append({"seed_id": seed.id, "reason": "group_budget"})
                continue
            new = [h for h in support if h.id not in seen]
            need = max(0, len(result) + len(new) - limit)
            victims = [h.id for h in reversed(result) if h.id in base_ids and h.id not in protected and h.id not in members]
            if trace["expanded_spans"] + len(new) > max_extra or len(victims) < need:
                trace["skipped"].append({"seed_id": seed.id, "reason": "candidate_budget"})
                continue
            evicted = set(victims[:need])
            result = [h for h in result if h.id not in evicted] + new
            seen.difference_update(evicted)
            seen.update(h.id for h in new)
            trace["evicted_ids"].extend(victims[:need])
            trace["expanded_spans"] += len(new)
            protected.update(members)
            ordered = [seed.id] + [h.id for h in support]
            groups.append({"seed_id": seed.id, "member_ids": ordered,
                           "required_ids": ordered[1:], "optional_ids": [],
                           "unavailable_dependencies": [], "witness": kind,
                           "source_raw_id": seed.source_raw_id,
                           "source_version": seed.source_version,
                           "support_intervals": intervals, "semantic_entailment": False})
    trace["returned_candidates"] = len(result)
    trace["hard_groups"] = len(groups)
    return result, groups, trace

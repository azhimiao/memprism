"""Pure, bounded lexical navigation over caller-gated exact source spans.

build_views(candidates, user_id) -> (JSON-friendly views, trace).
rank_views(candidates, views, intents) -> (the same candidate objects, trace).

These functions do not establish source validity, semantic field support or
question sufficiency. The caller must gate sources/permissions before calling
them and revalidate lifecycle/epoch before publishing whole original spans.
No retrieval, persistence, model calls, wall clock or generated prose is used.
Offsets, like segments.py, count Python Unicode code points, not UTF-16 units.
Only span_exact_v1 candidates are navigable; raw/derived rows stay unpromoted.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date
import hashlib
from itertools import islice
import json
import re

POLICY = "source_bound_lexical_views_v1"
MAX_CANDIDATES = 200
MAX_SPAN_CHARS = 1800
MAX_FIELDS = 13
MAX_RANGES_PER_FIELD = 4
KINDS = ("claim", "event", "preference", "constraint", "procedure", "task")
MAX_VIEWS = MAX_CANDIDATES * len(KINDS)
MAX_INTENTS = 16
TYPE_BOOST_FRACTION = 0.05
MAX_TYPE_BOOST = TYPE_BOOST_FRACTION / 61


def _rx(pattern):
    return re.compile(pattern, re.IGNORECASE)


# Signals refer to lexical surfaces, never semantic facts or instructions to run.
_NEGATION = _rx(
    r"\b(?:not|never|no|neither|without|cannot)\b|"
    r"\b(?:don|doesn|didn|isn|aren|wasn|weren|won|can|couldn|wouldn|shouldn|haven|hasn|hadn)['’]t\b|"
    r"不|没|未|无|非|禁止|不得|别"
)
_CONDITION = _rx(
    r"\b(?:if|unless|provided(?:\s+that)?|only\s+when|only\s+if|in\s+case|when|whenever)\b|"
    r"如果|假如|除非|只要|只有|若|当|的话|前提|条件"
)
_PLAN = _rx(
    r"\b(?:will|shall|going\s+to|plans?(?:\s+to)?|planning(?:\s+to)?|intend\w*|"
    r"scheduled(?:\s+to)?|tomorrow|next\s+(?:week|month|year))\b|计划|打算|准备|将|拟|明天|下周|下个月|明年"
)
_PAST = _rx(
    r"\b(?:visited|moved|joined|started|finished|attended|arrived|left|completed|"
    r"occurred|happened|was\s+born|has\s+been|have\s+been)\b|"
    r"参加了|搬到|搬去了|开始了|结束了|完成了|发生了|已经|曾经|去了|到达了"
)
_EXCEPTION = _rx(r"\b(?:unless|except(?:\s+when|\s+if|\s+for)?|otherwise|but)\b|除非|例外|除外|除了|否则|但是|但")
_STEP = _rx(r"\b(?:step\s*\d{1,3}|first|then|finally|next)\b|步骤\s*[一二三四五六七八九十\d]{1,3}|首先|然后|最后|接着|(?:^|\n)\s*\d{1,3}[.)、]")
_RELATIVE = _rx(
    r"\b(?:today|yesterday|tomorrow|(?:last|next|this)\s+(?:day|week|month|year)|"
    r"\d{1,4}\s+(?:days?|weeks?|months?|years?)\s+ago)\b|"
    r"今天|昨天|明天|前天|后天|上周|下周|本周|上个月|下个月|去年|今年|明年|\d{1,4}[天周月年]前"
)
_ABSOLUTE = _rx(r"(?<!\d)(?:\d{4}-\d{1,2}-\d{1,2}|\d{4}年\d{1,2}月\d{1,2}(?:日|号)?)(?!\d)")
_MODAL = _rx(r"\b(?:may|might|could|would|should|maybe|perhaps|hope\w*|wish\w*)\b|可能|也许|应该|希望|想要")
_REPORTED = _rx(r"\b(?:said|says|told|according\s+to|suggest\w*|recommend\w*|think\w*)\b|说|据说|听说|建议|推荐|认为|[\"“”「」]")
_QUESTION = _rx(r"[?？]")
_DEONTIC = _rx(r"\b(?:must|shall|should|rules?|required|prohibited)\b|规则|必须|应当|应该|禁止|不得")
_TYPE_PATTERNS = {
    "event": _rx(_PAST.pattern + "|" + _PLAN.pattern),
    "preference": _rx(r"\b(?:prefer\w*|favo[ur]+ite|likes?|liked|dislike\w*)\b|喜欢|偏好|讨厌|倾向"),
    "constraint": _rx(r"\b(?:must|shall|should|rule\w*|required|prohibited|never|cannot|avoid|allergic|budget|do\s+not)\b|规则|必须|应当|应该|禁止|不得|不能|过敏|禁忌|预算|不要"),
    "procedure": _rx(_STEP.pattern + r"|\b(?:procedure|workflow|execute)\b|流程|执行"),
    "task": _rx(r"\b(?:task\w*|todo|to-do|deadline|due|pending|completed)\b|任务|截止|待办|完成"),
}
_QUALIFIERS = {
    "negation": _NEGATION, "condition": _CONDITION, "plan": _PLAN,
    "exception": _EXCEPTION, "step": _STEP, "absolute_date": _ABSOLUTE,
    "relative_time": _RELATIVE, "modality": _MODAL, "reported_speech": _REPORTED,
    "question": _QUESTION,
}
# An initial explicit grammatical surface only; no pronoun/entity resolution.
# Complex/quoted/multiple sentences deliberately leave the subject unknown.
_SUBJECT_EN = re.compile(
    r"^\s*(?P<subject>I|[Ww]e|[Yy]ou|[Hh]e|[Ss]he|[Tt]hey|"
    r"(?:[Mm]y|[Oo]ur|[Hh]is|[Hh]er|[Tt]heir)\s+(?:friend|colleague|partner|sister|brother|mother|father)|"
    r"[A-Z][a-z]{1,31}(?:\s+[A-Z][a-z]{1,31}){0,2})\s+"
    r"(?=(?i:am|is|are|was|were|do|does|did|don['’]t|doesn['’]t|didn['’]t|"
    r"have|has|had|will|shall|must|may|might|can|cannot|can['’]t|could|should|would|"
    r"prefer\w*|like\w*|dislike\w*|plan\w*|intend\w*|live\w*|work\w*|visit\w*|"
    r"move\w*|join\w*|start\w*|finish\w*|attend\w*|arriv\w*|left|complete\w*)\b)"
)
_SUBJECT_ZH = re.compile(
    r"^\s*(?P<subject>(?:我|我们|他|她|他们|她们|你|你们)(?:的)?"
    r"(?:朋友|同事|伴侣|姐姐|妹妹|哥哥|弟弟|母亲|父亲)|我们|他们|她们|你们|我|他|她|你)"
    r"(?=不|没|未|喜欢|偏好|讨厌|计划|打算|准备|将|会|想|必须|应该|不能|已经|曾经|在|住|参加|搬|开始|完成|去了)"
)
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _get(candidate, name, default=None):
    return candidate.get(name, default) if isinstance(candidate, Mapping) else getattr(candidate, name, default)


def _bounded(items, cap, name):
    result = list(islice(iter(items), cap + 1))
    if len(result) > cap:
        raise ValueError(f"{name} exceeds {cap}")
    return result


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**53 - 1


def _identifier(value):
    return isinstance(value, str) and 0 < len(value) <= 512 and bool(value.strip())


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_problem(candidate, user_id):
    if _get(candidate, "user_id") != user_id:
        return "foreign_user"
    if _get(candidate, "deletion_state") != "active":
        return "inactive_candidate"
    if _get(candidate, "representation") != "span_exact_v1":
        return "not_exact_span"
    if not all(_identifier(_get(candidate, key)) for key in ("id", "source_raw_id", "user_id", "session_id")):
        return "missing_source_identity"
    content = _get(candidate, "content")
    if not isinstance(content, str) or len(content) > MAX_SPAN_CHARS or not content.strip():
        return "span_content_limit_or_empty"
    start, end = _get(candidate, "span_start"), _get(candidate, "span_end")
    if not _integer(start) or not _integer(end, 1) or end - start != len(content):
        return "invalid_offsets"
    if not _integer(_get(candidate, "source_version"), 1):
        return "invalid_version"
    if not all(isinstance(_get(candidate, key), str) and _HASH.fullmatch(_get(candidate, key))
               for key in ("source_sha256", "span_sha256", "fragment_hash")):
        return "invalid_hash_metadata"
    if _get(candidate, "role") not in ("user", "assistant") or not _integer(_get(candidate, "message_index")):
        return "invalid_source_metadata"
    stamp, arrival = _get(candidate, "source_timestamp"), _get(candidate, "recorded_at")
    if (stamp is not None and not _integer(stamp)) or not _integer(arrival):
        return "invalid_source_metadata"
    if type(_get(candidate, "created_at")) is not int or _get(candidate, "created_at") != (stamp if stamp is not None else arrival):
        return "inconsistent_source_time_metadata"
    try:
        if _digest(content) != _get(candidate, "span_sha256"):
            return "span_hash_mismatch"
        identity = [_get(candidate, key) for key in (
            "user_id", "source_raw_id", "source_version", "source_sha256",
            "span_start", "span_end", "span_sha256", "fragment_hash")]
        canonical_id = "span_" + _digest(json.dumps(identity, ensure_ascii=False, separators=(",", ":")))
        if canonical_id != _get(candidate, "id"):
            return "span_identity_mismatch"
    except UnicodeEncodeError:
        return "invalid_unicode"
    # Cannot verify the raw body, current version, permissions, fragment policy,
    # token limit or truthful source metadata without the caller's source gate.
    return None


def _source(candidate):
    return {
        "span_id": _get(candidate, "id"), "raw_id": _get(candidate, "source_raw_id"),
        "user_id": _get(candidate, "user_id"), "version": _get(candidate, "source_version"),
        "source_sha256": _get(candidate, "source_sha256"), "span_sha256": _get(candidate, "span_sha256"),
        "fragment_hash": _get(candidate, "fragment_hash"),
        "absolute_start": _get(candidate, "span_start"), "absolute_end": _get(candidate, "span_end"),
        "offset_unit": "unicode_codepoint",
        **{key: _get(candidate, key) for key in (
            "role", "session_id", "message_index", "source_timestamp", "recorded_at", "created_at")},
    }


def _matches(pattern, content):
    found = list(islice(pattern.finditer(content), MAX_RANGES_PER_FIELD + 1))
    return [(m.start(), m.end()) for m in found[:MAX_RANGES_PER_FIELD]], len(found) > MAX_RANGES_PER_FIELD


def _field(content, ranges, base, truncated=False):
    return {
        "values": [content[start:end] for start, end in ranges],
        "ranges": [{"start": start, "end": end, "absolute_start": base + start,
                    "absolute_end": base + end} for start, end in ranges],
        "lexical_status": "exact_quote" if ranges else "unknown",
        "field_supported": "unknown", "truncated": truncated,
    }


def _subject(content):
    if _REPORTED.search(content) or re.search(r"[.!?。！？;；\n\r].*\S", content.rstrip(".!?。！？ \t\r\n")):
        return []
    match = _SUBJECT_ZH.match(content) or _SUBJECT_EN.match(content)
    if not match or match["subject"].casefold() in {"if", "unless", "when", "then", "first", "finally", "rule", "step"}:
        return []
    return [match.span("subject")]


def _event_status(content, signals, boundary):
    past, plan = bool(_PAST.search(content)), bool(signals["plan"][0])
    if boundary == "hard_split_requires_context" or signals["reported_speech"][0] or signals["question"][0] or signals["modality"][0]:
        return "unknown"
    if signals["condition"][0]:
        return "conditional_unknown"
    if signals["negation"][0]:
        return "negated_unknown"
    if signals["exception"][0]:
        return "qualified_unknown"
    if plan and past:
        return "mixed_unknown"
    if _DEONTIC.search(content) or not _subject(content):
        return "unknown"
    if plan:
        return "planned"
    return "occurrence_cue" if past else "unknown"


def _dates(content, ranges):
    result = []
    for index, (start, end) in enumerate(ranges):
        try:
            normalized = date(*(int(part) for part in re.findall(r"\d+", content[start:end]))).isoformat()
        except ValueError:
            normalized = None
        result.append({"range_index": index, "iso_date": normalized,
                       "precision": "day" if normalized else "unknown"})
    return result


def build_views(candidates, user_id):
    """Return bounded, source-bound navigation; support remains unknown.

    Accepts EvidenceCandidate objects or mappings with the same source fields.
    Raises ValueError on an invalid user ID or >200 candidates, rather than
    silently truncating the candidate set. Bad individual candidates get no
    views and a content-free reason in trace. No raw/summary fallback exists.
    Every field value is an exact source quote; classifications are only hints.
    Relative dates remain unresolved, even with a source timestamp (no timezone
    or linguistic-granularity inference). Arrival/created_at are never anchors.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty identifier")
    candidates = _bounded(candidates, MAX_CANDIDATES, "candidates")
    if len(user_id) > 512:
        return [], {"policy": POLICY, "candidate_count":len(candidates), "view_count":0,
                    "source_valid":"unknown", "requires_caller_gate":True,
                    "reason":"identifier_outside_navigation_limit", "decisions":[],
                    "limits":{"candidates":MAX_CANDIDATES,"views":MAX_VIEWS,
                              "fields_per_view":MAX_FIELDS,"ranges_per_field":MAX_RANGES_PER_FIELD}}
    views, decisions = [], []
    for index, candidate in enumerate(candidates):
        problem = _candidate_problem(candidate, user_id)
        if problem:
            decisions.append({"candidate_index": index, "decision": "skip", "reason": problem})
            continue
        content, base = _get(candidate, "content"), _get(candidate, "span_start")
        source = _source(candidate)
        signals = {name: _matches(pattern, content) for name, pattern in _QUALIFIERS.items()}
        cues = {kind: _matches(pattern, content) for kind, pattern in _TYPE_PATTERNS.items()}
        kinds = [kind for kind in KINDS if kind in cues and cues[kind][0]] or ["claim"]
        for kind in kinds:
            kind_ranges, kind_clipped = cues.get(kind, ([], False))
            fields = {"statement": _field(content, [(0, len(content))], base),
                      "subject": _field(content, _subject(content), base),
                      "kind_cue": _field(content, kind_ranges, base, kind_clipped)}
            fields.update({name: _field(content, ranges, base, clipped)
                           for name, (ranges, clipped) in signals.items()})
            views.append({
                "policy": POLICY, "candidate_index": index, "kind": kind,
                "source": dict(source), "fields": fields,
                "navigation": {
                    "event_status": _event_status(content, signals, _get(candidate, "boundary_status")) if kind == "event" else "not_applicable",
                    "absolute_dates": _dates(content, signals["absolute_date"][0]),
                    "relative_time_status": ("unresolved_no_source_timestamp" if source["source_timestamp"] is None
                                             else "unresolved_timezone_or_granularity") if signals["relative_time"][0] else "absent",
                    "resolved_relative_date": None,
                    "context_status": "requires_context" if _get(candidate, "boundary_status") == "hard_split_requires_context" else "unknown",
                    "update_policy": "coexist_no_supersede",
                },
                "support": {"source_valid": "unknown", "field_supported": "unknown",
                            "set_sufficient": "unknown", "basis": "lexical_navigation",
                            "source_validation_owner": "caller_gate"},
            })
        decisions.append({"candidate_index": index, "decision": "navigate", "kinds": kinds})
    return views, {"policy": POLICY, "candidate_count": len(candidates), "view_count": len(views),
                   "source_valid": "unknown", "requires_caller_gate": True, "decisions": decisions,
                   "limits": {"candidates": MAX_CANDIDATES, "views": MAX_VIEWS,
                              "fields_per_view": MAX_FIELDS, "ranges_per_field": MAX_RANGES_PER_FIELD}}


def rank_views(candidates, views, intents):
    """Small, non-stacking type boost; preserve candidate identity/multiplicity.

    Recognized intents: personalization, procedure, temporal, the six kinds,
    event_occurred and event_planned. Temporal is status-neutral navigation;
    event_occurred only uses occurrence_cue, event_planned only planned; these
    specific status filters take precedence over generic temporal/event. Neither
    implies an event happened. Unknown intents do nothing. RRF-style rank scores
    here are ordering numbers, never support probabilities or source scores.

    Full canonical view equality prevents stale/foreign/tampered fields being
    used as ranking authority. Indices bind to the same input ordering used by
    build_views. Mixed-user inputs get no boost and must still be caller-gated.
    """
    candidates = _bounded(candidates, MAX_CANDIDATES, "candidates")
    supplied = _bounded(views, MAX_VIEWS, "views")
    names = [intents] if isinstance(intents, str) else _bounded(intents, MAX_INTENTS, "intents")
    if any(not isinstance(name, str) or len(name) > 64 for name in names):
        raise ValueError("intents must be bounded strings")
    names = set(names)
    wanted = names.intersection(KINDS)
    for name, kinds in {"personalization": {"constraint", "preference"},
                        "procedure": {"constraint", "procedure", "task"}, "temporal": {"event"},
                        "event_occurred": {"event"}, "event_planned": {"event"}}.items():
        if name in names:
            wanted.update(kinds)
    users = {_get(candidate, "user_id") for candidate in candidates if _identifier(_get(candidate, "user_id"))}
    single_user = len(users) == 1 and all(_identifier(_get(candidate, "user_id")) and
                                         _get(candidate, "user_id") in users for candidate in candidates)
    canonical, _ = build_views(candidates, next(iter(users))) if single_user else ([], {})
    expected = {(view["candidate_index"], view["kind"]): view for view in canonical}
    accepted, rejected, duplicates = {}, 0, 0
    for view in supplied:
        if not isinstance(view, dict) or type(view.get("candidate_index")) is not int or not isinstance(view.get("kind"), str):
            rejected += 1
            continue
        key = (view["candidate_index"], view["kind"])
        if key not in expected or view != expected[key]:
            rejected += 1
        elif key in accepted:
            duplicates += 1
        else:
            accepted[key] = view
    by_candidate = {}
    for (index, kind), view in accepted.items():
        if kind not in wanted:
            continue
        status = view["navigation"]["event_status"]
        if kind == "event" and names.intersection({"event_occurred", "event_planned"}):
            if not (("event_occurred" in names and status == "occurrence_cue") or
                    ("event_planned" in names and status == "planned")):
                continue
        by_candidate.setdefault(index, []).append((kind, status))
    scored, rows = [], []
    for index, candidate in enumerate(candidates):
        matches = sorted(by_candidate.get(index, []))
        base = 1 / (60 + index + 1)
        boost = base * TYPE_BOOST_FRACTION if matches else 0.0
        scored.append((base + boost, index, candidate))
        rows.append({"candidate_index": index, "base_rank_score": base, "type_boost": boost,
                     "rank_score": base + boost, "matched_kinds": [kind for kind, _ in matches],
                     "event_statuses": sorted({status for kind, status in matches if kind == "event"})})
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in scored], {
        "policy": POLICY, "triggered": any(row["type_boost"] for row in rows), "wanted": sorted(wanted),
        "candidate_count": len(candidates), "accepted_views": len(accepted), "rejected_views": rejected,
        "duplicate_views": duplicates, "scope_status": "single_user" if single_user else "unknown_or_mixed",
        "output_indices": [index for _, index, _ in scored], "rows": rows,
        "max_type_boost": MAX_TYPE_BOOST, "type_boost_fraction": TYPE_BOOST_FRACTION,
        "output_policy": "whole_original_span",
        "field_supported": "unknown", "set_sufficient": "unknown",
    }

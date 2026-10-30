"""Local evidence scheduling and conservative Answer-input accounting.

No answer generation, model dispatch, derived facts, or changes to indexed text.
The reserve covers unobserved platform instructions; it is not a reconstruction
of the private evaluator prompt.
"""
from __future__ import annotations

import json
import re

from app.segments import token_count


ANSWER_INPUT_LIMIT = 117760
INSTRUCTION_RESERVE = 4096
DEFAULT_EVIDENCE_LIMIT = 65536


class InvalidSearchInput(ValueError):
    """A deterministic request limit that retrying cannot repair."""


def evidence_budget(query, options, configured=None):
    input_tokens = token_count(query) + token_count(json.dumps(list(options), ensure_ascii=False))
    remaining = ANSWER_INPUT_LIMIT - INSTRUCTION_RESERVE - input_tokens
    if remaining <= 0:
        raise InvalidSearchInput("query_context_budget_exceeded")
    budget = min(configured if configured is not None else DEFAULT_EVIDENCE_LIMIT, remaining)
    return budget, {"answer_input_limit": ANSWER_INPUT_LIMIT, "query_options_tokens": input_tokens,
                    "instruction_reserve": INSTRUCTION_RESERVE, "effective_evidence_tokens": budget,
                    "accounting": "serialized_hits_plus_reserve_not_private_prompt"}


def _global_query(query):
    # Only explicit whole-history requests trigger source spreading. Ordinary
    # factual/rule queries keep the original retrieval order.
    return bool(re.search(r"总结|综述|概括|整体回顾|完整经历|所有阶段|"
                          r"\b(?:summari[sz]e|summary|overview|entire history|all stages|"
                          r"across (?:all )?sessions|over the (?:whole|entire))\b", query, re.I))


def state_time_query(query):
    """Expose source chronology for state questions without choosing a winner."""
    return bool(re.search(r"当前|现在|最新|更新|变更|变化|冲突|矛盾|"
                          r"\b(?:current(?:ly)?|now|updated?|changed?|conflict(?:ing)?|"
                          r"contradict(?:ion|ory|ing)?)\b", query, re.I))


def record_token_cost(hit, rendered_text):
    """One cost contract for structural admission and final packing."""
    encoded = json.dumps({"id": hit.id, "content": rendered_text, "score": hit.score,
                          "created_at": hit.created_at}, ensure_ascii=False, separators=(",", ":"))
    return token_count(encoded) + 16


def pack_units(candidates, groups, max_items, char_limit, token_limit, renderer=None, *, query="", spread_sources=None):
    """Pack each directed witness unit atomically, sharing dependencies once.

    A common table header must not turn 30 independent rows into one huge
    connected component. Only a seed's explicitly declared dependencies are
    mandatory. Permission-filtered dependencies suppress that seed, not other
    independent evidence. No admission of an unvalidated candidate occurs here.
    """
    registry = {hit.id: hit for hit in candidates}
    requirements = {}
    unavailable = set()
    for group in groups:
        seed = group["seed_id"]
        requirements.setdefault(seed, set()).update(group["member_ids"])
        requirements[seed].add(seed)
        if group.get("unavailable_dependencies"):
            unavailable.add(seed)
    order = list(dict.fromkeys(hit.id for hit in candidates))
    positions = {key: i for i, key in enumerate(order)}
    spread = _global_query(query) if spread_sources is None else spread_sources
    if spread:
        # First two recalled fragments per source before additional fragments;
        # no extra source search and no time-based recency override.
        counts = {}; first = []; rest = []
        for key in order:
            source = registry[key].source_raw_id or key
            counts[source] = counts.get(source, 0) + 1
            (first if counts[source] <= 2 else rest).append(key)
        order = first + rest
    rendered = {key: renderer(hit) if renderer else hit.content for key, hit in registry.items()}
    costs = {key: record_token_cost(hit, rendered[key]) for key, hit in registry.items()}
    kept = []; used = set(); chars = tokens = 0; omitted = []
    for key in order:
        if len(kept) >= max_items:
            break
        if key in used:
            continue
        required = set(requirements.get(key, {key}))
        # Dependencies can themselves be seeds; close only directed edges.
        pending = list(required)
        while pending:
            current = pending.pop()
            for dep in requirements.get(current, ()):
                if dep not in required:
                    required.add(dep); pending.append(dep)
        if required.intersection(unavailable) or not required <= registry.keys():
            omitted.append({"seed_id": key, "reason": "missing_dependency"})
            continue
        # Preserve original source order within the group for readable tables.
        members = [registry[item] for item in sorted(required - used, key=positions.__getitem__)]
        if len({item.source_raw_id for item in members}) == 1 and all(
                item.span_start is not None for item in members):
            members.sort(key=lambda item: (item.span_start, item.id))
        new_chars = sum(len(rendered[item.id]) for item in members)
        new_tokens = sum(costs[item.id] for item in members)
        if (len(kept) + len(members) > max_items or chars + new_chars > char_limit
                or tokens + new_tokens > token_limit):
            omitted.append({"seed_id": key, "reason": "unit_over_budget"})
            continue
        kept.extend(members); used.update(item.id for item in members)
        chars += new_chars; tokens += new_tokens
    return kept, chars, {"policy": "directed_evidence_units_v1", "source_spread": spread,
                         "omitted_units": omitted, "packed_record_tokens": tokens,
                         "packed_content_tokens": sum(token_count(rendered[item.id]) for item in kept)}

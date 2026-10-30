"""Lexical query text. Embeddings live in embeddings.py; vectors in store.py.

Phase A indexes original text plus deterministic tokens so FTS5 unicode61 can
match digits and two-character CJK terms. Search responses never read this
expanded text; they read RawMessage.content.
"""

from __future__ import annotations

import re

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_WORD_OR_NUMBER = re.compile(r"[A-Za-z0-9]+")
_POSITIONAL_RUN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


def build_lexical_text(content: str) -> str:
    extras: list[str] = []
    for run in _CJK_RUN.findall(content):
        if len(run) >= 2:
            extras.extend(run[index : index + 2] for index in range(len(run) - 1))
    extras.extend(re.findall(r"\d+", content))
    if not extras:
        return content
    return content + "\n" + " ".join(extras)


def build_match_query(query: str, *, policy: str = "legacy") -> str | None:
    if policy == "balanced_v1":
        return _build_balanced_match_query(query)
    if policy != "legacy":
        raise ValueError("Unknown lexical query policy")
    tokens: list[str] = []
    for word in _WORD_OR_NUMBER.findall(query):
        if word.isdigit() or len(word) >= 2:
            tokens.append(word)
    for run in _CJK_RUN.findall(query):
        if len(run) >= 2:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        unique.append(token)
    if not unique:
        return None
    return " OR ".join(_quote_fts_token(token) for token in unique[:128])


def _build_balanced_match_query(query: str) -> str | None:
    """Keep the legacy vocabulary, but remove language/prefix truncation bias.

    Deduplicate case-sensitively at first occurrence in original text order.
    Over 128 unique terms, reserve up to 64 slots each for ASCII words/numbers
    and CJK bigrams, transferring unused slots to the other group. Within each
    group choose ranks floor(i * (n - 1) / (k - 1)), including both endpoints,
    then emit in original order. This is uniform over unique-term ranks, not
    character offsets or term frequency. Full scanning/deduplication uses O(n)
    space; the MATCH expression, not input processing, is capped at 128 terms.

    No semantic importance is inferred: a middle keyword can fall between
    sampled ranks, and language reservation can displace useful ASCII terms.
    This policy is a coverage tradeoff, not a recall improvement guarantee.
    """
    unique: list[str] = []
    seen: set[str] = set()
    groups: list[list[int]] = [[], []]
    for match in _POSITIONAL_RUN.finditer(query):
        run = match.group()
        is_cjk = "\u4e00" <= run[0] <= "\u9fff"
        if is_cjk:
            terms = (run[i : i + 2] for i in range(len(run) - 1))
        else:
            terms = iter((run,)) if run.isdigit() or len(run) >= 2 else iter(())
        for token in terms:
            if token in seen:
                continue
            seen.add(token)
            groups[int(is_cjk)].append(len(unique))
            unique.append(token)
    if not unique:
        return None
    if len(unique) > 128:
        counts = [min(64, len(group)) for group in groups]
        for i, group in enumerate(groups):
            counts[i] += min(128 - sum(counts), len(group) - counts[i])
        selected: list[int] = []
        for group, count in zip(groups, counts):
            if count == 1:
                selected.append(group[0])
            elif count > 1:
                selected.extend(
                    group[i * (len(group) - 1) // (count - 1)]
                    for i in range(count)
                )
        unique = [unique[i] for i in sorted(selected)]
    return " OR ".join(_quote_fts_token(token) for token in unique)


def _quote_fts_token(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


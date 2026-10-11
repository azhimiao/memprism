"""Lexical query text. Embeddings live in embeddings.py; vectors in store.py.

Phase A indexes original text plus deterministic tokens so FTS5 unicode61 can
match digits and two-character CJK terms. Search responses never read this
expanded text; they read RawMessage.content.
"""

from __future__ import annotations

import re

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_WORD_OR_NUMBER = re.compile(r"[A-Za-z0-9]+")


def build_lexical_text(content: str) -> str:
    extras: list[str] = []
    for run in _CJK_RUN.findall(content):
        if len(run) >= 2:
            extras.extend(run[index : index + 2] for index in range(len(run) - 1))
    extras.extend(re.findall(r"\d+", content))
    if not extras:
        return content
    return content + "\n" + " ".join(extras)


def build_match_query(query: str) -> str | None:
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


def _quote_fts_token(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


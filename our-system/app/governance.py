"""Conservative governance proposals. Unrecognised language remains raw evidence."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, Sequence
from app.models import RawMessage

POLICY_VERSION = "bounded-governance-v1"


@dataclass(frozen=True)
class ClaimDraft:
    source_raw_id: str
    subject: str
    predicate: str
    object_text: str
    kind: str = "fact"
    scope: str = "self_report"
    modality: str = "asserted"
    polarity: str = "positive"
    valid_from: int | None = None
    explicit_update: bool = False
    single_valued: bool = False
    attributed_origin: str = "user"
    temporal_precision: str = "source_timestamp"
    rule_version: str = POLICY_VERSION


@dataclass(frozen=True)
class MutationDraft:
    source_raw_id: str
    operation: str
    target_text: str | None = None
    target_id: str | None = None
    rule_version: str = POLICY_VERSION


@dataclass(frozen=True)
class GovernancePlan:
    claims: tuple[ClaimDraft, ...] = ()
    mutations: tuple[MutationDraft, ...] = ()
    version: str = POLICY_VERSION


class GovernancePolicy(Protocol):
    def plan(self, messages: Sequence[RawMessage]) -> GovernancePlan | list: ...


class PassthroughGovernance:
    def plan(self, messages: Sequence[RawMessage]) -> list:
        return []


def normalized_statement(text: str) -> str:
    # Exact statement selector: no substring/topic/semantic erasure.
    return text.strip().rstrip("。.!！").strip()


def statement_digest(text: str) -> str:
    statement = normalized_statement(text)
    statement = re.sub(r"^(?:请记住|记住|(?i:please remember that |remember that ))", "", statement).strip()
    return hashlib.sha256(statement.encode("utf-8")).hexdigest()


def claim_key(user_id: str, claim: ClaimDraft) -> str:
    fields = [user_id, claim.subject, claim.predicate, claim.scope, claim.modality, claim.polarity]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode("utf-8")).hexdigest()


def _target(text: str) -> str | None:
    text = text.strip()
    pairs = {'"': '"', '“': '”', "'": "'", '「': '」'}
    if text[:1] in pairs:
        if len(text) < 3 or text[-1] != pairs[text[0]]:
            return None
        text = text[1:-1]
    text = normalized_statement(text)
    return text or None


def parse_mutation(message: RawMessage) -> MutationDraft | None:
    if message.role != "user":
        return None
    text = normalized_statement(message.content)
    id_match = re.fullmatch(r"(?:请)?(?:删除|忘记)记忆\s+(raw_[0-9a-f]{32})", text)
    if id_match:
        return MutationDraft(message.id, "erase", target_id=id_match[1])
    patterns = (
        ("erase", r"(?:请)?(?:忘记|忘掉|删除)(.+?)(?:这条信息|这条记忆|这条记录|这件事)"),
        ("erase", r"(?:请)?(?:忘记|忘掉|删除)([“\"「].+[”\"」])"),
        ("erase", r"(?i:(?:please )?(?:forget|erase|delete) (?:the (?:fact|memory|information) that )?)([\"“].+[\"”])"),
        ("erase", r"(?i:(?:please )?(?:forget|erase|delete) the (?:fact|memory|information) that )(.+)"),
        ("retract", r"(?:请)?撤回我之前说的(.+)"),
        ("retract", r"(?i:I retract (?:my statement|the claim) that )(.+)"),
        ("suppress", r"(?:请)?不要再检索([“\"「].+[”\"」])"),
        ("suppress", r"(?i:(?:please )?do not retrieve )([\"“].+[\"”])"),
    )
    for operation, pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match and (target := _target(match[1])):
            return MutationDraft(message.id, operation, target_text=target)
    return None


_SINGLE_ATTRIBUTES = {
    "居住地": "residence", "住所": "residence", "常住地": "residence",
    "时区": "timezone", "timezone": "timezone", "home city": "residence",
    "residence": "residence", "工作单位": "employer", "employer": "employer",
}
_QUALIFIERS = re.compile(
    r"可能|也许|打算|计划|如果|假如|据说|听说|曾经|以前|去年|前年|明年|下周|上周|昨天|明天|曾|不再|不是|并非|"
    r"旅游|出差|暂时|旅行|搬去玩|临时|周末|假期|将会|\d+[年月天日]|"
    r"\b(?:maybe|might|perhaps|plan|planning|if|suppose|reportedly|used to|not|never|tomorrow|yesterday|next|last|ago|until|since|during|temporarily|visit|vacation|weekend|will)\b",
    re.IGNORECASE,
)


def parse_claim(message: RawMessage) -> ClaimDraft | None:
    if message.role != "user":
        return None
    text = normalized_statement(message.content)
    if _QUALIFIERS.search(text) or any(c in text for c in '“”"「」\n\r；;!?！？'):
        return None
    effective_time = message.source_timestamp
    precision = "source_timestamp" if effective_time is not None else "unknown"
    dated = re.fullmatch(r"(\d{4}-\d{2}-\d{2})[：:]\s*(.+)", text)
    if dated:
        try:
            effective_time = int(datetime.strptime(dated[1], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            return None
        text = dated[2]
        precision = "day"
    patterns = (
        ("residence", False, r"我(?:目前|现在)?住在(.+)"),
        ("residence", True, r"我(?:现在|已经)?搬到(.+?)(?:了)?"),
        ("residence", False, r"(?i:I (?:currently |now )?live in )(.+)"),
        ("residence", True, r"(?i:I (?:have )?moved to )(.+)"),
        ("employer", False, r"我在(.+?)工作"),
        ("employer", False, r"(?i:I work (?:at|for) )(.+)"),
    )
    for predicate, update, pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            value = match[1].strip()
            if not _simple_value(value) or not _entity_value(value):
                return None
            return ClaimDraft(message.id, "speaker:user", predicate, value,
                              valid_from=effective_time, explicit_update=update, single_valued=True,
                              temporal_precision=precision)
    generic = re.fullmatch(r"我的(.+?)(是|改为|改成)(.+)", text)
    english = re.fullmatch(r"(?i:my )(.+?)(?i: (is|changed to) )(.+)", text)
    match = generic or english
    if match:
        attribute, verb, value = [part.strip() for part in match.groups()]
        if not _simple_value(attribute) or not _simple_value(value):
            return None
        predicate = _SINGLE_ATTRIBUTES.get(attribute.casefold(), "attribute:" + attribute.casefold())
        if attribute.casefold() in _SINGLE_ATTRIBUTES and not _entity_value(value):
            return None
        return ClaimDraft(message.id, "speaker:user", predicate, value,
                          valid_from=effective_time,
                          explicit_update=verb.casefold() in {"改为", "改成", "changed to"},
                          single_valued=attribute.casefold() in _SINGLE_ATTRIBUTES, temporal_precision=precision)
    return None


def _simple_value(value: str) -> bool:
    return bool(value) and len(value) <= 160 and not re.search(r"[，,。:：]|(?:\band\b|\bbut\b)|但是|然后|并且|之后|之前|前提|是假的|才是|不是真的|的话|除非|否则", value, re.IGNORECASE)


def _entity_value(value: str) -> bool:
    # The supported slot is a bare name, not a sentence with a temporal,
    # conditional, negated or causal tail. False negatives fall back to raw.
    return not re.search(r"是|的|再|才|若|要|当|没有|并未|条件|前提|因为|所以|之后|之前|几天|"
                         r"\b(?:in|at|on|by|for|from|to|when|because|after|before|unless|would|could|should|which|that|is|was|are|were|been|had|have|has)\b|"
                         r"\b(?:19|20)\d{2}\b", value,re.I)


class BoundedGovernance:
    """Full-sentence grammar, no fixture/gold/query access; no LLM capability claim."""
    def plan(self, messages: Sequence[RawMessage]) -> GovernancePlan:
        claims, mutations = [], []
        for message in messages:
            mutation = parse_mutation(message)
            if mutation:
                mutations.append(mutation)
            elif claim := parse_claim(message):
                claims.append(claim)
        return GovernancePlan(tuple(claims), tuple(mutations))


@dataclass(frozen=True)
class TemporalView:
    intent: str = "all"
    as_of: int | None = None


def knowledge_time_gate(query, candidates, view):
    """Source availability for an explicitly dated knowledge question.

    Event/state time and the time evidence became available are separate axes.
    A later report can describe an earlier event; ordinary historical questions
    therefore keep that report. Only explicit 'as of ... known/recorded' asks
    for a knowledge cutoff. Missing source timestamps stay unknown, never use
    arrival time or the server clock. A date boundary covers its complete UTC
    source day, matching the existing date parser's precision.
    """
    explicit = re.search(r'\bas\s+of\b|截至|截止到|截止|为止', query, re.I)
    knowledge = re.search(r'\b(?:known|knew|recorded|reported|available|information)\b|知道|已知|掌握|获知|已记录|记录', query, re.I)
    if view.intent != 'as_of' or view.as_of is None or not explicit or not knowledge:
        return list(candidates), {'active':False,'axis':'source_availability','reason':'no_explicit_knowledge_cutoff'}
    # temporal_view already returns the inclusive source-day end.
    cutoff = view.as_of
    kept, omitted, unknown = [], [], []
    for hit in candidates:
        if hit.source_timestamp is None:
            kept.append(hit);unknown.append(hit.id)
        elif hit.source_timestamp <= cutoff:
            kept.append(hit)
        else:
            omitted.append(hit.id)
    return kept, {'active':True,'axis':'source_availability','source_day_end_unix_ms':cutoff,
                  'precision':'UTC_source_day','omitted_future_source_ids':omitted,
                  'unknown_source_time_ids':unknown,'event_time_inferred':False}


def temporal_view(query: str) -> TemporalView:
    # No server clock, answer options or dataset identifiers.
    current = re.search(r"现在|目前|当前|\b(?:now|currently|current)\b",query,re.I)
    historical = re.search(r"以前|过去|曾经|历史|之前|撤回|去年|前年|\b(?:previous|history|historical|retracted|last year|used to)\b",query,re.I)
    # Mixed/negative requests do not grant historical use of retracted sources.
    if historical and (current or re.search(r"不要|不许|别|不包含|排除|无需|\b(?:not|never|without|exclude|ignore|don't)\b",query,re.I)):
        return TemporalView("current" if current else "all")
    if re.search(r"以前|过去|曾经|历史|之前|撤回|去年|前年|\b(?:previous|history|historical|retracted|last year|used to)\b", query, re.I):
        return TemporalView("history")
    dates = re.findall(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)", query)
    if len(dates) == 1:
        try:
            point = datetime.strptime(dates[0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return TemporalView("as_of", int(point.timestamp() * 1000) + 86399999)
        except ValueError:
            return TemporalView("history")
    if re.search(r"现在|目前|当前|\b(?:now|currently|current)\b", query, re.I):
        return TemporalView("current")
    return TemporalView()

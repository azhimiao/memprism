"""062 deterministic offsets and explicit structures; no generated facts."""
from __future__ import annotations

import csv
import re
from app import segments

VERSION = "source_blocks_062_v1"
STRUCTURE_SCAN_LIMIT = 128000
STRUCTURE_UNIT_LIMIT = 256
RULE = re.compile(r"\b(?:must|shall|required|unless|except|rules?|procedure|steps?)\b|必须|规则|步骤|除非|例外|前提", re.I)
EXPERIENCE = re.compile(r"\b(?:input|output|trial|experiment|observation|result|outcome)\b|输入|输出|试验|实验|观测|结果", re.I)
SENSITIVE = re.compile(r"\b(?:private|confidential|sensitive|password|diagnosis|medical|passport|secret)\b|隐私|保密|密码|诊断|病历|身份证|敏感", re.I)
WITHHOLD = re.compile(r"\b(?:do not|never|don't)\s+(?:share|disclose|reveal|publish)\b|不得披露|禁止公开|不要泄露|请勿公开", re.I)
HEADER = re.compile(r"^\s*(?:#{1,6}\s+|(?:Rules?|Policy|Procedure|Scope|规则|流程|范围)\s*[:：]|第[^\s]{1,12}[章节条])", re.I)
FIELD = re.compile(r"^\s*(subject|attribute|scope|value|effective|update|cardinality|rule id|reference|主体|属性|范围|值|生效日期|更新|规则编号|引用|单值)\s*[:：]\s*(.{1,160})\s*$", re.I | re.M)
_FIELD_NAMES = dict(zip(("主体", "属性", "范围", "值", "生效日期", "更新", "规则编号", "引用"),
                       ("subject", "attribute", "scope", "value", "effective", "update", "rule id", "reference")))
_FIELD_NAMES['单值'] = 'cardinality'


def fenced_ranges(text):
    result=[];start=None;marker=None
    for match in re.finditer(r"(?m)^\s*(`{3,}|~{3,}).*$",text):
        if start is None:
            start=match.start();marker=match[1]
        elif match[1][0]==marker[0] and len(match[1])>=len(marker):
            result.append((start,match.end()));start=None
    if start is not None:result.append((start,len(text)))
    return result


def explicit_fields(text):
    values = {}
    for match in FIELD.finditer(text):
        key = _FIELD_NAMES.get(match[1].lower(), match[1].lower())
        if key in values:  # ambiguous repeated fields are not a state identity
            return {}
        values[key] = match[2].strip()
    return values


def sections(text):
    """Exact paragraph/line spans. Small ordinary prose is sentence addressable."""
    result = []
    for match in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        lo, hi = match.span()
        body = text[lo:hi]
        if not (RULE.search(body) or EXPERIENCE.search(body) or "\n" in body):
            cuts = [m.end() for m in re.finditer(r"[.!?。！？](?:\s+|$)", body)]
            starts = [0] + cuts
            for start, end in zip(starts, cuts + [len(body)]):
                if body[start:end].strip():
                    result.append((lo + start, lo + end))
        else:
            # Lines retain table/record offsets; support units join dependencies.
            offset = lo
            for line in body.splitlines(keepends=True):
                end = offset + len(line)
                if line.strip():
                    result.append((offset, end))
                offset = end
    return result


def structure_units(text):
    """lo, hi, required ranges, kind; all ranges refer to the same original."""
    from app.evidence_units import _structures
    text = text[:STRUCTURE_SCAN_LIMIT]
    fences=fenced_ranges(text)
    units = list(_structures(text))[:STRUCTURE_UNIT_LIMIT]
    lines = [(m.start(), m.end(), m[0].rstrip("\r\n")) for m in re.finditer(r"[^\n]+(?:\n|$)", text)]
    for i, (lo, hi, line) in enumerate(lines):
        delimiter = "\t" if "\t" in line else "," if "," in line else None
        if delimiter is None or "`" in line or any(lo<b and hi>a for a,b in fences):
            continue
        try:
            header = next(csv.reader([line], delimiter=delimiter, strict=True))
        except (csv.Error, StopIteration):
            continue
        if not 2 <= len(header) <= 32 or not all(re.fullmatch(r"[\w ()%/.-]{1,60}", h.strip()) and re.search(r"[^\W\d_]", h) for h in header):
            continue
        if not any(EXPERIENCE.search(h) or re.search(r"date|time|unit|name|value|条件|参数|时间|单位|数值", h, re.I) for h in header):
            continue
        end = i + 1
        while end < len(lines) and lines[end][0] == lines[end - 1][1]:
            try:
                row = next(csv.reader([lines[end][2]], delimiter=delimiter, strict=True))
            except (csv.Error, StopIteration):
                break
            if len(row) != len(header):
                break
            units.append((lines[end][0], lines[end][1], [(lo, hi)], "observation_schema"))
            if len(units)>=STRUCTURE_UNIT_LIMIT:
                return units
            end += 1
    for match in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        body = match[0]
        if len(body) > 7000 or any(match.start()<b and match.end()>a for a,b in fences):
            continue
        fields = explicit_fields(body)
        kind = None
        if {"subject", "attribute", "scope", "value"} <= fields.keys():
            kind = "explicit_state"
        elif (RULE.search(body) or (fields.get('scope') and (fields.get('rule id') or fields.get('reference')))) and not re.search(r"\|\s*:?-{3,}", body):
            kind = "rule_context"
        elif len(EXPERIENCE.findall(body)) >= 2:
            kind = "observation_record"
        if kind:
            units.append((match.start(), match.end(), [match.span()], kind))
            if len(units)>=STRUCTURE_UNIT_LIMIT:
                break
    return units


def blocks(text):
    """All source text is retained in Raw. Directory entries are exact subspans."""
    units = structure_units(text)
    fences=fenced_ranges(text)
    spans = []
    guarded = [(m.start(),m.end(),bool(WITHHOLD.search(m[0])))
               for m in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*",text)
               if not any(m.start()<b and m.end()>a for a,b in fences)
               and (re.match(r"\s*(?:private|confidential|sensitive|隐私|保密|敏感)\s*[:：]",m[0],re.I) or WITHHOLD.search(m[0]))]
    # A record/section header owns its body across blank lines. Only an
    # unambiguous subsequent Markdown peer heading ends this restriction.
    headings=[h for h in re.finditer(r"(?m)^[ \t]*(#{1,6})[ \t]+[^\n]+",text)
              if not any(h.start()<b and h.end()>a for a,b in fences)]
    for line in re.finditer(r"(?m)^[ \t]*(?:#{1,6}[ \t]+)?[^\n]+[:：][ \t]*$",text):
        if not (WITHHOLD.search(line[0]) or re.search(r"\b(?:private|confidential|sensitive)\b|保密|隐私|敏感",line[0],re.I)):
            continue
        if any(line.start()<b and line.end()>a for a,b in fences):continue
        level_match=re.match(r"[ \t]*(#{1,6})",line[0])
        level=len(level_match[1]) if level_match else 6
        end=next((h.start() for h in headings if h.start()>line.start() and len(h[1])<=level),len(text))
        guarded.append((line.start(),end,bool(WITHHOLD.search(line[0]))))
    for lo, hi in sections(text):
        for start, end, _ in segments.boundaries(text[lo:hi]):
            a, b = lo + start, lo + end
            body = text[a:b]
            code = any(a<end and b>start for start,end in fences)
            relevant = [u for u in units if a < u[1] and b > u[0]]
            kind = 'code' if code else next((u[3] for u in relevant if u[3] == "explicit_state"),
                        relevant[0][3] if relevant else "text")
            inherited = [g for g in guarded if a<g[1] and b>g[0]]
            spans.append({"lo": a, "hi": b, "kind": kind,
                          "sensitive": bool((not code and SENSITIVE.search(body)) or inherited),
                          "withhold": bool((not code and WITHHOLD.search(body)) or any(g[2] for g in inherited))})
    return spans, units


def query_focus(query):
    """Use an explicit last question/task when present, never alter base recall."""
    labels = list(re.finditer(r"(?im)^\s*(?:question|task|request|问题|提问|任务)\s*[:：]\s*", query))
    if labels:
        return query[labels[-1].end():].strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", query) if p.strip()]
    if len(paragraphs) > 1 and re.search(r"[?？]|^(?:please |请|summari[sz]e|概括|总结)", paragraphs[-1], re.I):
        return paragraphs[-1]
    return query.strip()


def demand(query):
    focus = query_focus(query)
    summary = bool(re.search(r"^(?:please\s+)?(?:summari[sz]e|give (?:me )?(?:a |an )?(?:summary|overview)|review)\b|^(?:请)?(?:总结|概括|综述|回顾)|(?:can|could) you summari[sz]e", focus, re.I))
    global_scope = summary and bool(re.search(r"\b(?:all|entire|whole)\s+(?:my |the |our )?(?:history|conversations?|sessions?|stages?)\b|全部历史|所有会话|完整经历|所有阶段|整个历史", focus, re.I))
    state = bool(re.search(r"\b(?:current|currently|latest|conflict|contradiction|changed|history of)\b|当前|现在|最新|冲突|矛盾|变更|变化历程", focus, re.I))
    return {"focus": focus, "summary": summary, "global": global_scope,
            "state": state, "rule": bool(RULE.search(focus)), "experience": bool(EXPERIENCE.search(focus))}

"""Query-only capability hints and bounded optional routes.

These hints are not support probabilities, fact extraction or dataset labels.
No wall clock or answer gold is available to this module.
"""
from dataclasses import dataclass
import re

from app.governance import temporal_view
from app.segments import token_count


@dataclass(frozen=True)
class CapabilityPlan:
    temporal_intent:str
    as_of:int|None
    intents:tuple[str,...]
    option_queries:tuple[tuple[str,int],...]
    options_reason:str
    window_reason:str|None


def plan_query(query,options=None):
    view=temporal_view(query)
    kinds=[]
    for name,pattern in [
        ("personalization",r"偏好|喜欢|推荐|适合|饮食|过敏|\b(?:prefer|recommend|preference|allerg|diet|suitable)\w*\b"),
        ("procedure",r"步骤|流程|规则|程序|执行|格式|\b(?:step|procedure|rule|execute|format|json|workflow|code)\w*\b"),
        ("multi_hop",r"共同|分别|关系|多少|\b(?:both|between|relationship|how many|compare)\b"),
        ("temporal",r"何时|什么时候|先后|之前|之后|\b(?:when|before|after|chronolog|earliest|latest)\w*\b")]:
        if re.search(pattern,query,re.I):kinds.append(name)
    unique=list(dict.fromkeys(option.strip() for option in (options or []) if option.strip()))
    reason="bounded_options";routes=[]
    if not unique:reason="options_absent"
    elif len(unique)>8:reason="options_over_count_cap"
    elif any(token_count(option)>512 for option in unique):reason="options_over_token_cap"
    else:
        for i,option in enumerate(unique):
            routes.append((query+"\nCandidate option: "+option,100//len(unique)+(i<100%len(unique))))
    window_reason="relational_or_temporal_query" if {"multi_hop","temporal"}&set(kinds) else None
    return CapabilityPlan(view.intent,view.as_of,tuple(kinds),tuple(routes),reason,window_reason)


_TYPES=[
    ("constraint",r"\b(?:must not|never|cannot|allergic|contraindicat|avoid|budget|do not)\b|禁止|不得|不能|过敏|禁忌|预算|不要"),
    ("preference",r"\b(?:prefer|favorite|favourite|like|dislike|rather than)\b|喜欢|偏好|讨厌|倾向"),
    ("procedure",r"\b(?:step\s*\d|procedure|workflow|first|then|finally|execute)\b|步骤|流程|首先|然后|最后|执行"),
    ("event",r"\b(?:visited|moved|joined|started|finished|attended|yesterday|tomorrow)\b|参加|搬到|开始|结束|昨天|明天"),
    ("task",r"\b(?:task|todo|deadline|due|pending|completed)\b|任务|截止|待办|完成")]


def evidence_type_hints(text):
    """Navigation hints only. A conditional/negated sentence stays intact."""
    return tuple(name for name,pattern in _TYPES if re.search(pattern,text,re.I))


def typed_order(candidates,plan):
    wanted=set()
    if "personalization" in plan.intents:wanted.update(("constraint","preference"))
    if "procedure" in plan.intents:wanted.update(("constraint","procedure","task"))
    if "temporal" in plan.intents:wanted.add("event")
    if not wanted:return list(candidates),{"triggered":False,"reason":"no_type_intent"}
    # Small ranking increment on already recalled exact sources, never a global
    # scan or recent-source fallback. RRF remains a ranking score only.
    scored=[]
    for rank,hit in enumerate(candidates,1):
        types=set(evidence_type_hints(hit.content));boost=.25*bool(types&wanted)
        scored.append((1./(60+rank)+boost/60,rank,hit))
    scored.sort(key=lambda item:(-item[0],item[1],item[2].id))
    return [item[2] for item in scored],{"triggered":True,"wanted":sorted(wanted),"policy":"source_hint_rank_v1"}

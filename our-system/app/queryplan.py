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


def _scale_hit_shares(pairs, remaining):
    """Fit nominal hit requests into `remaining` without raising any request."""
    result={name:0 for name,_ in pairs}
    active=[(name,qty) for name,qty in pairs if qty>0]
    if remaining<=0 or not active:
        return result
    total=sum(qty for _,qty in active)
    if total<=remaining:
        result.update(dict(active))
        return result
    raw=[(name,qty*remaining/total) for name,qty in active]
    floors={name:int(value) for name,value in raw}
    leftover=remaining-sum(floors.values())
    order=sorted(raw,key=lambda item:(-(item[1]-int(item[1])),item[0]))
    for index in range(leftover):
        floors[order[index][0]]+=1
    result.update(floors)
    return result


def allocate_index_budget(budget,*,options_active,sentence,bridge,atom,option_total):
    """One hit budget for every route, assigned before any index call.

    Options-active searches keep the pre-registered original share (min(100, budget)).
    Sentence, bridge, atom and options spend only what remains and are reduced to fit.
    Without options, the original query keeps a positive share and the same auxiliary
    nominals use the leftover. This never raises `budget`.
    """
    if isinstance(budget,bool) or not isinstance(budget,int) or budget<1:
        raise ValueError("candidate budget must be a positive integer")
    if isinstance(option_total,bool) or not isinstance(option_total,int) or option_total<0:
        raise ValueError("option hit total must be a non-negative integer")
    sentence_nominal=min(50,budget//4) if sentence else 0
    bridge_nominal=min(50,budget//4) if bridge else 0
    atom_nominal=min(50,budget//4) if atom else 0
    if options_active:
        original=min(100,budget)
        aux=_scale_hit_shares((("sentence",sentence_nominal),("bridge",bridge_nominal),
            ("atom",atom_nominal),("options",option_total)),budget-original)
    else:
        names=(("sentence",sentence_nominal),("bridge",bridge_nominal),("atom",atom_nominal))
        aux_sum=sentence_nominal+bridge_nominal+atom_nominal
        if aux_sum<budget:
            original=budget-aux_sum
            aux={name:qty for name,qty in names}
            aux["options"]=0
        else:
            original=1
            aux=_scale_hit_shares(names,budget-1)
            aux["options"]=0
    planned={"original":original,"sentence":aux["sentence"],"bridge":aux["bridge"],
             "atom":aux["atom"],"options":aux["options"]}
    return planned


def scale_option_queries(option_queries,budget):
    """Redistribute an already accepted option plan. Never increases its hit total."""
    queries=tuple(option_queries)
    if budget<=0 or not queries:
        return ()
    total=sum(quota for _,quota in queries)
    if total<=0 or budget>=total:
        return queries
    raw=[(query,quota*budget/total) for query,quota in queries]
    floors=[int(value) for _,value in raw]
    leftover=budget-sum(floors)
    order=sorted(range(len(raw)),key=lambda i:(-(raw[i][1]-floors[i]),i))
    for index in range(leftover):
        floors[order[index]]+=1
    return tuple((query,quota) for (query,_),quota in zip(raw,floors) if quota>0)


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

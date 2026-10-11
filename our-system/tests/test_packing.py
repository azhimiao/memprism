"""Packing contracts. A/B quality is measured separately on public questions."""
from app.models import EvidenceCandidate
from app.retrieval import pack_evidence
from app.segments import token_count


def hit(id,text):
    return EvidenceCandidate(id=id,user_id="u",content=text,score=1,created_at=0,
                             deletion_state="active",source_raw_id=id,message_index=0)


def test_skip_keeps_later_whole_evidence_when_first_cannot_fit():
    hits=[hit("long","alpha "*30),hit("condition","Only if approved."),hit("negative","Do not execute.")]
    assert pack_evidence(hits,3,40,"prefix")[0]==[]
    kept,chars,trace=pack_evidence(hits,3,40,"skip")
    assert [h.id for h in kept]==["condition","negative"] and chars<=40 and trace["skipped_budget"]==1
    assert [h.content for h in kept]==["Only if approved.","Do not execute."]


def test_unicode_token_budget_and_item_budget_apply_without_truncating():
    hits=[hit("emoji","😀"*50),hit("short","alpha"),hit("more","beta")]
    kept,chars,trace=pack_evidence(hits,1,1000,"skip",10)
    assert [h.id for h in kept]==["short"] and sum(token_count(h.content) for h in kept)<=10
    assert trace["skipped_budget"]==1

from app.models import EvidenceCandidate
from app.set_selection import select_evidence


def hit(id,text,source=None):
    return EvidenceCandidate(id=id, user_id='u', content=text, score=1.,created_at=1,
        deletion_state='active',source_raw_id=source or id,message_index=0)


def test_fixed_pool_coverage_can_skip_large_distractor_without_generated_text():
    pool=[hit('a','filler '*80),hit('b','Alice knows Bob.'),hit('c','Bob moved to Oslo.')]
    kept,chars,trace=select_evidence(pool,'Where did Bob move?',2,60)
    assert [h.id for h in kept]==['b','c']
    assert chars==sum(len(h.content) for h in kept) and not trace['semantic_sufficiency']
    assert all(h in pool for h in kept)


def test_complete_group_with_exception_is_atomic_and_missing_member_omits_seed():
    pool=[hit('rule','Step 1: execute.'),hit('except','Except when input is missing.'),hit('other','Task: log status.')]
    group={'seed_id':'rule','member_ids':['rule','except']}
    kept,_,_=select_evidence(pool,'execute',1,100,groups=[group])
    assert all(h.id!='rule' for h in kept)
    kept,_,trace=select_evidence(pool[:1],'execute',3,100,groups=[group])
    assert kept==[] and trace['omitted_groups'][0]['reason']=='missing_dependency'


def test_unknown_query_never_adds_recent_or_external_source():
    assert select_evidence([], 'unseen topic',10,100)[0]==[]
    pool=[hit('a','A saved fact.')]
    assert select_evidence(pool,'unseen topic',10,100)[0]==pool


def test_overlapping_groups_cannot_split_and_budget_counts_actual_renderer():
    pool=[hit(x,'text') for x in 'abc']
    groups=[{'seed_id':'a','member_ids':['a','b']},{'seed_id':'b','member_ids':['b','c']}]
    assert select_evidence(pool,'text',2,100,groups=groups)[0]==[]
    kept,chars,_=select_evidence(pool,'text',3,20,renderer=lambda h:'tag:'+h.content,groups=groups)
    assert kept==[] and chars==0


def test_coverage_never_evicts_protected_long_high_rank_source_for_short_tail():
    pool=[hit('head','A detailed report describes alpha and its exact contextual conditions. '*4)]
    pool.extend(hit(str(i),'alpha') for i in range(1,10))
    kept,_,trace=select_evidence(pool,'alpha',4,1000)
    assert kept[0] is pool[0]
    assert 'head' in trace['protected_prefix_ids']
    assert trace['coverage_relative_bonus_cap']==.25

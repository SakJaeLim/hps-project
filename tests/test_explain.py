"""T12 · spec 07 — TDD Green. T21: RL 의사결정 설명 융합 추가."""
import pytest

def test_explain():
    from snct.knowledge.explain import explain
    from snct.common.schema import CandidatePlan
    txt = explain(CandidatePlan(engine="greedy"), [], evidence=[{"type":"doc","ref":"SOP","snippet":"..."}])
    assert isinstance(txt, str) and txt


# ── T21: RL 의사결정 설명 (reward_decomp 귀인 + 위반 + doc_refs 융합) ──
def _bl_r2_decision():
    from snct.data.sources.rl_results import RLResultStore
    return RLResultStore().get_decision("BL", 2)


@pytest.mark.rl_data
def test_explain_rl_decision_attributes_reward_terms():
    """설명은 reward_decomp 상위 ±기여항을 운영자 라벨로 인용해야 한다."""
    from snct.knowledge.explain import explain
    from snct.common.schema import CandidatePlan
    d = _bl_r2_decision()
    txt = explain(CandidatePlan(engine="rl"), [], evidence=[], decision=d)
    # 절댓값 1위 = R2_stack_full(스택 충진), 큰 양(+) 기여 = R7_completion(적재 완료)
    assert "스택 충진" in txt
    assert "적재 완료" in txt
    # 기여 방향(부호) 표기
    assert ("+" in txt) and ("−" in txt or "-" in txt)


@pytest.mark.rl_data
def test_explain_rl_decision_cites_regulations_and_kpi():
    from snct.knowledge.explain import explain
    from snct.common.schema import CandidatePlan
    d = _bl_r2_decision()
    txt = explain(CandidatePlan(engine="rl"), [], evidence=[], decision=d)
    assert "SOLAS" in txt          # doc_refs 인용
    assert "WBI" in txt or "wbi" in txt  # 운영지표 인용


@pytest.mark.rl_data
def test_explain_rl_decision_is_faithful():
    """설명에 등장하는 보상총합은 소스 값과 일치(환각 금지)."""
    from snct.knowledge.explain import explain
    from snct.common.schema import CandidatePlan
    d = _bl_r2_decision()
    txt = explain(CandidatePlan(engine="rl"), [], evidence=[], decision=d)
    # reward_total 정수부가 그대로 등장
    assert str(int(d.reward_total)) in txt.replace(",", "")

"""T22 · spec 04·07 — 설명 faithfulness 평가 하니스 TDD.

설명(rule-based)에 등장하는 모든 수치가 RL 결과의 사실(reward_decomp·kpi·violation)에
근거하는지 자동 채점한다. xai_grounding.json 전 레코드에 대해 회귀.
규칙 기반 설명은 새 수치를 만들지 않으므로 faithfulness == 1.0 이어야 한다.
"""
import pytest

pytestmark = pytest.mark.rl_data


def _store():
    from snct.data.sources.rl_results import RLResultStore
    return RLResultStore()


def test_score_decision_is_fully_grounded():
    """BL R4 설명의 모든 수치가 소스에 근거 → faithfulness 1.0, 위반 없음."""
    from snct.eval.faithfulness import score_decision
    d = _store().get_decision("BL", 4)
    rep = score_decision(d)
    assert rep["faithfulness"] == 1.0
    assert rep["n_unsupported"] == 0
    assert rep["doc_refs_cited"] is True


def test_harness_detects_hallucination():
    """음성 대조: 설명에 소스에 없는 수치(9999.0)를 끼워 넣으면 unsupported로 탐지."""
    from snct.eval.faithfulness import score_decision
    d = _store().get_decision("BL", 4)
    tampered = "보상 총합 9999.0 ... 무게균형 0.936"  # 9999.0은 어떤 소스값과도 불일치
    rep = score_decision(d, text=tampered)
    assert rep["n_unsupported"] >= 1
    assert rep["faithfulness"] < 1.0


def test_evaluate_all_grounding_records():
    """xai_grounding 전 레코드 집계 — 평균 faithfulness 1.0, 실패 0건."""
    from snct.eval.faithfulness import evaluate
    rep = evaluate(_store())
    assert rep["n"] >= 1
    assert rep["n"] == len(_store().load_xai_grounding())
    assert rep["mean_faithfulness"] == pytest.approx(1.0)
    assert rep["failures"] == []

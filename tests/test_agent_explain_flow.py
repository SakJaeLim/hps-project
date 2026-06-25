"""에이전트 흐름 · spec 02·07 — RL 의사결정 설명 진입점 TDD.

질의(policy/round 또는 자연어) → 근거 수집(RDB·LPG) → 설명 융합 → 자기검증(faithfulness).
"""
import pytest

pytestmark = pytest.mark.rl_data


def test_parse_decision_ref_from_question():
    from snct.agents.graph import parse_decision_ref
    ref = parse_decision_ref("BL 정책 4라운드는 왜 그렇게 적재했나?")
    assert ref is not None
    assert ref.policy == "BL" and ref.round_id == 4


def test_run_explanation_by_ids():
    from snct.agents.graph import run_explanation
    rec = run_explanation(policy="BL", round_id=4)
    # 설명에 귀인·규정·LPG 위반 컨테이너가 모두 인용됨
    assert "SOLAS_VI" in rec.rationale
    assert "Tier 정합" in rec.rationale
    assert "BL_R4" in rec.rationale  # 위반 컨테이너 ID
    # 식별/검증 체크 포함
    assert any("policy=BL" in c for c in rec.checks)
    assert any("round=4" in c for c in rec.checks)


def test_run_explanation_self_verifies_faithfulness():
    """흐름이 설명을 스스로 채점해 환각 0(faithfulness=1.0)을 체크로 남긴다."""
    from snct.agents.graph import run_explanation
    rec = run_explanation(policy="BL", round_id=4)
    assert any("faithfulness=1.0" in c for c in rec.checks)


def test_run_explanation_from_natural_language():
    from snct.agents.graph import run_explanation
    rec = run_explanation(question="EF 정책 2라운드 적재 사유 설명해줘")
    assert isinstance(rec.rationale, str) and rec.rationale
    assert any("policy=EF" in c for c in rec.checks)


def test_run_explanation_unknown_decision_is_graceful():
    from snct.agents.graph import run_explanation
    rec = run_explanation(policy="BL", round_id=999)
    assert "찾을 수 없" in rec.rationale or "없습니다" in rec.rationale

"""통합 · spec 07 — RL 설명에 LPG(컨테이너별 위반 규정) 근거 자동 인용 TDD.

T18(decision) + T19(LPG) + T21(explain) + T22(faithfulness)를 한 설명으로 묶는다.
규정 인용을 추가해도 faithfulness == 1.0(인용 숫자는 LPG 근거) 이어야 한다.
"""
import pytest

pytestmark = pytest.mark.rl_data


def _store():
    from snct.data.sources.rl_results import RLResultStore
    return RLResultStore()


def _lpg():
    from snct.knowledge.lpg_csv import LPGGraph
    return LPGGraph()


def test_lpg_violations_in_round():
    """(policy, round) 위반 슬롯 → 컨테이너·규정 조인."""
    rows = _lpg().violations_in_round("BL", 4)
    assert rows, "BL R4는 col_wt 위반 8건이 있어야 함"
    r = rows[0]
    assert {"container_id", "code", "rule", "source", "row"} <= set(r.keys())
    assert any(x["code"] == "SOLAS_VI" for x in rows)
    assert any("145" in (x["rule"] or "") for x in rows)


def test_explain_enriched_cites_container_violations():
    """lpg 제공 시 설명에 위반 컨테이너 ID와 규정 근거가 인용된다."""
    from snct.knowledge.explain import explain_rl_decision
    d = _store().get_decision("BL", 4)
    txt = "\n".join(explain_rl_decision(d, lpg=_lpg()))
    assert "SOLAS_VI" in txt
    # 실제 위반 컨테이너 ID가 인용됨
    cid = _lpg().violations_in_round("BL", 4)[0]["container_id"]
    assert cid in txt


def test_clean_round_has_no_violation_section():
    """위반 없는 라운드(BL R2)는 컨테이너 위반 섹션이 없다."""
    from snct.knowledge.explain import explain_rl_decision
    d = _store().get_decision("BL", 2)
    txt = "\n".join(explain_rl_decision(d, lpg=_lpg()))
    assert "위반 컨테이너" not in txt


def test_enriched_explanation_is_faithful():
    """규정 인용(145 MT 등)을 포함해도 모든 수치가 근거됨 → faithfulness 1.0."""
    from snct.eval.faithfulness import score_decision
    d = _store().get_decision("BL", 4)
    rep = score_decision(d, lpg=_lpg())
    assert rep["n_unsupported"] == 0
    assert rep["faithfulness"] == 1.0

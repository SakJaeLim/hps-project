"""T18 · spec 07 — RL 결과 로더 TDD.

강화학습 결과 자료(reward_decomp·kpi·slot_assignment·violation_log·xai_grounding)를
캐노니컬하게 적재하고, (policy, round_id) 단위 의사결정 근거를 융합 입력으로 제공한다.
"""
import pytest

pytestmark = pytest.mark.rl_data  # 실제 RL 결과 자료가 있어야 통과


def _store():
    from snct.data.sources.rl_results import RLResultStore
    return RLResultStore()


def test_store_discovers_artifacts():
    """기본 생성 시 RDB/RAG 산출물 위치를 자동 탐색한다."""
    s = _store()
    assert s.rdb_dir.is_dir()
    assert s.rag_dir.is_dir()


def test_load_reward_decomp_has_attribution_terms():
    """reward_decomp = R1~R15 항목별 ±기여(귀인) + reward_total."""
    df = _store().load_reward_decomp()
    assert {"policy", "round_id", "reward_total"} <= set(df.columns)
    # 보상 분해항이 최소 10개 이상(R1..R15)
    r_terms = [c for c in df.columns if c.startswith("R") and "_" in c]
    assert len(r_terms) >= 10
    assert df["policy"].dtype == object  # 문자열 정규화


def test_load_kpi_has_ops_metrics():
    df = _store().load_kpi()
    assert {"policy", "round_id", "reward", "osr", "wbi", "psr", "cwvr"} <= set(df.columns)


def test_load_xai_grounding_is_reference():
    """xai_grounding = 이미 융합된 정답 레퍼런스(rationale 포함)."""
    g = _store().load_xai_grounding()
    assert isinstance(g, list) and g
    rec = g[0]
    assert {"policy", "round_id", "rationale"} <= set(rec.keys())


def test_get_decision_fuses_sources_consistently():
    """(policy, round_id) 의사결정 = reward_decomp + kpi + violations + grounding 융합.
    동일 결정의 reward_total은 소스 간 일치해야 한다(사실 정합)."""
    s = _store()
    d = s.get_decision("BL", 2)
    assert d.policy == "BL" and d.round_id == 2
    # 귀인: 절댓값 상위 기여항이 정렬되어 제공됨
    assert d.top_contributions, "상위 보상 기여항이 비어 있음"
    name, val = d.top_contributions[0]
    assert isinstance(name, str) and isinstance(val, float)
    # 운영지표 존재
    assert "osr" in d.kpi and "wbi" in d.kpi
    # 사실 정합: reward_decomp.reward_total ≈ kpi.reward
    assert d.reward_total == pytest.approx(d.kpi["reward"], abs=0.01)


def test_get_decision_unknown_raises():
    s = _store()
    with pytest.raises(KeyError):
        s.get_decision("BL", 999)

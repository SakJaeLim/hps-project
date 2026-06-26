"""T19 · spec 07 — LPG(neo4j_kg) CSV 폴백 그래프 질의 TDD.

Neo4j 실연결 없이 neo4j_kg/*.csv를 직접 읽어 관계·제약 근거를 질의한다.
파라미터 템플릿 우선(specs/07): STACKED_ON · VIOLATES · Constraint.
"""
import pytest

pytestmark = pytest.mark.rl_data


def _g():
    from snct.knowledge.lpg_csv import LPGGraph
    return LPGGraph()


def test_stacked_on_returns_upper_container():
    """X 위에 쌓인 컨테이너(상위 Tier)를 반환. BL_R4_r0_t0 위 = BL_R4_r0_t1."""
    g = _g()
    res = g.stacked_on("BL_R4_r0_t0")
    ids = [r["container_id"] for r in res]
    assert "BL_R4_r0_t1" in ids
    # is_overstow 플래그 포함(재취급 여부)
    assert "is_overstow" in res[0]


def test_violations_of_container_joins_slot_constraint():
    """컨테이너 → 슬롯(ASSIGNED_TO) → 제약(VIOLATES) 조인. 제약 코드·근거 포함."""
    g = _g()
    res = g.violations_of("BL_R4_r1_t9")
    assert res, "위반이 있어야 함"
    codes = [r["code"] for r in res]
    assert "SOLAS_VI" in codes        # C_COL_WT → SOLAS_VI
    assert any("145" in (r.get("rule") or "") for r in res)


def test_violations_of_clean_container_is_empty():
    g = _g()
    assert g.violations_of("BL_R4_r0_t0") == []


def test_constraint_lookup_by_code():
    g = _g()
    c = g.constraint("SOLAS_VI")
    assert c["source"] == "SOLAS"
    assert "145" in c["rule"]


def test_ask_routes_relation_question():
    """자연어 질의 → 템플릿 라우팅 → graph 근거 반환(Evidence 계약)."""
    g = _g()
    res = g.ask("BL_R4_r1_t9 컨테이너는 무슨 규정을 위반했나?")
    assert res["sources"], "근거가 비어 있음"
    assert res["sources"][0]["type"] == "graph"
    assert "SOLAS" in res["answer"]

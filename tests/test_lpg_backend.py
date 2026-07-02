"""T28 · spec 07 — LPG 백엔드 팩토리(Neo4j ↔ CSV 폴백) TDD.

Neo4j가 떠 있으면 그래프DB로, 없으면 CSV 폴백으로 자동 전환한다.
인터페이스(stacked_on·violations_of·violations_in_round·constraint)는 동일.
라이브 Neo4j 없는 환경에서도 폴백·상태 보고가 깨지지 않아야 한다.
"""
import pytest

pytestmark = pytest.mark.rl_data


def test_status_reports_backend():
    from snct.knowledge.lpg import lpg_status
    s = lpg_status()
    assert s["backend"] in ("neo4j", "csv")
    assert "neo4j_connected" in s


def test_get_lpg_falls_back_to_csv_when_neo4j_down():
    """Neo4j 미연결 환경 → CSV LPGGraph 반환, 질의는 정상 동작."""
    from snct.knowledge.lpg import get_lpg, lpg_status
    if lpg_status()["neo4j_connected"]:
        pytest.skip("Neo4j가 실제로 연결됨 — 폴백 케이스 아님")
    g = get_lpg()
    from snct.knowledge.lpg_csv import LPGGraph
    assert isinstance(g, LPGGraph)
    # 동일 인터페이스로 질의 가능
    assert g.violations_in_round("BL", 4)


def test_backend_interface_is_uniform():
    """선택된 백엔드는 항상 동일 메서드 4종을 제공한다(덕타이핑)."""
    from snct.knowledge.lpg import get_lpg
    g = get_lpg()
    for m in ("stacked_on", "violations_of", "violations_in_round", "constraint"):
        assert callable(getattr(g, m))


def test_neo4j_lpg_unavailable_is_graceful():
    """Neo4jLPG는 서버가 없으면 is_available()=False (예외 없이)."""
    from snct.knowledge.lpg_neo4j import Neo4jLPG
    backend = Neo4jLPG(uri="bolt://localhost:7687", user="neo4j", password="wrongpass")
    assert backend.is_available() is False

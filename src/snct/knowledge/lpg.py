"""T28 · spec 07 — LPG 백엔드 팩토리 (Neo4j ↔ CSV 폴백).

Neo4j가 떠 있으면 그래프DB, 없으면 CSV 폴백을 반환한다. 상위 코드(explain·locator)는
동일 인터페이스(stacked_on·violations_of·violations_in_round·constraint)만 의존.
"""
from __future__ import annotations
import os


def _neo4j_backend():
    """환경설정으로 Neo4j 백엔드 인스턴스 생성. 연결되면 반환, 아니면 None."""
    try:
        from snct.knowledge.lpg_neo4j import Neo4jLPG
        backend = Neo4jLPG()
        return backend if backend.is_available() else None
    except Exception:
        return None


def get_lpg(prefer_neo4j: bool = True):
    """LPG 질의 백엔드 반환. Neo4j 가용 시 Neo4jLPG, 아니면 CSV LPGGraph."""
    if prefer_neo4j and os.environ.get("NEO4J_URI"):
        backend = _neo4j_backend()
        if backend is not None:
            return backend
    from snct.knowledge.lpg_csv import LPGGraph
    return LPGGraph()


def lpg_status() -> dict:
    """현재 LPG 백엔드 상태(대시보드 표시용)."""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    connected = False
    try:
        from snct.knowledge.lpg_neo4j import Neo4jLPG
        connected = Neo4jLPG().is_available()
    except Exception:
        connected = False
    return {
        "backend": "neo4j" if connected else "csv",
        "neo4j_connected": connected,
        "neo4j_uri": uri,
    }

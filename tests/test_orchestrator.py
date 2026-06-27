"""
orchestrator 스모크/유닛 테스트 — 교정 재시도·도구 에러격리·router 호환성 검증.
배치: tests/test_orchestrator.py
실행: PYTHONPATH=src pytest tests/test_orchestrator.py -v
      (langgraph 미설치여도 순수 파이썬 폴백으로 통과)
"""
import pytest
from snct.knowledge import orchestrator as orch


def test_returns_router_compatible_shape():
    """orchestrator.answer 가 기존 router.answer 와 호환되는 키를 반환한다."""
    r = orch.answer("DG 컨테이너 격리 규정 알려줘")
    assert {"answer", "sources", "used_path"}.issubset(r.keys())   # router.py 대체 호환
    assert isinstance(r["answer"], str)
    assert isinstance(r["sources"], list)


def test_routing_sql_for_listing():
    s = orch.route_node({"question": "빈 슬롯 목록 보여줘"})
    assert "sql" in s["routes"]


def test_routing_graph_for_regulation():
    s = orch.route_node({"question": "DG 격리 규정이 왜 필요해?"})
    assert "graph" in s["routes"]


def test_routing_default_when_no_match():
    s = orch.route_node({"question": "asldkfj"})
    assert s["routes"] == ["doc", "graph"]


def test_retry_escalates_routes_and_k():
    """교정 재시도: 시도할수록 경로 확장 + k 증가 (같은 재시도 금지)."""
    assert orch._retrieval_plan(0, ["doc"]) == (["doc"], 3)
    routes1, k1 = orch._retrieval_plan(1, ["doc"])
    assert set(routes1) == {"doc", "sql", "graph"} and k1 == 6
    routes2, k2 = orch._retrieval_plan(2, ["doc"])
    assert k2 == 10 and k2 > k1


def test_error_isolation_one_backend_down(monkeypatch):
    """도구 하나가 예외를 던져도 전체가 죽지 않고 답을 반환한다."""
    def boom(*a, **k):
        raise RuntimeError("backend down")
    monkeypatch.setattr(orch, "graph_ask", boom)        # graph 백엔드 장애 가정
    r = orch.answer("DG 격리 규정 관계 알려줘")           # graph 경로 포함 질의
    assert isinstance(r["answer"], str)                 # 예외 전파 없이 정상 반환


def test_faithfulness_zero_when_no_evidence(monkeypatch):
    """근거가 0건이면 faithfulness=0 (그라운딩 실패를 점수로 드러냄)."""
    monkeypatch.setattr(orch, "doc_retrieve", lambda q, k=3: [])
    monkeypatch.setattr(orch, "sql_ask", lambda q: {"sources": []})
    monkeypatch.setattr(orch, "graph_ask", lambda q: {"sources": []})
    s = {"question": "존재하지 않는 정보", "retries": 0}
    s = orch.faithfulness_node(orch.fuse_node(orch.retrieve_node(orch.route_node(s))))
    assert s["faithfulness"] == 0


def test_number_grounding_no_false_positive():
    """단어경계 매칭: '20'이 근거의 '2024'에 오탐 그라운딩되지 않는다."""
    s = {
        "question": "x",
        "evidence": [{"type": "doc", "ref": "D1", "snippet": "기준 연도는 2024년이다"}],
        "draft": "권장 풍속 한계는 20m/s 이다",
    }
    s = orch.faithfulness_node(s)
    assert s["faithfulness"] == 0        # '20'은 근거에 단어로 존재하지 않음


def test_evidence_dedup(monkeypatch):
    """동일 (type,ref) 근거는 1건으로 합쳐진다."""
    dup = [{"type": "doc", "ref": "SOP-001", "snippet": "a"}]
    monkeypatch.setattr(orch, "doc_retrieve", lambda q, k=3: dup * 3)
    monkeypatch.setattr(orch, "graph_ask", lambda q: {"sources": []})
    monkeypatch.setattr(orch, "sql_ask", lambda q: {"sources": []})
    s = orch.retrieve_node({"question": "안전 규칙", "routes": ["doc"], "retries": 0})
    assert len(s["evidence"]) == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

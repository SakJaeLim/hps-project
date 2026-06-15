"""T07 · spec 02 — TDD Red(미구현). 구현되면 xfail 제거 → Green 이어야 함."""
import pytest

@pytest.mark.xfail(reason="TDD Red — T07 미구현", strict=False)
def test_agent_flow():
    from snct.agents.graph import build_graph
    g = build_graph()
    out = g.invoke({"request": {"vessel_id": "V-1"}})
    assert "recommendation" in out or "rationale" in out

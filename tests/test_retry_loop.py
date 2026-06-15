"""T08 · spec 02 — TDD Red(미구현). 구현되면 xfail 제거 → Green 이어야 함."""
import pytest

@pytest.mark.xfail(reason="TDD Red — T08 미구현", strict=False)
def test_retry_loop():
    from snct.agents.graph import build_graph
    out = build_graph().invoke({"request": {"vessel_id": "V-1"}, "force_violation": True})
    assert out.get("retries", 0) >= 1  # 위반 시 계획 재시도

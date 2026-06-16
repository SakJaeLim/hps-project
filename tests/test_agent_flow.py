"""T07 · spec 02 — TDD Green."""
import pytest

def test_agent_flow():
    from snct.agents.graph import build_graph
    run_pipeline = build_graph()
    out = run_pipeline(question="DG 위험물 적재 규정", vessel_id="V-1")
    
    assert out.plan is not None
    assert isinstance(out.rationale, str)
    assert len(out.checks) > 0

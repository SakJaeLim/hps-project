"""T08 · spec 02 — TDD Green."""
import pytest

def test_retry_loop(monkeypatch):
    from snct.agents.graph import build_graph
    from snct.ontology.graph import Ontology
    from snct.common.schema import Violation
    
    # Force a violation to trigger retry
    def mock_validate(self, ys, plan):
        return [Violation(rule="mock", severity="error", container_id="test")]
    monkeypatch.setattr(Ontology, "validate", mock_validate)
    
    run_pipeline = build_graph()
    rec = run_pipeline(question="test", vessel_id="V-1")
    
    assert any("retries=2" in check for check in rec.checks)

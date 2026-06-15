"""T12 · spec 07 — TDD Red(미구현). 구현되면 xfail 제거 → Green 이어야 함."""
import pytest

@pytest.mark.xfail(reason="TDD Red — T12 미구현", strict=False)
def test_explain():
    from snct.knowledge.explain import explain
    from snct.common.schema import CandidatePlan
    txt = explain(CandidatePlan(engine="greedy"), [], evidence=[{"type":"doc","ref":"SOP","snippet":"..."}])
    assert isinstance(txt, str) and txt

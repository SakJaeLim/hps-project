"""T12 · spec 07 — TDD Green."""
import pytest

def test_explain():
    from snct.knowledge.explain import explain
    from snct.common.schema import CandidatePlan
    txt = explain(CandidatePlan(engine="greedy"), [], evidence=[{"type":"doc","ref":"SOP","snippet":"..."}])
    assert isinstance(txt, str) and txt

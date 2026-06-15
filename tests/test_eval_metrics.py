"""T15 · spec 04 — 평가 하니스(재취급·위반율·속도). TDD Red(미구현)."""
import pytest

@pytest.mark.xfail(reason="TDD Red — T15 평가 하니스 미구현", strict=False)
def test_eval_metrics():
    from snct.eval.harness import evaluate
    m = evaluate("greedy", instances=5, seed=0)
    assert {"rehandling", "violation_rate", "latency_s"} <= set(m)
    assert m["violation_rate"] < 0.01 and m["latency_s"] <= 5

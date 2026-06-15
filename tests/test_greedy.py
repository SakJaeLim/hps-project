"""T05 · spec 00 — TDD Red(미구현). 구현되면 xfail 제거 → Green 이어야 함."""
import pytest

@pytest.mark.xfail(reason="TDD Red — T05 미구현", strict=False)
def test_greedy():
    from snct.engine.greedy import GreedyStrategy
    from snct.common.schema import YardState
    cp = GreedyStrategy().plan(YardState(slots=[], queue=[]))
    assert cp.engine == "greedy" and isinstance(cp.assignments, list)

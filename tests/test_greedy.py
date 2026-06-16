"""T05 · spec 00 — TDD Green."""
import pytest

def test_greedy():
    from snct.engine.greedy import GreedyStrategy
    from snct.common.schema import YardState, Slot, Container
    ys = YardState(
        slots=[Slot(bay=1, row=1, tier=1, max_stack_weight=30.0)],
        queue=[Container(id="C1", weight_ton=20.0, size="40", type="GP", pod="LAX")]
    )
    cp = GreedyStrategy().plan(ys)
    assert cp.engine == "greedy"
    assert isinstance(cp.assignments, list)
    assert len(cp.assignments) > 0

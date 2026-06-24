"""T06 · spec ADR-0001 — TDD Green."""
import pytest

def test_rl_strategy():
    from snct.engine.base import get_strategy
    from snct.common.schema import YardState, Slot, Container
    ys = YardState(
        slots=[Slot(bay=1, row=1, tier=1, max_stack_weight=30.0)],
        queue=[Container(id="C1", weight_ton=20.0, size="40", type="GP", pod="LAX")]
    )
    cp = get_strategy("rl").plan(ys)
    assert cp.engine.startswith("rl")  # 원우 모델 통합 → 유효 배정

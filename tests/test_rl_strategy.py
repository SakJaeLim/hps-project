"""T06 · spec ADR-0001 — TDD Red(미구현). 구현되면 xfail 제거 → Green 이어야 함."""
import pytest

@pytest.mark.xfail(reason="TDD Red — T06 미구현", strict=False)
def test_rl_strategy():
    from snct.engine.base import get_strategy
    from snct.common.schema import YardState
    cp = get_strategy("rl").plan(YardState(slots=[], queue=[]))
    assert cp.engine == "rl"  # 원우 모델 통합 → 유효 배정

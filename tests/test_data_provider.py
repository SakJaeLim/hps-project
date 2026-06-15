"""T02 · spec 03,06 — TDD Red(미구현). 구현되면 xfail 제거 → Green 이어야 함."""
import pytest

def test_data_provider():
    from snct.data.provider import get_provider
    from snct.common.schema import YardState
    ys = get_provider("simulated").get_yard_state("V-1")
    assert isinstance(ys, YardState) and ys.slots and ys.queue

"""T04 · spec 01 — TDD Red(미구현). 구현되면 xfail 제거 → Green 이어야 함."""
import pytest

@pytest.mark.xfail(reason="TDD Red — T04 미구현", strict=False)
def test_constraints():
    from snct.ontology.graph import Ontology
    viols = Ontology().validate(plan=None)  # 제약 5종 점검 → 위반 목록
    assert isinstance(viols, list)

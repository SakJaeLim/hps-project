"""T03 · spec 01 — TDD Red(미구현). 구현되면 xfail 제거 → Green 이어야 함."""
import pytest

@pytest.mark.xfail(reason="TDD Red — T03 미구현", strict=False)
def test_ontology_schema():
    from snct.ontology.graph import Ontology
    onto = Ontology()
    assert onto.driver is not None  # Neo4j 스키마 v1 적재 후 노드 조회 가능

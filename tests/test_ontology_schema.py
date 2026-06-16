"""T03 · spec 01 — TDD Green."""
import pytest

def test_ontology_schema():
    from snct.ontology.graph import Ontology
    onto = Ontology()
    assert onto.G is not None  # in-memory graph is initialized
    assert len(onto.G.nodes) > 0  # Should have initial rules or nodes

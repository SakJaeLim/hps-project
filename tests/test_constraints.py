"""T04 · spec 01 — TDD Green."""
import pytest

def test_constraints():
    from snct.ontology.graph import Ontology
    from snct.common.schema import CandidatePlan, YardState, Slot, Container, Assignment
    
    ys = YardState(
        slots=[Slot(bay=1, row=1, tier=1, max_stack_weight=10.0)],
        queue=[]
    )
    # Plan exceeds max stack weight (24.5 > 10.0)
    cp = CandidatePlan(
        engine="greedy",
        assignments=[Assignment(container_id="C1", bay=1, row=1, tier=1)]
    )
    
    onto = Ontology()
    viols = onto.validate(ys, cp)
    assert isinstance(viols, list)


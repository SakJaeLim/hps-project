"""T01 · specs/03 — 캐노니컬 계약이 정의·인스턴스화 되는가. (구현됨: GREEN)"""
def test_contracts_instantiate():
    from snct.common.schema import (Container, Slot, YardState, Assignment,
                                     CandidatePlan, Violation, Recommendation)
    c = Container(id="SNCT-1", weight_ton=24.5, size="40", type="GP", pod="LAX", dg=False)
    s = Slot(bay=3, row=2, tier=4, max_stack_weight=30.0, dg_allowed=True)
    ys = YardState(slots=[s], queue=[c])
    cp = CandidatePlan(assignments=[Assignment("SNCT-1", 3, 2, 4)], engine="greedy")
    rec = Recommendation(plan=cp, violations=[Violation(rule="stack_weight", container_id="SNCT-1")],
                         rationale="ok", checks=["stack_weight"])
    assert ys.queue[0].id == "SNCT-1"
    assert cp.engine == "greedy" and cp.assignments[0].bay == 3
    assert rec.plan is cp and rec.violations[0].rule == "stack_weight"

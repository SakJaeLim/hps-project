"""L2 NetworkX 온톨로지 + 제약 Cypher 5종. (Neo4j → NetworkX 인메모리 대체)
별도 DB 서버 없이 동일한 제약 검증을 수행한다."""
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    
from snct.common.schema import YardState, CandidatePlan, Violation, Assignment


class Ontology:
    """In-memory knowledge graph for container terminal stowage constraints."""

    def __init__(self):
        if NETWORKX_AVAILABLE:
            self.G = nx.DiGraph()
            self._init_rules()

    def _init_rules(self):
        """Initialize constraint rule nodes in the knowledge graph."""
        rules = [
            ("RULE_STACK_WEIGHT", {"desc": "적재 중량 합계 ≤ 슬롯 최대 중량", "severity": "error"}),
            ("RULE_DG_BAY", {"desc": "DG 컨테이너 → DG 허용 Bay만 배치", "severity": "error"}),
            ("RULE_REEFER_BAY", {"desc": "Reefer → 전원공급 가능 Bay만 배치", "severity": "error"}),
            ("RULE_DISCHARGE", {"desc": "양하순서 역전 적재 금지", "severity": "warning"}),
            ("RULE_REHANDLING", {"desc": "재취급(Rehandling) 충돌 탐지", "severity": "warning"}),
        ]
        for rule_id, attrs in rules:
            self.G.add_node(rule_id, type="rule", **attrs)

    def build_graph(self, yard: YardState, plan: CandidatePlan):
        """Build the ontology graph from yard state and candidate plan."""
        # Add slot nodes
        for slot in yard.slots:
            sid = f"SLOT_{slot.bay}_{slot.row}_{slot.tier}"
            self.G.add_node(sid, type="slot",
                            bay=slot.bay, row=slot.row, tier=slot.tier,
                            max_stack_weight=slot.max_stack_weight,
                            dg_allowed=slot.dg_allowed,
                            reefer_capable=slot.reefer_capable)

        # Add container nodes and ASSIGNED_TO edges from the plan
        container_map = {c.id: c for c in yard.queue}
        for assignment in plan.assignments:
            cid = assignment.container_id
            c = container_map.get(cid)
            if c:
                self.G.add_node(cid, type="container",
                                weight=c.weight_ton, dg=c.dg, reefer=c.reefer,
                                pod=c.pod, discharge_order=c.discharge_order)
                sid = f"SLOT_{assignment.bay}_{assignment.row}_{assignment.tier}"
                self.G.add_edge(cid, sid, relation="ASSIGNED_TO")

        # Build STACKED_ON relationships (same bay+row, higher tier stacked on lower)
        slot_assignments: dict[tuple[int, int], list[tuple[int, str]]] = {}
        for a in plan.assignments:
            key = (a.bay, a.row)
            slot_assignments.setdefault(key, []).append((a.tier, a.container_id))

        for key, stack in slot_assignments.items():
            stack.sort(key=lambda x: x[0])  # sort by tier ascending
            for i in range(1, len(stack)):
                lower_cid = stack[i - 1][1]
                upper_cid = stack[i][1]
                self.G.add_edge(upper_cid, lower_cid, relation="STACKED_ON")

    def _check_stack_weight(self, yard: YardState, plan: CandidatePlan) -> list[Violation]:
        """R1: 적재 중량 합계 ≤ 슬롯 최대 중량."""
        violations = []
        container_map = {c.id: c for c in yard.queue}
        slot_map = {(s.bay, s.row, s.tier): s for s in yard.slots}

        # Accumulate weight per slot
        slot_weights: dict[tuple[int, int, int], float] = {}
        for a in plan.assignments:
            key = (a.bay, a.row, a.tier)
            c = container_map.get(a.container_id)
            if c:
                slot_weights[key] = slot_weights.get(key, 0.0) + c.weight_ton

        for key, total_weight in slot_weights.items():
            slot = slot_map.get(key)
            if slot and total_weight > slot.max_stack_weight:
                violations.append(Violation(
                    rule="stack_weight",
                    container_id=str(key),
                    detail=f"총 중량 {total_weight:.1f}t > 최대 {slot.max_stack_weight:.1f}t",
                    severity="error",
                ))
        return violations

    def _check_dg_bay(self, yard: YardState, plan: CandidatePlan) -> list[Violation]:
        """R2: DG 컨테이너 → DG 허용 Bay만."""
        violations = []
        container_map = {c.id: c for c in yard.queue}
        slot_map = {(s.bay, s.row, s.tier): s for s in yard.slots}

        for a in plan.assignments:
            c = container_map.get(a.container_id)
            slot = slot_map.get((a.bay, a.row, a.tier))
            if c and c.dg and slot and not slot.dg_allowed:
                violations.append(Violation(
                    rule="dg_bay",
                    container_id=a.container_id,
                    detail=f"DG 컨테이너가 DG 비허용 슬롯 BAY{a.bay}-ROW{a.row}-TIER{a.tier}에 배치됨",
                    severity="error",
                ))
        return violations

    def _check_reefer_bay(self, yard: YardState, plan: CandidatePlan) -> list[Violation]:
        """R3: Reefer → 전원공급 가능 Bay만."""
        violations = []
        container_map = {c.id: c for c in yard.queue}
        slot_map = {(s.bay, s.row, s.tier): s for s in yard.slots}

        for a in plan.assignments:
            c = container_map.get(a.container_id)
            slot = slot_map.get((a.bay, a.row, a.tier))
            if c and c.reefer and slot and not slot.reefer_capable:
                violations.append(Violation(
                    rule="reefer_bay",
                    container_id=a.container_id,
                    detail=f"Reefer 컨테이너가 전원 미지원 슬롯 BAY{a.bay}에 배치됨",
                    severity="error",
                ))
        return violations

    def _check_discharge(self, yard: YardState, plan: CandidatePlan) -> list[Violation]:
        """R4: 양하순서 역전 적재 금지 — 먼저 내릴 컨테이너가 아래에 있으면 위반."""
        violations = []
        container_map = {c.id: c for c in yard.queue}

        # Group by (bay, row) stack
        stacks: dict[tuple[int, int], list[tuple[int, str]]] = {}
        for a in plan.assignments:
            key = (a.bay, a.row)
            stacks.setdefault(key, []).append((a.tier, a.container_id))

        for key, stack in stacks.items():
            stack.sort(key=lambda x: x[0])  # tier ascending (bottom → top)
            for i in range(len(stack)):
                for j in range(i + 1, len(stack)):
                    lower_c = container_map.get(stack[i][1])
                    upper_c = container_map.get(stack[j][1])
                    if lower_c and upper_c:
                        # If lower container discharges earlier, it's blocked by upper
                        if lower_c.discharge_order < upper_c.discharge_order:
                            violations.append(Violation(
                                rule="discharge",
                                container_id=stack[i][1],
                                detail=f"{stack[i][1]}(양하순서 {lower_c.discharge_order})이 "
                                       f"{stack[j][1]}(양하순서 {upper_c.discharge_order}) 아래에 위치 — 양하 역전",
                                severity="warning",
                            ))
        return violations

    def _check_rehandling(self, yard: YardState, plan: CandidatePlan) -> list[Violation]:
        """R5: BLOCKS 관계 탐지 — 재취급 충돌 카운트."""
        violations = []
        container_map = {c.id: c for c in yard.queue}

        stacks: dict[tuple[int, int], list[tuple[int, str]]] = {}
        for a in plan.assignments:
            key = (a.bay, a.row)
            stacks.setdefault(key, []).append((a.tier, a.container_id))

        for key, stack in stacks.items():
            stack.sort(key=lambda x: x[0])
            for i in range(len(stack)):
                for j in range(i + 1, len(stack)):
                    lower_c = container_map.get(stack[i][1])
                    upper_c = container_map.get(stack[j][1])
                    if lower_c and upper_c and lower_c.pod != upper_c.pod:
                        # Different POD stacked → potential rehandling
                        violations.append(Violation(
                            rule="rehandling",
                            container_id=stack[j][1],
                            detail=f"{stack[j][1]}(POD:{upper_c.pod})이 "
                                   f"{stack[i][1]}(POD:{lower_c.pod}) 위에 적재 — 재취급 위험",
                            severity="warning",
                        ))
        return violations

    def validate(self, yard: YardState, plan: CandidatePlan) -> list[Violation]:
        """제약 5종 전체 검증. 위반 목록 반환."""
        if not NETWORKX_AVAILABLE:
            return []
        self.G = nx.DiGraph()
        self._init_rules()
        self.build_graph(yard, plan)

        violations = []
        violations.extend(self._check_stack_weight(yard, plan))
        violations.extend(self._check_dg_bay(yard, plan))
        violations.extend(self._check_reefer_bay(yard, plan))
        violations.extend(self._check_discharge(yard, plan))
        violations.extend(self._check_rehandling(yard, plan))
        return violations

    def get_graph_summary(self) -> dict:
        """Return a summary of the current ontology graph."""
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "containers": len([n for n, d in self.G.nodes(data=True) if d.get("type") == "container"]),
            "slots": len([n for n, d in self.G.nodes(data=True) if d.get("type") == "slot"]),
            "rules": len([n for n, d in self.G.nodes(data=True) if d.get("type") == "rule"]),
        }

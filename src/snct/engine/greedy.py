"""기준선 전략 — 단순 휴리스틱(specs/00 · ADR-0001). RL/CP 비교의 하한선·폴백.
규칙: 무거운 컨테이너 하단 · 동일 POD 묶기 · DG/Reefer bay 제약 · 반출순서 보존."""
from snct.engine.base import StowageStrategy
from snct.common.schema import YardState, CandidatePlan, Assignment


class GreedyStrategy(StowageStrategy):
    name = "greedy"

    def plan(self, yard: YardState) -> CandidatePlan:
        assignments: list[Assignment] = []
        # Track occupied slots (bay, row, tier) -> container_id
        occupied: dict[tuple[int, int, int], str] = {}
        for slot in yard.slots:
            if slot.occupied_by:
                occupied[(slot.bay, slot.row, slot.tier)] = slot.occupied_by

        # Sort containers: heavier first (Heavy-Down), then by POD grouping
        sorted_queue = sorted(
            yard.queue,
            key=lambda c: (-c.weight_ton, c.pod, c.discharge_order),
        )

        for container in sorted_queue:
            best_slot = None
            best_score = float("-inf")

            for slot in yard.slots:
                key = (slot.bay, slot.row, slot.tier)
                if key in occupied:
                    continue  # slot already taken

                score = 0.0

                # Rule 1: DG constraint — DG container MUST go to dg_allowed slot
                if container.dg and not slot.dg_allowed:
                    continue  # hard constraint → skip
                if container.dg and slot.dg_allowed:
                    score += 100  # bonus for correct DG placement

                # Rule 2: Reefer constraint — Reefer MUST go to reefer_capable slot
                if container.reefer and not slot.reefer_capable:
                    continue  # hard constraint → skip
                if container.reefer and slot.reefer_capable:
                    score += 100

                # Rule 3: Heavy-Down — prefer lower tiers for heavier containers
                # Lower tier number = lower position = better for heavy
                score += (10 - slot.tier) * (container.weight_ton / 5.0)

                # Rule 4: Weight capacity — must not exceed max stack weight
                if container.weight_ton > slot.max_stack_weight:
                    continue  # hard constraint

                # Bonus: POD grouping — prefer slots near same-POD containers
                for (b, r, t), cid in occupied.items():
                    for q_container in yard.queue:
                        if q_container.id == cid and q_container.pod == container.pod:
                            if b == slot.bay:
                                score += 20  # same bay bonus
                            if abs(r - slot.row) <= 1:
                                score += 10  # adjacent row bonus

                if score > best_score:
                    best_score = score
                    best_slot = slot

            if best_slot:
                key = (best_slot.bay, best_slot.row, best_slot.tier)
                occupied[key] = container.id
                assignments.append(
                    Assignment(
                        container_id=container.id,
                        bay=best_slot.bay,
                        row=best_slot.row,
                        tier=best_slot.tier,
                    )
                )

        # Compute objective metrics
        rehandling_count = 0
        weight_imbalance = 0.0
        if assignments:
            weights = [a.tier for a in assignments]
            weight_imbalance = max(weights) - min(weights) if weights else 0

        return CandidatePlan(
            assignments=assignments,
            engine="greedy",
            objective={
                "rehandling": rehandling_count,
                "weight_imbalance": weight_imbalance,
                "assigned": len(assignments),
                "unassigned": len(yard.queue) - len(assignments),
            },
        )

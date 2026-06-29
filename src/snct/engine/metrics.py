"""적재 계획 KPI 라이브 계산 — 엔진(greedy/RL) 무관, 배정 결과만으로 산출.

scratch/engine_comparison.py 의 compute_kpi 를 정식 모듈로 승격한 것이다.
xAI 화면이 RL 전용 배치 파일(reward_decomp/kpi.csv)에 의존하지 않고,
화면에 실제로 표시되는 그 계획(greedy/RL 공통)에서 동일한 KPI를 즉시 계산한다.
→ 배치-라이브 불일치 제거 + greedy 도 동일하게 설명 가능.
"""
from __future__ import annotations
from collections import defaultdict

import numpy as np

from snct.common.schema import YardState, CandidatePlan

# 양하항 반출 순서(근항=높은 번호가 아래 Tier에 있어야 정상). compute KPI 의 POD 역전 판정 기준.
POD_ORDER = {"ROTTERDAM": 1, "COLOMBO": 2, "SINGAPORE": 3, "NINGBO": 4, "SHANGHAI": 5, "BUSAN": 6}

# UI 라벨 + 방향(higher_better) + 단위. 대시보드 KPI 카드/색상에 사용.
KPI_META = {
    "assign_rate":        {"label": "배정률",            "unit": "%", "higher_better": True},
    "heavy_down_rate":    {"label": "Heavy-Down 준수율", "unit": "%", "higher_better": True},
    "pod_violation_rate": {"label": "POD 역전 위반률",   "unit": "%", "higher_better": False},
    "row_weight_std":     {"label": "Row 무게편차",      "unit": "t", "higher_better": False},
    "dg_violations":      {"label": "DG 제약위반",       "unit": "건", "higher_better": False},
    "avg_weight":         {"label": "평균 배정무게",     "unit": "t", "higher_better": None},
}


def compute_plan_kpi(yard: YardState, plan: CandidatePlan) -> dict:
    """배정 결과(plan)로 적재 품질 KPI 를 계산. 엔진 종류와 무관하게 동작한다."""
    n_containers = len(yard.queue)
    n_assigned = len(plan.assignments)
    cntr_map = {c.id: c for c in yard.queue}

    # 1) 배정률
    assign_rate = n_assigned / max(n_containers, 1) * 100.0

    # row → [(tier, weight)] 묶음
    row_tiers: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for a in plan.assignments:
        c = cntr_map.get(a.container_id)
        row_tiers[a.row].append((a.tier, c.weight_ton if c else 0.0))

    # 2) Heavy-Down 준수율: 같은 row 에서 아래 tier 가 더 무겁거나 같아야 정상
    hd_ok, hd_total = 0, 0
    for pairs in row_tiers.values():
        ps = sorted(pairs, key=lambda x: x[0])
        for i in range(len(ps) - 1):
            hd_total += 1
            if ps[i][1] >= ps[i + 1][1]:
                hd_ok += 1
    heavy_down_rate = (hd_ok / hd_total * 100.0) if hd_total else 100.0

    # 3) POD 역전 위반률: 아래(낮은 tier)가 근항(POD_ORDER 큰 값)이어야 정상
    pod_viol, pod_total = 0, 0
    for row in row_tiers:
        seq = sorted(
            [(a.tier, cntr_map.get(a.container_id)) for a in plan.assignments if a.row == row],
            key=lambda x: x[0],
        )
        for i in range(len(seq) - 1):
            c1, c2 = seq[i][1], seq[i + 1][1]
            if c1 and c2:
                pod_total += 1
                if POD_ORDER.get(str(c1.pod).upper(), 0) < POD_ORDER.get(str(c2.pod).upper(), 0):
                    pod_viol += 1
    pod_violation_rate = (pod_viol / pod_total * 100.0) if pod_total else 0.0

    # 4) Row 별 총무게 표준편차 (무게중심 불균형)
    row_weights: dict[int, float] = defaultdict(float)
    for a in plan.assignments:
        c = cntr_map.get(a.container_id)
        if c:
            row_weights[a.row] += c.weight_ton
    wt_vals = list(row_weights.values()) or [0.0]
    row_weight_std = float(np.std(wt_vals))

    # 5) 평균 배정무게
    assigned_ids = {a.container_id for a in plan.assignments}
    total_wt = sum(c.weight_ton for c in yard.queue if c.id in assigned_ids)
    avg_weight = total_wt / max(n_assigned, 1)

    # 6) DG 제약 위반 수 (DG 컨테이너가 dg_allowed 아닌 슬롯에 놓인 경우)
    slot_map = {(s.bay, s.row, s.tier): s for s in yard.slots}
    dg_violations = 0
    for a in plan.assignments:
        c = cntr_map.get(a.container_id)
        s = slot_map.get((a.bay, a.row, a.tier))
        if c and s and getattr(c, "dg", False) and not s.dg_allowed:
            dg_violations += 1

    return {
        "assign_rate": round(assign_rate, 1),
        "heavy_down_rate": round(heavy_down_rate, 1),
        "pod_violation_rate": round(pod_violation_rate, 1),
        "row_weight_std": round(row_weight_std, 2),
        "avg_weight": round(avg_weight, 2),
        "dg_violations": dg_violations,
        "n_assigned": n_assigned,
        "n_unassigned": n_containers - n_assigned,
        "n_total": n_containers,
    }

# 01. 온톨로지 명세

## 노드
Vessel, Bay, Row, Tier, Slot, Container, Berth, Yard, YardBlock

## 관계
(:Container)-[:ASSIGNED_TO]->(:Slot)
(:Slot)-[:IN_BAY]->(:Bay)-[:OF_VESSEL]->(:Vessel)
(:Container)-[:STACKED_ON]->(:Container)
(:Container)-[:BLOCKS]->(:Container)        // 재취급 관계
(:Vessel)-[:AT_BERTH]->(:Berth)

## 핵심 속성
Container{id, weight_ton, size, type, pod, dg, reefer, discharge_order}
Slot{bay,row,tier, max_stack_weight, dg_allowed, reefer_capable, size_class}

## 제약 Cypher (5종)
1) 적재중량: Σ STACKED_ON weight ≤ Slot.max_stack_weight
2) DG bay: dg=true → Slot.dg_allowed=true
3) Reefer bay: reefer=true → Slot.reefer_capable=true
4) 반출 순서: discharge_order 역전 적재 금지(아래가 먼저 나가면 위반)
5) 재취급: BLOCKS 관계 탐지 → 충돌 카운트

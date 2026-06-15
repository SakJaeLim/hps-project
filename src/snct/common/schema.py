"""캐노니컬 스키마(계약) — 출처 무관 공통 모델. specs/03 · specs/07."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Container:
    id: str
    weight_ton: float
    size: str          # "20" | "40" | "45"
    type: str          # "GP" | "RF" | "DG" ...
    pod: str           # port of discharge
    dg: bool = False
    reefer: bool = False
    discharge_order: int = 0

@dataclass
class Slot:
    bay: int; row: int; tier: int
    max_stack_weight: float
    dg_allowed: bool = False
    reefer_capable: bool = False
    size_class: str = "40"
    occupied_by: str | None = None

@dataclass
class YardState:
    slots: list[Slot] = field(default_factory=list)
    queue: list[Container] = field(default_factory=list)


@dataclass
class Assignment:
    container_id: str
    bay: int; row: int; tier: int

@dataclass
class CandidatePlan:
    """엔진(RL/CP/Greedy) 출력 — 컨테이너→슬롯 배정안. (계약)"""
    assignments: list[Assignment] = field(default_factory=list)
    engine: str = ""                                  # "rl" | "cp" | "greedy"
    objective: dict = field(default_factory=dict)     # rehandling, weight_imbalance ...

@dataclass
class Violation:
    """제약 검증 결과 1건. (계약)"""
    rule: str                                         # stack_weight|dg_bay|reefer_bay|discharge|rehandling
    container_id: str
    detail: str = ""
    severity: str = "error"                           # error | warning

@dataclass
class Recommendation:
    """운영자/대시보드로 가는 최종 산출. (계약)"""
    plan: CandidatePlan
    violations: list[Violation] = field(default_factory=list)
    rationale: str = ""
    checks: list[str] = field(default_factory=list)

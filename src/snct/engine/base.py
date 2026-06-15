"""L3 적재 최적화 — 전략 교체 가능 인터페이스 (ADR-0001).
RL(현 엔진·원우 모델)·CP(조건부 향후 비교군)·Greedy(기준선)를 같은 계약으로 갈아끼운다."""
from abc import ABC, abstractmethod
from snct.common.schema import YardState, CandidatePlan

class StowageStrategy(ABC):
    name: str = "base"
    @abstractmethod
    def plan(self, yard: YardState) -> CandidatePlan: ...

def get_strategy(name: str = "rl") -> "StowageStrategy":
    """기본 엔진 = rl (ADR-0001). cp는 벤치마크 통과 시에만 기본으로 승격."""
    if name == "greedy":
        from snct.engine.greedy import GreedyStrategy
        return GreedyStrategy()
    elif name == "cp":
        from snct.engine.cp_sat import CPStrategy
        return CPStrategy()
    else:
        from snct.engine.rl_policy import RLStrategy
        return RLStrategy()

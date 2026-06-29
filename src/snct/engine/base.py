"""L3 적재 최적화 — 전략 교체 가능 인터페이스 (ADR-0001).
RL(현 엔진·원우 모델)·CP(조건부 향후 비교군)·Greedy(기준선)를 같은 계약으로 갈아끼운다."""
from abc import ABC, abstractmethod
from snct.common.schema import YardState, CandidatePlan

class StowageStrategy(ABC):
    name: str = "base"
    @abstractmethod
    def plan(self, yard: YardState) -> CandidatePlan: ...

def get_strategy(name: str = "rl", deterministic: bool = True) -> "StowageStrategy":
    """기본 엔진 = rl (ADR-0001). cp는 벤치마크 통과 시에만 기본으로 승격.
    deterministic: RL 정책 추론 방식. True=argmax(최적·재현성), False=표집(매 실행 다른 대안)."""
    if name == "greedy":
        from snct.engine.greedy import GreedyStrategy
        return GreedyStrategy()
    elif name == "cp":
        from snct.engine.cp_sat import CPStrategy
        return CPStrategy()
    elif name == "rl_sf":
        from snct.engine.rl_policy import RLStrategy
        return RLStrategy(model_type="SF", deterministic=deterministic)
    elif name == "rl_ef":
        from snct.engine.rl_policy import RLStrategy
        return RLStrategy(model_type="EF", deterministic=deterministic)
    else: # "rl" or "rl_bl"
        from snct.engine.rl_policy import RLStrategy
        return RLStrategy(model_type="BL", deterministic=deterministic)

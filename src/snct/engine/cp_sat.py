"""향후 비교군(현재 미채택) — OR-Tools CP-SAT 최적화 (ADR-0001).
RL vs CP head-to-head 벤치마크에서 CP가 '유의미하게' 우월할 때에만 기본 엔진으로 교체.
지금은 인터페이스만 확보하여 향후 비교를 코드 변경 최소로 가능케 한다."""
from snct.engine.base import StowageStrategy
from snct.common.schema import YardState, CandidatePlan

class CPStrategy(StowageStrategy):
    name = "cp"
    def plan(self, yard: YardState) -> CandidatePlan:
        raise NotImplementedError("향후: ortools.sat 모델링 — 벤치마크 후 채택 결정")

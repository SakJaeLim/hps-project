"""L1 데이터 어댑터. 상위 레이어는 DataProvider 인터페이스만 의존한다. (specs/03, specs/06)"""
from abc import ABC, abstractmethod
from snct.common.schema import YardState, Container, Slot

class DataProvider(ABC):
    @abstractmethod
    def get_yard_state(self, vessel_id: str) -> YardState: ...
    @abstractmethod
    def get_container_queue(self, vessel_id: str): ...

class SimulatedProvider(DataProvider):
    """선박제원 + 본선 플래닝 기준 기반 합성 데이터(기본 구현, 실데이터 도착 전)."""
    def get_yard_state(self, vessel_id: str) -> YardState:
        # Return a mock YardState with some slots and containers for verification
        slots = [
            Slot(bay=1, row=1, tier=1, max_stack_weight=30.0),
            Slot(bay=1, row=2, tier=1, max_stack_weight=30.0, reefer_capable=True),
            Slot(bay=1, row=3, tier=1, max_stack_weight=30.0, dg_allowed=True)
        ]
        queue = [
            Container(id="CNTR-001", weight_ton=24.5, size="40", type="GP", pod="LAX"),
            Container(id="CNTR-002", weight_ton=18.0, size="40", type="RF", pod="ROTTERDAM", reefer=True),
            Container(id="CNTR-003", weight_ton=12.0, size="20", type="DG", pod="SINGAPORE", dg=True)
        ]
        return YardState(slots=slots, queue=queue)
        
    def get_container_queue(self, vessel_id: str):
        return [
            Container(id="CNTR-001", weight_ton=24.5, size="40", type="GP", pod="LAX"),
            Container(id="CNTR-002", weight_ton=18.0, size="40", type="RF", pod="ROTTERDAM", reefer=True)
        ]

class LiveProvider(DataProvider):
    """실데이터(운영사) → 캐노니컬 스키마. specs/06_data_sources 참조.
    원천 파서는 src/snct/data/sources/* 에 둔다. (EDIFACT/CSV/XML/PDF)"""
    # 정형 원천 → 캐노니컬
    def load_baplie(self, path): raise NotImplementedError("EDIFACT BAPLIE → Container·Slot·ASSIGNED_TO")
    def load_coprar(self, path): raise NotImplementedError("EDIFACT COPRAR → 양·적하 작업목록(WorkOrder)")
    def load_movins(self, path): raise NotImplementedError("EDIFACT MOVINS → 적재 순서·세그리게이션")
    def load_bay_plan(self, path): raise NotImplementedError("Bay Plan → Vessel/Bay/Row/Tier/Slot 구조")
    def load_yard_inventory(self, path): raise NotImplementedError("Yard Inventory → YardState(점유·STACKED_ON)")
    def load_equipment_ops(self, path): raise NotImplementedError("Equipment Operation → 재취급 실측(평가)")
    def load_gate_tx(self, path): raise NotImplementedError("Gate Transaction → 컨테이너 큐(반입)")
    def load_ais(self, path): raise NotImplementedError("AIS → Vessel eta·status")
    # 캐노니컬 인터페이스
    def get_yard_state(self, vessel_id):
        slots = [Slot(bay=1, row=1, tier=1, max_stack_weight=30.0)]
        queue = [Container(id="CNTR-001", weight_ton=24.5, size="40", type="GP", pod="LAX")]
        return YardState(slots=slots, queue=queue)
    def get_container_queue(self, vessel_id):
        return [Container(id="CNTR-001", weight_ton=24.5, size="40", type="GP", pod="LAX")]

def get_provider(name: str = "simulated") -> DataProvider:
    return {"simulated": SimulatedProvider, "live": LiveProvider}[name]()

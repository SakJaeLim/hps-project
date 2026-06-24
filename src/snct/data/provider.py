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
        import os
        import re
        import pandas as pd
        
        # Determine target sheet name based on vessel_id
        v_upper = str(vessel_id).upper()
        if "LV1" in v_upper or "LEVEL1" in v_upper:
            sheet_name = "R1_Lv1"
        elif "LV2" in v_upper or "LEVEL2" in v_upper:
            sheet_name = "R2_Lv2"
        elif "LV3" in v_upper or "LEVEL3" in v_upper:
            sheet_name = "R3_Lv3"
        else:
            # Default to LV4 (10R x 10T)
            sheet_name = "R4_Lv4"
            
        base_dir = os.environ.get("SNCT_BASE_DIR", r"c:\Users\lione\Desktop\aSSIST\19_Project\12_hps-project-main")
        xlsx_path = os.path.join(base_dir, "data", "RL", "강화학습 결과 자료", "single_bay_6pod_ppo_v13_3way_BayPlan_Distributions_seed42.xlsx")
        
        if os.path.exists(xlsx_path):
            try:
                df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
                
                # Find row with header "Tier\Row"
                header_rows = df[df[0] == "Tier\\Row"].index.tolist()
                if len(header_rows) >= 2:
                    pod_header_idx = header_rows[0]
                    weight_header_idx = header_rows[1]
                    
                    # Read rows below pod header until we find "Col ΣWT"
                    pod_rows = []
                    for r in range(pod_header_idx + 1, len(df)):
                        first_val = str(df.iloc[r, 0]).strip()
                        if "Col" in first_val or pd.isna(df.iloc[r, 0]):
                            break
                        pod_rows.append(df.iloc[r])
                        
                    pod_df = pd.DataFrame(pod_rows)
                    pod_df.columns = df.iloc[pod_header_idx]
                    pod_df.set_index("Tier\\Row", inplace=True)
                    
                    # Filter columns to only include rows (e.g. R0, R1, etc.)
                    row_cols = [c for c in pod_df.columns if str(c).startswith("R")]
                    pod_df = pod_df[row_cols]
                    
                    # Read weight rows
                    weight_rows = []
                    for r in range(weight_header_idx + 1, len(df)):
                        first_val = str(df.iloc[r, 0]).strip()
                        if pd.isna(df.iloc[r, 0]) or first_val == "" or "선적" in first_val:
                            break
                        weight_rows.append(df.iloc[r])
                        
                    weight_df = pd.DataFrame(weight_rows)
                    weight_df.columns = df.iloc[weight_header_idx]
                    weight_df.set_index("Tier\\Row", inplace=True)
                    weight_df = weight_df[row_cols]
                    
                    n_rows = len(row_cols)
                    n_tiers = len(pod_df)
                    tier_labels = list(pod_df.index)[::-1]  # T0, T1, etc.
                    
                    slots = []
                    queue = []
                    
                    POD_ID_TO_NAME = {
                        1: "BUSAN",
                        2: "SHANGHAI",
                        3: "NINGBO",
                        4: "SINGAPORE",
                        5: "COLOMBO",
                        6: "ROTTERDAM"
                    }
                    
                    def parse_pod_id(val):
                        if pd.isna(val):
                            return 0
                        match = re.search(r'\((\d+)\)', str(val))
                        if match:
                            return int(match.group(1))
                        return 0
                        
                    # Create all slots
                    for r_idx in range(n_rows):
                        for t_idx in range(n_tiers):
                            slots.append(
                                Slot(
                                    bay=1,
                                    row=r_idx + 1,
                                    tier=t_idx + 1,
                                    max_stack_weight=145.0,
                                    dg_allowed=True,
                                    reefer_capable=True
                                )
                            )
                    
                    # Parse containers
                    cnt = 0
                    for t_idx, tier_label in enumerate(tier_labels):
                        for r_idx, col_label in enumerate(row_cols):
                            pod_val = pod_df.loc[tier_label, col_label]
                            pod_id = parse_pod_id(pod_val)
                            if pod_id > 0:
                                wt_val = weight_df.loc[tier_label, col_label]
                                weight = float(wt_val) if not pd.isna(wt_val) else 15.0
                                pod_name = POD_ID_TO_NAME.get(pod_id, "ROTTERDAM")
                                
                                cnt += 1
                                # Alternate dg/reefer for mock validation test if needed
                                dg_flag = (pod_id == 6 and cnt % 5 == 0)
                                reefer_flag = (pod_id == 4 and cnt % 4 == 0)
                                queue.append(
                                    Container(
                                        id=f"CNTR-{sheet_name}-{cnt:03d}",
                                        weight_ton=weight,
                                        size="40",
                                        type="GP",
                                        pod=pod_name,
                                        dg=dg_flag,
                                        reefer=reefer_flag
                                    )
                                )
                    return YardState(slots=slots, queue=queue)
            except Exception as e:
                print(f"[SimulatedProvider] Error loading excel: {e}")
                
        # Fallback if file missing or parse error
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
        ys = self.get_yard_state(vessel_id)
        return ys.queue

class SNCTLiveProvider(DataProvider):
    """실데이터(SNCT 운영사) → 캐노니컬 스키마. specs/06_data_sources 참조.
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
    return {"simulated": SimulatedProvider, "snct_live": SNCTLiveProvider}[name]()

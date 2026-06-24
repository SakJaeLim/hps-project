# 적재 계획(Planning) 강화학습 연동 및 변경 내역

## 1. 사용 중인 RL 모델 파일

### 📁 모델 파일 위치
```
data/RL/강화학습 결과 자료/single_bay_6pod_ppo_v13_3way_ALL_models_seed42/
├── single_bay_6pod_ppo_v13_3way_BL_seed42.zip   ← rl_bl (Baseline)
├── single_bay_6pod_ppo_v13_3way_SF_seed42.zip   ← rl_sf (Safety First)
└── single_bay_6pod_ppo_v13_3way_EF_seed42.zip   ← rl_ef (Efficiency First)
```

### 📊 분석 데이터 파일 (Curriculum 학습 결과)
```
data/RL/강화학습 결과 자료/
└── single_bay_6pod_ppo_v13_3way_BayPlan_Distributions_seed42.xlsx
    ├── R1_Lv1 시트  → Level 1 (4R × 4T,  16 slots)
    ├── R2_Lv2 시트  → Level 2 (6R × 6T,  36 slots)
    ├── R3_Lv3 시트  → Level 3 (8R × 8T,  64 slots)
    └── R4_Lv4 시트  → Level 4 (10R × 10T, 100 slots)
```

### 🔑 엔진별 모델 매핑

| 대시보드 선택값 | 모델 파일 | 학습 특성 |
|---|---|---|
| `greedy` | 모델 파일 없음 (규칙 기반) | Heavy-Down + POD 그룹핑 단순 점수 |
| `rl_bl` | `..._BL_seed42.zip` | Baseline PPO — 기본 보상 함수 |
| `rl_sf` | `..._SF_seed42.zip` | Safety First — 안전 규정 위반 패널티 강화 |
| `rl_ef` | `..._EF_seed42.zip` | Efficiency First — 처리 속도·재취급 최소화 |

---

## 2. 전체 연동 흐름 (Data Flow)

```
[대시보드 UI]
   ↓ 사용자: 엔진 선택 + 커리큘럼 레벨 선택 + "계획 수립" 버튼
   
[dashboard/dashboard_app.py]
   ↓ POST /plan { engine: "rl_sf", vessel_id: "VESSEL-LV4", question: "..." }
   
[src/snct/api/app_api.py  →  /plan endpoint]
   ↓ run_pipeline(engine="rl_sf", vessel_id="VESSEL-LV4")
   
[src/snct/agents/graph.py  →  recognize()]
   ↓ get_provider("simulated").get_yard_state("VESSEL-LV4")
   
[src/snct/data/provider.py  →  SimulatedProvider]
   ↓ Excel 파싱: single_bay_6pod_ppo_v13_3way_BayPlan_Distributions_seed42.xlsx
   ↓ "LV4" → R4_Lv4 시트 → 100개 Container + 100개 Slot 생성
   
[src/snct/agents/graph.py  →  plan_step()]
   ↓ get_strategy("rl_sf")
   
[src/snct/engine/base.py  →  get_strategy()]
   ↓ RLStrategy(model_type="SF") 반환
   
[src/snct/engine/rl_policy.py  →  RLStrategy.__init__()]
   ↓ PPO.load("..._SF_seed42.zip", custom_objects={관측/행동 공간})
   ↓ 79차원 관측 벡터 생성 → PPO.predict() → 행동(0~9 Row 인덱스)
   ↓ 제약 위반 시 Greedy Fallback 동작
   
[src/snct/agents/graph.py  →  validate() → explain()]
   ↓ 온톨로지 제약 검증 (중량·DG·Reefer) + xAI 근거 설명 생성
   
[src/snct/api/app_api.py  →  응답 JSON]
   ↓ { assignments, slots, violations, rationale, engine, latency_ms }
   
[dashboard/dashboard_app.py  →  draw_bay_plan_fig()]
   ↓ Matplotlib으로 POD 분포 그리드 + 무게 분포 히트맵 렌더링
   ↓ st.pyplot(fig) 로 대시보드에 실시간 표출
```

---

## 3. 기존 코드 대비 변경 파일 목록 및 변경 내역

### 3-1. `src/snct/engine/rl_policy.py` — **[NEW 신규 생성]**

| 항목 | 내용 |
|---|---|
| **변경 유형** | 신규 파일 생성 |
| **기존 상태** | 해당 파일 없음. `base.py`의 `get_strategy("rl")`이 미구현 상태 |
| **변경 후** | 완전한 PPO 추론 엔진 구현 |

**주요 구현 내용:**
- NumPy 2.x → 1.x 하위 호환성 패치 3종 (모듈 로딩 시 자동 적용)
  ```python
  sys.modules['numpy._core'] = np.core
  sys.modules['numpy.random._pcg64'] = np.random
  _pickle.BitGenerators[PCG64] = PCG64
  ```
- `custom_objects`를 통한 Gymnasium 공간 불일치 우회
  ```python
  PPO.load(model_path, device="cpu", custom_objects={
      "observation_space": Box(shape=(79,)),
      "action_space": Discrete(10)
  })
  ```
- 79차원 정규화 관측 벡터 `_to_obs()` 생성 로직 (Stack 특성 60 + 현재 컨테이너 7 + 글로벌 6 + 잔여 POD 분포 6)
- PPO 행동 → 슬롯 매핑 후 제약 위반 시 Greedy Fallback 자동 전환

---

### 3-2. `src/snct/engine/base.py` — **[MODIFY 수정]**

| 항목 | 기존 | 변경 후 |
|---|---|---|
| `get_strategy("rl")` | 미구현 또는 더미 | `RLStrategy(model_type="BL")` 반환 |
| `get_strategy("rl_sf")` | 없음 | `RLStrategy(model_type="SF")` 반환 |
| `get_strategy("rl_ef")` | 없음 | `RLStrategy(model_type="EF")` 반환 |
| `get_strategy("rl_bl")` | 없음 | `RLStrategy(model_type="BL")` 반환 |

```diff
- # rl 전략 없음 또는 미구현
+ elif name == "rl_sf":
+     from snct.engine.rl_policy import RLStrategy
+     return RLStrategy(model_type="SF")
+ elif name == "rl_ef":
+     from snct.engine.rl_policy import RLStrategy
+     return RLStrategy(model_type="EF")
+ else: # "rl" or "rl_bl"
+     from snct.engine.rl_policy import RLStrategy
+     return RLStrategy(model_type="BL")
```

---

### 3-3. `src/snct/data/provider.py` — **[MODIFY 수정]**

| 항목 | 기존 | 변경 후 |
|---|---|---|
| 슬롯 수 | 하드코딩 3개 (Row 1~3, Tier 1) | Excel 파싱으로 동적 생성 (최대 100개) |
| 컨테이너 수 | 하드코딩 3개 | Excel 파싱으로 동적 생성 (최대 100개) |
| Curriculum 지원 | 없음 | `vessel_id`의 LV1/LV2/LV3/LV4 접미사로 시트 자동 선택 |
| 데이터 출처 | 인라인 코드 | `single_bay_6pod_ppo_v13_3way_BayPlan_Distributions_seed42.xlsx` |

```diff
- slots = [
-     Slot(bay=1, row=1, tier=1, max_stack_weight=30.0),
-     Slot(bay=1, row=2, tier=1, max_stack_weight=30.0, reefer_capable=True),
-     Slot(bay=1, row=3, tier=1, max_stack_weight=30.0, dg_allowed=True)
- ]
- queue = [
-     Container(id="CNTR-001", weight_ton=24.5, ...),
-     ...
- ]
+ # vessel_id에서 커리큘럼 레벨 판단 → Excel 시트 로드
+ v_upper = str(vessel_id).upper()
+ if "LV1" in v_upper: sheet_name = "R1_Lv1"   # 4R × 4T
+ elif "LV2" in v_upper: sheet_name = "R2_Lv2"  # 6R × 6T
+ elif "LV3" in v_upper: sheet_name = "R3_Lv3"  # 8R × 8T
+ else: sheet_name = "R4_Lv4"                   # 10R × 10T (기본값)
+
+ # Excel → Slot 100개 + Container 100개 동적 파싱 생성
```

---

### 3-4. `src/snct/api/app_api.py` — **[MODIFY 수정]**

| 항목 | 기존 | 변경 후 |
|---|---|---|
| `/plan` 응답 필드 | `assignments`, `violations`, `rationale` | **`weight_ton`, `pod` 필드 추가** |
| `slots` 반환 | 없음 | **전체 슬롯 레이아웃 반환 추가** |

```diff
  "assignments": [
      {"container_id": a.container_id, "bay": a.bay, "row": a.row, "tier": a.tier,
+      "weight_ton": cntr_lookup[a.container_id].weight_ton,
+      "pod": cntr_lookup[a.container_id].pod}
      for a in recommendation.plan.assignments
  ],
+ "slots": [
+     {"bay": s.bay, "row": s.row, "tier": s.tier,
+      "dg_allowed": s.dg_allowed, "reefer_capable": s.reefer_capable}
+     for s in yard_state.slots
+ ],
```

---

### 3-5. `dashboard/dashboard_app.py` — **[MODIFY 수정]**

| 항목 | 기존 | 변경 후 |
|---|---|---|
| 엔진 선택지 | `["greedy"]` | `["greedy", "rl_bl", "rl_sf", "rl_ef"]` |
| 커리큘럼 선택 | 없음 | Level 1 ~ Level 4 드롭다운 + Vessel ID 자동 매핑 |
| Bay Plan 시각화 | 없음 | `draw_bay_plan_fig()` 함수 신규 추가 |
| 한글 폰트 | 깨짐(Tofu) | Malgun Gothic 설정으로 정상화 |
| 축 좌표 표기 | `R1`, `T1` (1-based) | `R0`, `T0` (0-based, Excel과 동일) |
| 그림 슈퍼타이틀 | 없음 | `PPO 강화학습 최적 배정 분포 (LV4 - 10R × 10T)` |

**신규 추가 함수 `draw_bay_plan_fig(res)`:**
```python
# API 응답(res)에서 assignments + slots를 받아
# 왼쪽: POD 분포 컬러 그리드 (6색 + 범례)
# 오른쪽: 무게 분포 Blues 히트맵 + Colorbar
# → Matplotlib Figure 반환 → st.pyplot(fig)으로 렌더
```

---

## 4. 적재 계획 파이프라인 요약도

```
┌─────────────────────────────────────────────────────────┐
│                   대시보드 (Streamlit)                    │
│  [엔진선택] [커리큘럼레벨] [작업지시] [계획수립버튼]           │
└──────────────────────┬──────────────────────────────────┘
                       │ POST /plan
┌──────────────────────▼──────────────────────────────────┐
│              FastAPI 백엔드 (app_api.py)                  │
│  /plan → run_pipeline(engine, vessel_id, question)       │
└────────┬─────────────────────────────┬───────────────────┘
         │                             │
┌────────▼──────────┐       ┌──────────▼──────────────────┐
│  SimulatedProvider │       │       Agent Pipeline         │
│  (provider.py)    │       │  recognize → plan → validate │
│  Excel 파싱        │       │  → explain (graph.py)        │
│  LV1~LV4 슬롯·큐   │       └──────────┬───────────────────┘
└───────────────────┘                  │
                              ┌─────────▼──────────────────┐
                              │      RL Strategy            │
                              │    (rl_policy.py)           │
                              │  BL / SF / EF .zip 로드     │
                              │  79dim obs → PPO.predict()  │
                              │  Greedy Fallback 안전망      │
                              └────────────────────────────┘
```

---

## 5. requirements.txt 추가 항목

```diff
+ stable-baselines3>=2.0.0   # PPO 모델 로딩용
```

# PPO 강화학습(RL) 엔진 연동 및 적재 계획 다변화 구현 계획서

현재 뼈대(Greedy Fallback Mock)만 구성되어 있는 RL 적재 계획을 `/data/RL` 폴더 내에 저장된 실제 학습 완료 파일들(`single_bay_6pod_ppo_v13_3way_BL_seed42`, `SF`, `EF`)과 연동하기 위한 설계 및 개발 로드맵입니다.

---

## User Review Required

> [!IMPORTANT]
> **1. 라이브러리 추가 설치 필요**
> PPO 모델을 로드하기 위해서는 `stable-baselines3` 패키지가 설치되어야 합니다. 로컬 개발 환경에서 `pip install stable-baselines3` 실행이 가능해야 합니다.
>
> **2. 관측 및 행동 공간 불일치 조율 (Adapter 구현)**
> 기존 코드의 `StowageEnv`는 9슬롯(39차원 obs, Discrete 9 actions) 기준이지만, 원우님이 학습하신 v13 PPO 모델은 10열 최대 적재 환경(`SingleBayStowageEnv` - 79차원 obs, Discrete 10 actions)을 기반으로 동작합니다. 따라서 **PPO 전용 관측 변환기(Observation Adapter) 및 행동 매퍼(Action Mapper)**를 연동 엔진 내에 구축해야 합니다.

---

## Proposed Changes

### 1. 의존성 패키지 설치
#### [MODIFY] [requirements.txt](file:///c:/Users/lione/Desktop/aSSIST/19_Project/12_hps-project-main/requirements.txt)
* `stable-baselines3` 라이브러리를 추가하여 가중치 파일 로드를 가능하게 합니다.

```diff
  # --- serving / API / UI ---
  fastapi
  uvicorn
  streamlit
  pydantic
+ stable-baselines3
```

---

### 2. 적재 환경 및 모델 어댑터 개발
#### [MODIFY] [rl_policy.py](file:///c:/Users/lione/Desktop/aSSIST/19_Project/12_hps-project-main/src/snct/engine/rl_policy.py)
* `DQNetwork` 구조 대신 `stable_baselines3.PPO` 모듈을 로드하여 사용합니다.
* `RLStrategy` 내에 `BL`, `SF`, `EF` 모델 파일을 선택하여 로드할 수 있도록 파라미터를 확장합니다.
* `StowageEnv`의 스택/야드 구조와 노트북의 79차원 observation 매핑 함수(`_obs`)를 추가하여 PPO 모델의 입력값 규격을 완벽하게 맞춥니다.
* PPO가 출력하는 `action` (10개 열 중 1개 선택)을 실제 슬롯(Bay, Row, Tier) 정보로 변환하여 적재 계획을 구축하는 룰을 삽입합니다.

```python
# 개념 예시 코드
from stable_baselines3 import PPO

class PPOStowageStrategy(StowageStrategy):
    def __init__(self, policy_type: str = "BL"):
        # 모델 파일 경로 설정
        checkpoint_dir = r"data/RL/강화학습 결과 자료/single_bay_6pod_ppo_v13_3way_ALL_models_seed42"
        model_path = os.path.join(checkpoint_dir, f"single_bay_6pod_ppo_v13_3way_{policy_type}_seed42")
        
        # stable-baselines3를 통한 PPO 모델 로드 (.zip 파일 직접 로드 가능)
        self.model = PPO.load(model_path, device="cpu")
        
    def plan(self, yard: YardState) -> CandidatePlan:
        # 1. 79차원 observation 변환
        # 2. model.predict(obs, deterministic=True) 호출
        # 3. Action(열 인덱스) -> YardState 슬롯 배정
        # 4. CandidatePlan 반환
```

---

### 3. 최적화 엔진 탐색 확장
#### [MODIFY] [base.py](file:///c:/Users/lione/Desktop/aSSIST/19_Project/12_hps-project-main/src/snct/engine/base.py)
* `get_strategy`에서 `"rl_bl"`, `"rl_sf"`, `"rl_ef"` 호출 시 각각의 가중치를 불러오도록 분기합니다.

```python
def get_strategy(name: str = "rl") -> "StowageStrategy":
    if name == "greedy":
        ...
    elif name in ["rl", "rl_bl"]:
        return PPOStowageStrategy(policy_type="BL")
    elif name == "rl_sf":
        return PPOStowageStrategy(policy_type="SF")
    elif name == "rl_ef":
        return PPOStowageStrategy(policy_type="EF")
```

---

### 4. 대시보드 및 API 연동
#### [MODIFY] [dashboard_app.py](file:///c:/Users/lione/Desktop/aSSIST/19_Project/12_hps-project-main/dashboard/dashboard_app.py)
* 대시보드 화면 내 "적재 계획 (Planning)" 페이지의 "최적화 엔진 선택" 셀렉트 박스에 선택지를 확장합니다.
  * 기존: `["greedy", "rl"]`
  * 변경: `["greedy", "rl_bl (기본)", "rl_sf (안전우선)", "rl_ef (효율우선)"]`

---

## Verification Plan

### Automated Tests
1. **PPO 로드 및 추론 검증 테스트 (`tests/test_rl_strategy.py`)**
   * PPO 모델을 로드하여 mock 야드 정보에 대해 정상적으로 오류 없이 적재 계획(`CandidatePlan`)을 수립하는지 검증합니다.
   * `pytest tests/test_rl_strategy.py` 실행하여 통과 확인.

### Manual Verification
1. **대시보드 수립 테스트**
   * streamlit 페이지에서 각 정책(BL, SF, EF)을 선택하여 "계획 수립" 실행.
   * 각 정책에 따라 무게 중심 제약 위반(WBI) 및 재취급률(OSR) 결과가 다르게 수립되는지 설명(xAI) 및 슬롯 배정 현황 시각화로 직접 대조 확인.

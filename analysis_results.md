# 강화학습 최적 배정 분포 시각화 보고서

PPO 강화학습 결과물인 `single_bay_6pod_ppo_v13_3way_BayPlan_Distributions_seed42.xlsx` 엑셀 파일의 데이터를 파싱하여, 각 커리큘럼 레벨별(Level 1 ~ Level 4)로 **POD(양하항) 분포**와 **중량(Weight) 분포**를 시각화한 결과입니다.

---

## 📊 단계별 배정 분포 (Curriculum Level 1 ~ 4)

````carousel
### 1단계: Curriculum Level 1 (4R × 4T)
![Level 1 Distribution](/C:/Users/lione/.gemini/antigravity-ide/brain/b0e3cabf-f710-4efe-b4e9-5c01b26f03f7/bay_plan_lv1.png)
* **특징:** 가장 단순한 환경(4열 4단)으로, 6개 POD 중 대표 POD가 1~2개 층씩 나란히 수평 밴딩을 이루며 배치됩니다.

<!-- slide -->
### 2단계: Curriculum Level 2 (6R × 6T)
![Level 2 Distribution](/C:/Users/lione/.gemini/antigravity-ide/brain/b0e3cabf-f710-4efe-b4e9-5c01b26f03f7/bay_plan_lv2.png)
* **특징:** 환경의 크기가 확장(6열 6단)되었으나, 여전히 같은 POD가 같은 층(Tier)에 평평하게 정렬되고 무거운 화물들이 안정적으로 아래로 배치됩니다.

<!-- slide -->
### 3단계: Curriculum Level 3 (8R × 8T)
![Level 3 Distribution](/C:/Users/lione/.gemini/antigravity-ide/brain/b0e3cabf-f710-4efe-b4e9-5c01b26f03f7/bay_plan_lv3.png)
* **특징:** 8열 8단으로 확장되면서, 각 목적지 포트별 수평 레이어 구조가 뚜렷해지고 중량 구배가 하단에서 상단으로 갈수록 점차 옅어지는 양상을 띱니다.

<!-- slide -->
### 4단계: Curriculum Level 4 (10R × 10T)
![Level 4 Distribution](/C:/Users/lione/.gemini/antigravity-ide/brain/b0e3cabf-f710-4efe-b4e9-5c01b26f03f7/bay_plan_lv4.png)
* **특징:** 최종 시나리오 규격(10열 10단)입니다. 100% 적재 상황에서도 동일 POD 수평 밴딩이 매우 칼같이 지켜지고 있고, 하부 층에는 어두운 파란색(고중량 18~20t)이, 상부 층에는 연한 파란색(저중량 10~13t)이 배치되어 선박의 복원성(COG)이 극대화되는 배정 형태를 보여줍니다.
````

---

## 💡 주요 강화학습(PPO) 패턴 분석

1. **POD 수평 밴딩 (Same-Tier Banding)**
   * 좌측 POD 분포 그림을 보면, 목적지 항구별로 동일한 층(Tier)에 수평으로 일렬 배치가 이루어지고 있습니다. 
   * 최하단에는 가장 멀리 가는 `Rotterdam(6)`이 깔리고, 위로 갈수록 목적지 순서에 따라 층이 나뉘어 쌓여 선박이 중간 기항지에 도달했을 때 불필요하게 컨테이너를 다시 들었다 놓는 **재취급(Rehandling/Overstow) 발생률을 극도로 억제**합니다.

2. **중량 중심 구배 (Heavy-Down & Weight Balance)**
   * 우측 중량 분포 그림에서 하부 층(T0 ~ T3)에는 주로 짙은 색의 무거운 컨테이너들이 밀집해 있으며, 위로 올라갈수록 색상이 연해집니다.
   * 이는 Heavy-Down / Light-Up 규칙이 학습 과정의 보상 함수(R6, R9, R11)에 의해 최적 반영되어 **선박 전체의 무게중심(COG)을 낮추고 안정성을 확보**한 결과입니다.

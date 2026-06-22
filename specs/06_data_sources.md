# 06. 실데이터 소스 매핑 (운영사 제공)

데이터 어댑터(`LiveProvider`)가 아래 원천을 **캐노니컬 스키마로 정규화**한다.
상위 레이어(온톨로지·엔진·에이전트·대시보드)는 변경 없음. 도착 전까지 `SimulatedProvider`로 진행하고,
도착분부터 우선순위대로 교체한다. (어댑터 패턴 — ADR/03_data_spec)

## A. 정형 데이터 (운영/EDI) → L1·L2·L3·평가

| 데이터 | 정의 | 캐노니컬 매핑 | 레이어 | 핵심 용도 |
|---|---|---|---|---|
| Vessel Call History | 과거 입출항 이력(선석·스케줄·항차) | Vessel · Berth · Call | L1·L2·평가 | 실 선박/선석 엔티티, 시뮬 시드, 평가 시나리오 |
| Bay Plan | 선박 베이 구조·배치 | Vessel→Bay→Row→Tier→Slot | L2·L3 | 적재 인스턴스 구조, 슬롯 온톨로지, RL 입력 |
| **BAPLIE** ★ | 컨테이너별 본선 위치(점유/공실)+중량·POD·DG·Reefer | Container · Slot · ASSIGNED_TO | L1·L2·L3·평가 | 실 적재 상태·정답 → RL 입력, 평가 기준 |
| COPRAR | 양·적하 지시(선사→터미널, 재적재 포함) | Container · WorkOrder(load/discharge/shift) | L1·L3 | 계획 요청의 작업 목록(무엇을 적재) |
| MOVINS | 적재 지시(순서·세그리게이션·재적재) | StowageInstruction(sequence/segregation) | L2·L3 | 적재 제약·순서 → 엔진 제약/보상 |
| **Yard Inventory** ★ | 야드 현재 재고(블록·베이·행·단) | YardState · Slot.occupied_by · STACKED_ON | L1·L2·L3 | 실 야드 상태(엔진 대상), 재취급(무엇이 무엇 위에) |
| Gate Transaction | 게이트 반출입(차량·컨테이너·시각) | GateEvent(arrival/departure) | L1·평가 | 컨테이너 큐·반입 시점, 처리량 |
| **Equipment Operation** ★ | 장비 작업 로그(이동·시각·생산성) | MoveEvent · rehandling · crane | L3·평가 | 재취급 실측·크레인 간섭 → 평가/보상 |
| AIS | 선박 위치·이동 | Vessel(eta·status) | L1·L4 | 입항 예측·실시간 상태(상태인식 에이전트) |

★ = 코어 실데이터(이게 있으면 실제 적재·재취급을 실문제로 구동·평가 가능)

## B. 비정형 데이터 (규정/절차) → L5 RAG·SLM

| 데이터 | 정의 | 핵심 용도 | 민감도 |
|---|---|---|---|
| IMO Code (IMDG) | 위험물 분류·격리 규정 | DG 격리 제약(L2/L3) + RAG 근거 | 공개 |
| ISPS Code | 선박·항만 보안 | 보안 절차 RAG·SOP 맥락 | **보안 민감 — 접근통제** |
| SOLAS | 해상인명안전(복원성·VGM·라싱) | 복원성·중량 제약 근거(L3) + RAG | 공개 |
| Terminal SOP | 표준운영절차 | **RAG 핵심 코퍼스** + SLM 파인튜닝 데이터 | 영업 민감 |
| Safety Manuals | 안전 매뉴얼 | 안전 제약·절차 RAG + SLM 데이터 | 영업 민감 |
| Accident Reports | 사고 보고서 | 위험 패턴·"왜 위험" 설명·경보 근거 | **개인·법적 민감 — 비식별** |

## 이 데이터가 새로 푸는 것 (이전엔 시뮬레이션뿐)
1. **실 적재 인스턴스**: BAPLIE+Bay Plan+Yard Inventory+COPRAR/MOVINS → RL 모델을 실문제에서 구동. (원우 모델 학습·평가용 실데이터원)
2. **실 평가**: Equipment Operation 로그 = 실제 재취급 수 → "재취급 감소"를 시뮬이 아닌 **실측 대비** 검증(검증 메모의 과장 리스크 해소).
3. **그라운딩된 설명**: SOP+Safety+Accident+IMO/ISPS/SOLAS = 설명 에이전트의 근거 코퍼스 → 환각↓·규정 인용(프로젝트 차별점 강화).

## 수집·정규화 (ETL · src/snct/data/sources)
- EDIFACT 파서 필요: BAPLIE/COPRAR/MOVINS (버전 확인 필수 — 예 D.95B/D.16A, SMDG MIG 참조).
- 포맷 이질성(EDI/CSV/XML/PDF) → 원천별 파서로 정규화 후 캐노니컬 스키마 적재.
- 규정 PDF → 청크 + 임베딩(BGE-M3) + Vector DB(RAG).

## 거버넌스 (필수)
- 민감 데이터: ISPS · Accident Reports · Gate Transaction(차량/운전자 PII) · BAPLIE(영업) → **비식별·접근통제·협약 범위**.
- 저장: `data/real/`(gitignore), 공유 샘플은 합성/가명. ISPS·사고보고서는 별도 접근통제 폴더.
- 실데이터 커밋 절대 금지.

## 도착 우선순위 (조각조각 들어올 때)
- **Tier 1 (코어 실데이터 데모)**: Yard Inventory · BAPLIE · Bay Plan · COPRAR/MOVINS
- **Tier 2 (평가·현실성)**: Equipment Operation · Vessel Call History · Gate Transaction
- **Tier 3 (실시간·보강)**: AIS
- **RAG 코퍼스(병렬)**: SOP · Safety · IMO/SOLAS 먼저 → ISPS · Accident는 접근통제 후

## 전산팀에 함께 요청할 메타정보 (포맷 미정 대응 — 받자마자 파싱 가능하게)
각 데이터셋별: ① 포맷·스키마(EDI 버전 / CSV 헤더 정의), ② 샘플 파일 1건, ③ 기간(과거 N개월), ④ 비식별 범위, ⑤ 데이터 사전(필드 정의), ⑥ 갱신 주기.

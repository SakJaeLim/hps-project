# TASK 분해 (PRD → 원자적 작업)

원자적 작업: 독립 실행 · 명확한 입력/출력 · **완료조건 = 해당 테스트 통과**(TDD Green).
TDD: 각 작업은 `tests/`의 실패하는 테스트(Red)로 시작 → 최소 구현으로 통과(Green) → 리팩터.
담당: PL · FE(원우1) · ON(원우2,온톨로지) · DM(원우3,도메인).

| ID | 작업 | 담당 | 완료조건(test) | spec | 주차 |
|----|------|------|----------------|------|------|
| T01 | 캐노니컬 스키마 정의(Container/Slot/YardState/CandidatePlan/Violation/Recommendation) | PL | `tests/test_schema.py` | 03 | W1 |
| T02 | 데이터 어댑터 IF + SimulatedProvider(시뮬 생성기) | ON/PL | `tests/test_data_provider.py` | 03,06 | W1 |
| T03 | Neo4j 스키마 v1 + 샘플 적재 | ON | `tests/test_ontology_schema.py` | 01 | W1 |
| T04 | 제약 Cypher 5종(중량·DG·Reefer·반출순서·재취급) | ON | `tests/test_constraints.py` | 01 | W2 |
| T05 | Greedy 적재 플래너(기준선) | PL | `tests/test_greedy.py` | 00,ADR-0001 | W1–2 |
| T06 | StowageEnv 골격 + RL 모델 통합(원우 모델 어댑터) | PL/DM | `tests/test_rl_strategy.py` | ADR-0001 | W2–3 |
| T07 | LangGraph 4노드 흐름(인식→계획→검증→설명) | PL | `tests/test_agent_flow.py` | 02 | W2 |
| T08 | 검증 위반 시 계획 재시도 루프 | PL | `tests/test_retry_loop.py` | 02 | W3 |
| T09 | 문서 RAG 검색(규정·SOP) | PL/DM | `tests/test_rag_docs.py` | 06,07 | W2–3 |
| T10 | GraphRAG text2Cypher 템플릿(관계·재취급) | ON | `tests/test_graphrag.py` | 07 | W3 |
| T11 | NL2SQL 읽기전용 골격(조건부) | PL | `tests/test_nl2sql_readonly.py` | 07 | W3 |
| T12 | 설명 합성(xAI: 계획+검증사실+근거 융합) | PL | `tests/test_explain.py` | 07 | W3 |
| T13 | FastAPI POST /plan | PL | `tests/test_api_plan.py` | 00 | W2 |
| T14 | Streamlit 4화면(메인·요청·결과·경보) | FE | 수동검증 + `tests/test_dashboard_smoke.py` | 00 | W2–3 |
| T15 | 평가 하니스(재취급·위반율·응답속도) | PL/DM | `tests/test_eval_metrics.py` | 04 | W3 |
| T16 | 도메인 SLM LoRA 파인튜닝 1회 | DM/PL | 정성평가(환각↓·근거↑) | 07 | W3 |
| T17 | 통합·시연·문서화(M3) | 전원 | E2E 데모 통과 | 00 | W4 |

> 마일스톤: M1(W2말)=T01–T05,T07,T13 E2E 슬라이스 · M2(W3말)=T04,T06,T08–T12,T15 통합 · M3(W4말)=T16–T17.

## 설명가능 RL 적재 에이전트 (xAI-RL) — specs/07 추가분

> 전략: **규칙 기반 융합 우선 + LPG는 CSV 폴백**. RL 결과 자료(`data/강화학습 결과 자료/`)를 근거 소스로 사용.

| ID | 작업 | 담당 | 완료조건(test) | spec | 주차 | 상태 |
|----|------|------|----------------|------|------|------|
| T18 | RL 결과 로더(reward_decomp·kpi·slot_assignment·violation_log·xai_grounding 적재) | PL/DM | `tests/test_rl_result_loader.py` | 07 | W3 | ✅ |
| T19 | LPG CSV 폴백 그래프 질의(STACKED_ON·VIOLATES·Constraint·violations_in_round) | ON | `tests/test_lpg_csv_query.py` | 07 | W3 | ✅ |
| T20 | NL2SQL 읽기전용 이식(sqi agent 패턴 → RDB CSV/DuckDB, `RLAnalyst`, 가드레일) | PL | `tests/test_nl2sql_readonly.py` | 07 | W3 | ✅ |
| T21 | 설명 합성기 재작성(reward_decomp 귀인 + kpi + doc_refs 융합, `explain_rl_decision`) | PL | `tests/test_explain.py` | 07 | W3 | ✅ |
| T22 | 설명 faithfulness 평가(수치 근거율·환각 탐지, `eval/faithfulness.py`) | PL/DM | `tests/test_explain_faithfulness.py` | 04,07 | W3–4 | ✅ |
| T24 | 설명↔LPG 통합(컨테이너별 위반 규정 자동 인용, faithfulness 1.0 유지) | PL | `tests/test_explain_integration.py` | 07 | W3–4 | ✅ |
| T25 | 설명 흐름 진입점(`run_explanation`: 질의→근거수집→설명→자기검증) | PL | `tests/test_agent_explain_flow.py` | 02,07 | W4 | ✅ |
| T26 | 대시보드/API 연결(정책·라운드 선택 → 설명 표시, `POST /explain`) | FE/PL | `tests/test_api_explain.py` + 수동검증 | 00 | W4 | ✅ |
| T27 | 컨테이너 위치 조회(자연어 → slot_assignment 위치 + 적층/반출 가능 여부, `POST /locate` + 대시보드) | PL/FE | `tests/test_locator.py` | 07 | W4 | ✅ |
| T28 | LPG Neo4j 실연결(`Neo4jLPG`·`import_kg`) + 백엔드 팩토리(Neo4j↔CSV 자동 폴백) + 대시보드 상태/질의 | ON/PL | `tests/test_lpg_backend.py` | 07 | W4 | ✅ |
| T23 | (후순위) 설명 표현용 SLM LoRA — 사실 카드→문장 다듬기 | DM | 정성평가(환각0·근거유지) | 07 | W4 | ⬜ |

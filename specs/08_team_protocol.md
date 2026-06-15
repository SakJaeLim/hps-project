# 08. 팀 역할·협업 프로토콜 (4인)

## 인원
- **PL (나)** — 아키텍처 총괄·핵심 개발·통합·평가·근거 검증 (+ PM/조율)
- **원우1 · 프론트엔드** — L6 대시보드·시각화·발표/시연
- **원우2 · 온톨로지/데이터** — L1 데이터 어댑터·캐노니컬 스키마·L2 Neo4j·Cypher·text2Cypher
- **원우3 · 도메인 전문가** — 규칙·RL 모델 공급·SLM 데이터셋·평가·규정 코퍼스·현실성 검증

## 역할: 전체 확장 구조 vs 1개월 현실 범위
| 인원 | 전체 확장 구조 담당 | 1개월 현실 범위 |
|---|---|---|
| **PL** | 아키텍처·ADR / L4 오케스트레이션 / L3 엔진 통합 / L5 지식접근(router·NL2SQL·explain·xAI) / L6 API / 통합·평가·반박 총괄 | LangGraph 4노드 + RL 모델 통합 + Greedy 기준선 + 문서RAG 연결 + explain + FastAPI + 통합. NL2SQL/GraphRAG는 골격·조건부 |
| **원우1 (프론트)** | L6 대시보드 전체·UX / 시각화 / 발표·시연 / (향후) React 고도화 | Streamlit 4화면(메인·요청·결과·경보) + mock 계약 병렬개발 + 발표자료·시연영상 |
| **원우2 (온톨로지)** | L1 어댑터·캐노니컬 스키마·EDIFACT 파싱·DuckDB / L2 Neo4j·Cypher·text2Cypher | Neo4j 스키마 v1 + 제약 Cypher 5종 + 시뮬 데이터 생성기 + text2Cypher 템플릿 2~3종. 실데이터 파서는 도착분만 |
| **원우3 (도메인)** | 규칙·RL 모델 공급·SLM 데이터셋·평가·규정 코퍼스 큐레이션·현실성 검증 | 제약 규칙표 + 선박제원 전사 + RL 모델·env 인수인계 + instruction 데이터셋 v1 + 평가 시나리오 3종 + SOP 코퍼스 정리 |

## RACI (레이어 × 인원)  — R 책임 · A 최종책임 · C 자문 · I 통보
| 워크스트림 | PL | 원우1 | 원우2 | 원우3 |
|---|---|---|---|---|
| L1 데이터 어댑터 | C | I | **A·R** | C |
| L2 온톨로지(Neo4j·Cypher) | C | I | **A·R** | C |
| L3 적재 엔진(통합) | **A·R** | I | I | C |
| L4 오케스트레이션 | **A·R** | I | I | I |
| L5 지식접근(RAG·NL2SQL) | **A·R** | I | C(text2Cypher) | C(코퍼스) |
| L6 대시보드 / API | C / R | **A·R** / I | I | I |
| 규칙·RL모델·데이터셋 | C | I | C | **A·R** |
| 평가·검증 | A | C | C | R |
| 아키텍처·ADR·통합 | **A·R** | C | C | C |

## 협업 규칙
- **계약 우선**: YardState·CandidatePlan·Violation·Recommendation 동결 후 mock으로 병렬개발(서로 대기 없음).
- **GitHub flow** (main 보호·feature/*·PR 리뷰) · **SDD**(명세 먼저) · **AI-Native**(바이브코딩).
- 주간 동기화 1회 + 계약 변경은 명세 PR 우선.

## 근거 기반 의사결정·반박 프로토콜 (Evidence over Authority)
**원칙**: 모든 핵심 주장(특히 도메인·아키텍처 결정)은 **근거를 동반하거나 '가정'으로 명시**한다. 결정은 근거로 판가름하며 권위/연차가 아니다. 도메인 전문가의 직관은 귀중하지만, **검증 가능해야 시스템에 인코딩**할 수 있다. (이 규칙은 PL 자신의 주장에도 동일 적용)

**왜 도메인 주장도 반박 대상인가**: (i) 현장 관행이 최신 기술/연구와 어긋날 수 있고(예: RL vs CP), (ii) 관행이 최적이 아닐 수 있으며, (iii) 모호한 주장은 코드/제약으로 옮길 수 없다.

**프로세스**: `주장(Claim) → 근거/가정 분류 → 반증·교차검증 → 결정(ADR) → 재검토(새 근거 시)`

**PL의 반박 준비 도구(항상 지참)**: 국제 표준(IMO/IMDG·SOLAS·ISPS·SMDG MIG), 학술 벤치마크(CPMP·MBPP·NL2SQL·text2Cypher), 실데이터·운영 로그, 상용 사례(Navis N4). → 검증 메모/ADR로 기록.

**기록**: `specs/decisions/ADR-*` (결정) + `specs/decisions/CHALLENGE-LOG.md` (주장↔반증 추적).

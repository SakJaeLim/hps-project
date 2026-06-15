# ADR-0002 · 구현 언어·기술 스택 — Python 단일 런타임

- 상태: 채택(Accepted) · 2026-06 · 관련: ADR-0001, specs/architecture.md

## 결정
구현 언어는 **Python 단일 런타임**. 데이터 질의는 **Cypher**(그래프)·**SQL**(관계형), 입력 교환은 **EDIFACT**.

## 근거
1. **AI/ML 생태계가 Python 네이티브** — LangGraph, Gymnasium/Stable-Baselines3(RL), transformers·PEFT·Unsloth(SLM), sentence-transformers(임베딩), neo4j-graphrag, OR-Tools(CP). 원우 RL 모델도 PyTorch/SB3.
2. **수업·팀 정합** — 이론.pdf "모든 코드는 파이썬", workshop 노트북 전부 Python → 학습곡선 최소.
3. **단일 언어 = 통합비용↓** — 계약(YardState·CandidatePlan…)을 한 런타임에서 공유, 4인 병렬개발이 언어 경계 없이.
4. **AI-Native(바이브코딩) 성숙도** — Cursor/Claude 코드생성 품질이 Python에서 가장 높음.
5. **UI+API도 Python** — FastAPI(REST·async·Pydantic) + Streamlit(빠른 대시보드)로 1개월 내 완결.

## 컴포넌트별 언어/기술
| 레이어 | 언어/기술 |
|---|---|
| L6 대시보드 | Python · Streamlit |
| L6 API | Python · FastAPI |
| L4 오케스트레이션 | Python · LangGraph |
| L3 엔진 | Python · PyTorch/SB3(RL) · OR-Tools(CP, 향후) |
| L2 온톨로지 | Cypher @ Neo4j (드라이버 Python) |
| L5 지식 접근 | Python · LangChain · neo4j-graphrag · SQL |
| L1 데이터 | Python · EDIFACT 파서 · pandas/DuckDB · SQL |

## 대안 기각
- **Java/C++**: 실 TOS(Navis 등)는 Java지만 재구현이 목표가 아님 — EDI·데이터로 연동.
- **JS/TS 백엔드**: ML 라이브러리 공백. (프론트 고도화 시 React를 FastAPI REST 위에 얹는 것은 향후 옵션)
- **Rust/Go**: ML 생태계·1개월 제약상 과함.

## 결과
패키징은 uv/pyproject. GPU 파트(SLM 파인튜닝·RL 추론)는 별도 환경/런팟으로 분리 가능.

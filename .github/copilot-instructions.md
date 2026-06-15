# Copilot / AI 작업 지침 — SNCT Decision Platform

이 저장소는 **AI-Native SDD**로 개발한다. AI(에이전트)는 아래를 항상 준수한다.

## 단일 진실원천(SSOT)과 순서
1. `PRD.md` = 무엇/왜. 2. `specs/` = 어떻게(설계·ADR). 3. `TASK.md` = 원자적 작업. 4. `tests/` = 완료조건(TDD). 5. `src/` = 구현.
- 기능 변경은 **상류(PRD/specs) 먼저** 수정한 뒤 TASK·test·코드를 따라간다.
- 작업 완료 기준 = 해당 테스트 Green. 과잉 구현 금지(테스트만 통과).

## TDD (필수)
- 새 작업은 `tdd-red`로 실패 테스트 → `tdd-green` 최소 구현 → `tdd-refactor`.
- 테스트는 `tests/`에 두고 `pytest`로 검증.

## 아키텍처 계약 (변경 시 specs/07·schema 먼저)
- 계약: `YardState · CandidatePlan · Violation · Recommendation` (common/schema.py).
- 엔진은 `StowageStrategy` 교체형: rl(기본)·greedy·cp (ADR-0001).
- 오케스트레이션은 **결정론적**(LangGraph). LLM은 설명에만. 운영자 승인(HITL).
- 지식접근: 문서RAG·NL2SQL(읽기전용)·GraphRAG 라우팅 + 사실 근거 융합(specs/07).

## 규약
- 언어: Python 단일 런타임(ADR-0002). 질의 Cypher/SQL. NL2SQL은 **읽기전용·검증·LIMIT**.
- 실데이터는 `data/real/`(git 제외). 민감자료(ISPS·사고보고서·PII) 접근통제.
- 커밋: `commit-helper` 규약. PR은 1인 이상 리뷰.

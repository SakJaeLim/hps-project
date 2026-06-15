# SDD 운영 가이드 (AI-Native, 5일차 기준)

## 산출물 체인 (단일 진실원천 = PRD)
```
PRD.md ──▶ specs/ ──▶ TASK.md ──▶ tests/ (TDD Red) ──▶ src/ (Green) ──▶ refactor
(무엇/왜)   (어떻게·설계)  (원자적 작업)   (완료조건)        (최소구현)
```
- 상류(PRD/specs)를 먼저 바꾼다. 코드만 바꾸지 않는다.
- 작업 완료 = 해당 테스트 Green.

## TDD 사이클 (수업 5일차)
1. **Red** (`tdd-red` skill): 작업의 완료조건을 실패하는 테스트로 작성 → `pytest`로 실패 확인.
2. **Green** (`tdd-green` skill): 테스트를 통과하는 최소 구현.
3. **Refactor** (`tdd-refactor` skill): 테스트 유지하며 정리.
> 현재 `tests/`의 테스트는 모두 **Red(미구현)** 상태이며 `xfail`로 표시되어 있다. 구현되면 `xfail` 제거 → Green 이어야 한다.

## AI-Native 스캐폴딩 (`.github/`)
- `copilot-instructions.md` — 프로젝트 전체 AI 지침(이 SDD 체인·계약·규약).
- `skills/` — prd-writer · task-breakdown · tdd-red/green/refactor · code-review · commit-helper (수업 스킬팩).
- `instructions/` — 파일 범위 지침(test·python).
- `prompts/` — 재사용 프롬프트.

## 트레이서빌리티 (작업 ↔ spec ↔ test ↔ 코드)
| TASK | spec | test | src |
|------|------|------|-----|
| T01 | 03 | test_schema | common/schema.py |
| T02 | 03,06 | test_data_provider | data/provider.py |
| T03–T04 | 01 | test_ontology_schema, test_constraints | ontology/graph.py |
| T05–T06 | 00,ADR-0001 | test_greedy, test_rl_strategy | engine/{greedy,rl_policy,env,base}.py |
| T07–T08 | 02 | test_agent_flow, test_retry_loop | agents/graph.py |
| T09–T12 | 06,07 | test_rag_docs, test_graphrag, test_nl2sql_readonly, test_explain | knowledge/*.py |
| T13 | 00 | test_api_plan | api/main.py |
| T15 | 04 | test_eval_metrics | (eval 하니스 — 신규) |

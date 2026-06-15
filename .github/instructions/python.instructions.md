---
applyTo: "src/**/*.py"
---
# 파이썬 구현 지침
- 계약(common/schema.py) 타입을 입출력으로 사용. 모듈은 상단에 관련 `specs/` 또는 `ADR-` 명시.
- 부수효과 분리: 외부 IO(Neo4j·DB·LLM)는 어댑터/도구 경계 뒤로.
- NL2SQL은 읽기전용(SELECT)만 실행·검증. text2Cypher는 템플릿 우선.
- 최소 구현 우선(테스트 통과). 추측 기능 추가 금지.

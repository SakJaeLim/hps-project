---
applyTo: "tests/**/*.py"
---
# 테스트 지침 (TDD)
- 프레임워크: pytest. 파일명 `test_*.py`, 함수 `test_*`.
- 테스트는 **요구사항의 표현**이다. TASK의 완료조건을 1:1로 검증한다.
- Red 단계: 미구현 작업은 `@pytest.mark.xfail(reason="TDD Red — Tnn 미구현", strict=False)`.
  구현 완료 시 `xfail` 제거 → 반드시 Green.
- 외부 의존(neo4j·gymnasium 등)은 테스트 함수 내부에서 import(수집 실패 방지) 또는 모킹.
- 결정적 테스트: 시드 고정. NL2SQL/Cypher 생성은 템플릿/모킹으로 검증.

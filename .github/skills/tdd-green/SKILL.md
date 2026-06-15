---
name: tdd-green
description: 실패하는 pytest 테스트를 통과시키는 최소한의 구현 코드를 작성합니다.
---
# Skill: tdd-green

## 트리거 조건

다음 중 하나가 포함된 요청이 들어올 때 이 Skill을 사용합니다:
- "테스트 통과시켜줘" / "Green 단계 시작"
- "tdd-green Skill 사용해서"
- "구현 코드 작성해줘" (테스트 파일이 존재하는 컨텍스트)
- "FAILED 테스트를 통과시켜줘"

## 역할

실패 상태(`FAILED`)의 pytest 테스트를 읽고, 테스트를 통과시키는 **최소한의 구현 코드**를 작성합니다.  
이 Skill이 완료된 후 `pytest`를 실행하면 모든 테스트가 `PASSED` 상태여야 합니다.

## 입력

| 항목 | 필수 여부 | 없을 경우 |
|------|----------|----------|
| `test_<모듈명>.py` | 필수 | "테스트 파일을 먼저 작성하세요 (tdd-red Skill 사용)" 출력 후 중단 |

## 출력

**파일**: `<모듈명>.py`
- 모든 테스트를 통과하는 구현 파일
- 실행 시 반드시 `PASSED` 상태

## 구현 원칙

### 핵심 규칙 (우선순위 순)

1. **최소한만 구현**: 테스트를 통과하는 가장 짧고 단순한 코드
2. **과잉 설계 금지**: 클래스, 상속, 디자인 패턴을 미리 도입하지 않음
3. **함수 크기 제한**: 함수 하나에 10줄을 넘기지 않음
4. **미래 기능 금지**: 테스트에 없는 기능을 미리 구현하지 않음
5. **외부 라이브러리 금지**: 표준 라이브러리만 사용

### 좋은 Green 코드 vs 나쁜 Green 코드

```python
# 나쁜 예: 과잉 설계 (Green 단계에서 이렇게 하지 마세요)
class TodoRepository:
    def __init__(self, storage_backend=None):
        self._items = []
        self._storage = storage_backend or InMemoryStorage()

# 좋은 예: 최소 구현
def add(items: list[str], item: str) -> list[str]:
    if not item:
        raise ValueError("항목은 빈 문자열일 수 없습니다.")
    return items + [item]
```

## 제약사항

- **`test_*.py` 파일을 절대 수정하지 않습니다.** 테스트를 변경하여 통과시키는 방법은 허용되지 않습니다.
- 전역 변수(mutable global state)를 사용하지 않습니다.
- PASSED가 되지 않는 테스트가 있으면 오류 메시지를 분석하여 재시도합니다.
- 구현 완료 후 반드시 `pytest test_<모듈명>.py -v`를 실행합니다.

## 실행 확인

Skill 완료 후 출력 예시:
```
PASSED test_calculator.py::test_add_positive_numbers
PASSED test_calculator.py::test_add_with_zero
PASSED test_calculator.py::test_add_invalid_input
3 passed, 0 failed ← 이 상태가 정상입니다.
```

## 이전 / 다음 Skill

- **이전**: `tdd-red` Skill의 출력 (`test_*.py`)을 입력으로 사용
- **다음**: 이 Skill의 출력 (`<모듈명>.py`)은 `tdd-refactor` Skill의 입력으로 사용

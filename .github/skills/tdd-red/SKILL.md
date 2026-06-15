---
name: tdd-red
description: TASK 항목을 입력받아 실패하는 pytest 테스트를 먼저 작성합니다. 구현 코드는 작성하지 않습니다.
---
# Skill: tdd-red

## 트리거 조건

다음 중 하나가 포함된 요청이 들어올 때 이 Skill을 사용합니다:
- "테스트 먼저 작성해줘" / "실패하는 테스트 만들어줘"
- "Red 단계 시작"
- "tdd-red Skill 사용해서"
- "TDD 시작", "TASK T0X에 대한 테스트 작성"

## 역할

TASK 항목을 입력받아 **실패하는 pytest 테스트**를 먼저 작성합니다.  
구현 코드는 작성하지 않습니다. 이 Skill이 완료된 후 `pytest`를 실행하면 모든 테스트가 `FAILED` 상태여야 합니다.

## 입력

| 항목 | 필수 여부 | 없을 경우 |
|------|----------|----------|
| TASK ID 또는 기능 설명 | 필수 | 사용자에게 질문 |
| `TASK.md` | 선택 | 기능 설명으로 직접 진행 |

## 출력

**파일 1**: `test_<모듈명>.py`
- pytest 테스트 파일
- 실행 시 반드시 `FAILED` 상태

**파일 2**: `<모듈명>.py`
- 빈 함수 껍데기만 포함 (`pass` 또는 `raise NotImplementedError()`)
- `ImportError` 방지 목적으로만 존재

## 테스트 작성 규칙

### 구조

```python
def test_함수명_시나리오():
    # Given: 사전 조건
    ...
    # When: 실행
    result = 함수명(...)
    # Then: 검증
    assert result == expected
```

### 필수 케이스 (TASK당 최소 3개)

| 케이스 유형 | 목적 | 예시 |
|-----------|------|------|
| Happy Path | 정상 동작 | `add(items, "할 일")` → 항목 포함 목록 |
| Edge Case | 경계값 | `add(items, "")` → `ValueError` |
| Error Case | 잘못된 입력 | `add(items, None)` → `TypeError` |

### 파일명 규칙

- 테스트 파일: `test_<모듈명>.py` (예: `test_calculator.py`)
- 구현 파일: `<모듈명>.py` (예: `calculator.py`)

## 제약사항

- **구현 코드를 절대 작성하지 않습니다.** 함수 본문은 `pass` 또는 `raise NotImplementedError()`만 허용됩니다.
- **테스트가 PASSED 상태가 되면 안 됩니다.** PASSED가 나오면 즉시 "테스트가 의미 없습니다. 재작성이 필요합니다."라고 알립니다.
- 이 Skill 완료 후 반드시 `pytest test_<모듈명>.py -v`를 실행하여 FAILED를 확인합니다.
- `test_*.py` 파일에 실제 구현이 포함되지 않도록 합니다.

## 실행 확인

Skill 완료 후 출력 예시:
```
FAILED test_calculator.py::test_add_positive_numbers - AssertionError
FAILED test_calculator.py::test_add_with_zero - AssertionError
FAILED test_calculator.py::test_add_invalid_input - Failed: DID NOT RAISE
3 failed, 0 passed ← 이 상태가 정상입니다.
```

## 이전 / 다음 Skill

- **이전**: `task-breakdown` Skill의 TASK 항목을 입력으로 사용
- **다음**: 이 Skill의 출력 (`test_*.py`)은 `tdd-green` Skill의 입력으로 사용

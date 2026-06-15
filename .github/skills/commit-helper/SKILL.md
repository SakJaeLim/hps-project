---
name: commit-helper
description: Git 커밋 메시지를 Conventional Commits 형식으로 작성할 때 사용하세요. 변경사항을 분석하고 커밋 메시지를 제안합니다.
---
# Git 커밋 메시지 작성 가이드

## Conventional Commits 형식

`<type>(<scope>): <description>`

## 사용 가능한 타입

- feat: 새로운 기능 추가
- fix: 버그 수정
- docs: 문서만 변경
- refactor: 코드 리팩터링 (기능 변경 없음)
- test: 테스트 추가 또는 수정
- chore: 빌드, 패키지, 설정 변경

## 작성 절차

1. 변경된 파일과 내용을 파악한다
2. 변경의 주요 목적을 한 줄로 요약한다
3. 적절한 type을 선택한다
4. description은 한국어 50자 이내로 간결하게 작성한다
5. 필요 시 본문(body)에 Why/What/How를 설명한다

## 완료 기준

- [ ] type이 정확한가?
- [ ] description이 50자 이내인가?
- [ ] 한국어 (또는 영어)로 일관성 있게 작성했는가?


## 커밋 템플릿

자세한 형식은 예를 들어 ./template.txt 같은 상대 경로 파일을 참조하도록 작성할 수 있습니다.
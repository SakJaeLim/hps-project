# PortSLM Safety SOP SFT 데이터셋 샘플

## 파일
- `portslm_safety_sft_sample.jsonl` — 시드 샘플 30건 (UTF-8, 1줄=1 JSON)

## 스키마
| 필드 | 설명 |
|---|---|
| type | safety_regulation_qa / hazard_diagnosis / procedure_guidance |
| instruction | 작업 지시 |
| input | 질문 또는 작업 상황 |
| output | 근거(안전 규정·통제 우선순위) 인용 정답 |
| meta | difficulty(easy/medium/hard), topic/violation 태그 |

## 구성 (30건)
- 안전 규정 Q&A 10 · 위험 진단/경보 10 · 비상/절차 안내 10
- 난이도: easy 6 · medium 16 · hard 8

## 세 유형의 학습 효과
- safety_regulation_qa: 규정·기준 지식(통제 우선순위, PTW, DG, 자동화구역 등)
- hazard_diagnosis: 상황 입력 → 위험요인·필요 통제·작업 중지 판정(안전 경보)
- procedure_guidance: 비상·고위험 작업의 단계별 절차 안내

## 샘플 수 권장
- 본 30건은 포맷·근거 인용 스타일 확정용 시드.
- 실제 LoRA 학습용 최종셋 권장: 유형별 150~500건, 합계 500~1,500건.
- 확장: 상황 파라미터 치환, 표현 다양화, 사고보고서(비식별) 기반 위험진단 자동 생성.

## 거버넌스
- 사내 Safety Manual·사고보고서 원문은 NDA·비식별 범위에서만 학습 사용.
- 공개 표준(ILO 항만 안전보건 실무규약·IMDG·CSC·ISPS) 근거는 자유 인용.
- output의 판정은 규정 기반 논리이며, 실제 적용 시 해당 터미널 인가 안전매뉴얼이 우선.

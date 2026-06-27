# PortSLM 적재계획 SFT 데이터셋 샘플

## 파일
- `portslm_stowage_sft_sample.jsonl` — 시드 샘플 30건 (UTF-8, 1줄 = 1 JSON)

## 스키마
| 필드 | 설명 |
|---|---|
| type | recommend_with_reason / regulation_qa / violation_diagnosis |
| instruction | 작업 지시 (시스템·유저 프롬프트로 활용) |
| input | 컨테이너 속성·계획 등 입력 컨텍스트 |
| output | 근거(SOP 조항)를 인용한 정답 |
| meta | difficulty(easy/medium/hard), constraints/topic/violation 태그 |

## 구성 (30건)
- 유형별 10건씩 균등: 슬롯 추천 10 · 규정 Q&A 10 · 위반 진단 10
- 난이도 분포: easy 7 · medium 13 · hard 10

## 샘플 수 권장
- 본 30건은 포맷·근거 인용 스타일을 확정하는 시드. 
- 실제 LoRA 학습용 최종셋 권장 규모: 유형별 150~500건, 합계 500~1,500건.
- 확장 방법: 이 30건을 템플릿으로 (1) 컨테이너 속성 파라미터 치환 합성, 
  (2) 동일 케이스의 표현 다양화(paraphrase), (3) 시뮬레이터 생성 상태로 자동 라벨링.

## 학습 시 변환 예 (Alpaca식)
prompt = instruction + "\n\n" + input  → completion = output
(또는 chat 포맷: system=instruction, user=input, assistant=output)

## 거버넌스
- output의 근거 인용은 공개 가능한 본선 플래닝 기준·IMDG·SOLAS 범위로 작성.
- 사내 SOP 원문 직접 인용은 NDA 범위 내에서만 학습에 사용.

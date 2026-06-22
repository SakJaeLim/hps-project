# 의사결정 지원 플랫폼 (decision-platform)

온톨로지 기반 컨테이너 터미널 운영 의사결정 지원 플랫폼 — **1개월 MVP**.
멀티에이전트(LangGraph) · 적재 최적화(Greedy→RL) · 도메인 SLM(LoRA)+RAG · Neo4j 온톨로지 · Streamlit 대시보드.

> 상세 계획은 `../../00_프로젝트_관리/실행계획서_의사결정지원플랫폼.docx` 참조.

## 아키텍처 (레이어)
| 레이어 | 내용 | 위치 |
|---|---|---|
| L1 데이터 | 어댑터(Simulated/Live), 캐노니컬 스키마 | `src/snct/data` |
| L2 온톨로지 | Neo4j 그래프 + Cypher 제약 | `src/snct/ontology` |
| L3 최적화 | Greedy 플래너 + Gym Env(+RL) | `src/snct/engine` |
| L4 에이전트 | LangGraph 인식·계획·검증·설명 | `src/snct/agents` |
| L5 도메인모델 | RAG + SLM LoRA 파인튜닝 | `src/snct/slm` |
| L6 앱 | FastAPI + Streamlit | `src/snct/api`, `dashboard` |

## 데이터 전략
`DataProvider` 추상 인터페이스 → `SimulatedProvider`(기본) / `LiveProvider`(실데이터).
**시뮬레이션 우선**, 실데이터 도착 시 정제 스크립트만 추가해 어댑터 교체. `data/real/`는 git 추적 제외.

## 시작하기
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
# Neo4j 기동 후 .env 설정 (.env.example 참고)
streamlit run dashboard/app.py
```

## 개발 방법론
- **SDD**: `specs/` 명세 먼저 작성·승인 후 구현.
- **GitHub flow**: `main` 보호, `feature/*` 브랜치, PR 1인 이상 리뷰.
- **AI Native**: Cursor/Claude + Context7 MCP로 명세 기반 바이브코딩.

## 팀
도메인 전문가 · 기획/온톨로지 · **PL(아키텍처·핵심개발)** · 프론트엔드

## SDD (Spec-Driven Development, 5일차 AI-Native)
단일 진실원천 체인: **PRD.md → specs/ → TASK.md → tests/(TDD Red) → src/(Green)**.
- `PRD.md` 무엇/왜 · `specs/` 어떻게 · `TASK.md` 원자적 작업 · `tests/` 완료조건 · `src/` 구현.
- TDD: `tests/`는 현재 미구현이 `xfail`(Red). 구현 시 xfail 제거 → Green. `pytest`로 검증.
- AI-Native: `.github/`(copilot-instructions·skills·instructions). 상세 `docs/SDD.md`.

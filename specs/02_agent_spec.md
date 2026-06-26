# 02. 에이전트 명세 (LangGraph)

## 상태(State)
{request, yard_state, candidate_plan, violations, rationale, retries}

## 노드
- recognize: 데이터 어댑터 → 온톨로지 반영, yard_state 적재
- plan: get_strategy(기본 'rl', ADR-0001).plan(yard) → candidate_plan  # rl=원우모델 / greedy=기준선 / cp=향후
- validate: validate_constraints(Cypher) → violations
- explain: knowledge.router(문서RAG·NL2SQL·GraphRAG) 근거 수집 → explain() 융합 → rationale (specs/07)
## 엣지
recognize→plan→validate→(violations? plan : explain)→END ; 최대 retries=3

## 도구(tools)
query_ontology(graph), run_stowage_engine, validate_constraints, search_docs(RAG), query_db(NL2SQL)

## 설명 흐름 (xAI-RL, specs/07) — `run_explanation`
기존 계획 흐름과 별개로, **이미 산출된 RL 의사결정을 설명**하는 진입점.
- 상태: {question | (policy, round_id), decision, lpg, rationale, checks}
- 노드: recognize(질의→`RLDecisionRef` 파싱) → collect(RDB `RLResultStore` + LPG `LPGGraph`) →
  explain(`explain(decision, lpg)` 융합) → verify(`faithfulness.score_decision` 자기검증)
- 산출: `Recommendation{rationale, checks=[policy, round, faithfulness]}`
- 가드: 파싱 실패·미존재 의사결정은 그레이스풀 메시지 반환.

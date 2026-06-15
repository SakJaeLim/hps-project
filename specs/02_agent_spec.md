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

# PortSLM SOLAS 통합 데이터 패키지

원천: Focus on IMO — SOLAS 1974 (IMO, October 1998) / 표준: SOLAS 1974 (as amended)

## 산출물
| 단계 | 파일 | 건수 |
|---|---|---|
| 설계서 | PortSLM_SOLAS_통합데이터파이프라인_설계서.docx | — |
| ② 청킹 | chunks/solas_chunks.jsonl | 29 (章13+개정14+개념2) |
| ③ SFT | sft/solas_sft.jsonl | 17 |
| ④ 임베딩 | embedding/vector_schema.json, vector_records.jsonl | 29 |
| ⑤ GraphRAG | graphrag/nodes_*.csv(7), rels_*.csv(7), cypher_templates.cypher, neo4j_import.sh | 노드 76 |

## SOLAS 특성 반영
- 이중 축: 章(Chapter) + 개정(Amendment) + 계기 사건(Event)
- 인과 보존: Amendment-TRIGGERED_BY->Event (Herald·Estonia 등)
- port_relevance 태그로 적재 관련 규정(II-1·VI·VII·XII) 사전 필터

## COPRAR 패키지와 통합
- 동일 BGE-M3(1024d) → 단일/멀티 컬렉션 통합 검색
- Constraint(stability·weight·securing·DG) 동일 키 → 다중 근거 설명
  ("적재 규칙=COPRAR" + "안전 근거=SOLAS")

## 적재
- 벡터DB: BGE-M3로 embedding_input 임베딩 → Chroma/pgvector
- KG: `bash graphrag/neo4j_import.sh`

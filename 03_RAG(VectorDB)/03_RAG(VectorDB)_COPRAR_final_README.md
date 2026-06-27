# PortSLM COPRAR 통합 데이터 패키지

원천: COPRAR EDIFACT User Guide v2.3 (Valenciaport PCS) / 표준: UN/EDIFACT D.00B · SMDG v2.0

## 산출물
| 단계 | 파일 | 건수 |
|---|---|---|
| 설계서 | PortSLM_COPRAR_통합데이터파이프라인_설계서.docx | — |
| ② 청킹 | chunks/portslm_coprar_chunks.jsonl | 26 |
| ③ SFT | sft/portslm_coprar_sft.jsonl | 21 |
| ④ 임베딩 | embedding/vector_schema.json, vector_records.jsonl | 26 |
| ⑤ GraphRAG | graphrag/nodes_*.csv(6), rels_*.csv(5), cypher_templates.cypher, neo4j_import.sh | 노드 203 |

## 핵심 키 연계
chunk_id == 벡터DB id == KG Segment.vector_ref  → 그래프 추론에서 원문 청크 즉시 검색

## 적재
- 벡터DB: BGE-M3(1024d)로 embedding_input 임베딩 → Chroma/pgvector
- KG: `bash graphrag/neo4j_import.sh`

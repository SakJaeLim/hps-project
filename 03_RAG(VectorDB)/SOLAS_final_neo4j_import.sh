#!/usr/bin/env bash
# Neo4j 적재 (SOLAS KG)
neo4j-admin database import full solas \
  --nodes=Document=graphrag/nodes_document.csv \
  --nodes=Chapter=graphrag/nodes_chapter.csv \
  --nodes=Amendment=graphrag/nodes_amendment.csv \
  --nodes=Event=graphrag/nodes_event.csv \
  --nodes=EvidenceChunk=graphrag/nodes_chunk.csv \
  --nodes=Code=graphrag/nodes_code.csv \
  --nodes=Constraint=graphrag/nodes_constraint.csv \
  --relationships=DESCRIBED_IN=graphrag/rels_described_in.csv \
  --relationships=DESCRIBES=graphrag/rels_describes.csv \
  --relationships=AMENDS=graphrag/rels_amends.csv \
  --relationships=TRIGGERED_BY=graphrag/rels_triggered_by.csv \
  --relationships=REFERENCES_CODE=graphrag/rels_references_code.csv \
  --relationships=SUPPORTS=graphrag/rels_supports.csv \
  --relationships=EVIDENCED_BY=graphrag/rels_evidenced_by.csv \
  --overwrite-destination

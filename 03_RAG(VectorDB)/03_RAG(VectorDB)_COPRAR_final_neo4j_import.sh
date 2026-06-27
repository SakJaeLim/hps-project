#!/usr/bin/env bash
# Neo4j 적재 (neo4j-admin database import full)
neo4j-admin database import full coprar \
  --nodes=Document=graphrag/nodes_document.csv \
  --nodes=SegmentGroup=graphrag/nodes_segmentgroup.csv \
  --nodes=Segment=graphrag/nodes_segment.csv \
  --nodes=Element=graphrag/nodes_element.csv \
  --nodes=CodeValue=graphrag/nodes_codevalue.csv \
  --nodes=Constraint=graphrag/nodes_constraint.csv \
  --relationships=PART_OF=graphrag/rels_part_of.csv \
  --relationships=DESCRIBED_IN=graphrag/rels_described_in.csv \
  --relationships=HAS_ELEMENT=graphrag/rels_has_element.csv \
  --relationships=USES_CODE=graphrag/rels_uses_code.csv \
  --relationships=SUPPORTED_BY=graphrag/rels_supported_by.csv \
  --overwrite-destination

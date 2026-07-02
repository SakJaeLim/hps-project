// ════ Neo4j import (LOAD CSV) — 단일베이 적재계획 지식그래프 ════
CREATE CONSTRAINT vessel_id IF NOT EXISTS FOR (n:Vessel) REQUIRE n.vessel_id IS UNIQUE;
CREATE CONSTRAINT bay_id    IF NOT EXISTS FOR (n:Bay) REQUIRE n.bay_id IS UNIQUE;
CREATE CONSTRAINT row_id    IF NOT EXISTS FOR (n:Row) REQUIRE n.row_id IS UNIQUE;
CREATE CONSTRAINT tier_id   IF NOT EXISTS FOR (n:Tier) REQUIRE n.tier_id IS UNIQUE;
CREATE CONSTRAINT slot_id   IF NOT EXISTS FOR (n:Slot) REQUIRE n.slot_id IS UNIQUE;
CREATE CONSTRAINT cont_id   IF NOT EXISTS FOR (n:Container) REQUIRE n.container_id IS UNIQUE;
CREATE CONSTRAINT port_id   IF NOT EXISTS FOR (n:Port) REQUIRE n.pod_id IS UNIQUE;
CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (n:Policy) REQUIRE n.policy IS UNIQUE;
CREATE CONSTRAINT cons_id   IF NOT EXISTS FOR (n:Constraint) REQUIRE n.cons_id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///Vessel.csv' AS r CREATE (:Vessel {vessel_id:r.`vessel_id:ID(Vessel)`, voyage:r.voyage});
LOAD CSV WITH HEADERS FROM 'file:///Bay.csv' AS r CREATE (:Bay {bay_id:r.`bay_id:ID(Bay)`, vessel:r.vessel});
LOAD CSV WITH HEADERS FROM 'file:///Row.csv' AS r CREATE (:Row {row_id:r.`row_id:ID(Row)`, policy:r.policy, round_id:toInteger(r.`round_id:int`), row:toInteger(r.`row:int`)});
LOAD CSV WITH HEADERS FROM 'file:///Tier.csv' AS r CREATE (:Tier {tier_id:r.`tier_id:ID(Tier)`, row_id:r.row_id, tier:toInteger(r.`tier:int`)});
LOAD CSV WITH HEADERS FROM 'file:///Slot.csv' AS r CREATE (:Slot {slot_id:r.`slot_id:ID(Slot)`, policy:r.policy, round_id:toInteger(r.`round_id:int`), row:toInteger(r.`row:int`), tier:toInteger(r.`tier:int`), is_bottom:toInteger(r.`is_bottom:int`), is_top:toInteger(r.`is_top:int`)});
LOAD CSV WITH HEADERS FROM 'file:///Container.csv' AS r CREATE (:Container {container_id:r.`container_id:ID(Container)`, policy:r.policy, round_id:toInteger(r.`round_id:int`), pod_id:toInteger(r.`pod_id:int`), weight_mt:toFloat(r.`weight_mt:float`), row:toInteger(r.`row:int`), tier:toInteger(r.`tier:int`)});
LOAD CSV WITH HEADERS FROM 'file:///Port.csv' AS r CREATE (:Port {pod_id:toInteger(r.`pod_id:ID(Port)`), pod_name:r.pod_name, distance_rank:toInteger(r.`distance_rank:int`)});
LOAD CSV WITH HEADERS FROM 'file:///Policy.csv' AS r CREATE (:Policy {policy:r.`policy:ID(Policy)`, desc:r.desc});
LOAD CSV WITH HEADERS FROM 'file:///Constraint.csv' AS r CREATE (:Constraint {cons_id:r.`cons_id:ID(Constraint)`, code:r.code, rule:r.rule, source:r.source});

LOAD CSV WITH HEADERS FROM 'file:///REL_HAS_ROW.csv' AS r MATCH (a:Bay {bay_id:r.`:START_ID(Bay)`}),(b:Row {row_id:r.`:END_ID(Row)`}) CREATE (a)-[:HAS_ROW]->(b);
LOAD CSV WITH HEADERS FROM 'file:///REL_HAS_TIER.csv' AS r MATCH (a:Row {row_id:r.`:START_ID(Row)`}),(b:Tier {tier_id:r.`:END_ID(Tier)`}) CREATE (a)-[:HAS_TIER]->(b);
LOAD CSV WITH HEADERS FROM 'file:///REL_HAS_SLOT.csv' AS r MATCH (a:Tier {tier_id:r.`:START_ID(Tier)`}),(b:Slot {slot_id:r.`:END_ID(Slot)`}) CREATE (a)-[:HAS_SLOT]->(b);
LOAD CSV WITH HEADERS FROM 'file:///REL_ASSIGNED_TO.csv' AS r MATCH (a:Container {container_id:r.`:START_ID(Container)`}),(b:Slot {slot_id:r.`:END_ID(Slot)`}) CREATE (a)-[:ASSIGNED_TO]->(b);
LOAD CSV WITH HEADERS FROM 'file:///REL_HAS_POD.csv' AS r MATCH (a:Container {container_id:r.`:START_ID(Container)`}),(b:Port {pod_id:toInteger(r.`:END_ID(Port)`)}) CREATE (a)-[:HAS_POD]->(b);
LOAD CSV WITH HEADERS FROM 'file:///REL_STACKED_ON.csv' AS r MATCH (a:Container {container_id:r.`:START_ID(Container)`}),(b:Container {container_id:r.`:END_ID(Container)`}) CREATE (a)-[:STACKED_ON {is_overstow:toInteger(r.`is_overstow:int`)}]->(b);
LOAD CSV WITH HEADERS FROM 'file:///REL_VIOLATES.csv' AS r MATCH (a:Slot {slot_id:r.`:START_ID(Slot)`}),(b:Constraint {cons_id:r.`:END_ID(Constraint)`}) CREATE (a)-[:VIOLATES]->(b);
LOAD CSV WITH HEADERS FROM 'file:///REL_ACHIEVED.csv' AS r MATCH (a:Policy {policy:r.`:START_ID(Policy)`}),(b:Bay {bay_id:r.`:END_ID(Bay)`}) CREATE (a)-[:ACHIEVED {round_id:toInteger(r.`round_id:int`), reward:toFloat(r.`reward:float`), osr:toFloat(r.`osr:float`), wbi:toFloat(r.`wbi:float`), vpr:toFloat(r.`vpr:float`), psr:toFloat(r.`psr:float`), cwvr:toFloat(r.`cwvr:float`)}]->(b);

// ════ text2Cypher 검증 쿼리 예시 ════
// Q. "SF 정책에서 재취급(overstow)이 발생한 슬롯은?"
// MATCH (s:Slot {policy:'SF'})-[:VIOLATES]->(c:Constraint {cons_id:'C_OVERSTOW'})
// RETURN s.round_id, s.row, s.tier;
// Q. "이 배정이 컬럼 무게 제약(SOLAS)을 위반하는가?"
// MATCH (s:Slot)-[:VIOLATES]->(c:Constraint {code:'SOLAS_VI'}) RETURN s.policy, s.round_id, s.row;
// Q. "오버스토우 관계 체인 (Container STACKED_ON, is_overstow=1)"
// MATCH (up:Container)-[r:STACKED_ON {is_overstow:1}]->(down:Container)
// RETURN up.policy, up.row, up.tier, up.pod_id AS upper_pod, down.pod_id AS lower_pod;

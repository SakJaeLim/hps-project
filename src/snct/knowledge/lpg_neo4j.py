"""T28 · spec 07 — LPG Neo4j 백엔드 (그래프DB 실연결).

CSV 폴백(`lpg_csv.LPGGraph`)과 동일 인터페이스를 Cypher로 구현한다.
Neo4j가 없으면 `is_available()`가 False → 팩토리가 CSV로 폴백.
KG 적재는 neo4j_kg/*.csv를 드라이버로 MERGE(Neo4j import 디렉터리 설정 불필요).
"""
from __future__ import annotations
import os
from pathlib import Path

from snct.data.sources.rl_results import RLResultStore


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


class Neo4jLPG:
    """neo4j_kg 그래프에 대한 Cypher 질의자. (CSV LPGGraph와 동일 인터페이스)"""

    def __init__(self, uri: str | None = None, user: str | None = None,
                 password: str | None = None, kg_dir: str | Path | None = None):
        self.uri = uri or _env("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or _env("NEO4J_USER", "neo4j")
        self.password = password or _env("NEO4J_PASSWORD", "neo4j")
        self.kg_dir = Path(kg_dir) if kg_dir else None
        self._driver = None

    # ── 연결 ────────────────────────────────────────────────
    def _connect(self):
        if self._driver is None:
            import neo4j
            self._driver = neo4j.GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def is_available(self) -> bool:
        """서버 연결 가능 여부. 드라이버 미설치·서버 미실행·인증실패 시 False."""
        try:
            self._connect().verify_connectivity()
            return True
        except Exception:
            self._driver = None
            return False

    def close(self):
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _run(self, cypher: str, **params) -> list[dict]:
        with self._connect().session() as s:
            return [dict(r) for r in s.run(cypher, **params)]

    # ── 적재(import) ────────────────────────────────────────
    def import_kg(self, kg_dir: str | Path | None = None) -> dict:
        """neo4j_kg CSV를 드라이버로 적재(MERGE). 질의에 필요한 핵심 노드/관계만.
        → {containers, slots, constraints, assigned, violates, stacked} 카운트."""
        import pandas as pd
        from snct.knowledge.lpg_csv import _read, _strip_col

        kg = Path(kg_dir) if kg_dir else (self.kg_dir or RLResultStore().kg_dir)

        def load(name):
            df = _read(kg / name)
            df.columns = [_strip_col(c) for c in df.columns]
            return df

        cont = load("Container.csv")
        slot = load("Slot.csv")
        cons = load("Constraint.csv")
        assigned = load("REL_ASSIGNED_TO.csv")
        violates = load("REL_VIOLATES.csv")
        stacked = load("REL_STACKED_ON.csv")

        with self._connect().session() as s:
            s.run("CREATE CONSTRAINT container_id IF NOT EXISTS "
                  "FOR (n:Container) REQUIRE n.container_id IS UNIQUE")
            s.run("CREATE CONSTRAINT slot_id IF NOT EXISTS "
                  "FOR (n:Slot) REQUIRE n.slot_id IS UNIQUE")
            s.run("CREATE CONSTRAINT cons_id IF NOT EXISTS "
                  "FOR (n:Constraint) REQUIRE n.cons_id IS UNIQUE")

            for _, r in cont.iterrows():
                s.run("MERGE (c:Container {container_id:$id}) "
                      "SET c.policy=$p, c.round_id=$rd, c.row=$row, c.tier=$tier",
                      id=r["container_id"], p=str(r.get("policy")),
                      rd=int(r.get("round_id", 0)), row=int(r.get("row", 0)), tier=int(r.get("tier", 0)))
            for _, r in slot.iterrows():
                s.run("MERGE (sl:Slot {slot_id:$id}) SET sl.policy=$p, sl.round_id=$rd, sl.row=$row, sl.tier=$tier",
                      id=r["slot_id"], p=str(r.get("policy")), rd=int(r.get("round_id", 0)),
                      row=int(r.get("row", 0)), tier=int(r.get("tier", 0)))
            for _, r in cons.iterrows():
                s.run("MERGE (k:Constraint {cons_id:$id}) SET k.code=$code, k.rule=$rule, k.source=$src",
                      id=r["cons_id"], code=r["code"], rule=r["rule"], src=r["source"])
            for _, r in assigned.iterrows():
                s.run("MATCH (c:Container {container_id:$c}),(sl:Slot {slot_id:$s}) MERGE (c)-[:ASSIGNED_TO]->(sl)",
                      c=r["START"], s=r["END"])
            for _, r in violates.iterrows():
                s.run("MATCH (sl:Slot {slot_id:$s}),(k:Constraint {cons_id:$k}) MERGE (sl)-[:VIOLATES]->(k)",
                      s=r["START"], k=r["END"])
            for _, r in stacked.iterrows():
                s.run("MATCH (a:Container {container_id:$a}),(b:Container {container_id:$b}) "
                      "MERGE (a)-[r:STACKED_ON]->(b) SET r.is_overstow=$ov",
                      a=r["START"], b=r["END"], ov=int(r.get("is_overstow", 0) or 0))

        return {"containers": len(cont), "slots": len(slot), "constraints": len(cons),
                "assigned": len(assigned), "violates": len(violates), "stacked": len(stacked)}

    # ── 템플릿 질의 (CSV LPGGraph와 동일 시그니처) ──────────
    def stacked_on(self, container_id: str) -> list[dict]:
        rows = self._run(
            "MATCH (a:Container)-[r:STACKED_ON]->(b:Container {container_id:$id}) "
            "RETURN a.container_id AS container_id, coalesce(r.is_overstow,0) AS is_overstow",
            id=container_id)
        return [{"container_id": r["container_id"], "is_overstow": int(r["is_overstow"])} for r in rows]

    def violations_of(self, container_id: str) -> list[dict]:
        return self._run(
            "MATCH (c:Container {container_id:$id})-[:ASSIGNED_TO]->(:Slot)-[:VIOLATES]->(k:Constraint) "
            "RETURN k.cons_id AS cons_id, k.code AS code, k.rule AS rule, k.source AS source",
            id=container_id)

    def violations_in_round(self, policy: str, round_id: int) -> list[dict]:
        return self._run(
            "MATCH (sl:Slot {policy:$p, round_id:$rd})-[:VIOLATES]->(k:Constraint) "
            "OPTIONAL MATCH (c:Container)-[:ASSIGNED_TO]->(sl) "
            "RETURN c.container_id AS container_id, sl.slot_id AS slot_id, "
            "sl.row AS row, sl.tier AS tier, k.code AS code, k.rule AS rule, k.source AS source",
            p=str(policy), rd=int(round_id))

    def constraint(self, code_or_id: str) -> dict:
        rows = self._run(
            "MATCH (k:Constraint) WHERE k.code=$x OR k.cons_id=$x "
            "RETURN k.cons_id AS cons_id, k.code AS code, k.rule AS rule, k.source AS source",
            x=code_or_id)
        if not rows:
            raise KeyError(f"제약 없음: {code_or_id}")
        return rows[0]

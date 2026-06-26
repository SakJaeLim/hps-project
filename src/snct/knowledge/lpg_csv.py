"""T19 · spec 07 — LPG(neo4j_kg) CSV 폴백 그래프 질의.

Neo4j 실연결 없이 RL 결과의 neo4j_kg/*.csv를 직접 읽어 관계·제약 근거를 질의한다.
설계(specs/07): 핵심 질의는 **파라미터 템플릿 우선**. 자유질의 text2Cypher는 후속.
Neo4j 도입 시 동일 인터페이스(stacked_on/violations_of/constraint)를 Cypher로 교체.

그래프 스키마(neo4j_kg):
  Container --ASSIGNED_TO--> Slot --VIOLATES--> Constraint
  Container(상위 Tier) --STACKED_ON--> Container(하위 Tier)  [is_overstow:0|1]
"""
from __future__ import annotations
from pathlib import Path
import re

import pandas as pd

from snct.data.sources.rl_results import RLResultStore


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _strip_col(name: str) -> str:
    """neo4j-admin 헤더 정규화: 'container_id:ID(Container)' → 'container_id',
    ':START_ID(Container)' → 'START', ':END_ID(Constraint)' → 'END', ':TYPE'→'TYPE'."""
    if name.startswith(":START_ID"):
        return "START"
    if name.startswith(":END_ID"):
        return "END"
    if name.startswith(":"):
        return name[1:].split("(")[0]
    return name.split(":")[0]


class LPGGraph:
    """neo4j_kg CSV에 대한 읽기 전용 그래프 질의자. (csv 폴백)"""

    def __init__(self, kg_dir: str | Path | None = None):
        self.kg_dir = Path(kg_dir) if kg_dir else RLResultStore().kg_dir
        if not self.kg_dir.is_dir():
            raise FileNotFoundError(f"neo4j_kg 폴더 없음: {self.kg_dir}")
        self._cache: dict[str, pd.DataFrame] = {}

    def _t(self, fname: str) -> pd.DataFrame:
        if fname not in self._cache:
            df = _read(self.kg_dir / fname)
            df.columns = [_strip_col(c) for c in df.columns]
            self._cache[fname] = df
        return self._cache[fname]

    # ── 템플릿 질의 ─────────────────────────────────────────
    def stacked_on(self, container_id: str) -> list[dict]:
        """container_id '위에' 직접 쌓인 컨테이너(상위 Tier)들. is_overstow 포함."""
        rel = self._t("REL_STACKED_ON.csv")
        upper = rel[rel["END"] == container_id]
        return [
            {"container_id": r["START"], "is_overstow": int(r.get("is_overstow", 0) or 0)}
            for _, r in upper.iterrows()
        ]

    def violations_of(self, container_id: str) -> list[dict]:
        """컨테이너 → 슬롯(ASSIGNED_TO) → 제약(VIOLATES) 조인 → 위반 제약 상세."""
        assigned = self._t("REL_ASSIGNED_TO.csv")
        slots = assigned[assigned["START"] == container_id]["END"].tolist()
        if not slots:
            return []
        viol = self._t("REL_VIOLATES.csv")
        cons_ids = viol[viol["START"].isin(slots)]["END"].tolist()
        if not cons_ids:
            return []
        cdf = self._t("Constraint.csv")
        out = []
        for cid in cons_ids:
            row = cdf[cdf["cons_id"] == cid]
            if not row.empty:
                r = row.iloc[0]
                out.append({"cons_id": cid, "code": r["code"], "rule": r["rule"], "source": r["source"]})
        return out

    def violations_in_round(self, policy: str, round_id: int) -> list[dict]:
        """(policy, round_id)에서 위반한 슬롯 → 컨테이너·제약 조인.
        → [{container_id, slot_id, row, tier, code, rule, source}]."""
        slots = self._t("Slot.csv").copy()
        
        # 타입 불일치 및 문자열 공백(BOM 등) 완벽 차단용 가드
        if "policy" in slots.columns:
            slots["policy"] = slots["policy"].astype(str).str.strip()
        if "round_id" in slots.columns:
            slots["round_id"] = pd.to_numeric(slots["round_id"], errors="coerce")
            
        target_policy = str(policy).strip()
        target_round = float(round_id)
        
        sel = slots[(slots["policy"] == target_policy) & (slots["round_id"] == target_round)]
        if sel.empty:
            return []
            
        slot_ids = set(sel["slot_id"].astype(str).str.strip())
        if not slot_ids:
            return []
            
        viol = self._t("REL_VIOLATES.csv").copy()
        viol["START"] = viol["START"].astype(str).str.strip()
        viol = viol[viol["START"].isin(slot_ids)]
        if viol.empty:
            return []
            
        assigned = self._t("REL_ASSIGNED_TO.csv").copy()
        assigned["START"] = assigned["START"].astype(str).str.strip()
        assigned["END"] = assigned["END"].astype(str).str.strip()
        
        cdf = self._t("Constraint.csv").copy()
        cdf["cons_id"] = cdf["cons_id"].astype(str).str.strip()
        
        slot_loc = {str(r["slot_id"]).strip(): (r.get("row"), r.get("tier")) for _, r in sel.iterrows()}
        out = []
        for _, vr in viol.iterrows():
            slot_id = str(vr["START"]).strip()
            cons_id = str(vr["END"]).strip()
            
            cont = assigned[assigned["END"] == slot_id]["START"]
            container_id = cont.iloc[0] if not cont.empty else None
            
            crow = cdf[cdf["cons_id"] == cons_id]
            code = crow.iloc[0]["code"] if not crow.empty else cons_id
            rule = crow.iloc[0]["rule"] if not crow.empty else ""
            source = crow.iloc[0]["source"] if not crow.empty else ""
            row, tier = slot_loc.get(slot_id, (None, None))
            out.append({
                "container_id": container_id, "slot_id": slot_id,
                "row": row, "tier": tier,
                "code": code, "rule": rule, "source": source,
            })
        return out

    def constraint(self, code_or_id: str) -> dict:
        """제약을 code 또는 cons_id로 조회."""
        cdf = self._t("Constraint.csv")
        row = cdf[(cdf["code"] == code_or_id) | (cdf["cons_id"] == code_or_id)]
        if row.empty:
            raise KeyError(f"제약 없음: {code_or_id}")
        r = row.iloc[0]
        return {"cons_id": r["cons_id"], "code": r["code"], "rule": r["rule"], "source": r["source"]}

    # ── 자연어 라우팅 ───────────────────────────────────────
    @staticmethod
    def _extract_container_id(question: str) -> str | None:
        m = re.search(r"[A-Z]{2}_R\d+_r\d+_t\d+", question)
        return m.group(0) if m else None

    def ask(self, question: str) -> dict:
        """자연어 → 템플릿 라우팅 → {answer, sources:[Evidence{type:'graph'}]}."""
        cid = self._extract_container_id(question)
        sources, lines = [], []

        if cid and any(k in question for k in ("위반", "규정", "제약", "violat")):
            for v in self.violations_of(cid):
                lines.append(f"{cid} → [{v['code']}/{v['source']}] {v['rule']}")
                sources.append({"type": "graph", "ref": v["cons_id"],
                                "snippet": f"{cid} VIOLATES {v['code']}: {v['rule']}"})
        elif cid and any(k in question for k in ("위", "쌓", "stack", "재취급", "overstow")):
            for s in self.stacked_on(cid):
                tag = " (재취급/overstow)" if s["is_overstow"] else ""
                lines.append(f"{cid} 위에 적재: {s['container_id']}{tag}")
                sources.append({"type": "graph", "ref": s["container_id"],
                                "snippet": f"{s['container_id']} STACKED_ON {cid}{tag}"})

        if not sources:
            return {"answer": "관련 그래프 근거를 찾을 수 없습니다.", "sources": []}
        return {"answer": "\n".join(lines), "sources": sources}

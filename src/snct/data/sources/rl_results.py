"""T18 · spec 07 — 강화학습 결과 자료 로더 (xAI-RL 근거 소스).

RL(PPO) 학습이 산출한 설명 근거를 캐노니컬하게 적재한다:
  - RDB(정형): reward_decomp(R1~R15 ±기여=귀인)·kpi·slot_assignment·violation_log
  - RAG(비정형): xai_grounding(이미 융합된 정답 레퍼런스)·rag_chunks
  - LPG(그래프): neo4j_kg/*.csv  (T19에서 CSV 폴백 질의)

설계 원칙(specs/07): 설명은 자유 생성이 아니라 **이 사실들의 검색·융합**이다.
폴더명에 seed/version이 박혀 있어 글롭으로 자동 탐색한다.
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from snct.common.schema import RLDecision

# reward_decomp 컬럼(R1~R15) → 운영자 친화 라벨 (설명 표현은 explain.py가 사용)
REWARD_LABELS = {
    "R1_valid": "유효 배치",
    "R2_stack_full": "스택 충진",
    "R3_overstow": "재취급(overstow) 회피",
    "R4_order": "반출 순서",
    "R5_weight_bal": "행간 무게 균형",
    "R6_cog": "무게중심(COG)",
    "R7_completion": "적재 완료",
    "R8_pod_band": "동일 POD 밴드 정렬",
    "R9_col_wt": "컬럼 중량 제한(SOLAS)",
    "R10_tier_match": "Tier 정합",
    "R11_wt_inversion": "중량 역전 회피",
    "R12_col_order": "컬럼 순서",
    "R13_tier_band": "Same-Tier 밴드 정렬",
    "R14_empty_row": "빈 행 패널티",
    "R15_vstack_pod": "동일 POD 수직 적재",
}


def _repo_root() -> Path:
    # src/snct/data/sources/rl_results.py → parents[4] = repo root
    return Path(__file__).resolve().parents[4]


def default_results_dir() -> Path:
    # 깃 저장소 내 실제 경로는 data/RL/강화학습 결과 자료 입니다.
    nfc_path = _repo_root() / "data" / "RL" / "강화학습 결과 자료"
    if nfc_path.is_dir():
        return nfc_path
    
    # OS별 자소분리(NFD) 폴더명 대응
    nfd_path = _repo_root() / "data" / "RL" / "강화학습 결과 자료"
    if nfd_path.is_dir():
        return nfd_path
        
    return nfc_path


def _read_csv(path: Path) -> pd.DataFrame:
    # CSV 다수에 UTF-8 BOM(﻿)이 있어 utf-8-sig로 읽는다.
    return pd.read_csv(path, encoding="utf-8-sig")


class RLResultStore:
    """RL 결과 자료에 대한 읽기 전용 접근자. (csv/json 폴백, 외부 DB 불필요)"""

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else default_results_dir()
        if not self.base_dir.is_dir():
            raise FileNotFoundError(f"RL 결과 자료 폴더 없음: {self.base_dir}")
        self.rdb_dir = self._discover("*RDB_LPG*/rdb")
        self.kg_dir = self._discover("*RDB_LPG*/neo4j_kg")
        self.rag_dir = self._discover_dir("*RAG*")

    # ── 탐색 ────────────────────────────────────────────────
    def _discover(self, pattern: str) -> Path:
        for p in sorted(self.base_dir.glob(pattern)):
            if p.is_dir():
                return p
        raise FileNotFoundError(f"산출물 폴더 없음: {self.base_dir}/{pattern}")

    def _discover_dir(self, pattern: str) -> Path:
        # .zip 제외, 디렉터리만
        for p in sorted(self.base_dir.glob(pattern)):
            if p.is_dir():
                return p
        raise FileNotFoundError(f"산출물 폴더 없음: {self.base_dir}/{pattern}")

    # ── 로딩 ────────────────────────────────────────────────
    def _load(self, rel: str) -> pd.DataFrame:
        df = _read_csv(self.rdb_dir / rel)
        if "policy" in df.columns:
            df["policy"] = df["policy"].astype(str)
        if "round_id" in df.columns:
            df["round_id"] = pd.to_numeric(df["round_id"], errors="coerce").astype("Int64")
        return df

    def load_reward_decomp(self) -> pd.DataFrame:
        return self._load("reward_decomp.csv")

    def load_kpi(self) -> pd.DataFrame:
        return self._load("kpi.csv")

    def load_slot_assignment(self) -> pd.DataFrame:
        return self._load("slot_assignment.csv")

    def load_violation_log(self) -> pd.DataFrame:
        return self._load("violation_log.csv")

    def load_xai_grounding(self) -> list[dict]:
        with open(self.rag_dir / "xai_grounding.json", encoding="utf-8") as f:
            return json.load(f)

    # ── 융합 ────────────────────────────────────────────────
    def get_decision(self, policy: str, round_id: int) -> RLDecision:
        """(policy, round_id) 의사결정 근거를 융합해 RLDecision으로 반환.
        존재하지 않으면 KeyError."""
        rd = self.load_reward_decomp()
        row = rd[(rd["policy"] == str(policy)) & (rd["round_id"] == int(round_id))]
        if row.empty:
            raise KeyError(f"의사결정 없음: policy={policy}, round_id={round_id}")
        row = row.iloc[0]

        # 귀인: R-term 절댓값 내림차순(0 제외)
        contribs = []
        for col in rd.columns:
            if col in REWARD_LABELS:
                val = float(row[col])
                if val != 0.0:
                    contribs.append((col, val))
        contribs.sort(key=lambda kv: abs(kv[1]), reverse=True)

        # KPI
        kpi_df = self.load_kpi()
        krow = kpi_df[(kpi_df["policy"] == str(policy)) & (kpi_df["round_id"] == int(round_id))]
        kpi = {} if krow.empty else {
            k: (float(v) if pd.notna(v) else None)
            for k, v in krow.iloc[0].items()
            if k not in ("policy", "round_id", "level")
        }

        # 위반 로그
        vl = self.load_violation_log()
        vrows = vl[(vl["policy"] == str(policy)) & (vl["round_id"] == int(round_id))]
        violations = vrows.to_dict("records")

        # 근거(grounding) — 이미 융합된 rationale/doc_refs
        rationale, doc_refs = [], []
        for rec in self.load_xai_grounding():
            if str(rec.get("policy")) == str(policy) and int(rec.get("round_id", -1)) == int(round_id):
                rationale = rec.get("rationale", [])
                doc_refs = rec.get("doc_refs", [])
                break

        return RLDecision(
            policy=str(policy),
            round_id=int(round_id),
            reward_total=float(row["reward_total"]),
            top_contributions=contribs,
            kpi=kpi,
            violations=violations,
            rationale=rationale,
            doc_refs=doc_refs,
        )

"""T27 · spec 07 — 컨테이너 위치 조회 (완성된 적재계획 slot_assignment).

자연어로 "X 컨테이너 어디 있어?" → 위치(bay/row/tier) + 적층 정보(반출 가능 여부).
근거 소스: slot_assignment(RDB, 위치) + LPGGraph.stacked_on(LPG, 위에 쌓인 것).

실데이터(TOS) 확장 시 container_id가 ISO 번호가 되면 _CID_RE만 교체하면 동작.
"""
from __future__ import annotations
import re

from snct.data.sources.rl_results import RLResultStore

# 컨테이너 ID 패턴(현 합성데이터). 실데이터는 ISO 번호 r"[A-Z]{4}\d{7}"로 교체.
_CID_RE = re.compile(r"[A-Z]{2}_R\d+_r\d+_t\d+")


def locate(container_id: str, store: RLResultStore | None = None) -> dict | None:
    """컨테이너ID → 위치(bay/row/tier) + 맥락(POD·중량·최하단/최상단). 없으면 None."""
    store = store or RLResultStore()
    df = store.load_slot_assignment()
    hit = df[df["container_id"] == container_id]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return {
        "container_id": container_id,
        "vessel": r["vessel"],
        "voyage": r["voyage"],
        "bay": r["bay"],
        "policy": r["policy"],
        "round_id": int(r["round_id"]),
        "row": int(r["row"]),
        "tier": int(r["tier"]),
        "pod": r["pod_name"],
        "weight_mt": float(r["weight_mt"]),
        "is_bottom": bool(r["is_bottom"]),
        "is_top": bool(r["is_top"]),
    }


def extract_container_id(question: str) -> str | None:
    m = _CID_RE.search(question or "")
    return m.group(0) if m else None


def where_is(question: str) -> dict:
    """자연어 질의 → {answer, sources}. 위치 + 위에 쌓인 컨테이너(반출 가능 여부)."""
    cid = extract_container_id(question)
    if not cid:
        return {"answer": "질문에서 컨테이너 ID를 찾지 못했습니다.", "sources": []}

    loc = locate(cid)
    if loc is None:
        return {"answer": f"{cid}는 적재계획에 없습니다.", "sources": []}

    # 위에 적층된 컨테이너 = 반출 시 재취급 필요 여부 (LPG: Neo4j 또는 CSV 폴백)
    on_top = []
    try:
        from snct.knowledge.lpg import get_lpg
        on_top = get_lpg().stacked_on(cid)
    except Exception:
        on_top = []

    if loc["is_top"] or not on_top:
        access = "바로 반출 가능(최상단)"
    else:
        blockers = ", ".join(x["container_id"] for x in on_top)
        overstow = any(x.get("is_overstow") for x in on_top)
        tag = " ⚠️overstow" if overstow else ""
        access = f"위에 {len(on_top)}개 적재됨 → 재취급 필요: {blockers}{tag}"

    answer = (
        f"📍 {cid} 위치: {loc['vessel']} / {loc['bay']} / "
        f"ROW {loc['row']} · TIER {loc['tier']} "
        f"(POD={loc['pod']}, {loc['weight_mt']}t) — {access}"
    )
    return {
        "answer": answer,
        "sources": [{"type": "sql", "ref": "slot_assignment", "snippet": loc}],
    }

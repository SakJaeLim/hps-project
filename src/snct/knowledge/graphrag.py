"""GraphRAG / text2Cypher (온톨로지) — 지식그래프 기반 Q&A.
NetworkX 그래프에서 키워드 기반 서브그래프 탐색 + 템플릿 매칭."""
import networkx as nx

# 도메인 지식 그래프 (항만 용어 관계)
_KG = nx.DiGraph()

# 노드: 항만 도메인 핵심 개념
_concepts = [
    ("Container", {"type": "concept", "desc": "적재 대상 화물 단위"}),
    ("DG_Container", {"type": "concept", "desc": "위험물(Dangerous Goods) 컨테이너"}),
    ("Reefer_Container", {"type": "concept", "desc": "냉동(Refrigerated) 컨테이너"}),
    ("Slot", {"type": "concept", "desc": "선박 내 적재 위치 (Bay-Row-Tier)"}),
    ("Bay", {"type": "concept", "desc": "선박 길이 방향 구획"}),
    ("IMDG_Code", {"type": "regulation", "desc": "국제해상위험물운송규칙"}),
    ("SOLAS", {"type": "regulation", "desc": "해상인명안전협약"}),
    ("Heavy_Down", {"type": "principle", "desc": "무거운 컨테이너 하단 배치 원칙"}),
    ("Light_Up", {"type": "principle", "desc": "가벼운 컨테이너 상단 배치 원칙"}),
    ("POD_Grouping", {"type": "principle", "desc": "동일 양하항 컨테이너 그룹핑"}),
    ("Rehandling", {"type": "concept", "desc": "재취급 — 불필요한 컨테이너 이동"}),
    ("Segregation", {"type": "concept", "desc": "위험물 격리 규정"}),
    ("Vessel_Define", {"type": "concept", "desc": "선박 정의 — Bay/Slot 구성 및 DG/Reefer 지정"}),
    ("BAPLIE", {"type": "concept", "desc": "선적 배치도 EDIFACT 메시지"}),
    ("COPINO", {"type": "concept", "desc": "게이트 반출입 확인 메시지"}),
    ("PTW", {"type": "concept", "desc": "작업허가(Permit To Work)"}),
    ("LOTO", {"type": "concept", "desc": "에너지 차단(Lock Out Tag Out)"}),
    ("SOP", {"type": "regulation", "desc": "터미널 안전 표준 작업 절차"}),
    ("COG", {"type": "concept", "desc": "무게중심(Center of Gravity)"}),
    ("GM", {"type": "concept", "desc": "복원 메타센트릭 높이"}),
]
for node_id, attrs in _concepts:
    _KG.add_node(node_id, **attrs)

# 엣지: 관계
_relations = [
    ("DG_Container", "Container", "IS_A"),
    ("Reefer_Container", "Container", "IS_A"),
    ("Container", "Slot", "ASSIGNED_TO"),
    ("Slot", "Bay", "BELONGS_TO"),
    ("DG_Container", "IMDG_Code", "GOVERNED_BY"),
    ("DG_Container", "Segregation", "REQUIRES"),
    ("DG_Container", "Vessel_Define", "CONSTRAINED_BY"),
    ("Reefer_Container", "Vessel_Define", "CONSTRAINED_BY"),
    ("Heavy_Down", "COG", "ENSURES"),
    ("Heavy_Down", "SOLAS", "GOVERNED_BY"),
    ("Light_Up", "COG", "ENSURES"),
    ("POD_Grouping", "Rehandling", "MINIMIZES"),
    ("BAPLIE", "Vessel_Define", "DESCRIBES"),
    ("PTW", "SOP", "PART_OF"),
    ("LOTO", "SOP", "PART_OF"),
]
for src, dst, rel in _relations:
    _KG.add_edge(src, dst, relation=rel)


# Template queries (Cypher-like → NetworkX equivalents)
TEMPLATES = {
    "stacked_on": lambda g, cid: [
        {"container": n, "relation": "STACKED_ON"}
        for n in g.predecessors(cid)
        if g.edges[n, cid].get("relation") == "STACKED_ON"
    ] if cid in g else [],
    "rehandling_conflict": lambda g, _: [
        {"blocker": u, "blocked": v}
        for u, v, d in g.edges(data=True)
        if d.get("relation") == "BLOCKS"
    ],
    "related_regulations": lambda g, concept: [
        {"regulation": n, "relation": g.edges[concept, n].get("relation", "")}
        for n in g.successors(concept)
        if g.nodes[n].get("type") == "regulation"
    ] if concept in g else [],
}


def _keyword_search(question: str) -> list[dict]:
    """Find relevant graph nodes by keyword matching."""
    results = []
    q_lower = question.lower()
    for node_id, attrs in _KG.nodes(data=True):
        desc = attrs.get("desc", "").lower()
        if node_id.lower() in q_lower or any(w in q_lower for w in desc.split() if len(w) > 2):
            # Get neighbors for context
            neighbors = []
            for n in _KG.successors(node_id):
                rel = _KG.edges[node_id, n].get("relation", "")
                neighbors.append({"node": n, "relation": rel, "desc": _KG.nodes[n].get("desc", "")})
            for n in _KG.predecessors(node_id):
                rel = _KG.edges[n, node_id].get("relation", "")
                neighbors.append({"node": n, "relation": rel, "desc": _KG.nodes[n].get("desc", "")})

            results.append({
                "type": "graph",
                "ref": node_id,
                "snippet": attrs.get("desc", ""),
                "neighbors": neighbors[:5],
            })
    return results[:5]


def ask(question: str) -> dict:
    """→ {answer, sources:[{type:'graph', ref, snippet}]}.
    템플릿 매칭 우선 → 미스 시 키워드 기반 서브그래프 탐색."""

    q_lower = question.lower()
    sources = []
    # 규정 질의 → related_regulations 템플릿 실사용
    if any(w in q_lower for w in ["규정", "regulation", "imdg", "solas", "격리"]):
        for node_id, attrs in _KG.nodes(data=True):
            if node_id.lower() in q_lower:
                for r in TEMPLATES["related_regulations"](_KG, node_id):
                    sources.append({
                        "type": "graph",
                        "ref": r["regulation"],
                        "snippet": f"({r['relation']}) {node_id} → {r['regulation']}",
                        "neighbors": [],
                    })
    # 폴백: 키워드 서브그래프 탐색
    if not sources:
        sources = _keyword_search(question)

    if not sources:
        return {"answer": "관련 지식그래프 정보를 찾을 수 없습니다.", "sources": []}

    # Compile answer from graph context
    answer_parts = []
    for src in sources:
        answer_parts.append(f"[{src['ref']}] {src['snippet']}")
        for nb in src.get("neighbors", []):
            answer_parts.append(f"  → ({nb['relation']}) {nb['node']}: {nb['desc']}")

    return {
        "answer": "\n".join(answer_parts),
        "sources": sources,
    }

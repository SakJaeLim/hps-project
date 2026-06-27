# DEPRECATED: orchestrator.py(LangGraph)로 대체됨. 호환을 위해 남겨두며 다음 PR에서 제거 예정.
"""지식 접근 라우터 — 질문을 doc/sql/graph 경로로 라우팅하고 근거를 모은다. specs/07."""
from enum import Enum


class KPath(str, Enum):
    DOC = "doc"
    SQL = "sql"
    GRAPH = "graph"


# Keyword-based routing rules
_ROUTE_RULES = {
    KPath.SQL: [
        "몇 개", "목록", "현황", "조회", "리스트", "카운트", "count",
        "작업", "이력", "크레인", "빈 슬롯", "가용", "status",
    ],
    KPath.GRAPH: [
        "관계", "연결", "왜", "이유", "원인", "규정", "제약",
        "IMDG", "SOLAS", "격리", "segregation", "ontology",
    ],
    KPath.DOC: [
        "규칙", "SOP", "절차", "안전", "원칙", "Heavy-Down", "Light-Up",
        "BAPLIE", "COPINO", "PTW", "LOTO", "어떻게", "방법",
        "추천", "슬롯", "적재", "배치", "DG", "Reefer", "위험물", "냉동",
    ],
}


def route(question: str) -> list[KPath]:
    """Classify question type → select knowledge paths (can be parallel)."""
    q_lower = question.lower()
    matched = set()

    for path, keywords in _ROUTE_RULES.items():
        for kw in keywords:
            if kw.lower() in q_lower:
                matched.add(path)
                break

    if not matched:
        # Default: doc + graph
        matched = {KPath.DOC, KPath.GRAPH}

    return list(matched)


def answer(question: str) -> dict:
    """→ {answer, sources[], used_path[]}. 선택 경로 호출 + 근거 수집."""
    paths = route(question)
    all_sources = []
    answer_parts = []

    for path in paths:
        try:
            if path == KPath.DOC:
                from snct.knowledge.rag_docs import retrieve
                docs = retrieve(question, k=3)
                all_sources.extend(docs)
                for doc in docs:
                    answer_parts.append(f"[문서 {doc['ref']}] {doc.get('title', '')}: {doc['snippet']}")

            elif path == KPath.SQL:
                from snct.knowledge.nl2sql import ask as sql_ask
                result = sql_ask(question)
                all_sources.extend(result.get("sources", []))
                answer_parts.append(f"[SQL] {result['answer']}")

            elif path == KPath.GRAPH:
                from snct.knowledge.graphrag import ask as graph_ask
                result = graph_ask(question)
                all_sources.extend(result.get("sources", []))
                answer_parts.append(f"[그래프] {result['answer']}")
        except Exception as e:
            answer_parts.append(f"[{path.value}] 오류: {e}")

    return {
        "answer": "\n\n".join(answer_parts) if answer_parts else "관련 정보를 찾을 수 없습니다.",
        "sources": all_sources,
        "used_path": [p.value for p in paths],
    }

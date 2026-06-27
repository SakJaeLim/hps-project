"""
PortSLM 지식 오케스트레이터 (LangGraph) — v2 (router.py 대체용).

route → retrieve → fuse → faithfulness → (교정 재시도 | finalize)
Corrective-RAG 패턴: 검증 실패 시 '같은 걸 재시도'가 아니라 경로 확장·k 증가로 교정한다.

■ router.py 대체: answer(question)이 기존 router.answer와 호환되는 dict를 반환
  → agents/graph.py 의 `from snct.knowledge.router import answer as knowledge_answer` 를
    `from snct.knowledge.orchestrator import answer as knowledge_answer` 로만 바꾸면 됨.

■ 스왑 지점(아래 import): 이 3개 모듈 '내부'만 실백엔드로 교체. 그래프는 안 건드림.
    rag_docs.retrieve : TF-IDF → BGE-M3(1024d) + Chroma(BYO embeddings)
    nl2sql.ask        : DuckDB 템플릿 → Neon 실 NL2SQL
    graphrag.ask      : NetworkX → Neo4j 벡터/Cypher

■ VESSL 배포 전제 — 설정은 전부 env로(하드코딩 금지):
    CHROMA_DIR, NEO4J_URI/USER/PASSWORD, DATABASE_URL, EMBED_MODEL, SLM_MODEL_PATH
  데이터가 VESSL 스토리지에 있으므로 경로도 env에서 읽는다. 오케스트레이터 자체는 설정 불요.

설치: pip install langgraph langchain-core   (없어도 순수 파이썬 폴백으로 동작)
스모크: python -m snct.knowledge.orchestrator
"""
from __future__ import annotations
from typing import TypedDict, Literal, Callable
import re

# ── 지식 도구(인터페이스) — 실연결 시 '이 모듈 내부'만 교체 ──────────────────────
try:
    from snct.knowledge.rag_docs import retrieve as doc_retrieve   # TODO: BGE-M3 + Chroma(PersistentClient, query_embeddings)
    from snct.knowledge.nl2sql import ask as sql_ask               # TODO: Neon 실 NL2SQL
    from snct.knowledge.graphrag import ask as graph_ask           # TODO: Neo4j 벡터/Cypher
except Exception:
    def doc_retrieve(q, k=3): return [{"type": "doc", "ref": "SOP-STUB", "snippet": "(stub doc)"}]
    def sql_ask(q):           return {"answer": "(stub)", "sources": []}
    def graph_ask(q):         return {"answer": "(stub)", "sources": []}

THRESHOLD = 80
MAX_RETRIES = 2

# 선택: LLM 라우터/판정을 끼울 훅 (없으면 규칙 기반). VESSL에서 SLM 붙이면 주입.
llm_router: Callable[[str], list[str]] | None = None     # (question) -> ["doc","sql","graph"]
llm_judge: Callable[[str, str], int] | None = None       # (draft, evidence_text) -> 0~100


class KState(TypedDict, total=False):
    question: str
    routes: list[str]
    evidence: list[dict]
    draft: str
    faithfulness: int
    retries: int
    answer: str


# ── ② 라우팅 (규칙 기본 + LLM 훅) ───────────────────────────────────────────────
_RULES = {
    "sql":   ["몇", "목록", "현황", "조회", "카운트", "count", "작업", "이력", "가용", "빈 슬롯", "통계"],
    "graph": ["관계", "연결", "왜", "이유", "규정", "제약", "imdg", "solas", "격리", "segregation"],
    "doc":   ["규칙", "sop", "절차", "안전", "원칙", "어떻게", "방법", "배치", "추천"],
}

def route_node(s: KState) -> KState:
    if llm_router is not None:                       # TODO: 모호 질의는 LLM 분류로(하이브리드)
        try:
            s["routes"] = llm_router(s["question"]) or ["doc", "graph"]
            return s
        except Exception:
            pass
    q = s["question"].lower()
    routes = [p for p, kws in _RULES.items() if any(k in q for k in kws)]
    s["routes"] = routes or ["doc", "graph"]
    return s


# ── ③ 검색 (교정형: 시도할수록 넓게/깊게) + 도구별 에러 격리 ─────────────────────
def _retrieval_plan(attempt: int, base: list[str]) -> tuple[list[str], int]:
    if attempt == 0:
        return base, 3
    if attempt == 1:
        return ["doc", "sql", "graph"], 6            # 교정 1: 경로 전체 확장
    return ["doc", "sql", "graph"], 10               # 교정 2: k 증가 (+ TODO: 쿼리 재작성)

def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        return {"__error__": str(e)}

def retrieve_node(s: KState) -> KState:
    routes, k = _retrieval_plan(s.get("retries", 0), s.get("routes", ["doc", "graph"]))
    s["routes"] = routes
    ev: list[dict] = []
    if "doc" in routes:
        r = _safe(doc_retrieve, s["question"], k=k)
        ev += r if isinstance(r, list) else []
    if "sql" in routes:
        r = _safe(sql_ask, s["question"])
        ev += r.get("sources", []) if isinstance(r, dict) and "__error__" not in r else []
    if "graph" in routes:
        r = _safe(graph_ask, s["question"])
        ev += r.get("sources", []) if isinstance(r, dict) and "__error__" not in r else []
    # 근거 중복 제거 (type+ref 기준)
    seen, dedup = set(), []
    for e in ev:
        key = (e.get("type"), e.get("ref"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(e)
    s["evidence"] = dedup
    return s


# ── ④ 융합 (지금은 결정론적 구조화 답변; SLM은 TODO) ────────────────────────────
def fuse_node(s: KState) -> KState:
    ev = s.get("evidence", [])
    if not ev:
        s["draft"] = "관련 근거를 찾지 못했습니다."
        return s
    by_type: dict[str, list[dict]] = {}
    for e in ev:
        by_type.setdefault(e.get("type", "etc"), []).append(e)
    label = {"doc": "📄 문서", "sql": "📊 운영데이터", "graph": "🕸️ 관계/규정"}
    parts = []
    for t, items in by_type.items():
        parts.append(f"### {label.get(t, t)}")
        for e in items[:5]:
            parts.append(f"- [{e.get('ref', '')}] {str(e.get('snippet', ''))[:160]}")
    # TODO(SLM): 위 팩트카드를 Qwen2.5-VL-3B LoRA로 자연어 정제. 신규 수치 생성 금지, 근거 내 수치만.
    s["draft"] = "\n".join(parts)
    return s


# ── ⑤ 자기검증 (v1: 수치 그라운딩 + 근거 존재. 실버전은 LLM-judge) ───────────────
_NUM = re.compile(r"(?<!\d)\d+(?:\.\d+)?(?!\d)")     # 단어경계 — "20"이 "2024"에 오탐되지 않게
def faithfulness_node(s: KState) -> KState:
    draft = s.get("draft", "")
    ev = s.get("evidence", [])
    ev_text = " ".join(str(e.get("snippet", "")) for e in ev)
    if llm_judge is not None:                        # TODO: Self-RAG식 faithfulness/answer-relevance
        try:
            s["faithfulness"] = max(0, min(100, int(llm_judge(draft, ev_text))))
            return s
        except Exception:
            pass
    nums = _NUM.findall(draft)
    ev_nums = set(_NUM.findall(ev_text))
    if not ev:
        s["faithfulness"] = 0
    elif not nums:
        s["faithfulness"] = 100                      # 수치 주장 없음 → 그라운딩 위반 없음
    else:
        grounded = sum(1 for n in nums if n in ev_nums)
        s["faithfulness"] = round(100 * grounded / len(nums))
    return s


def bump_retry(s: KState) -> KState:
    s["retries"] = s.get("retries", 0) + 1
    return s

def finalize_node(s: KState) -> KState:
    s["answer"] = (
        f"{s.get('draft', '')}\n\n"
        f"— Faithfulness {s.get('faithfulness', 0)}/100 · 경로 {s.get('routes')} · 재시도 {s.get('retries', 0)}"
    )
    return s

def needs_retry(s: KState) -> Literal["retry", "ok"]:
    if s.get("faithfulness", 0) < THRESHOLD and s.get("retries", 0) < MAX_RETRIES:
        return "retry"
    return "ok"


# ── 그래프 조립 ────────────────────────────────────────────────────────────────
def build_graph():
    from langgraph.graph import StateGraph, END
    g = StateGraph(KState)
    for name, fn in [("route", route_node), ("retrieve", retrieve_node), ("fuse", fuse_node),
                     ("faithfulness", faithfulness_node), ("bump", bump_retry), ("finalize", finalize_node)]:
        g.add_node(name, fn)
    g.set_entry_point("route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "fuse")
    g.add_edge("fuse", "faithfulness")
    g.add_conditional_edges("faithfulness", needs_retry, {"retry": "bump", "ok": "finalize"})
    g.add_edge("bump", "retrieve")        # ← 교정 재시도(경로 확장·k 증가)
    g.add_edge("finalize", END)
    return g.compile()


def answer(question: str) -> dict:
    """router.answer 호환. → {answer, sources, used_path, routes, faithfulness, retries}."""
    state: KState = {"question": question, "retries": 0}
    try:
        out = build_graph().invoke(state)
    except Exception:                                 # langgraph 미설치 → 동일 흐름 폴백
        s = faithfulness_node(fuse_node(retrieve_node(route_node(state))))
        while needs_retry(s) == "retry":
            s = faithfulness_node(fuse_node(retrieve_node(bump_retry(s))))
        out = finalize_node(s)
    routes = out.get("routes", [])
    return {
        "answer": out.get("answer", ""),
        "sources": out.get("evidence", []),
        "used_path": routes,          # router.py 호환 키
        "routes": routes,
        "faithfulness": out.get("faithfulness"),
        "retries": out.get("retries", 0),
    }


if __name__ == "__main__":
    for q in ["DG 컨테이너 격리 규정 알려줘", "빈 슬롯 목록 보여줘", "재취급은 왜 생기지?"]:
        r = answer(q)
        print(f"\nQ: {q}\n  used_path={r['used_path']} faithfulness={r['faithfulness']} retries={r['retries']}")
        print("  " + r["answer"][:280].replace("\n", "\n  "))

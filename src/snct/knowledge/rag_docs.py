"""
문서 RAG (비정형) — BGE-M3(1024d) 임베딩 + Chroma 벡터검색.  배치: src/snct/knowledge/rag_docs.py

기존 TF-IDF 인메모리 스텁을 대체. retrieve(query, k) 시그니처/반환형은 동일하게 유지
(→ orchestrator / router 무수정). 모델·DB 미가용 시 인라인 TF-IDF로 자동 폴백(스모크/CI 안전).

VESSL 환경변수(코드에 하드코딩 금지):
    CHROMA_DIR     Chroma 영속 경로 (예: /data/chroma).            기본 ./.chroma
    EMBED_MODEL    임베딩 모델.                                     기본 BAAI/bge-m3
    RAG_CORPUS     적재용 chunks.jsonl 경로(build_index에서 사용).  기본 data/rag/chunks.jsonl
    RAG_COLLECTION Chroma 컬렉션명.                                 기본 portslm_docs

적재(최초 1회):  python -c "from snct.knowledge.rag_docs import build_index; build_index()"
"""
from __future__ import annotations
import os
import json
import functools

CHROMA_DIR = os.environ.get("CHROMA_DIR", "./.chroma")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
COLLECTION = os.environ.get("RAG_COLLECTION", "portslm_docs")
CORPUS_PATH = os.environ.get("RAG_CORPUS", "data/rag/chunks.jsonl")

# ── BGE-M3 임베딩 (lazy load; GPU면 fp16) ───────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _model():
    from FlagEmbedding import BGEM3FlagModel       # pip install FlagEmbedding
    use_fp16 = os.environ.get("EMBED_FP16", "1") == "1"
    return BGEM3FlagModel(EMBED_MODEL, use_fp16=use_fp16)

def embed(texts: list[str]) -> list[list[float]]:
    """BGE-M3 dense 임베딩 (n, 1024). 정규화 출력 → Chroma cosine과 정합."""
    vecs = _model().encode(texts, batch_size=12, max_length=1024)["dense_vecs"]
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]

# ── Chroma 컬렉션 (lazy) ────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _collection():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

# ── 적재 ────────────────────────────────────────────────────────────────────────
def build_index(corpus_path: str | None = None) -> int:
    """chunks.jsonl → BGE-M3 임베딩 → Chroma upsert. 반환: 적재 건수.
    chunk 스키마는 도메인별로 다양 → text/id를 관대하게 해석."""
    path = corpus_path or CORPUS_PATH
    ids, docs, metas = [], [], []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            text = o.get("text") or o.get("content") or o.get("page_content") or ""
            if not text.strip():
                continue
            ref = str(o.get("chunk_id") or o.get("id") or f"chunk-{i}")
            meta = o.get("metadata") if isinstance(o.get("metadata"), dict) else {}
            meta = {k: v for k, v in {**meta,
                    "title": o.get("title") or o.get("doc_title") or meta.get("title", ""),
                    "doc_id": o.get("doc_id", meta.get("doc_id", "")),
                    }.items() if isinstance(v, (str, int, float, bool))}
            ids.append(ref); docs.append(text); metas.append(meta or {"title": ""})
    if not docs:
        return 0
    col = _collection()
    B = 64
    for s in range(0, len(docs), B):
        col.upsert(ids=ids[s:s+B], embeddings=embed(docs[s:s+B]),
                   documents=docs[s:s+B], metadatas=metas[s:s+B])
    return len(docs)

# ── 검색 (시그니처/반환형 = 기존 스텁과 동일) ───────────────────────────────────
def retrieve(query: str, k: int = 4) -> list[dict]:
    """→ [{type:'doc', ref, title, snippet, score}]. (orchestrator/router 호환)"""
    try:
        col = _collection()
        if col.count() == 0:                       # 인덱스 미적재 → 폴백
            return _fallback_retrieve(query, k)
        res = col.query(query_embeddings=embed([query]), n_results=k,
                        include=["documents", "metadatas", "distances"])
        out = []
        for ref, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            out.append({
                "type": "doc",
                "ref": ref,
                "title": (meta or {}).get("title", ""),
                "snippet": (doc or "")[:200],
                "score": round(1.0 - float(dist), 4),   # cosine distance → 유사도
            })
        return out
    except Exception:
        return _fallback_retrieve(query, k)        # 모델/DB 미가용 → 폴백

# ── 폴백: 인라인 TF-IDF (의존성/인덱스 없을 때만; 스모크·CI 안전망) ──────────────
_FALLBACK_CORPUS = [
    {"id": "SOP-001", "title": "Heavy-Down & Light-Up 원칙",
     "content": "무거운 컨테이너는 선박 하부에, 가벼운 컨테이너는 상부에 적재해 무게중심(COG)을 낮추고 복원성(GM)을 확보한다. 위반 시 SOLAS 검사 대상."},
    {"id": "SOP-002", "title": "DG 위험물 격리 (IMDG Code)",
     "content": "위험물(DG)은 IMDG Code Class별 격리거리를 준수하고 DG 허용 Bay에만 배치한다. 격리 위반 시 선적 거부."},
    {"id": "SOP-003", "title": "Reefer 지정위치",
     "content": "Reefer는 전원 공급 가능한 Reefer Bay에만 배치한다. 일반 Bay 배치 시 화물 손상·클레임."},
    {"id": "SOP-004", "title": "양하순서 및 Rehandling 최소화",
     "content": "먼저 양하할 컨테이너가 아래에 적재되면 재취급이 발생한다. 동일 POD를 같은 Bay에 그룹핑한다."},
    {"id": "SOP-005", "title": "PTW/LOTO 안전 작업",
     "content": "위험 작업은 작업허가(PTW)와 에너지 차단(LOTO)을 선행한다. 풍속 20m/s 이상 크레인 작업 중지."},
]

@functools.lru_cache(maxsize=1)
def _tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer()
    mat = vec.fit_transform([f"{d['title']} {d['content']}" for d in _FALLBACK_CORPUS])
    return vec, mat

def _fallback_retrieve(query: str, k: int) -> list[dict]:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    vec, mat = _tfidf()
    sims = cosine_similarity(vec.transform([query]), mat).flatten()
    out = []
    for idx in np.argsort(sims)[::-1][:k]:
        if sims[idx] > 0.01:
            d = _FALLBACK_CORPUS[idx]
            out.append({"type": "doc", "ref": d["id"], "title": d["title"],
                        "snippet": d["content"][:200], "score": float(sims[idx])})
    return out


if __name__ == "__main__":
    print("build_index:", build_index() if os.path.exists(CORPUS_PATH) else "(corpus 없음 → 폴백 사용)")
    for hit in retrieve("DG 위험물 격리 규정"):
        print(f"  [{hit['ref']}] {hit['title']} (score={hit['score']})")

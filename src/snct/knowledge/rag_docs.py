"""문서 RAG (비정형) — 규정·SOP·사고보고서를 TF-IDF 임베딩으로 시맨틱 검색.
외부 벡터DB 없이 scikit-learn TfidfVectorizer 기반 인메모리 검색."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 인메모리 문서 코퍼스 — 터미널 안전 규정 및 본선 플래닝 SOP
_CORPUS = [
    {
        "id": "SOP-001",
        "title": "Heavy-Down & Light-Up 원칙",
        "content": "무거운 컨테이너(Heavy)는 선박 하부(Down)에, 가벼운 컨테이너(Light)는 상부(Up)에 "
                   "적재하여 선박의 무게중심(COG)을 낮추고 복원성(GM)을 확보해야 한다. "
                   "이를 위반할 경우 선박 전복 위험이 있으며, SOLAS 규정에 따라 강제 검사 대상이 된다.",
    },
    {
        "id": "SOP-002",
        "title": "DG 위험물 컨테이너 격리 규정 (IMDG Code)",
        "content": "위험물(DG) 컨테이너는 IMDG Code에 따라 Class별 격리 거리를 준수해야 한다. "
                   "Class 3(인화성 액체)는 발화원 및 기관실로부터 격리해야 하며, "
                   "DG 적재 가능 Bay(Vessel Define에 등록)에만 배치해야 한다. "
                   "격리 위반 시 선적 거부 및 벌금 부과 대상이다.",
    },
    {
        "id": "SOP-003",
        "title": "Reefer 냉동 컨테이너 지정위치 규칙",
        "content": "Reefer(냉동) 컨테이너는 전원 공급이 가능한 Reefer Bay에만 배치할 수 있다. "
                   "전원 플러그 연결이 불가능한 일반 Bay에 배치할 경우 냉동화물 손상 및 "
                   "클레임이 발생한다. Reefer 지정위치 제약은 POD 그룹핑보다 우선한다.",
    },
    {
        "id": "SOP-004",
        "title": "양하순서 및 Rehandling 최소화",
        "content": "먼저 양하(discharge)할 컨테이너가 나중에 양하할 컨테이너 아래에 적재되면 "
                   "재취급(Rehandling)이 발생하여 작업 시간과 비용이 증가한다. "
                   "양하순서를 보존하여 적재 역전을 방지하고, "
                   "동일 양하항(POD) 컨테이너를 같은 Bay에 그룹핑하여 작업 효율을 높인다.",
    },
    {
        "id": "SOP-005",
        "title": "안전 작업 허가 및 에너지 차단 (PTW/LOTO)",
        "content": "컨테이너 터미널 안전 규정(SOP §3.2)에 따라 모든 위험 작업은 "
                   "작업허가(Permit To Work, PTW) 및 에너지 차단(Lock Out Tag Out, LOTO) "
                   "절차를 선행해야 한다. 크레인 작업 시 바람 속도 제한(풍속 20m/s 이상 작업 중지), "
                   "야간 작업 시 조명 확보 기준을 준수해야 한다.",
    },
    {
        "id": "SOP-006",
        "title": "BAPLIE/COPINO/MOVINS 메시지 규격",
        "content": "BAPLIE는 선박 적재 배치도(Bay Plan), COPINO는 게이트 반출입 확인, "
                   "MOVINS는 적재 지시를 전달하는 EDIFACT 표준 메시지이다. "
                   "본선 플래닝 시 BAPLIE를 기준으로 현재 적재 상태를 파악하고, "
                   "MOVINS로 적재 순서를 전달한다.",
    },
    {
        "id": "SOP-007",
        "title": "선박 복원성(COG/GM) 및 중량 분포",
        "content": "적재 계획 수립 시 선박의 무게중심(Center of Gravity, COG)과 "
                   "복원 메타센트릭 높이(GM)를 계산하여 적정 범위 내에 있는지 확인해야 한다. "
                   "편하중(Listing)이 발생하지 않도록 좌우 균형을 맞추고, "
                   "종방향 트림(Trim)도 허용 범위 내로 관리해야 한다.",
    },
]

# Build TF-IDF index on initialization
_vectorizer = TfidfVectorizer()
_corpus_texts = [f"{doc['title']} {doc['content']}" for doc in _CORPUS]
_tfidf_matrix = _vectorizer.fit_transform(_corpus_texts)


def retrieve(query: str, k: int = 4) -> list[dict]:
    """Retrieve top-k most relevant SOP documents for the query.
    → [{type:'doc', ref, snippet}]."""
    query_vec = _vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:k]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0.01:  # minimum relevance threshold
            doc = _CORPUS[idx]
            results.append({
                "type": "doc",
                "ref": doc["id"],
                "title": doc["title"],
                "snippet": doc["content"][:200],
                "score": float(similarities[idx]),
            })
    return results

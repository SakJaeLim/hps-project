# 07. 지식 접근 레이어 (Knowledge Access) — L5

"RAG"는 단일 기능이 아니라 **3개 소스에 대한 라우팅 + 근거 융합(xAI)** 이다.
(이전 표기 "RAG+SLM"을 본 명세로 구체화한다.)

## 3개 검색 경로
1. **문서 RAG (비정형)** — SOP·Safety·IMO(IMDG)·ISPS·SOLAS·Accident → 청크 → BGE-M3 임베딩 → 벡터DB → 시맨틱 검색.
   용도: 규정 근거·절차 Q&A, 설명 그라운딩. (모듈 `knowledge/rag_docs.py`)
2. **NL2SQL (정형 운영 DB)** — 자연어 → 스키마 링킹 → SQL(**읽기전용**) → 실행 → 결과/요약.
   용도: 운영 수치("위험물 컨테이너 수", "야드 점유율", "게이트 처리량").
   *조건부*: 실 DB가 있으면 직접, 없으면 정형 피드(Yard/Gate/Equipment/BAPLIE)를 **DuckDB/Postgres에 적재**해 질의 가능하게.
   가드레일: 읽기전용 · 스키마 링킹 · SQL 검증 · LIMIT · 미리보기 · (변경 쿼리 차단). (모듈 `knowledge/nl2sql.py`)
3. **GraphRAG / text2Cypher (온톨로지)** — 자연어 → Cypher → Neo4j.
   용도: 관계·제약·재취급 충돌("이 컨테이너 위에 뭐가 쌓였나", "충돌 어디").
   **핵심 질의는 파라미터 템플릿 우선**, 자유질의만 LLM text2cypher(neo4j-graphrag Text2CypherRetriever). (모듈 `knowledge/graphrag.py`)

## 라우터 + 융합 (xAI)
- 라우터가 질문 유형에 따라 경로 선택(병렬 가능). (`knowledge/router.py`)
- 설명 합성기가 **[엔진의 계획 + Cypher 제약 검증 사실 + 문서 근거 + 운영 수치]를 융합**해 근거 인용 자연어 설명(rationale) 생성 = xAI. 모든 주장은 사실 소스에 근거 → 환각↓. (`knowledge/explain.py`)

## 신뢰성 (검증 반영)
- 생성 정확도 한계: NL2SQL는 복잡 질의에서 오류(BIRD 벤치 ~59%), text2cypher는 관계 방향·최신 문법에 취약 → **스키마 링킹·few-shot·읽기전용** 필수, 핵심 질의는 템플릿.
- **도메인 SLM 파인튜닝의 실용 용도**: 질문-스키마-쿼리(SQL/Cypher) 쌍으로 파인튜닝하면 생성 품질↑ (트랙1 파인튜닝이 설명뿐 아니라 쿼리 생성에도 기여).

## 계약(산출물)
- `KnowledgeQuery(question)` → `{answer, sources[], used_path}`
- `Evidence{type: doc|sql|graph, ref, snippet}` — 설명의 근거 단위.

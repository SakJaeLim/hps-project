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

---

## 설명가능 RL 적재 에이전트 (xAI-RL)

> 목적: RL(PPO)이 산출한 적재 계획에 대해 **"왜 이렇게 적재했는가"**를 사실 근거와 함께 설명한다.
> 원칙: 설명은 **자유 생성이 아니라 RL이 이미 뽑아둔 근거의 검색·융합**이다. 모든 수치 주장은 소스 태깅 → 환각↓.

### 근거 소스 (강화학습 결과 자료 = 이미 가공됨)
RL 학습 단계에서 설명 근거가 구조화되어 산출됨 (`data/강화학습 결과 자료/`):
- **RDB (정형)** `RDB_LPG_*/rdb/`: `reward_decomp.csv`(R1~R15 항목별 ±기여 = **귀인/attribution**), `kpi.csv`(reward·osr·wbi·psr·cwvr), `slot_assignment.csv`(컨테이너→bay/row/tier), `violation_log.csv`.
- **LPG (그래프)** `RDB_LPG_*/neo4j_kg/`: Vessel→Bay→Row→Tier→Slot→Container, `STACKED_ON(is_overstow)`, `VIOLATES`, `Constraint`(SOLAS/ISPS…), `import_cypher.cypher`. **MVP는 CSV 직접 적재(DuckDB/pandas) 폴백**, Neo4j 실연결은 후속(ADR 동일 패턴).
- **RAG (비정형)** `RAG_*/`: `rag_chunks.jsonl`·`rationale_chunks.csv`(라운드별 근거 청크), `xai_grounding.json`(이미 융합된 정답 레퍼런스).
- **SFT** `*_sft_seed42.jsonl`: 설명 **표현 다듬기**용 LoRA 데이터.

### 설명 합성 (규칙 기반 우선)
`explain()`은 다음을 **결정론적 규칙**으로 융합한다 (LLM 자유생성 아님):
1. `reward_decomp`에서 절댓값 상위 ±기여항을 정렬 → "왜 좋/나쁜가"의 주요인.
2. LPG에서 해당 (policy, round) 위반 사실(`VIOLATES`)·재취급(`STACKED_ON.is_overstow`) 인용.
3. `doc_refs`(SOLAS_VI/ISPS/SOP…) 규정 근거 인용.
4. `kpi`(osr/wbi/psr/cwvr) 결과 지표 요약.
- **SLM 역할 한정**: 위 사실 카드를 입력받아 **문장 표현만** 다듬는다(후순위). 새 수치 생성 금지.
- 검증: `xai_grounding.json`을 정답 레퍼런스로 사용해 합성 결과의 사실 일치(faithfulness)를 회귀 테스트.

### 계약 확장
- `RLDecisionRef{policy, round_id}` — 설명 대상 식별자.
- `RLDecision{policy, round_id, reward_total, top_contributions, kpi, violations, rationale, doc_refs}` — 근거 융합체.
- `Evidence.type`에 `reward|kpi` 추가 (RDB 근거 단위).
- `explain(plan, violations, evidence, decision=None, lpg=None)` — decision 제공 시 RL 설명 우선,
  lpg(LPGGraph) 동시 제공 시 위반 컨테이너별 규정 자동 인용.

### 구현 모듈 (현행)
- `data/sources/rl_results.py` — `RLResultStore`: RL 결과 자료 글롭 탐색·로딩·`get_decision()` 융합. (T18)
- `knowledge/lpg_csv.py` — `LPGGraph`: neo4j_kg CSV 폴백 질의(`stacked_on`·`violations_of`·`violations_in_round`·`constraint`). (T19)
- `knowledge/lpg_neo4j.py` + `knowledge/lpg.py` — `Neo4jLPG`(Cypher 실연결·`import_kg`) + `get_lpg()`/`lpg_status()` 팩토리. Neo4j 가용 시 그래프DB, 아니면 CSV 폴백(동일 인터페이스). (T28)
- `knowledge/nl2sql.py` — `RLAnalyst`: RL RDB(kpi·reward_decomp·violation_log·slot_assignment) → DuckDB, 읽기전용 가드레일(`_guard_and_limit`). (T20)
- `knowledge/explain.py` — `explain_rl_decision()`: 귀인+지표+규정+위반+근거 규칙 융합(새 수치 생성 없음). (T21·T24)
- `eval/faithfulness.py` — `score_decision`/`evaluate`: 설명 수치 근거율 채점·환각 탐지(식별자 숫자 제외, LPG 인용 숫자 인정). (T22·T24)
- `agents/graph.py` — `run_explanation()`: 질의→근거수집(RDB·LPG)→설명→faithfulness 자기검증. (T25)
- `knowledge/locator.py` — `locate()`/`where_is()`: 완성된 적재계획(slot_assignment)에서 컨테이너 위치(bay/row/tier) + 적층/반출 가능 여부(LPG `stacked_on`) 조회. 실데이터 ISO 번호 확장은 `_CID_RE`만 교체. (T27)

### 신뢰성 측정 (현행 기준)
- 규칙 기반 설명 → 12개 RL 의사결정 전수 **faithfulness == 1.0**(환각 0). 음성 대조(가짜 수치 주입)로 하니스 유효성 검증.
- SLM(T23) 도입 시 동일 `eval/faithfulness.py`로 환각을 정량 회귀.

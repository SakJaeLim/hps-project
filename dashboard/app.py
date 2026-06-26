"""L6 Streamlit 운영자 대시보드 — 메인·요청·결과·경보 + RL 적재 설명(xAI-RL). 프론트엔드 담당."""
import sys
import pathlib

# src/ 패키지 경로 등록 (streamlit run dashboard/app.py 실행 가정)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

# .env 로드 (Neo4j/Neon 접속정보) — 있으면 자동 적용, 없으면 CSV 폴백
try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

import streamlit as st

st.set_page_config(page_title="SNCT 의사결정 지원", layout="wide")
page = st.sidebar.radio("화면", ["메인 대시보드", "계획 요청", "결과", "RL 적재 설명", "컨테이너 위치 조회", "Neo4j 그래프", "경보"])
st.title("SNCT 컨테이너 터미널 의사결정 지원")


@st.cache_data(show_spinner=False)
def _available_decisions():
    """RL 결과에서 (policy, round) 선택지 로드. 데이터 없으면 빈 목록."""
    try:
        from snct.data.sources.rl_results import RLResultStore
        kpi = RLResultStore().load_kpi()
        policies = sorted(kpi["policy"].astype(str).unique().tolist())
        rounds = sorted(int(r) for r in kpi["round_id"].dropna().unique())
        return policies, rounds
    except Exception as e:
        return [], [], str(e)


if page == "메인 대시보드":
    st.info("야드·선석 실시간 현황 (온톨로지 기반). TODO(W1 와이어프레임 → W2 연동)")
elif page == "계획 요청":
    st.text_input("선박 ID"); st.button("적재 계획 생성")
elif page == "결과":
    st.write("권고안 + 근거 + 재취급 위험 표시. TODO(W2)")
elif page == "RL 적재 설명":
    st.subheader("RL 적재 의사결정 설명 (xAI-RL)")
    st.caption("강화학습이 왜 그렇게 적재했는지 — reward 귀인·운영지표·규정·위반 컨테이너를 사실 근거로 설명합니다.")
    avail = _available_decisions()
    policies, rounds = avail[0], avail[1]
    if not policies:
        st.error(f"RL 결과 자료를 찾을 수 없습니다. ({avail[2] if len(avail) > 2 else 'data/강화학습 결과 자료'})")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        policy = c1.selectbox("정책", policies)
        round_id = c2.selectbox("라운드", rounds)
        with_lpg = c3.checkbox("LPG 위반 컨테이너 상세 포함", value=True)
        if st.button("설명 생성", type="primary"):
            from snct.agents.graph import run_explanation
            rec = run_explanation(policy=policy, round_id=int(round_id), with_lpg=with_lpg)
            # faithfulness 배지
            faith = next((c.split("=")[1] for c in rec.checks if c.startswith("faithfulness=")), None)
            if faith is not None:
                (st.success if float(faith) >= 1.0 else st.warning)(
                    f"근거 충실도(faithfulness) = {faith}  ·  {' · '.join(rec.checks)}"
                )
            st.markdown(rec.rationale)
elif page == "컨테이너 위치 조회":
    st.subheader("컨테이너 위치 조회")
    st.caption("적재계획 완료 후 — 컨테이너가 어디 있는지, 바로 반출 가능한지 확인합니다.")
    q = st.text_input("질문 또는 컨테이너 ID", placeholder="예: BL_R4_r0_t0 어디 있어?")
    if st.button("위치 조회", type="primary") and q.strip():
        from snct.knowledge.locator import where_is
        res = where_is(q)
        if res["sources"]:
            st.success(res["answer"])
            st.json(res["sources"][0]["snippet"])
        else:
            st.warning(res["answer"])
elif page == "Neo4j 그래프":
    st.subheader("Neo4j 지식그래프 (LPG)")
    from snct.knowledge.lpg import lpg_status, get_lpg
    status = lpg_status()

    # 1) 연결 상태
    if status["neo4j_connected"]:
        st.success(f"✅ Neo4j 연결됨 — {status['neo4j_uri']} (그래프DB 질의 사용)")
    else:
        st.warning(
            f"⚠️ Neo4j 미연결 — CSV 폴백 사용 중 (URI: {status['neo4j_uri']})\n\n"
            "그래프DB로 보려면: ① Neo4j 기동 → ② .env에 NEO4J_URI/USER/PASSWORD 설정 → ③ 아래 'KG 적재'"
        )
        with st.expander("Neo4j 빠른 기동 (Docker)"):
            st.code(
                "docker run -d --name neo4j -p7474:7474 -p7687:7687 \\\n"
                "  -e NEO4J_AUTH=neo4j/test1234 neo4j:5\n\n"
                "# .env\nNEO4J_URI=bolt://localhost:7687\nNEO4J_USER=neo4j\nNEO4J_PASSWORD=test1234",
                language="bash",
            )

    # 2) KG 적재 (연결 시)
    if status["neo4j_connected"]:
        if st.button("neo4j_kg CSV → Neo4j 적재"):
            from snct.knowledge.lpg_neo4j import Neo4jLPG
            with st.spinner("적재 중..."):
                counts = Neo4jLPG().import_kg()
            st.success(f"적재 완료: {counts}")

    # 3) 질의 (백엔드 무관 동일 인터페이스)
    st.divider()
    st.caption(f"질의 백엔드: **{status['backend'].upper()}**")
    g = get_lpg()
    c1, c2 = st.columns(2)
    with c1:
        cid = st.text_input("컨테이너 위반 규정 조회", placeholder="BL_R4_r1_t9")
        if st.button("위반 규정 조회") and cid.strip():
            st.json(g.violations_of(cid.strip()))
    with c2:
        cid2 = st.text_input("위에 쌓인 컨테이너(STACKED_ON)", placeholder="BL_R4_r0_t0")
        if st.button("적층 조회") and cid2.strip():
            st.json(g.stacked_on(cid2.strip()))
else:
    st.warning("충돌·위험 경보. TODO(W3)")

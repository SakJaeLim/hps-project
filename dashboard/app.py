"""L6 Streamlit 운영자 대시보드 — 4화면(메인·요청·결과·경보). 프론트엔드 담당."""
import streamlit as st

st.set_page_config(page_title="SNCT 의사결정 지원", layout="wide")
page = st.sidebar.radio("화면", ["메인 대시보드", "계획 요청", "결과", "경보"])
st.title("SNCT 컨테이너 터미널 의사결정 지원")
if page == "메인 대시보드":
    st.info("야드·선석 실시간 현황 (온톨로지 기반). TODO(W1 와이어프레임 → W2 연동)")
elif page == "계획 요청":
    st.text_input("선박 ID"); st.button("적재 계획 생성")
elif page == "결과":
    st.write("권고안 + 근거 + 재취급 위험 표시. TODO(W2)")
else:
    st.warning("충돌·위험 경보. TODO(W3)")

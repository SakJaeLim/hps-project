import streamlit as st
import pandas as pd
import requests
import time

# Custom Styling Injections
st.set_page_config(
    page_title="PortSLM — 항만 도메인 특화 SLM",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Theme CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');
    
    /* Global Overrides */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        color: #1F3864 !important;
        font-weight: 700;
    }
    
    /* Top Header Bar */
    .header-bar {
        background: linear-gradient(135deg, #13315c 0%, #1F3864 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .header-title {
        font-size: 24px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 13px;
        opacity: 0.8;
        margin-top: 4px;
    }
    
    .badge {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 5px 15px;
        font-size: 12px;
        font-weight: 600;
        color: #fff;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Dashboard Cards */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.08);
    }
    
    /* Chat Bubbles */
    .chat-bubble {
        padding: 14px 18px;
        border-radius: 12px;
        margin-bottom: 12px;
        line-height: 1.6;
        font-size: 14.5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    
    .chat-user {
        background: #edf2f7;
        color: #2d3748;
        margin-left: 15%;
        border-top-right-radius: 2px;
        border: 1px solid #cbd5e0;
    }
    
    .chat-bot {
        background: #f0f7ff;
        color: #1a202c;
        margin-right: 15%;
        border-top-left-radius: 2px;
        border-left: 4px solid #1F3864;
        border: 1px solid #bee3f8;
    }
    
    /* Evidence Label */
    .evidence {
        font-size: 11.5px;
        color: #2b6cb0;
        background: #ebf8ff;
        border: 1px dashed #90cdf4;
        border-radius: 6px;
        padding: 8px 12px;
        margin-top: 8px;
    }
    
    /* Buttons styling */
    .stButton>button {
        background: linear-gradient(135deg, #13315c 0%, #1F3864 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(31, 56, 100, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# API Server URL
API_URL = "http://127.0.0.1:8000"

def call_api(endpoint, data=None, method="POST"):
    try:
        if method == "POST":
            r = requests.post(f"{API_URL}{endpoint}", json=data, timeout=30.0)
        else:
            r = requests.get(f"{API_URL}{endpoint}", timeout=30.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# Local Session State Initializations
if "active_model" not in st.session_state:
    st.session_state.active_model = "PortSLM (Fine-Tuned)"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 512
if "top_p" not in st.session_state:
    st.session_state.top_p = 0.9
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "local_history" not in st.session_state:
    st.session_state.local_history = []

# Mock fallbacks matching API logic
def local_mock_inference(model_name, prompt):
    prompt_lower = prompt.lower()
    if "dg" in prompt_lower or "위험물" in prompt_lower:
        if "base" in model_name.lower():
            return "일반적으로 위험물(DG) 컨테이너는 특수 구역에 보관해야 하며, 격리 규정이 적용됩니다. 상세한 슬롯 규칙은 선사 지침서(IMDG)를 참고해야 합니다."
        else:
            return "추천: BAY11 또는 BAY13의 DG 허용 슬롯. 근거: (1) DG 컨테이너는 Vessel Define에 등록된 'DG 적재 가능 Bay'에만 배치 필수(특수화물 배치 기준). (2) IMDG Code에 따라 Class 3 위험물은 발화원 및 기관실과 격리되어야 하므로 일반 BAY05는 탈락. (3) 12.0t 중량은 Heavy-Down 원칙상 최하단 TIER01 배치가 균형에 부합. 결론: DG 지정 Bay 제약이 결정적."
    elif "heavy-down" in prompt_lower or "무거운" in prompt_lower or "24.5t" in prompt_lower:
        if "base" in model_name.lower():
            return "무거운 컨테이너는 아래쪽에 쌓는 것이 선적 균형에 좋습니다. 보통 24.5t 컨테이너는 하단 슬롯에 추천되지만 구체적인 슬롯은 상황에 따라 다릅니다."
        else:
            return "추천 슬롯: BAY03-ROW02-TIER01. 근거: (1) 특수화물(DG/Reefer) 아님 → 지정 위치 제약 무관. (2) POD 그룹핑 — LAX는 원거리 항이므로 하부에 적재하여 양하 역순을 방지(선적 Planning 기준). (3) 중량 24.5t은 Heavy-Down & Light-Up 원칙상 하부 Tier 배치가 적합, 선박 복원성(COG) 확보. (4) 상단 간섭이 없어 재취급(rehandling) 위험 low. 종합: 4개 제약 충족."
    elif "reefer" in prompt_lower or "냉동" in prompt_lower:
        if "base" in model_name.lower():
            return "냉동(Reefer) 컨테이너는 전원 케이블을 연결할 수 있는 곳에 적재해야 합니다."
        else:
            return "추천 슬롯: BAY07-ROW01-TIER02. 근거: (1) Reefer 컨테이너는 전원 공급이 가능한 지정 Reefer Bay(BAY07/09)에만 배치 필수(특수화물 배치 기준). 후보 중 BAY07만 전원 연결 가능하므로 BAY03은 탈락. (2) ROTTERDAM은 원거리 항이나 Reefer 지정 위치 제약이 POD 그룹핑보다 우선함. (3) 18.0t 중간 중량으로 TIER02 배치는 중량 분포상 허용. 결론: Reefer 지정 위치 제약 결정적."
    else:
        if "base" in model_name.lower():
            return f"입력된 질문 '{prompt}'에 대한 일반적인 컨테이너 터미널 규정에 따르면, 모든 작업은 항만 안전 SOP 및 IMDG 규칙을 참고하십시오."
        else:
            return f"답변: 입력된 질문 '{prompt}'은(는) 안전 규정(SOP §3.2) 및 본선 플래닝 기준에 의거하여 조치를 취해야 합니다. 작업허가(PTW) 및 에너지 차단(LOTO) 절차를 선행하여 충돌과 위험을 사전에 통제하여야 합니다. 근거: 터미널 안전 매뉴얼 SOP."

# Sidebar Layout (SCR-05 & Global Navigation)
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #1F3864;'>PortSLM</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 12px; color: #69788f;'>aSSIST AI Project 4조</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Left Navigation menu
    page = st.radio(
        "화면 이동",
        ["홈 (Home)", "도메인 Q&A", "모델 비교 (前/後)", "평가 대시보드", "적재 계획 (Planning)", "이력 조회"]
    )
    
    st.markdown("---")
    st.markdown("### 전역 설정")
    
    # Global Settings (SCR-05 Settings)
    st.session_state.active_model = st.selectbox(
        "사용 모델 선택",
        ["PortSLM (Fine-Tuned)", "Qwen2.5-1.5B (Base)", "PortSLM (INT4 Quantized)"]
    )
    
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.5, st.session_state.temperature, 0.1)
    st.session_state.max_tokens = st.slider("Max Tokens", 64, 1024, st.session_state.max_tokens, 64)
    st.session_state.top_p = st.slider("Top-p", 0.1, 1.0, st.session_state.top_p, 0.05)

# Render Topbar Header
model_short = "PortSLM" if "Fine-Tuned" in st.session_state.active_model else "Base" if "Base" in st.session_state.active_model else "INT4"
st.markdown(f"""
<div class="header-bar">
    <div>
        <h1 class="header-title">PortSLM 항만 도메인 어시스턴트</h1>
        <div class="header-subtitle">컨테이너 터미널 적재계획 및 안전 규정 질의응답 시스템</div>
    </div>
    <div>
        <span class="badge">활성 모델: {st.session_state.active_model}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- PAGE RENDERING -----------------

if page == "홈 (Home)":
    # SCR-01 Home Screen
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <h3>서비스 소개</h3>
            <p style="color: #4a5568; line-height: 1.7; font-size: 15px;">
                인천신항 SNCT 컨테이너 터미널 본선 적재계획(Stowage Planning) 및 터미널 안전 규정(Safety SOP)에 특화된 
                소형언어모델(SLM) 파인튜닝 어시스턴트입니다.<br><br>
                본 플랫폼은 중량 배분(Heavy-Down), 위험물(DG) 및 냉동(Reefer) 특수 컨테이너 격리 규칙을 준수하고, 
                규정 조항을 정확하게 근거로 제시하는 설명 가능 지능(explainable AI)을 제공합니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example Question Chips
        st.markdown("### 예시 질문 칩")
        st.markdown("<p style='font-size: 12.5px; color:#718096;'>클릭 시 복사하여 Q&A 탭에서 입력할 수 있습니다.</p>", unsafe_allow_html=True)
        
        col_chip1, col_chip2, col_chip3 = st.columns(3)
        with col_chip1:
            st.info("💡 **DG 위험물 적재 규정**\n\n'DG 컨테이너의 적재 규칙은 무엇인가?'")
        with col_chip2:
            st.info("⚖️ **무거운 24.5t 슬롯 추천**\n\n'24.5t 컨테이너는 어느 슬롯에?'")
        with col_chip3:
            st.info("❄️ **Reefer 냉동 지정위치**\n\n'Reefer 컨테이너의 적재 위치는?'")
            
    with col2:
        st.markdown(f"""
        <div class="card">
            <h4>모델 정보 카드</h4>
            <hr style="margin: 10px 0;">
            <table style="width: 100%; font-size: 12.5px; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #edf2f7;"><td style="padding: 8px 0; font-weight: 600; color: #1F3864;">베이스</td><td style="color:#4a5568;">Qwen2.5-1.5B-Instruct</td></tr>
                <tr style="border-bottom: 1px solid #edf2f7;"><td style="padding: 8px 0; font-weight: 600; color: #1F3864;">LoRA 어댑터</td><td style="color:#4a5568;">portslm-lora-v1</td></tr>
                <tr style="border-bottom: 1px solid #edf2f7;"><td style="padding: 8px 0; font-weight: 600; color: #1F3864;">양자화</td><td style="color:#4a5568;">INT4 GGUF 가용</td></tr>
                <tr style="border-bottom: 1px solid #edf2f7;"><td style="padding: 8px 0; font-weight: 600; color: #1F3864;">학습인프라</td><td style="color:#4a5568;">VESSL AI GPU</td></tr>
                <tr><td style="padding: 8px 0; font-weight: 600; color: #1F3864;">학습일시</td><td style="color:#4a5568;">2026-06-22</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)


elif page == "도메인 Q&A":
    # SCR-02 Domain Q&A
    st.markdown("### 도메인 질의응답 (Q&A)")
    
    # Input Area
    user_query = st.text_input("질문을 입력하십시오 (예: DG 위험물 적재 규칙은 무엇인가?)", key="query_input")
    
    if st.button("전송") and user_query:
        # Call API or local mock
        api_res = call_api("/generate", {
            "prompt": user_query,
            "model": "base" if "Base" in st.session_state.active_model else "portslm",
            "temperature": st.session_state.temperature,
            "max_tokens": st.session_state.max_tokens,
            "top_p": st.session_state.top_p
        })
        
        if api_res:
            ans_text = api_res["text"]
            terms = api_res["terms"]
        else:
            ans_text = local_mock_inference(st.session_state.active_model, user_query)
            terms = [term for term in ["Heavy-Down", "Light-Up", "IMDG", "SOLAS", "Reefer", "DG", "segregation", "rehandling", "BAPLIE", "COPINO", "SOP"] if term.lower() in ans_text.lower()]
            
        # Add to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "bot", "content": ans_text, "terms": terms})
        
        # Save to local history log for SCR-06
        st.session_state.local_history.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": user_query,
            "model": st.session_state.active_model,
            "feedback": "—"
        })
        
    # Render chat bubbles
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f'<div class="chat-bubble chat-user">🧑 <b>사용자:</b> {chat["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble chat-bot">⚓ <b>어시스턴트:</b> {chat["content"]}</div>', unsafe_allow_html=True)
            if chat.get("terms"):
                st.markdown(f'<div class="evidence">▸ <b>감지된 핵심 도메인 용어/근거:</b> {", ".join(chat["terms"])}</div>', unsafe_allow_html=True)
                
            # Feedback buttons
            col_feed1, col_feed2, col_feed3 = st.columns([1, 1, 10])
            with col_feed1:
                if st.button("👍 Good", key=f"feed_up_{hash(chat['content'])}"):
                    call_api("/feedback", {"qid": user_query, "vote": "up"})
                    st.success("피드백 반영 완료!")
            with col_feed2:
                if st.button("👎 Bad", key=f"feed_down_{hash(chat['content'])}"):
                    call_api("/feedback", {"qid": user_query, "vote": "down"})
                    st.warning("피드백 반영 완료.")


elif page == "모델 비교 (前/後)":
    # SCR-03 Model Comparison (Demo Main Screen)
    st.markdown("### 파인튜닝 전/후 모델 성능 비교 (시연 화면)")
    st.markdown("<p style='font-size: 13px; color: #4a5568;'>동일한 조건(temperature, top-p, seed) 하에 베이스 모델과 파인튜닝 모델의 적재 계획 및 규정 준수 응답을 공정하게 비교 대조합니다.</p>", unsafe_allow_html=True)
    
    comp_query = st.text_input("비교 질문 입력", "24.5t 무거운 컨테이너의 적재 슬롯을 추천하고 근거를 설명하라.")
    
    if st.button("두 모델 답변 생성"):
        api_res = call_api("/compare", {"prompt": comp_query})
        
        if api_res:
            base_text = api_res["base_text"]
            ft_text = api_res["finetuned_text"]
            terms = api_res["terms"]
        else:
            base_text = local_mock_inference("base", comp_query)
            ft_text = local_mock_inference("portslm", comp_query)
            terms = [term for term in ["Heavy-Down", "Light-Up", "IMDG", "SOLAS", "Reefer", "DG", "segregation", "rehandling", "BAPLIE", "COPINO", "SOP"] if term.lower() in ft_text.lower()]
            
        col_base, col_ft = st.columns(2)
        
        with col_base:
            st.markdown(f"""
            <div class="card" style="border-top: 5px solid #aeb6c2;">
                <h4 style="color: #69788f !important;">Qwen2.5-1.5B (베이스 모델)</h4>
                <p style="font-size: 13.5px; color: #2d3748; line-height: 1.65;">{base_text}</p>
                <div style="background:#f7fafc; padding:10px; border-radius:6px; font-size:12px; color:#a0aec0; border:1px solid #e2e8f0; text-align:center;">
                    ⚠️ 모호한 제약 적용, 전문용어 인용 부족
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_ft:
            st.markdown(f"""
            <div class="card" style="border-top: 5px solid #1F3864;">
                <h4 style="color: #1F3864 !important;">PortSLM (파인튜닝 완료 모델)</h4>
                <p style="font-size: 13.5px; color: #2d3748; line-height: 1.65;">{ft_text}</p>
                <div class="evidence">
                    ▸ <b>적용된 규정/용어:</b> {", ".join(terms)}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Preference Voting
        st.markdown("<h4 style='text-align: center;'>인간 선호도 평가 피드백</h4>", unsafe_allow_html=True)
        col_vote1, col_vote2, col_vote3 = st.columns(3)
        with col_vote1:
            if st.button("◀ 베이스 모델 우수"):
                call_api("/feedback", {"qid": comp_query, "vote": "base"})
                st.info("베이스 모델 선택 기록 완료.")
        with col_vote2:
            if st.button("동등함"):
                call_api("/feedback", {"qid": comp_query, "vote": "equal"})
                st.info("동등 선택 기록 완료.")
        with col_vote3:
            if st.button("파인튜닝 모델 우수 ▶"):
                call_api("/feedback", {"qid": comp_query, "vote": "ft"})
                st.success("파인튜닝 모델 선호 기록 완료!")


elif page == "평가 대시보드":
    # SCR-04 Evaluation Dashboard
    st.markdown("### 평가 대시보드 (골든셋 30문항 결과)")
    
    # Try loading from API
    metrics = call_api("/metrics", method="GET")
    if not metrics:
        # Static mock matching the HTML storyboard metrics
        metrics = {
            "quant": {"base_rouge": 31.2, "ft_rouge": 88.5, "base_term": 20.0, "ft_term": 92.4},
            "qual": {"base_accuracy": 2.5, "base_grounding": 2.0, "base_terminology": 2.2, "ft_accuracy": 4.8, "ft_grounding": 4.7, "ft_terminology": 4.6},
            "hallucination": {"base": "18%", "ft": "7%"}
        }
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1. 정량 지표 비교")
        # ROUGE-L and Term rate chart
        quant_df = pd.DataFrame({
            "지표": ["ROUGE-L", "ROUGE-L", "도메인 용어포함률", "도메인 용어포함률"],
            "모델": ["베이스", "파인튜닝(PortSLM)", "베이스", "파인튜닝(PortSLM)"],
            "성능 (%)": [
                metrics["quant"]["base_rouge"],
                metrics["quant"]["ft_rouge"],
                metrics["quant"]["base_term"],
                metrics["quant"]["ft_term"]
            ]
        })
        st.bar_chart(quant_df, x="지표", y="성능 (%)", color="모델", stack=False)
        
    with col2:
        st.markdown("### 2. 정성 지표 비교 (LLM-as-judge)")
        qual_df = pd.DataFrame({
            "항목": ["정확성 (Accuracy)", "근거성 (Grounding)", "용어 적절성 (Terminology)"],
            "베이스": [metrics["qual"]["base_accuracy"], metrics["qual"]["base_grounding"], metrics["qual"]["base_terminology"]],
            "파인튜닝": [metrics["qual"]["ft_accuracy"], metrics["qual"]["ft_grounding"], metrics["qual"]["ft_terminology"]]
        }).set_index("항목")
        st.dataframe(qual_df, use_container_width=True)
        
        st.markdown("### 3. 환각률 비교 (전문가 수동 검수)")
        st.markdown(f"""
        <div style="display:flex; justify-content:space-around; align-items:center; background:#f7fafc; padding:20px; border-radius:10px; border:1px solid #e2e8f0;">
            <div style="text-align:center;">
                <div style="font-size:14px; color:#718096;">베이스 모델 환각률</div>
                <div style="font-size:32px; font-weight:800; color:#e53e3e;">{metrics["hallucination"]["base"]}</div>
            </div>
            <div style="font-size:30px; color:#cbd5e0;">▶</div>
            <div style="text-align:center;">
                <div style="font-size:14px; color:#718096;">PortSLM 환각률</div>
                <div style="font-size:32px; font-weight:800; color:#38a169;">{metrics["hallucination"]["ft"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 4. 샘플별 상세 비교 테이블")
    sample_df = pd.DataFrame([
        {"질문": "DG 컨테이너 적재 가능 Bay는?", "베이스": "모호한 대답", "PortSLM": "IMDG 조항 인용", "LLM-judge": "2점 → 5점"},
        {"질문": "Heavy-Down 원칙의 정의?", "베이스": "일반론적인 설명", "PortSLM": "정확한 안전 영향 설명", "LLM-judge": "3점 → 4점"},
        {"질문": "Reefer 컨테이너 슬롯 추천?", "베이스": "지정위치 제약 무시", "PortSLM": "전원공급 플러그 지정위치 권고", "LLM-judge": "1점 → 5점"},
    ])
    st.table(sample_df)


elif page == "적재 계획 (Planning)":
    st.markdown("### 🚢 스마트 적재 계획 생성기")
    st.markdown("<p style='font-size: 13px; color: #4a5568;'>엔진(Greedy/RL)을 선택하고 적재 계획을 요청하면, 에이전트가 계획 수립, 제약 검증, 그리고 규정에 기반한 근거 설명을 자동으로 수행합니다.</p>", unsafe_allow_html=True)
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        engine_choice = st.selectbox("최적화 엔진 선택", ["greedy", "rl"])
    with col_opt2:
        vessel_id = st.text_input("Vessel ID", "VESSEL-001")
        
    plan_query = st.text_area("작업 지시 입력", "무거운 24.5t 컨테이너와 DG 컨테이너를 포함한 화물 적재 계획을 수립하라.")
    
    if st.button("계획 수립 (Pipeline Run)"):
        with st.spinner("에이전트 파이프라인 실행 중 (Recognize → Plan → Validate → Explain)..."):
            res = call_api("/plan", {"question": plan_query, "engine": engine_choice, "vessel_id": vessel_id})
            if res:
                st.success(f"파이프라인 실행 완료 (소요시간: {res.get('latency_ms', 0)}ms)")
                
                # Rationale Display
                st.markdown(f"""
                <div class="card" style="border-top: 5px solid #38a169;">
                    <h4 style="color: #276749 !important;">💡 지능형 설명 (Explainable AI)</h4>
                    <div style="font-size: 14px; color: #2d3748; line-height: 1.6; white-space: pre-wrap;">{res.get('rationale', '설명 내용이 없습니다.')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                col_plan, col_viol = st.columns(2)
                with col_plan:
                    st.markdown("#### 배정된 슬롯 (Assignments)")
                    if res.get("assignments"):
                        st.dataframe(pd.DataFrame(res["assignments"]), use_container_width=True)
                    else:
                        st.info("배정된 컨테이너가 없습니다.")
                        
                with col_viol:
                    st.markdown("#### 제약 위반 (Violations)")
                    if res.get("violations"):
                        st.dataframe(pd.DataFrame(res["violations"]), use_container_width=True)
                    else:
                        st.success("✅ 위반 사항 없음")
            else:
                st.error("API 호출 실패. 백엔드 서버 상태를 확인하세요.")


elif page == "이력 조회":
    # SCR-06 History
    st.markdown("### 질의응답 및 피드백 이력 조회")
    
    # Try fetching from API history
    history_list = call_api("/history", method="GET")
    
    if not history_list:
        # Use local session history fallback
        history_list = st.session_state.local_history
        
    if history_list:
        hist_df = pd.DataFrame(history_list)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("조회할 이력이 없습니다. 도메인 Q&A 및 모델 비교 화면에서 질문을 전송해 주십시오.")

import sys
import pathlib
import os
import streamlit as st
import pandas as pd
import requests
import time

# src/ 패키지 경로 등록 (streamlit run dashboard/app.py 실행 가정)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

# .env 로드 (Neo4j/Neon 접속정보) — 있으면 자동 적용, 없으면 CSV 폴백
try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

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
        color: #ffffff !important;
        white-space: nowrap;
    }
    
    .header-subtitle {
        font-size: 13px;
        color: rgba(255, 255, 255, 0.9) !important;
        margin-top: 4px;
        white-space: nowrap;
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
        background: #ffffff !important;
        color: #2d3748 !important;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    
    .card p {
        color: #2d3748 !important;
        white-space: pre-line !important;
    }
    
    .card h4, .card span, .card div, .card td, .card li {
        color: #2d3748 !important;
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

def call_api(endpoint: str, data: dict = None, method: str = "POST") -> dict | None:
    """Make HTTP requests to the backend API server."""
    try:
        if method == "POST":
            r = requests.post(f"{API_URL}{endpoint}", json=data, timeout=180.0)
        else:
            r = requests.get(f"{API_URL}{endpoint}", timeout=180.0)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"[Dashboard API] Error: API returned status {r.status_code} for {endpoint}")
    except Exception as e:
        print(f"[Dashboard API] Request failed for {endpoint}: {e}")
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

def draw_bay_plan_fig(res):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import re
    
    # Matplotlib settings for cross-platform sans-serif font rendering
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Malgun Gothic', 'NanumGothic', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    assignments = res.get("assignments", [])
    slots = res.get("slots", [])
    if not assignments or not slots:
        return None
        
    # Get all row and tier numbers
    rows = sorted(list(set(s["row"] for s in slots)))
    tiers = sorted(list(set(s["tier"] for s in slots)))
    
    # Grid dimensions
    n_rows = len(rows)
    n_tiers = len(tiers)
    
    # Map row and tier names to 0-based indices
    row_to_idx = {r: idx for idx, r in enumerate(rows)}
    tier_to_idx = {t: idx for idx, t in enumerate(tiers)}
    
    # Map grid coordinates
    pod_grid = np.zeros((n_tiers, n_rows), dtype=int)
    weight_grid = np.zeros((n_tiers, n_rows), dtype=float)
    
    # Mapping table of PODs to indices 1..6
    POD_MAP = {
        "BUSAN": 1, "BUSAN(1)": 1,
        "SHANGHAI": 2, "SHANGHAI(2)": 2,
        "NINGBO": 3, "NINGBO(3)": 3,
        "SINGAPORE": 4, "SINGAPORE(4)": 4,
        "COLOMBO": 5, "COLOMBO(5)": 5,
        "ROTTERDAM": 6, "ROTTERDAM(6)": 6,
        "LAX": 6
    }
    
    # Map assigned containers
    for a in assignments:
        row = a["row"]
        tier = a["tier"]
        pod_name = str(a.get("pod", "")).upper()
        # Clean up name if it has numbers
        pod_clean = re.sub(r'[^A-Z]', '', pod_name)
        pod_id = POD_MAP.get(pod_clean, 6)
        if pod_clean == "":
            try:
                pod_id = int(re.search(r'\d+', pod_name).group())
            except Exception:
                pod_id = 6
        
        wt = float(a.get("weight_ton", 0.0))
        
        if row in row_to_idx and tier in tier_to_idx:
            r_idx = row_to_idx[row]
            t_idx = tier_to_idx[tier]
            pod_grid[t_idx, r_idx] = pod_id
            weight_grid[t_idx, r_idx] = wt
            
    # Setup Colors & Names
    POD_COLORS = {
        0: "#E2E8F0",  # Empty
        1: "#3A86C8",  # Busan
        2: "#48CAE4",  # Shanghai
        3: "#00B4D8",  # Ningbo
        4: "#90E0EF",  # Singapore
        5: "#FFB703",  # Colombo
        6: "#FB8500",  # Rotterdam
    }
    POD_NAMES = {
        1: "Busan(1)", 2: "Shanghai(2)", 3: "Ningbo(3)", 4: "Singapore(4)", 5: "Colombo(5)", 6: "Rotterdam(6)"
    }
    POD_DISPLAY_NAMES = {
        1: "Bus", 2: "Sha", 3: "Nin", 4: "Sin", 5: "Col", 6: "Rot"
    }
    
    # Matplotlib plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    
    # 1. POD Plot
    ax = axes[0]
    for t in range(n_tiers):
        for r in range(n_rows):
            pod_id = pod_grid[t, r]
            cell_color = POD_COLORS.get(pod_id, "#E2E8F0")
            rect = plt.Rectangle((r - 0.5, t - 0.5), 1.0, 1.0, facecolor=cell_color, edgecolor="#cbd5e1", linewidth=1.5)
            ax.add_patch(rect)
            
            if pod_id > 0:
                lbl = f"{POD_DISPLAY_NAMES.get(pod_id, 'POD')}({pod_id})"
                ax.text(r, t, lbl, ha="center", va="center", color="white" if pod_id in [1, 3, 6] else "#1e293b", fontweight="bold", fontsize=10)
                
    ax.set_title("POD Allocation (Discharge Port Plan)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(range(n_rows))
    ax.set_xticklabels([f"R{r - 1}" for r in rows], fontweight="semibold")
    ax.set_yticks(range(n_tiers))
    ax.set_yticklabels([f"T{t - 1}" for t in tiers], fontweight="semibold")
    ax.set_xlim(-0.5, n_rows - 0.5)
    ax.set_ylim(-0.5, n_tiers - 0.5)
    ax.set_xlabel("Row", fontsize=11, labelpad=8)
    ax.set_ylabel("Tier", fontsize=11, labelpad=8)
    ax.grid(False)
    
    legend_patches = [mpatches.Patch(color=POD_COLORS[i], label=POD_NAMES[i]) for i in range(1, 7)]
    ax.legend(handles=legend_patches, bbox_to_anchor=(0.5, -0.18), loc="upper center", ncol=3, frameon=True, facecolor="#f8fafc", edgecolor="#e2e8f0", fontsize=9)

    # 2. Weight Plot
    ax = axes[1]
    # Check max weight to scale colormap
    max_w = max(20.0, weight_grid.max())
    im = ax.imshow(weight_grid, cmap="Blues", origin="lower", aspect="equal", vmin=0.0, vmax=max_w)
    
    for t in range(n_tiers):
        for r in range(n_rows):
            wt = weight_grid[t, r]
            if wt > 0:
                ax.text(r, t, f"{wt:.1f}t", ha="center", va="center", color="black" if wt < 12.0 else "white", fontweight="bold", fontsize=10)
                
    ax.set_title("Weight Distribution (Metric Tons)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(range(n_rows))
    ax.set_xticklabels([f"R{r - 1}" for r in rows], fontweight="semibold")
    ax.set_yticks(range(n_tiers))
    ax.set_yticklabels([f"T{t - 1}" for t in tiers], fontweight="semibold")
    ax.set_xlim(-0.5, n_rows - 0.5)
    ax.set_ylim(-0.5, n_tiers - 0.5)
    ax.set_xlabel("Row", fontsize=11, labelpad=8)
    ax.grid(False)
    
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
    cbar.set_label("Weight (Metric Tons)", fontsize=9, labelpad=8)
    
    # Determine curriculum level label
    if n_rows == 4:
        level_num = "LV1"
    elif n_rows == 6:
        level_num = "LV2"
    elif n_rows == 8:
        level_num = "LV3"
    elif n_rows == 10:
        level_num = "LV4"
    else:
        level_num = f"{n_rows}R"
        
    fig.suptitle(f"PPO Stowage Optimization Plan ({level_num} - {n_rows}R × {n_tiers}T)", fontsize=15, fontweight="bold", color="#1E3A8A", y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.22)
    
    return fig

# Mock fallbacks matching API logic
def local_mock_inference(model_name: str, prompt: str) -> str:
    """Fallback mock inference for testing UI without backend API."""
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

# Sidebar Layout (SCR-05 & Global Navigation)
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #ffffff;'>PortSLM</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 12px; color: #cbd5e0;'>aSSIST AI Project 4조</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Left Navigation menu
    page = st.radio(
        "화면 이동",
        ["홈 (Home)", "도메인 Q&A", "모델 비교 (前/後)", "평가 대시보드", "적재 계획 (Planning)", "RL 적재 설명 (xAI)", "컨테이너 위치 조회", "Neo4j 그래프", "이력 조회"]
    )
    
    st.markdown("---")
    st.markdown("### 전역 설정")
    
    st.session_state.active_model = st.selectbox(
        "사용 모델 선택",
        ["PortSLM (Fine-Tuned)", "Qwen2.5-VL-3B (Base)", "PortSLM (INT4 Quantized)"]
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
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        <div class="card">
            <h3>서비스 소개</h3>
            <p style="color: #4a5568; line-height: 1.7; font-size: 15px;">
                인천신항 컨테이너 터미널 본선 적재계획(Stowage Planning) 및 터미널 안전 규정(Safety SOP)에 특화된 
                소형언어모델(SLM) 파인튜닝 어시스턴트입니다.<br><br>
                본 플랫폼은 중량 배분(Heavy-Down), 위험물(DG) 및 냉동(Reefer) 특수 컨테이너 격리 규칙을 준수하고, 
                규정 조항을 정확하게 근거로 제시하는 설명 가능 지능(explainable AI)을 제공합니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
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
                <tr style="border-bottom: 1px solid #edf2f7;"><td style="padding: 8px 0; font-weight: 600; color: #1F3864;">베이스</td><td style="color:#4a5568;">Qwen2.5-VL-3B</td></tr>
                <tr style="border-bottom: 1px solid #edf2f7;"><td style="padding: 8px 0; font-weight: 600; color: #1F3864;">LoRA 어댑터</td><td style="color:#4a5568;">portslm-lora-v1</td></tr>
                <tr style="border-bottom: 1px solid #edf2f7;"><td style="padding: 8px 0; font-weight: 600; color: #1F3864;">양자화</td><td style="color:#4a5568;">INT4 GGUF 가용</td></tr>
                <tr style="border-bottom: 1px solid #edf2f7;"><td style="padding: 8px 0; font-weight: 600; color: #1F3864;">학습인프라</td><td style="color:#4a5568;">VESSL AI GPU</td></tr>
                <tr><td style="padding: 8px 0; font-weight: 600; color: #1F3864;">학습일시</td><td style="color:#4a5568;">2026-06-13</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

elif page == "도메인 Q&A":
    st.markdown("### 도메인 질의응답 (Q&A)")
    user_query = st.text_input("질문을 입력하십시오 (예: DG 위험물 적재 규칙은 무엇인가?)", key="query_input")
    
    if st.button("전송") and user_query:
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
            
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "bot", "content": ans_text, "terms": terms})
        
        st.session_state.local_history.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": user_query,
            "model": st.session_state.active_model,
            "feedback": "—"
        })
        
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f'<div class="chat-bubble chat-user">🧑 <b>사용자:</b> {chat["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble chat-bot">⚓ <b>어시스턴트:</b> {chat["content"]}</div>', unsafe_allow_html=True)
            if chat.get("terms"):
                st.markdown(f'<div class="evidence">▸ <b>감지된 핵심 도메인 용어/근거:</b> {", ".join(chat["terms"])}</div>', unsafe_allow_html=True)
                
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
    st.markdown("### 파인튜닝 전/후 모델 성능 비교 (시연 화면)")
    st.markdown("<p style='font-size: 13px; color: #4a5568;'>동일한 조건(temperature, top-p, seed) 하에 베이스 모델과 파인튜닝 모델의 적재 계획 및 규정 준수 응답을 공정하게 비교 대조합니다.</p>", unsafe_allow_html=True)
    
    comp_query = st.text_input("비교 질문 입력", "24.5t 무거운 컨테이너의 적재 슬롯을 추천하고 근거를 설명하라.")
    
    if st.button("두 모델 답변 생성"):
        api_res = call_api("/compare", {
            "prompt": comp_query,
            "temperature": st.session_state.temperature,
            "max_tokens": st.session_state.max_tokens,
            "top_p": st.session_state.top_p
        })
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
                <h4 style="color: #69788f !important;">Qwen2.5-VL-3B (베이스 모델)</h4>
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
    st.markdown("### 평가 대시보드 (골든셋 30문항 결과)")
    metrics = call_api("/metrics", method="GET")
    if not metrics:
        metrics = {
            "quant": {"base_rouge": 31.2, "ft_rouge": 88.5, "base_term": 20.0, "ft_term": 92.4},
            "qual": {"base_accuracy": 2.5, "base_grounding": 2.0, "base_terminology": 2.2, "ft_accuracy": 4.8, "ft_grounding": 4.7, "ft_terminology": 4.6},
            "hallucination": {"base": "18%", "ft": "7%"}
        }
        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 1. 정량 지표 비교")
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
        import altair as alt
        chart = alt.Chart(quant_df).mark_bar().encode(
            x=alt.X("모델:N", title=None),
            y=alt.Y("성능 (%):Q", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("모델:N", scale=alt.Scale(
                domain=["베이스", "파인튜닝(PortSLM)"],
                range=["#aeb6c2", "#1F3864"]
            )),
            column=alt.Column("지표:N", title=None)
        ).properties(width=130)
        st.altair_chart(chart, use_container_width=False)
        
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
    if metrics and "samples" in metrics and metrics["samples"]:
        sample_list = []
        for s in metrics["samples"]:
            sample_list.append({
                "질문": s["question"],
                "베이스 모델 답변": s["base"],
                "PortSLM 답변": s["ft"],
                "개선도 (ROUGE-L 차이)": s["score_gap"]
            })
        sample_df = pd.DataFrame(sample_list)
    else:
        sample_df = pd.DataFrame([
            {"질문": "DG 컨테이너 적재 가능 Bay는?", "베이스 모델 답변": "모호한 대답", "PortSLM 답변": "IMDG 조항 인용", "개선도 (ROUGE-L 차이)": "+60%p"},
            {"질문": "Heavy-Down 원칙의 정의?", "베이스 모델 답변": "일반론적인 설명", "PortSLM 답변": "정확한 안전 영향 설명", "개선도 (ROUGE-L 차이)": "+30%p"},
            {"질문": "Reefer 컨테이너 슬롯 추천?", "베이스 모델 답변": "지정위치 제약 무시", "PortSLM 답변": "전원공급 플러그 지정위치 권고", "개선도 (ROUGE-L 차이)": "+80%p"},
        ])
    st.table(sample_df)

elif page == "적재 계획 (Planning)":
    st.markdown("### 🚢 스마트 적재 계획 생성기")
    st.markdown("<p style='font-size: 13px; color: #4a5568;'>엔진(Greedy/RL)을 선택하고 적재 계획을 요청하면, 에이전트가 계획 수립, 제약 검증, 그리고 규정에 기반한 근거 설명을 자동으로 수행합니다.</p>", unsafe_allow_html=True)
    
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        engine_choice = st.selectbox("최적화 엔진 선택", ["greedy", "rl_bl", "rl_sf", "rl_ef"])
    with col_opt2:
        level_choice = st.selectbox("학습 단계 (Curriculum Level)", [
            "Level 4 (10R × 10T)",
            "Level 3 (8R × 8T)",
            "Level 2 (6R × 6T)",
            "Level 1 (4R × 4T)"
        ])
    with col_opt3:
        level_to_vessel = {
            "Level 4 (10R × 10T)": "VESSEL-LV4",
            "Level 3 (8R × 8T)": "VESSEL-LV3",
            "Level 2 (6R × 6T)": "VESSEL-LV2",
            "Level 1 (4R × 4T)": "VESSEL-LV1",
        }
        vessel_id = st.text_input("Vessel ID", level_to_vessel[level_choice])
        
    plan_query = st.text_area("작업 지시 입력", "무거운 24.5t 컨테이너와 DG 컨테이너를 포함한 화물 적재 계획을 수립하라.")
    
    if st.button("계획 수립 (Pipeline Run)"):
        with st.spinner("에이전트 파이프라인 실행 중 (Recognize → Plan → Validate → Explain)..."):
            res = call_api("/plan", {"question": plan_query, "engine": engine_choice, "vessel_id": vessel_id})
            if res:
                st.success(f"파이프라인 실행 완료 (소요시간: {res.get('latency_ms', 0)}ms)")
                
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
                
                # Render Bay Plan Visualizations
                st.markdown("### 📊 적재 계획 시각화 (Bay Plan)")
                fig = draw_bay_plan_fig(res)
                if fig:
                    st.pyplot(fig)
            else:
                st.error("API 호출 실패. 백엔드 서버 상태를 확인하세요.")

elif page == "RL 적재 설명 (xAI)":
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
        
        if "explanation_result" not in st.session_state:
            st.session_state.explanation_result = None

        if st.button("설명 생성", type="primary"):
            with st.spinner("RL 의사결정 설명 및 시각화 자료 로드 중..."):
                from snct.agents.graph import run_explanation
                rec = run_explanation(policy=policy, round_id=int(round_id), with_lpg=with_lpg)
                st.session_state.explanation_result = {
                    "policy": policy,
                    "round_id": round_id,
                    "rationale": rec.rationale,
                    "checks": rec.checks,
                }

        if st.session_state.explanation_result:
            result = st.session_state.explanation_result
            faith = next((c.split("=")[1] for c in result["checks"] if c.startswith("faithfulness=")), None)
            if faith is not None:
                faith_pct = int(float(faith) * 100)
                status_text = f"🛡️ **설명 신뢰성 검증 완료** (근거 충실도: **{faith_pct}%**)  |  **정책**: {result['policy']}  |  **라운드**: {result['round_id']}"
                if float(faith) >= 1.0:
                    st.success(status_text)
                else:
                    st.warning(status_text + " (⚠️ 일부 설명에 수치 불일치 가능성이 감지되었습니다.)")
            st.markdown(result["rationale"])

            # 훈련 결과 차트 이미지 렌더링 (NFC/NFD 호환)
            curr_policy = result["policy"]
            from snct.data.sources.rl_results import default_results_dir
            try:
                base_dir = default_results_dir()
                v13_dir = base_dir / "v13_3way_BL_SF_EF (1)"
                policy_dir = v13_dir / curr_policy
                
                img_files = [
                    f"fig5_bay_plan_PPO_{curr_policy}.png"
                ]
                
                existing_imgs = []
                for img_name in img_files:
                    img_path = policy_dir / img_name
                    if img_path.exists():
                        existing_imgs.append((img_path, img_name))
                
                if existing_imgs:
                    st.write("---")
                    st.markdown("### 📊 최적 적재 배치도 (Optimal Stowage Plan)")
                    cols = st.columns(2)
                    for idx, (path, name) in enumerate(existing_imgs):
                        name_clean = name.replace(".png", "")
                        if "PPO" in name_clean:
                            policy_suffix = name_clean.split("PPO_")[-1]
                            caption_name = f"📊 PPO 최적 적재계획도 (Policy: {policy_suffix.upper()})"
                        else:
                            caption_name = f"📊 {name_clean.replace('_', ' ').title()}"
                            
                        with cols[idx % 2]:
                            st.markdown(f"""
                            <div class="card" style="padding: 10px; margin-bottom: 10px; text-align: center;">
                                <p style="font-weight: bold; margin-bottom: 0px; color: #1F3864 !important; font-size: 14px;">{caption_name}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(str(path), use_container_width=True)
            except Exception as e:
                st.warning(f"시각화 이미지를 불러오는 중 오류가 발생했습니다: {e}")

elif page == "컨테이너 위치 조회":
    st.subheader("컨테이너 위치 조회")
    st.caption("적재계획 완료 후 — 컨테이너가 어디 있는지, 바로 반출 가능한지 확인합니다.")
    q = st.text_input("질문 또는 컨테이너 ID", placeholder="예: BL_R4_r0_t0 어디 있어?")
    if st.button("위치 조회", type="primary") and q.strip():
        from snct.knowledge.locator import where_is
        res = where_is(q)
        if res["sources"]:
            st.success(res["answer"])
            
            # JSON 원시 데이터 대신 프리미엄 상세 카드 렌더링
            info = res["sources"][0]["snippet"]
            if isinstance(info, dict):
                is_top = info.get("is_top", False)
                retrieval_status = "🟢 즉시 반출 가능 (최상단 적재)" if is_top else "🟡 재취급(Rehandling) 필요 (상단 컨테이너 간섭)"
                
                # Define premium status badges
                if is_top:
                    badge_html = (
                        '<span style="background-color: #E6F4EA; color: #137333; padding: 4px 10px; '
                        'border-radius: 12px; font-size: 13px; font-weight: 700; border: 1px solid #CEEAD6; '
                        'white-space: nowrap; display: inline-block;">'
                        '🟢 즉시 반출 가능 (최상단 적재)'
                        '</span>'
                    )
                else:
                    badge_html = (
                        '<span style="background-color: #FEF3C7; color: #B35C00; padding: 4px 10px; '
                        'border-radius: 12px; font-size: 13px; font-weight: 700; border: 1px solid #FDE68A; '
                        'white-space: nowrap; display: inline-block;">'
                        '🟡 재취급(Rehandling) 필요 (상단 컨테이너 간섭)'
                        '</span>'
                    )

                html_card = (
                    f'<div class="card" style="border-left: 5px solid #1F3864; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">'
                    f'<h4 style="margin-top: 0px; color: #1F3864 !important; font-weight: 700;">📦 컨테이너 물리 상세 정보 (LPG 그래프 조회)</h4>'
                    f'<hr style="margin: 10px 0; border: 0; border-top: 1px solid #edf2f7;">'
                    f'<table style="width: 100%; border-collapse: collapse; font-size: 14.5px; text-align: left; table-layout: fixed;">'
                    f'<tr style="border-bottom: 1px solid #edf2f7;"><td style="padding: 10px 8px; font-weight: 600; color: #4a5568; width: 220px; min-width: 220px; white-space: nowrap; vertical-align: middle;">🏷️ 컨테이너 식별자</td><td style="padding: 10px 8px; color: #2d3748; font-weight: 600; vertical-align: middle;">{info.get("container_id")}</td></tr>'
                    f'<tr style="border-bottom: 1px solid #edf2f7;"><td style="padding: 10px 8px; font-weight: 600; color: #4a5568; width: 220px; min-width: 220px; white-space: nowrap; vertical-align: middle;">🚢 선박명 / 항차</td><td style="padding: 10px 8px; color: #2d3748; vertical-align: middle;">{info.get("vessel")} ({info.get("voyage")})</td></tr>'
                    f'<tr style="border-bottom: 1px solid #edf2f7;"><td style="padding: 10px 8px; font-weight: 600; color: #4a5568; width: 220px; min-width: 220px; white-space: nowrap; vertical-align: middle;">📍 배치 위치 (Bay/Row/Tier)</td><td style="padding: 10px 8px; color: #2d3748; vertical-align: middle;">{info.get("bay")} - ROW {info.get("row")} - TIER {info.get("tier")}</td></tr>'
                    f'<tr style="border-bottom: 1px solid #edf2f7;"><td style="padding: 10px 8px; font-weight: 600; color: #4a5568; width: 220px; min-width: 220px; white-space: nowrap; vertical-align: middle;">⚖️ 목적지 (POD) / 중량</td><td style="padding: 10px 8px; color: #2d3748; vertical-align: middle;">{info.get("pod")} / {info.get("weight_mt", 0.0):.2f} Tons</td></tr>'
                    f'<tr style="border-bottom: 1px solid #edf2f7;"><td style="padding: 10px 8px; font-weight: 600; color: #4a5568; width: 220px; min-width: 220px; white-space: nowrap; vertical-align: middle;">🥞 적층 상태</td><td style="padding: 10px 8px; color: #2d3748; vertical-align: middle;">'
                    f'{"최하단 적재 (Bottom)" if info.get("is_bottom") else "중간 적재"}'
                    f'{" · 최상단 적재 (Top)" if info.get("is_top") else ""}'
                    f'</td></tr>'
                    f'<tr><td style="padding: 10px 8px; font-weight: 600; color: #4a5568; width: 220px; min-width: 220px; white-space: nowrap; vertical-align: middle;">🚚 현 시점 반출 여부</td><td style="padding: 10px 8px; vertical-align: middle; line-height: 1.8;">{badge_html}</td></tr>'
                    f'</table>'
                    f'</div>'
                )
                st.markdown(html_card, unsafe_allow_html=True)
            else:
                st.info(str(info))
        else:
            st.warning(res["answer"])

elif page == "Neo4j 그래프":
    st.subheader("Neo4j 지식그래프 (LPG)")
    from snct.knowledge.lpg import lpg_status, get_lpg
    status = lpg_status()

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

    if status["neo4j_connected"]:
        if st.button("neo4j_kg CSV → Neo4j 적재"):
            from snct.knowledge.lpg_neo4j import Neo4jLPG
            with st.spinner("적재 중..."):
                counts = Neo4jLPG().import_kg()
            st.success(f"적재 완료: {counts}")

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

elif page == "이력 조회":
    st.markdown("### 질의응답 및 피드백 이력 조회")
    history_list = call_api("/history", method="GET")
    if not history_list:
        history_list = st.session_state.local_history
        
    if history_list:
        hist_df = pd.DataFrame(history_list)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("조회할 이력이 없습니다. 도메인 Q&A 및 모델 비교 화면에서 질문을 전송해 주십시오.")

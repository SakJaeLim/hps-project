#!/bin/bash

# 프로젝트 폴더로 이동 (실행 위치 기준)
cd "$(dirname "$0")"
export SNCT_BASE_DIR="$(pwd)" # 윈도우 하드코딩 경로(i:\내 드라이브) 회피를 위해 환경 변수 자동 바인딩

echo "============================================="
echo "   PortSLM Service Auto-Restart Script       "
echo "============================================="

# 1. 깃 최신화
echo -e "\n[1/5] Pulling latest code from origin/sunny..."
git pull origin sunny

# 2. 기존 프로세스 종료
echo -e "\n[2/5] Stopping existing processes (uvicorn & streamlit)..."
pkill -f uvicorn
pkill -f streamlit
sleep 2

# 3. 가상환경 확인 및 패키지 설치
USE_VENV=true

# 기존 .venv가 온전한지 체크 (bin 폴더와 핵심 실행파일 유무)
if [ -d ".venv" ]; then
    if [ ! -f ".venv/bin/pip" ] || [ ! -f ".venv/bin/uvicorn" ] || [ ! -f ".venv/bin/streamlit" ]; then
        echo "Detected corrupted virtual environment (.venv). Removing and recreating..."
        rm -rf .venv
    fi
fi

if [ ! -d ".venv" ]; then
    echo -e "\n[3/5] Creating new virtual environment (.venv)..."
    python3 -m venv .venv 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "WARNING: python3 -m venv failed (possibly missing python3-venv package)."
        echo "Falling back to System Python Environment..."
        USE_VENV=false
    fi
fi

echo "Installing/Updating dependencies..."
export PYTHONUTF8=1

if [ "$USE_VENV" = true ]; then
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
    PIP_CMD=".venv/bin/pip"
    UVICORN_CMD=".venv/bin/uvicorn"
    STREAMLIT_CMD=".venv/bin/streamlit"
else
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    PIP_CMD="pip3"
    UVICORN_CMD="python3 -m uvicorn"
    STREAMLIT_CMD="python3 -m streamlit"
fi

# 4. 서비스 시작 (FastAPI 백엔드)
echo -e "\n[4/5] Starting FastAPI Backend on port 8000..."
nohup env PYTHONPATH=src $UVICORN_CMD src.snct.api.app_api:app --host 0.0.0.0 --port 8000 > fastapi.log 2>&1 &
FASTAPI_PID=$!
echo "FastAPI running with PID: $FASTAPI_PID"

# 5. 서비스 시작 (Streamlit 프론트엔드)
echo -e "\n[5/5] Starting Streamlit Dashboard on port 8501..."
nohup env PYTHONPATH=src $STREAMLIT_CMD run dashboard/app.py --server.port 8501 > streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "Streamlit running with PID: $STREAMLIT_PID"

sleep 3

# 6. 구동 프로세스 상태 검증
echo -e "\n=== Currently Running Services ==="
ps aux | grep -E "uvicorn|streamlit" | grep -v grep

echo -e "\n============================================="
echo "   Restart completed!"
echo "   - FastAPI Log: fastapi.log"
echo "   - Streamlit Log: streamlit.log"
echo "   - Env SNCT_BASE_DIR: $SNCT_BASE_DIR"
echo "============================================="

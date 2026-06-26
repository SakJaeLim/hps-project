#!/bin/bash

# 프로젝트 폴더로 이동 (실행 위치 기준)
cd "$(dirname "$0")"

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
if [ ! -d ".venv" ]; then
    echo -e "\n[3/5] Virtual environment (.venv) not found. Creating one..."
    python3 -m venv .venv
else
    echo -e "\n[3/5] Virtual environment (.venv) detected."
fi

echo "Installing/Updating dependencies..."
export PYTHONUTF8=1
.venv/bin/pip install -r requirements.txt

# 4. 서비스 시작 (FastAPI 백엔드)
echo -e "\n[4/5] Starting FastAPI Backend on port 8000..."
nohup .venv/bin/uvicorn src.snct.api.app_api:app --host 0.0.0.0 --port 8000 > fastapi.log 2>&1 &
FASTAPI_PID=$!
echo "FastAPI running with PID: $FASTAPI_PID"

# 5. 서비스 시작 (Streamlit 프론트엔드)
echo -e "\n[5/5] Starting Streamlit Dashboard on port 8501..."
nohup .venv/bin/streamlit run dashboard/app.py --server.port 8501 > streamlit.log 2>&1 &
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
echo "============================================="

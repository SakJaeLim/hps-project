#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

# Start FastAPI backend in background
python -m uvicorn snct.api.app_api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit dashboard in foreground
streamlit run dashboard/dashboard_app.py --server.port 8501 --server.address 0.0.0.0

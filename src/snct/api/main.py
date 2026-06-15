"""L6 FastAPI — 에이전트 그래프 호출 엔드포인트. specs/00 · 계약 Recommendation."""
from fastapi import FastAPI
app = FastAPI(title="SNCT Decision Support")

@app.get("/health")
def health(): return {"ok": True}

@app.post("/plan")
def plan(req: dict):
    """{vessel_id, containers[]} → {slots[], rationale, checks[]}  TODO(W2)."""
    return {"status": "not_implemented"}

"""T13 · spec 00 — TDD Green."""
import pytest
from fastapi.testclient import TestClient

def test_api_plan():
    from snct.api.app_api import app
    client = TestClient(app)
    r = client.post("/plan", json={"question": "Test", "vessel_id": "V-1"})
    assert r.status_code == 200
    assert "assignments" in r.json()

"""T13 · spec 00 — TDD Red(미구현). 구현되면 xfail 제거 → Green 이어야 함."""
import pytest

@pytest.mark.xfail(reason="TDD Red — T13 미구현", strict=False)
def test_api_plan():
    from fastapi.testclient import TestClient
    from snct.api.main import app
    r = TestClient(app).post("/plan", json={"vessel_id": "V-1", "containers": []})
    assert r.status_code == 200 and "slots" in r.json()

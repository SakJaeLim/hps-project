"""T26 · spec 00·07 — POST /explain 엔드포인트 TDD.

TestClient(anyio) 환경 이슈를 피해 엔드포인트 함수를 직접 호출해 계약을 검증한다.
"""
import pytest

pytestmark = pytest.mark.rl_data


def test_explain_endpoint_by_ids():
    from snct.api.app_api import explain_endpoint, ExplainRequest
    res = explain_endpoint(ExplainRequest(policy="BL", round_id=4))
    assert "SOLAS_VI" in res["rationale"]
    assert any("faithfulness=1.0" in c for c in res["checks"])
    assert "latency_ms" in res


def test_explain_endpoint_natural_language():
    from snct.api.app_api import explain_endpoint, ExplainRequest
    res = explain_endpoint(ExplainRequest(question="EF 정책 2라운드 적재 사유"))
    assert isinstance(res["rationale"], str) and res["rationale"]
    assert any("policy=EF" in c for c in res["checks"])


def test_explain_endpoint_unknown_is_graceful():
    from snct.api.app_api import explain_endpoint, ExplainRequest
    res = explain_endpoint(ExplainRequest(policy="BL", round_id=999))
    assert "없습니다" in res["rationale"] or "찾을 수 없" in res["rationale"]

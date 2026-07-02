"""T27 · spec 07 — 컨테이너 위치 조회(완성된 적재계획 slot_assignment) TDD.

자연어 질의 → 컨테이너 위치(bay/row/tier) + 적층 정보(반출 가능 여부)를 반환.
근거 소스: slot_assignment(RDB) + LPGGraph.stacked_on(LPG).
"""
import pytest

pytestmark = pytest.mark.rl_data


def test_locate_returns_position():
    from snct.knowledge.locator import locate
    loc = locate("BL_R2_r0_t0")
    assert loc is not None
    assert loc["bay"] == "BAY_01"
    assert loc["row"] == 0 and loc["tier"] == 0
    assert loc["pod"] == "Rotterdam"
    assert loc["is_top"] is True       # 최상단


def test_locate_unknown_returns_none():
    from snct.knowledge.locator import locate
    assert locate("NO_SUCH_CONTAINER") is None


def test_where_is_top_container_is_directly_accessible():
    from snct.knowledge.locator import where_is
    res = where_is("BL_R2_r0_t0 컨테이너 어디 있어?")
    assert "BAY_01" in res["answer"]
    assert "반출 가능" in res["answer"]
    assert res["sources"][0]["type"] == "sql"


def test_where_is_blocked_container_needs_reshuffle():
    """위에 적층된 컨테이너 → 재취급 필요 + 위 컨테이너 ID 명시."""
    from snct.knowledge.locator import where_is
    res = where_is("BL_R4_r0_t0 위치 알려줘")
    assert "재취급" in res["answer"]
    assert "BL_R4_r0_t1" in res["answer"]   # 위에 쌓인 컨테이너


def test_where_is_no_id_is_graceful():
    from snct.knowledge.locator import where_is
    res = where_is("오늘 날씨 어때?")
    assert res["sources"] == []


def test_locate_endpoint_by_container_id():
    from snct.api.app_api import locate_endpoint, LocateRequest
    res = locate_endpoint(LocateRequest(container_id="BL_R2_r0_t0"))
    assert "BAY_01" in res["answer"]
    assert res["sources"][0]["ref"] == "slot_assignment"


def test_locate_endpoint_natural_language():
    from snct.api.app_api import locate_endpoint, LocateRequest
    res = locate_endpoint(LocateRequest(question="BL_R4_r0_t0 위치 알려줘"))
    assert "재취급" in res["answer"]

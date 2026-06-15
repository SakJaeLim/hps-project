"""T11 · spec 07 — TDD Red(미구현). 구현되면 xfail 제거 → Green 이어야 함."""
import pytest

@pytest.mark.xfail(reason="TDD Red — T11 미구현", strict=False)
def test_nl2sql_readonly():
    from snct.knowledge import nl2sql
    with pytest.raises(Exception):
        nl2sql.run_readonly("DELETE FROM yard")  # 변경 쿼리 차단(읽기전용)
    res = nl2sql.ask("오늘 위험물 컨테이너 수")
    assert res["sources"][0]["type"] == "sql"

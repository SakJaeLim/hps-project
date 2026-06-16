"""T11 · spec 07 — TDD Green."""
import pytest

def test_nl2sql_readonly():
    from snct.knowledge import nl2sql
    with pytest.raises(Exception):
        nl2sql.run_readonly("DELETE FROM yard")  # 변경 쿼리 차단(읽기전용)
    res = nl2sql.ask("오늘 위험물 컨테이너 수")
    assert res["sources"][0]["type"] == "sql"

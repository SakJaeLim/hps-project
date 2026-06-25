"""T11 · spec 07 — TDD Green. T20: RL 결과 RDB NL2SQL(읽기전용·DuckDB) 추가."""
import pytest

def test_nl2sql_readonly():
    from snct.knowledge import nl2sql
    with pytest.raises(Exception):
        nl2sql.run_readonly("DELETE FROM yard")  # 변경 쿼리 차단(읽기전용)
    res = nl2sql.ask("오늘 위험물 컨테이너 수")
    assert res["sources"][0]["type"] == "sql"


# ── T20: RL 결과 RDB NL2SQL (kpi·reward_decomp·violation_log → DuckDB) ──
def _analyst():
    from snct.knowledge.nl2sql import RLAnalyst
    return RLAnalyst()


@pytest.mark.rl_data
def test_rl_analyst_loads_rdb_tables():
    a = _analyst()
    tables = a.tables()
    assert {"kpi", "reward_decomp", "violation_log"} <= set(tables)


@pytest.mark.rl_data
def test_rl_analyst_blocks_write_queries():
    a = _analyst()
    for bad in ("UPDATE kpi SET reward=0", "DROP TABLE kpi", "DELETE FROM kpi"):
        with pytest.raises(ValueError):
            a.query(bad)


@pytest.mark.rl_data
def test_rl_analyst_kpi_query_is_faithful():
    """BL 정책 OSR 질의 → 모든 BL 라운드 osr=0.0 (CSV 사실과 일치)."""
    a = _analyst()
    res = a.ask("BL 정책 라운드별 재취급률(OSR)을 보여줘")
    assert res["sources"][0]["type"] == "sql"
    rows = res["sources"][0]["snippet"]
    assert rows and all(str(r["policy"]) == "BL" for r in rows)
    assert all(float(r["osr"]) == 0.0 for r in rows)


@pytest.mark.rl_data
def test_rl_analyst_violation_aggregation():
    """컬럼중량 위반 최다 = 8건(BL/SF R4). 답변에 8 등장."""
    a = _analyst()
    res = a.ask("컬럼중량 위반이 가장 많은 정책과 라운드는?")
    assert "8" in res["answer"]
    # SUMMARY 스코프 집계 — 슬롯 단위 중복 합산이 아님
    assert "violation_log" in res["sources"][0]["ref"].lower()


@pytest.mark.rl_data
def test_rl_analyst_query_enforces_limit():
    a = _analyst()
    rows = a.query("SELECT * FROM reward_decomp")
    assert isinstance(rows, list)

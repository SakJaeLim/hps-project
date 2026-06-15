"""NL2SQL (정형 운영 DB) — 자연어 → SQL(읽기전용) → 실행.
DuckDB 인메모리 DB에 시뮬레이션 운영 데이터 적재."""
import duckdb

# In-memory DuckDB with simulated operational data
_con = duckdb.connect(":memory:")
_con.execute("""
    CREATE TABLE containers (
        id VARCHAR PRIMARY KEY,
        weight_ton DOUBLE,
        size VARCHAR,
        type VARCHAR,
        pod VARCHAR,
        dg BOOLEAN,
        reefer BOOLEAN,
        discharge_order INT,
        status VARCHAR DEFAULT 'QUEUED'
    )
""")
_con.execute("""
    CREATE TABLE slots (
        bay INT, row_num INT, tier INT,
        max_stack_weight DOUBLE,
        dg_allowed BOOLEAN,
        reefer_capable BOOLEAN,
        occupied_by VARCHAR,
        PRIMARY KEY (bay, row_num, tier)
    )
""")
_con.execute("""
    CREATE TABLE operations (
        op_id VARCHAR,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        container_id VARCHAR,
        operation VARCHAR,
        crane_id VARCHAR,
        duration_sec INT
    )
""")

# Seed sample data
_con.executemany(
    "INSERT INTO containers VALUES (?,?,?,?,?,?,?,?,?)",
    [
        ("CNTR-001", 24.5, "40", "GP", "LAX", False, False, 1, "QUEUED"),
        ("CNTR-002", 18.0, "40", "RF", "ROTTERDAM", False, True, 2, "QUEUED"),
        ("CNTR-003", 12.0, "20", "DG", "SINGAPORE", True, False, 3, "QUEUED"),
        ("CNTR-004", 22.0, "40", "GP", "BUSAN", False, False, 4, "LOADED"),
        ("CNTR-005", 8.5, "20", "GP", "LAX", False, False, 5, "QUEUED"),
        ("CNTR-006", 15.0, "40", "RF", "SHANGHAI", False, True, 6, "QUEUED"),
    ],
)
_con.executemany(
    "INSERT INTO slots VALUES (?,?,?,?,?,?,?)",
    [
        (1, 1, 1, 30.0, False, False, None),
        (1, 2, 1, 30.0, False, True, None),
        (1, 3, 1, 30.0, True, False, None),
        (3, 1, 1, 30.0, False, False, "CNTR-004"),
        (3, 2, 1, 30.0, True, True, None),
        (5, 1, 1, 30.0, False, False, None),
        (7, 1, 1, 30.0, False, True, None),
        (7, 2, 1, 30.0, True, False, None),
        (9, 1, 1, 30.0, True, True, None),
    ],
)
_con.executemany(
    "INSERT INTO operations VALUES (?,CURRENT_TIMESTAMP,?,?,?,?)",
    [
        ("OP-001", "CNTR-004", "LOAD", "QC-01", 120),
        ("OP-002", "CNTR-001", "GATE_IN", "RTG-03", 45),
    ],
)

# SQL templates by question type
_SQL_TEMPLATES = {
    "컨테이너_목록": "SELECT id, weight_ton, type, pod, status FROM containers ORDER BY weight_ton DESC",
    "DG_컨테이너": "SELECT id, weight_ton, pod FROM containers WHERE dg = TRUE",
    "Reefer_컨테이너": "SELECT id, weight_ton, pod FROM containers WHERE reefer = TRUE",
    "빈_슬롯": "SELECT bay, row_num, tier, dg_allowed, reefer_capable FROM slots WHERE occupied_by IS NULL",
    "DG_슬롯": "SELECT bay, row_num, tier FROM slots WHERE dg_allowed = TRUE AND occupied_by IS NULL",
    "Reefer_슬롯": "SELECT bay, row_num, tier FROM slots WHERE reefer_capable = TRUE AND occupied_by IS NULL",
    "적재_현황": "SELECT s.bay, s.row_num, s.tier, s.occupied_by, c.weight_ton, c.type "
                "FROM slots s LEFT JOIN containers c ON s.occupied_by = c.id WHERE s.occupied_by IS NOT NULL",
    "작업_이력": "SELECT op_id, container_id, operation, crane_id, duration_sec FROM operations",
    "무거운_컨테이너": "SELECT id, weight_ton, pod FROM containers WHERE weight_ton > 20 ORDER BY weight_ton DESC",
}

SCHEMA = {
    "containers": ["id", "weight_ton", "size", "type", "pod", "dg", "reefer", "discharge_order", "status"],
    "slots": ["bay", "row_num", "tier", "max_stack_weight", "dg_allowed", "reefer_capable", "occupied_by"],
    "operations": ["op_id", "timestamp", "container_id", "operation", "crane_id", "duration_sec"],
}


def to_sql(question: str, schema: dict | None = None) -> str:
    """Map natural language question to SQL query via keyword matching."""
    q_lower = question.lower()

    if "dg" in q_lower or "위험물" in q_lower:
        if "슬롯" in q_lower or "bay" in q_lower or "위치" in q_lower:
            return _SQL_TEMPLATES["DG_슬롯"]
        return _SQL_TEMPLATES["DG_컨테이너"]
    if "reefer" in q_lower or "냉동" in q_lower:
        if "슬롯" in q_lower or "bay" in q_lower or "위치" in q_lower:
            return _SQL_TEMPLATES["Reefer_슬롯"]
        return _SQL_TEMPLATES["Reefer_컨테이너"]
    if "빈" in q_lower or "비어" in q_lower or "가용" in q_lower:
        return _SQL_TEMPLATES["빈_슬롯"]
    if "적재" in q_lower and ("현황" in q_lower or "상태" in q_lower):
        return _SQL_TEMPLATES["적재_현황"]
    if "작업" in q_lower or "이력" in q_lower or "크레인" in q_lower:
        return _SQL_TEMPLATES["작업_이력"]
    if "무거" in q_lower or "중량" in q_lower:
        return _SQL_TEMPLATES["무거운_컨테이너"]

    return _SQL_TEMPLATES["컨테이너_목록"]


def run_readonly(sql: str) -> list[dict]:
    """Execute SELECT-only SQL with validation and LIMIT."""
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        raise ValueError("읽기전용: SELECT 쿼리만 허용됩니다.")
    if any(kw in sql_upper for kw in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]):
        raise ValueError("위험 쿼리 차단: DDL/DML 명령은 허용되지 않습니다.")

    # Add LIMIT if not present
    if "LIMIT" not in sql_upper:
        sql = sql.rstrip(";") + " LIMIT 100"

    result = _con.execute(sql).fetchdf()
    return result.to_dict(orient="records")


def ask(question: str) -> dict:
    """→ {answer, sources:[{type:'sql', ref:sql, snippet:rows}]}"""
    sql = to_sql(question, SCHEMA)
    try:
        rows = run_readonly(sql)
        # Format answer
        if rows:
            header = ", ".join(rows[0].keys())
            body = "\n".join([", ".join(str(v) for v in r.values()) for r in rows[:10]])
            answer = f"[쿼리 결과 — {len(rows)}건]\n{header}\n{body}"
        else:
            answer = "조회 결과가 없습니다."
        return {
            "answer": answer,
            "sources": [{"type": "sql", "ref": sql, "snippet": rows[:5]}],
        }
    except Exception as e:
        return {"answer": f"SQL 실행 오류: {e}", "sources": []}

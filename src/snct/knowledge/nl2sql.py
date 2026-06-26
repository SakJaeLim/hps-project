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
    """Map natural language question to SQL query dynamically by parsing parameters."""
    import re
    q_lower = question.lower()
    
    # 1. 특정 컨테이너 ID 파싱
    cntr_match = re.search(r"CNTR-\d+", question, re.IGNORECASE)
    if cntr_match:
        cid = cntr_match.group(0).upper()
        return f"SELECT id, weight_ton, size, type, pod, status FROM containers WHERE id = '{cid}'"
        
    # 2. 목적항(POD) 파싱
    pods = {
        "LAX": ["lax", "로스앤젤레스", "로스"],
        "ROTTERDAM": ["rotterdam", "로테르담"],
        "SINGAPORE": ["singapore", "싱가포르", "싱가폴"],
        "BUSAN": ["busan", "부산"],
        "SHANGHAI": ["shanghai", "상하이", "상해"],
        "NINGBO": ["ningbo", "닝보"],
        "COLOMBO": ["colombo", "콜롬보"]
    }
    target_pod = None
    for pod_code, names in pods.items():
        if any(name in q_lower for name in names):
            target_pod = pod_code
            break

    # 3. 중량 기준값 파싱 (예: 15t, 24톤 등)
    weight_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:톤|t|ton)", q_lower)
    weight_val = float(weight_match.group(1)) if weight_match else None
    
    # DG(위험물) 컨테이너/슬롯 처리
    if "dg" in q_lower or "위험물" in q_lower:
        if "슬롯" in q_lower or "bay" in q_lower or "위치" in q_lower or "가용" in q_lower or "빈" in q_lower:
            return "SELECT bay, row_num, tier FROM slots WHERE dg_allowed = TRUE AND occupied_by IS NULL"
        
        where_clauses = ["dg = TRUE"]
        if target_pod:
            where_clauses.append(f"pod = '{target_pod}'")
        if weight_val:
            op = ">=" if any(w in q_lower for w in ["이상", "초과", "무거운", "초과하는"]) else "<="
            where_clauses.append(f"weight_ton {op} {weight_val}")
        return f"SELECT id, weight_ton, pod, status FROM containers WHERE {' AND '.join(where_clauses)} ORDER BY weight_ton DESC"

    # Reefer(냉동) 컨테이너/슬롯 처리
    if "reefer" in q_lower or "냉동" in q_lower:
        if "슬롯" in q_lower or "bay" in q_lower or "위치" in q_lower or "가용" in q_lower or "빈" in q_lower:
            return "SELECT bay, row_num, tier FROM slots WHERE reefer_capable = TRUE AND occupied_by IS NULL"
        
        where_clauses = ["reefer = TRUE"]
        if target_pod:
            where_clauses.append(f"pod = '{target_pod}'")
        if weight_val:
            op = ">=" if any(w in q_lower for w in ["이상", "초과", "무거운", "초과하는"]) else "<="
            where_clauses.append(f"weight_ton {op} {weight_val}")
        return f"SELECT id, weight_ton, pod, status FROM containers WHERE {' AND '.join(where_clauses)} ORDER BY weight_ton DESC"

    # 일반 가용(빈) 슬롯 처리
    if "빈" in q_lower or "비어" in q_lower or "가용" in q_lower:
        return "SELECT bay, row_num, tier, dg_allowed, reefer_capable FROM slots WHERE occupied_by IS NULL"

    # 현재 야드 적재 현황 처리
    if "적재" in q_lower and ("현황" in q_lower or "상태" in q_lower):
        return ("SELECT s.bay, s.row_num, s.tier, s.occupied_by, c.weight_ton, c.type "
                "FROM slots s JOIN containers c ON s.occupied_by = c.id")

    # 크레인 작업 이력 처리
    if "작업" in q_lower or "이력" in q_lower or "크레인" in q_lower:
        crane_match = re.search(r"qc-\d+|rtg-\d+", q_lower)
        if crane_match:
            crane_id = crane_match.group(0).upper()
            return f"SELECT op_id, container_id, operation, crane_id, duration_sec FROM operations WHERE crane_id = '{crane_id}'"
        return "SELECT op_id, container_id, operation, crane_id, duration_sec FROM operations"

    # 기본 컨테이너 목록 조회 (필터 포함)
    where_clauses = []
    if target_pod:
        where_clauses.append(f"pod = '{target_pod}'")
    if weight_val:
        op = ">=" if any(w in q_lower for w in ["이상", "초과", "무거운", "초과하는"]) else "<="
        where_clauses.append(f"weight_ton {op} {weight_val}")
        
    if where_clauses:
        return f"SELECT id, weight_ton, type, pod, status FROM containers WHERE {' AND '.join(where_clauses)} ORDER BY weight_ton DESC"
        
    return "SELECT id, weight_ton, type, pod, status FROM containers ORDER BY weight_ton DESC"


def _guard_and_limit(sql: str, limit: int = 100) -> str:
    """읽기전용 가드레일: SELECT만 허용 · DDL/DML 차단 · LIMIT 강제. (spec 07)"""
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        raise ValueError("읽기전용: SELECT 쿼리만 허용됩니다.")
    if any(kw in sql_upper for kw in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]):
        raise ValueError("위험 쿼리 차단: DDL/DML 명령은 허용되지 않습니다.")
    if "LIMIT" not in sql_upper:
        sql = sql.rstrip(";") + f" LIMIT {limit}"
    return sql


def run_readonly(sql: str) -> list[dict]:
    """Execute SELECT-only SQL with validation and LIMIT."""
    sql = _guard_and_limit(sql)
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


# ────────────────────────────────────────────────────────────────────
# T20 · RL 결과 RDB NL2SQL — sqi agent 패턴 이식 (CSV → DuckDB, 읽기전용)
# ────────────────────────────────────────────────────────────────────
class RLAnalyst:
    """강화학습 결과 RDB(kpi·reward_decomp·violation_log·slot_assignment)를
    DuckDB에 적재하고 자연어로 질의한다. 가드레일은 _guard_and_limit 공유. (spec 07)"""

    def __init__(self, store=None):
        import duckdb
        from snct.data.sources.rl_results import RLResultStore

        self.store = store or RLResultStore()
        self.con = duckdb.connect(":memory:")
        loaders = {
            "kpi": self.store.load_kpi,
            "reward_decomp": self.store.load_reward_decomp,
            "violation_log": self.store.load_violation_log,
            "slot_assignment": self.store.load_slot_assignment,
        }
        self._tables = []
        for name, fn in loaders.items():
            df = fn().copy()
            if "round_id" in df.columns:
                df["round_id"] = df["round_id"].astype("int64")
            self.con.register(f"_{name}_df", df)
            self.con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}_df")
            self._tables.append(name)

    def tables(self) -> list[str]:
        return list(self._tables)

    def query(self, sql: str) -> list[dict]:
        """읽기전용 가드레일 적용 후 실행."""
        sql = _guard_and_limit(sql)
        return self.con.execute(sql).fetchdf().to_dict(orient="records")

    @staticmethod
    def _policy_filter(question: str) -> str:
        for p in ("BL", "SF", "EF"):
            if p in question.upper():
                return f" WHERE policy = '{p}'"
        return ""

    def to_sql(self, question: str) -> str:
        """RL 운영 질의 → SQL 템플릿(읽기전용). 핵심 질의는 템플릿 우선(spec 07)."""
        import re
        q = question.lower()
        
        # 정책 파싱 (BL, SF, EF)
        policy_match = re.search(r"\b(BL|SF|EF)\b", question, re.IGNORECASE)
        p = policy_match.group(1).upper() if policy_match else None
        
        # 라운드 파싱 (예: 1라운드, round 2, R3 등)
        round_match = re.search(r"(\d+)\s*(?:라운드|round|R\b|회차)", question, re.IGNORECASE)
        r = int(round_match.group(1)) if round_match else None

        where_clauses = []
        if p:
            where_clauses.append(f"policy = '{p}'")
        if r:
            where_clauses.append(f"round_id = {r}")
            
        where_str = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        and_where_str = f" AND {' AND '.join(where_clauses)}" if where_clauses else ""

        if "위반" in q and any(k in question for k in ("많", "최대", "가장", "top")):
            return (f"SELECT policy, round_id, n_col_wt_viol, n_overstow "
                    f"FROM violation_log WHERE scope = 'SUMMARY'{and_where_str} "
                    f"ORDER BY n_col_wt_viol DESC, n_overstow DESC")
        if any(k in q for k in ("보상 기여", "기여", "reward_decomp", "분해")):
            return f"SELECT * FROM reward_decomp{where_str} ORDER BY round_id"
        if any(k in q for k in ("osr", "재취급", "wbi", "무게균형", "psr", "지표", "kpi", "보상", "reward")):
            return f"SELECT policy, round_id, reward, osr, wbi, psr, cwvr FROM kpi{where_str} ORDER BY round_id"
        if any(k in q for k in ("배정", "슬롯", "적재 위치", "slot")):
            return f"SELECT vessel, policy, round_id, row, tier, container_id, pod_name, weight_mt FROM slot_assignment{where_str} ORDER BY round_id, row, tier"
        return f"SELECT policy, round_id, reward, osr, wbi, psr, cwvr FROM kpi{where_str} ORDER BY round_id"

    def ask(self, question: str) -> dict:
        """→ {answer, sources:[{type:'sql', ref:sql, snippet:rows}]}."""
        sql = self.to_sql(question)
        try:
            rows = self.query(sql)
            if rows:
                header = ", ".join(str(k) for k in rows[0].keys())
                body = "\n".join(", ".join(str(v) for v in r.values()) for r in rows[:10])
                answer = f"[RL RDB 결과 — {len(rows)}건]\n{header}\n{body}"
            else:
                answer = "조회 결과가 없습니다."
            return {"answer": answer, "sources": [{"type": "sql", "ref": sql, "snippet": rows[:5]}]}
        except Exception as e:
            return {"answer": f"SQL 실행 오류: {e}", "sources": []}

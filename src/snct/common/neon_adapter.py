"""Neon PostgreSQL 클라우드 DB 연동 어댑터 (neon_adapter.py)

.env의 DATABASE_URL을 활용하여 Neon RDB 인프라와 직접 SQL 세션을 맺고
Pandas DataFrame 형태로 데이터를 쿼리해 옵니다. (실패 시 None 반환 및 CSV 폴백 유도)
"""
from __future__ import annotations
import os
import urllib.parse
from sqlalchemy import create_engine
import pandas as pd

def get_database_url() -> str | None:
    """환경변수에서 DATABASE_URL을 조회하여 반환"""
    # .env.example 또는 실제 .env 파일 등에서 로드
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return None
    return db_url

class NeonAdapter:
    def __init__(self, db_url: str | None = None):
        self.db_url = db_url or get_database_url()
        self._engine = None

    def _get_engine(self):
        """SQLAlchemy 엔진 캐싱 로드"""
        if self._engine is None:
            if not self.db_url:
                raise ValueError("DATABASE_URL 환경 변수가 선언되어 있지 않습니다.")
            # psycopg2를 연동하도록 dialect 설정 보완
            conn_url = self.db_url
            if conn_url.startswith("postgres://"):
                conn_url = conn_url.replace("postgres://", "postgresql://", 1)
            self._engine = create_engine(conn_url, connect_args={"sslmode": "require"})
        return self._engine

    def is_available(self) -> bool:
        """Neon DB 연결 가능 여부 테스트"""
        if not self.db_url:
            return False
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                # 간단한 연결 체크 쿼리 실행
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"[Neon DB Connection Fail] {e}")
            self._engine = None
            return False

    def query_table(self, table_name: str, policy: str | None = None, round_id: int | None = None) -> pd.DataFrame | None:
        """Neon DB 테이블로부터 조건부 데이터를 쿼리하여 DataFrame으로 반환"""
        if not self.is_available():
            return None
            
        try:
            engine = self._get_engine()
            # 테이블명 인젝션 방어용 정형 테이블 제한
            valid_tables = ["reward_decomp", "kpi", "slot_assignment", "violation_log", "xai_grounding"]
            if table_name not in valid_tables:
                raise ValueError(f"해당 테이블명은 지원하지 않습니다: {table_name}")
                
            query = f'SELECT * FROM "{table_name}"'
            params = {}
            conditions = []
            
            if policy:
                conditions.append('"policy" = :policy')
                params["policy"] = str(policy)
            if round_id is not None:
                conditions.append('"round_id" = :round_id')
                params["round_id"] = int(round_id)
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            df = pd.read_sql_query(query, con=engine, params=params)
            return df
        except Exception as e:
            print(f"[Neon Query Fail] Table '{table_name}' : {e}")
            return None

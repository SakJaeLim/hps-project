"""평가 하니스 — RL/Greedy vs 기준선(랜덤·단순룰) 비교. specs/04 · TASK T15.
지표: 재취급 수 · 제약 위반율 · 계획 산출 시간. 프로토콜: 고정 시드·홀드아웃·다회 반복."""
def evaluate(strategy_name: str = "rl", instances: int = 20, seed: int = 0) -> dict:
    """→ {rehandling, violation_rate, latency_s, baseline_delta}. TODO(W3)."""
    raise NotImplementedError

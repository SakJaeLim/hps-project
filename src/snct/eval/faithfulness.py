"""T22 · spec 04·07 — 설명 faithfulness 평가 하니스.

설명(explain_rl_decision)에 등장하는 모든 수치가 RL 결과의 사실값에 근거하는지 채점한다.
근거 없는 수치(환각)를 unsupported로 탐지 → faithfulness = 1 - unsupported/total.

규칙 기반 설명은 사실값만 인용하므로 1.0이어야 하며, 본 하니스는 회귀 가드로 동작한다.
(SLM(T23) 도입 시 동일 하니스로 환각을 정량 측정.)
"""
from __future__ import annotations
import re

from snct.common.schema import RLDecision
from snct.knowledge.explain import explain_rl_decision

# 정량 수치 토큰만 추출(절댓값). 식별자 내부 숫자(BL_R4_r1_t9, SOLAS_VI)는 제외 —
# 앞뒤가 단어문자/점이면 식별자로 보고 건너뛴다. 예: "+1052.0", "0.936", "145 MT".
_NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")

# 표현 반올림 허용(±기여 .1f → 0.05, kpi .3f → 0.0005). 넉넉히 0.05.
_TOL = 0.05


def extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUM_RE.findall(text.replace(",", ""))]


def source_values(decision: RLDecision, lpg=None) -> set[float]:
    """설명이 인용 가능한 사실 수치의 절댓값 집합.
    lpg 제공 시 위반 규정(Constraint) 인용 숫자(예: 145 MT)도 근거로 포함."""
    vals: set[float] = {abs(float(decision.reward_total)), float(decision.round_id)}
    for _term, v in decision.top_contributions:
        vals.add(abs(float(v)))
    for v in decision.kpi.values():
        if isinstance(v, (int, float)):
            vals.add(abs(float(v)))
    for row in decision.violations:
        for key in ("n_overstow", "n_col_wt_viol", "n_empty_rows", "row", "tier"):
            v = row.get(key)
            if isinstance(v, (int, float)):
                vals.add(abs(float(v)))
    if lpg is not None:
        try:
            for r in lpg.violations_in_round(decision.policy, decision.round_id):
                vals.update(abs(x) for x in extract_numbers(str(r.get("rule", ""))))
        except Exception:
            pass
    return vals


def _is_supported(num: float, allowed: set[float]) -> bool:
    # +1e-6: .1f 반올림 경계(정확히 0.05)에서 부동소수점 표현 오차 흡수.
    return any(abs(num - a) <= max(_TOL, abs(a) * 1e-4) + 1e-6 for a in allowed)


def score_decision(decision: RLDecision, text: str | None = None, lpg=None) -> dict:
    """설명의 수치 근거율을 채점. text 미지정 시 규칙 기반 설명을 생성해 평가.
    lpg 제공 시 LPG 규정 인용까지 포함한 설명을 생성·채점한다."""
    if text is None:
        text = "\n".join(explain_rl_decision(decision, lpg=lpg))
    allowed = source_values(decision, lpg=lpg)
    numbers = extract_numbers(text)
    unsupported = [n for n in numbers if not _is_supported(n, allowed)]
    n_total = len(numbers)
    faithfulness = 1.0 if n_total == 0 else 1.0 - len(unsupported) / n_total
    return {
        "policy": decision.policy,
        "round_id": decision.round_id,
        "n_numbers": n_total,
        "n_unsupported": len(unsupported),
        "unsupported": unsupported,
        "faithfulness": faithfulness,
        "doc_refs_cited": bool(decision.doc_refs) and all(ref in text for ref in decision.doc_refs),
    }


def evaluate(store=None) -> dict:
    """xai_grounding 전 레코드에 대해 설명을 생성·채점하고 집계한다."""
    if store is None:
        from snct.data.sources.rl_results import RLResultStore
        store = RLResultStore()

    reports, failures = [], []
    for rec in store.load_xai_grounding():
        d = store.get_decision(rec["policy"], int(rec["round_id"]))
        rep = score_decision(d)
        reports.append(rep)
        if rep["faithfulness"] < 1.0:
            failures.append(rep)

    n = len(reports)
    mean_f = sum(r["faithfulness"] for r in reports) / n if n else 0.0
    min_f = min((r["faithfulness"] for r in reports), default=0.0)
    return {
        "n": n,
        "mean_faithfulness": mean_f,
        "min_faithfulness": min_f,
        "failures": failures,
        "reports": reports,
    }

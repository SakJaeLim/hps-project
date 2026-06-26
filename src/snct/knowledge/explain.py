"""xAI 설명 합성 — [엔진 계획 + Cypher 제약 검증 사실 + 문서 근거 + 운영 수치]를 융합해
근거 인용 자연어 설명(rationale)을 생성. 모든 주장은 사실 소스에 근거(환각↓). specs/07."""
from snct.common.schema import CandidatePlan, Violation, RLDecision

# 운영지표 코드 → 표기·해석 (값은 decision.kpi의 사실에서만 채움)
_KPI_LABELS = {
    "osr": ("OSR(재취급률)", "낮을수록 양하 시 불필요한 재취급↓"),
    "wbi": ("WBI(무게균형지수)", "높을수록 행간 하중이 고르게 분산"),
    "psr": ("PSR(POD 순수도)", "높을수록 동일 목적항이 한 행에 그룹핑"),
    "cwvr": ("CWVR(컬럼중량 위반율)", "낮을수록 SOLAS 컬럼중량 제한 준수"),
}


def _fmt(v: float) -> str:
    """+12.0 / −9.0 형태로 부호 명시(환각 없는 사실 수치)."""
    s = f"{abs(v):.1f}"
    return f"+{s}" if v >= 0 else f"−{s}"


def explain_rl_decision(decision: RLDecision, lpg=None) -> list[str]:
    """RL (policy, round_id) 의사결정의 '왜'를 사실 근거로 융합. 새 수치 생성 없음.
    귀인(reward_decomp 상위 ±기여) + 운영지표(kpi) + 규정(doc_refs) + 근거(rationale).
    lpg(선택, LPGGraph) 제공 시 위반 컨테이너별 규정 근거(T19)를 자동 인용."""
    from snct.data.sources.rl_results import REWARD_LABELS

    parts = [f"\n## RL 적재 의사결정 설명 — 정책 {decision.policy} / 라운드 {decision.round_id}"]
    parts.append(f"- 보상 총합(reward_total): {decision.reward_total:.3f}")

    # 1) 귀인: 왜 이 점수인가 (상위 기여항 ±)
    if decision.top_contributions:
        parts.append("\n### 주요 기여 요인 (reward 분해 = 귀인)")
        for term, val in decision.top_contributions[:6]:
            label = REWARD_LABELS.get(term, term)
            direction = "계획 품질을 높임" if val >= 0 else "감점 요인"
            parts.append(f"  • {label}: {_fmt(val)} — {direction}")

    # 2) 운영지표(kpi) 인용
    metrics = [(k, decision.kpi[k]) for k in _KPI_LABELS if k in decision.kpi and decision.kpi[k] is not None]
    if metrics:
        parts.append("\n### 운영지표 (사실 근거)")
        for k, v in metrics:
            label, interp = _KPI_LABELS[k]
            parts.append(f"  • {label} = {v:.3f} — {interp}")

    # 3) 규정 근거(doc_refs)
    if decision.doc_refs:
        parts.append("\n### 규정 근거")
        parts.append("  📚 " + ", ".join(decision.doc_refs))

    # 4) 위반 사실(violation_log) — 라운드 집계(SUMMARY)만. 컨테이너별 상세는 4b LPG 섹션.
    summary_viol = [
        v for v in decision.violations
        if str(v.get("scope", "")).upper() == "SUMMARY"
        and ((v.get("n_overstow") or 0) or (v.get("n_col_wt_viol") or 0))
    ]
    if summary_viol:
        parts.append("\n### 검증 위반(사실·집계)")
        for v in summary_viol:
            parts.append(
                f"  ❌ overstow={v.get('n_overstow')}, col_wt_viol={v.get('n_col_wt_viol')}"
            )

    # 4b) 위반 컨테이너 상세(LPG) — 어느 컨테이너가 어떤 규정을 위반했나
    if lpg is not None:
        try:
            rows = lpg.violations_in_round(decision.policy, decision.round_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows = []
        if rows:
            parts.append("\n### 위반 컨테이너 상세 (LPG 그래프 근거)")
            for r in rows:
                cid = r.get("container_id") or r.get("slot_id")
                parts.append(f"  ❌ {cid} → {r['code']}({r['source']}): {r['rule']}")

    # 5) 사전 융합 근거(rationale) — RL이 산출한 자연어 근거 그대로 인용
    if decision.rationale:
        parts.append("\n### 근거 요약")
        for r in decision.rationale:
            parts.append(f"  - {r}")

    return parts


def explain(
    plan: CandidatePlan,
    violations: list[Violation],
    evidence: list[dict],
    decision: RLDecision | None = None,
    lpg=None,
) -> str:
    """Synthesize an explainable rationale from plan, violations, and evidence.
    evidence = router가 모은 [{type:doc|sql|graph, ref, snippet}]. → rationale 문자열.
    decision(선택) = RL 의사결정 근거(reward_decomp·kpi·doc_refs 융합, specs/07 xAI-RL).
    lpg(선택) = LPGGraph. decision과 함께 주면 컨테이너별 위반 규정을 인용."""

    parts = []

    # 0. RL 의사결정 설명(있으면 최상단) — 사실 귀인 우선
    if decision is not None:
        parts.extend(explain_rl_decision(decision, lpg=lpg))
        # live plan 배정이 없으면 RL 설명만으로 자기완결 — 제네릭 계획/결론 섹션(빈 plan 기준)은
        # 모순(예: decision 위반 8건 vs "위반 없음")을 만들므로 생략한다.
        if not plan.assignments:
            return "\n".join(parts)
        parts.append("")

    # 1. Plan summary
    n_assigned = len(plan.assignments)
    engine = plan.engine or "unknown"
    parts.append(f"## 적재 계획 요약 (엔진: {engine})")
    parts.append(f"- 배정 완료: {n_assigned}건")
    if plan.objective:
        for key, val in plan.objective.items():
            parts.append(f"- {key}: {val}")

    # 2. Assignment details
    if plan.assignments:
        parts.append("\n### 슬롯 배정 상세")
        for a in plan.assignments[:10]:  # Limit display
            parts.append(f"  • {a.container_id} → BAY{a.bay:02d}-ROW{a.row:02d}-TIER{a.tier:02d}")

    # 3. Constraint validation results
    if violations:
        errors = [v for v in violations if v.severity == "error"]
        warnings = [v for v in violations if v.severity == "warning"]

        parts.append(f"\n### 제약 검증 결과")
        parts.append(f"- ❌ 위반(Error): {len(errors)}건")
        parts.append(f"- ⚠️ 경고(Warning): {len(warnings)}건")

        for v in errors:
            parts.append(f"  ❌ [{v.rule}] {v.container_id}: {v.detail}")
        for v in warnings[:5]:
            parts.append(f"  ⚠️ [{v.rule}] {v.container_id}: {v.detail}")
    else:
        parts.append("\n### 제약 검증 결과")
        parts.append("- ✅ 모든 제약 조건 충족 — 위반 없음")

    # 4. Evidence citations
    if evidence:
        parts.append("\n### 근거 인용")
        seen_refs = set()
        for ev in evidence:
            ref = ev.get("ref", "")
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            ev_type = ev.get("type", "unknown")
            snippet = ev.get("snippet", "")
            if isinstance(snippet, str) and snippet:
                parts.append(f"  📎 [{ev_type.upper()}:{ref}] {snippet[:150]}")
            elif isinstance(snippet, list):
                parts.append(f"  📎 [{ev_type.upper()}:{ref}] {len(snippet)}건 조회")

    # 5. Conclusion
    parts.append("\n### 결론")
    if violations:
        error_count = sum(1 for v in violations if v.severity == "error")
        if error_count > 0:
            parts.append(f"⛔ {error_count}건의 제약 위반이 발견되었습니다. 적재 계획 수정이 필요합니다.")
        else:
            parts.append("⚠️ 경고 사항이 있으나 적재 가능합니다. 운영자 검토를 권장합니다.")
    else:
        parts.append("✅ 모든 제약 조건을 충족하는 최적 적재 계획이 수립되었습니다.")

    return "\n".join(parts)

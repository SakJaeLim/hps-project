"""xAI 설명 합성 — [엔진 계획 + Cypher 제약 검증 사실 + 문서 근거 + 운영 수치]를 융합해
근거 인용 자연어 설명(rationale)을 생성. 모든 주장은 사실 소스에 근거(환각↓). specs/07."""
from snct.common.schema import CandidatePlan, Violation


def explain(plan: CandidatePlan, violations: list[Violation], evidence: list[dict]) -> str:
    """Synthesize an explainable rationale from plan, violations, and evidence.
    evidence = router가 모은 [{type:doc|sql|graph, ref, snippet}]. → rationale 문자열."""

    parts = []

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

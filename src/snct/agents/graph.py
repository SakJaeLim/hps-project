"""L4 멀티에이전트 파이프라인: 인식→계획→검증→설명 (위반 시 계획으로 재시도).
LangGraph 없이 순수 Python으로 동일한 4노드 파이프라인 구현."""
from dataclasses import dataclass, field
from snct.common.schema import YardState, CandidatePlan, Violation, Recommendation
from snct.data.provider import get_provider
from snct.engine.base import get_strategy
from snct.ontology.graph import Ontology
from snct.knowledge.orchestrator import answer as knowledge_answer  # router 상위호환(Corrective-RAG+faithfulness)
from snct.knowledge.explain import explain


@dataclass
class PipelineState:
    """Shared state flowing through the pipeline."""
    question: str = ""
    vessel_id: str = "VESSEL-001"
    yard_state: YardState | None = None
    plan: CandidatePlan | None = None
    violations: list[Violation] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    rationale: str = ""
    engine_name: str = "greedy"
    deterministic: bool = True   # RL 추론: True=argmax(최적), False=표집(매 실행 다른 대안)
    retry_count: int = 0
    max_retries: int = 2


def recognize(state: PipelineState) -> PipelineState:
    """Node 1: Recognize — 질문 유형 인식 및 야드 상태 로드."""
    # Load yard state from data provider
    provider = get_provider("simulated")
    state.yard_state = provider.get_yard_state(state.vessel_id)

    # Collect knowledge evidence for the question
    try:
        knowledge_result = knowledge_answer(state.question)
        state.evidence = knowledge_result.get("sources", [])
    except Exception:
        state.evidence = []

    return state


def plan_step(state: PipelineState) -> PipelineState:
    """Node 2: Plan — 적재 엔진 호출하여 배정 계획 생성."""
    if state.yard_state is None:
        return state

    strategy = get_strategy(state.engine_name, deterministic=state.deterministic)
    state.plan = strategy.plan(state.yard_state)
    return state


def validate(state: PipelineState) -> PipelineState:
    """Node 3: Validate — 온톨로지 기반 제약 검증."""
    if state.yard_state is None or state.plan is None:
        return state

    ontology = Ontology()
    state.violations = ontology.validate(state.yard_state, state.plan)
    return state


def explain_step(state: PipelineState) -> PipelineState:
    """Node 4: Explain — 근거 인용 설명 생성."""
    if state.plan is None:
        state.rationale = "적재 계획을 생성할 수 없습니다."
        return state

    state.rationale = explain(state.plan, state.violations, state.evidence)
    return state


def run_pipeline(
    question: str,
    vessel_id: str = "VESSEL-001",
    engine: str = "greedy",
    deterministic: bool = True,
) -> Recommendation:
    """Execute the full 4-node pipeline with retry on constraint violations.

    Flow: Recognize → Plan → Validate → (if errors → retry Plan) → Explain
    deterministic: RL 추론 방식 (True=argmax 최적/재현성, False=표집/매 실행 다른 대안).
    """
    state = PipelineState(
        question=question,
        vessel_id=vessel_id,
        engine_name=engine,
        deterministic=deterministic,
    )

    # Node 1: Recognize
    state = recognize(state)

    # Node 2 + 3: Plan → Validate (with retry loop)
    for attempt in range(state.max_retries + 1):
        state.retry_count = attempt

        # Node 2: Plan
        state = plan_step(state)

        # Node 3: Validate
        state = validate(state)

        # Check for hard errors
        errors = [v for v in state.violations if v.severity == "error"]
        if not errors:
            break  # No hard violations → proceed to explain

        # If errors and we have retries left, try alternative engine
        if attempt < state.max_retries:
            if state.engine_name == "rl":
                state.engine_name = "greedy"  # fallback to greedy
            # On retry, the same engine might produce different results
            # (in practice, greedy is deterministic, but RL might vary)

    # Node 4: Explain
    state = explain_step(state)

    return Recommendation(
        plan=state.plan or CandidatePlan(),
        violations=state.violations,
        rationale=state.rationale,
        checks=[f"engine={state.engine_name}", f"retries={state.retry_count}"],
    )


def build_graph():
    """Build the pipeline graph (API compatibility).
    Returns the run_pipeline function as the executable graph."""
    return run_pipeline


# ────────────────────────────────────────────────────────────────────
# 설명가능 RL 흐름 (spec 07 xAI-RL): 질의 → 근거수집(RDB·LPG) → 설명 → 자기검증
# ────────────────────────────────────────────────────────────────────
import re

_POLICY_RE = re.compile(r"\b(BL|SF|EF)\b", re.IGNORECASE)
_ROUND_RE = re.compile(r"(\d+)\s*(?:라운드|round|R\b|회차)", re.IGNORECASE)


def parse_decision_ref(question: str):
    """자연어 질의에서 (policy, round_id)를 추출. 실패 시 None. → RLDecisionRef."""
    from snct.common.schema import RLDecisionRef

    pm = _POLICY_RE.search(question or "")
    rm = _ROUND_RE.search(question or "")
    if not pm or not rm:
        return None
    return RLDecisionRef(policy=pm.group(1).upper(), round_id=int(rm.group(1)))


def run_explanation(
    question: str | None = None,
    policy: str | None = None,
    round_id: int | None = None,
    with_lpg: bool = True,
) -> Recommendation:
    """RL 의사결정 설명 흐름. policy/round_id 직접 지정 또는 question에서 파싱.
    근거 수집(RDB·LPG) → explain 융합 → faithfulness 자기검증 → Recommendation."""
    # 1) 인식: 의사결정 참조 결정
    if policy is None or round_id is None:
        ref = parse_decision_ref(question or "")
        if ref is None:
            return Recommendation(
                plan=CandidatePlan(engine="rl"),
                rationale="질의에서 정책(BL/SF/EF)과 라운드를 인식하지 못했습니다.",
                checks=["parse=failed"],
            )
        policy, round_id = ref.policy, ref.round_id

    # 2) 근거 수집 + 설명
    from snct.data.sources.rl_results import RLResultStore
    from snct.knowledge.explain import explain

    store = RLResultStore()
    try:
        decision = store.get_decision(policy, round_id)
    except KeyError:
        return Recommendation(
            plan=CandidatePlan(engine="rl"),
            rationale=f"해당 의사결정을 찾을 수 없습니다 (policy={policy}, round={round_id}).",
            checks=[f"policy={policy}", f"round={round_id}", "found=false"],
        )

    lpg = None
    if with_lpg:
        try:
            from snct.knowledge.lpg import get_lpg
            lpg = get_lpg()  # Neo4j 가용 시 그래프DB, 아니면 CSV 폴백
        except Exception as e:
            import traceback
            traceback.print_exc()
            lpg = None

    rationale = explain(CandidatePlan(engine="rl"), [], evidence=[], decision=decision, lpg=lpg)

    # 3) 자기검증: 설명의 수치 근거율(환각 가드)
    checks = [f"policy={policy}", f"round={round_id}"]
    try:
        from snct.eval.faithfulness import score_decision
        rep = score_decision(decision, text=rationale, lpg=lpg)
        checks.append(f"faithfulness={rep['faithfulness']:.1f}")
    except Exception:
        pass

    return Recommendation(plan=CandidatePlan(engine="rl"), rationale=rationale, checks=checks)

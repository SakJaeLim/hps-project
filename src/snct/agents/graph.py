"""L4 멀티에이전트 파이프라인: 인식→계획→검증→설명 (위반 시 계획으로 재시도).
LangGraph 없이 순수 Python으로 동일한 4노드 파이프라인 구현."""
from dataclasses import dataclass, field
from snct.common.schema import YardState, CandidatePlan, Violation, Recommendation
from snct.data.provider import get_provider
from snct.engine.base import get_strategy
from snct.ontology.graph import Ontology
from snct.knowledge.orchestrator import answer as knowledge_answer
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

    strategy = get_strategy(state.engine_name)
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
) -> Recommendation:
    """Execute the full 4-node pipeline with retry on constraint violations.

    Flow: Recognize → Plan → Validate → (if errors → retry Plan) → Explain
    """
    state = PipelineState(
        question=question,
        vessel_id=vessel_id,
        engine_name=engine,
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

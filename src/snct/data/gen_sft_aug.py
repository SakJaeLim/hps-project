"""결정형 SFT 데이터 증강 생성기 (v1 레시피 확장: A 스케일업 + B 근거 심화).

v1은 RL slot_assignment.csv(337행)에서 결정 데이터를 만들었는데, 그 CSV는 이미
전량 소진됐고 weight도 10~19.76t로 좁다(24.5t 같은 고중량 미포함). 이 생성기는:

  A) 합성 시나리오(다양한 무게·POD·DG/Reefer·고중량 포함) YardState 생성 →
     엔진(get_strategy)으로 **실제 배정**을 받아 ground-truth 슬롯 확보(환각 0).
  B) 그 배정을 DG/Reefer·POD·Heavy-Down·컬럼중량(SOLAS)·rehandling 까지 반영한
     **다양한 표현의 근거**로 설명 → 고정 4문장 템플릿의 과적합을 완화.

산출은 v1과 동일한 {type,instruction,input,output,meta} 스키마의
recommend_with_reason 예시. gen_sft.py 의 v1 데이터와 합쳐 ~1000건을 만든다.
"""
import os
import json
import argparse
import random

from snct.common.schema import Container, Slot, YardState
from snct.engine.base import get_strategy

POD_ORDER = {"Busan": 1, "Shanghai": 2, "Ningbo": 3, "Singapore": 4, "Colombo": 5, "Rotterdam": 6}
PODS = list(POD_ORDER.keys())
SIZES = ["20", "40", "45"]
INSTRUCTION = "다음 컨테이너의 적재 슬롯을 제약 조건에 맞게 추천하고, 근거를 본선 플래닝 SOP 조항으로 설명하라."


def _pick(rng, variants):
    return variants[rng.randrange(len(variants))]


def make_scenario(rng, idx, id_prefix="AUG"):
    """다양한 단일 베이 YardState 생성 (행6~10 × tier4~6, DG/Reefer 슬롯 포함)."""
    n_rows = rng.randint(6, 10)
    n_tiers = rng.randint(4, 6)
    dg_rows = {0, 1}            # 하부 두 행은 DG 허용
    reefer_rows = {2, 3}       # 다음 두 행은 Reefer 가능
    slots = []
    for r in range(n_rows):
        for t in range(n_tiers):
            slots.append(Slot(
                bay=1, row=r, tier=t,
                max_stack_weight=145.0,  # SOLAS 컬럼 한계 (per-slot 상한으로도 충분)
                dg_allowed=(r in dg_rows),
                reefer_capable=(r in reefer_rows),
                size_class="40",
            ))

    n_ctn = rng.randint(6, min(12, n_rows * n_tiers))
    queue = []
    for i in range(n_ctn):
        pod = _pick(rng, PODS)
        # 무게: 8~26t (고중량 >20t 포함 — 24.5t 데모 케이스 커버)
        weight = round(rng.uniform(8.0, 26.0), 1)
        is_dg = rng.random() < 0.15
        is_rf = (not is_dg) and rng.random() < 0.15
        ctype = "DG" if is_dg else ("RF" if is_rf else "GP")
        queue.append(Container(
            id=f"{id_prefix}{idx:04d}-C{i:02d}",
            weight_ton=weight,
            size=_pick(rng, SIZES),
            type=ctype,
            pod=pod,
            dg=is_dg,
            reefer=is_rf,
            discharge_order=POD_ORDER[pod],
        ))
    return YardState(slots=slots, queue=queue)


def build_rationale(rng, c, slot, col_cum_weight):
    """배정된 (c→slot) 을 실제 속성 기반으로 설명. 표현 다양화 + 제약 심화(B)."""
    reasons = []

    # 1) 특수화물(DG/Reefer) — 지정위치 제약
    if c.dg:
        reasons.append(_pick(rng, [
            f"DG(위험물) → dg_allowed 슬롯 필수, IMDG 격리·접근성 요건을 충족하는 위치.",
            f"위험물(DG)이므로 IMDG Code상 허용 구역(dg_allowed)에만 적재 가능 — 해당 슬롯이 요건 충족.",
        ]))
    elif c.reefer:
        reasons.append(_pick(rng, [
            f"Reefer → 전원 공급 가능(reefer_capable) 슬롯에 배정하여 냉동 유지.",
            f"냉동 컨테이너이므로 전원 접속이 되는 reefer_capable 슬롯이 필수 — 해당 위치 적합.",
        ]))
    else:
        reasons.append(_pick(rng, [
            "특수화물 아님 → 지정위치 제약 무관.",
            "일반화물(GP)로 DG/Reefer 지정 제약이 없어 배치 자유도 높음.",
        ]))

    # 2) POD 그룹핑 / 양하순서 — 방향 중립 서술(tier 방향은 무게 근거가 전담)로 모순 차단.
    far = POD_ORDER[c.pod] >= 4
    if far:
        reasons.append(_pick(rng, [
            f"POD 그룹핑 — {c.pod}은 원거리(후순위) 양하항으로, 동일 POD 묶음과 양하 역순(overstow) 방지를 고려한 배치.",
            f"{c.pod}은 후순위 양하항 → 동일 POD를 묶어 후순위 반출 시 간섭을 최소화.",
        ]))
    else:
        reasons.append(_pick(rng, [
            f"POD 그룹핑 — {c.pod}은 근거리(선순위) 양하항으로, 우선 반출과 동일 POD 묶음을 고려한 배치.",
            f"{c.pod}은 선순위 양하항 → 동일 POD를 묶어 우선 양하 동선을 단순화.",
        ]))

    # 3) 중량(Heavy-Down / Light-Up) — 고중량 강조
    if c.weight_ton >= 20.0:
        reasons.append(_pick(rng, [
            f"중량 {c.weight_ton:.1f}t(고중량) → Heavy-Down 원칙상 최하부 Tier 배치로 복원성(GM) 확보.",
            f"{c.weight_ton:.1f}t의 고중량이라 무게중심을 낮추기 위해 하부 적재가 필수적.",
        ]))
    elif c.weight_ton >= 17.0:
        reasons.append(_pick(rng, [
            f"중량 {c.weight_ton:.1f}t → Heavy-Down 원칙상 하부 Tier 배치가 적합, 복원성에 기여.",
            f"{c.weight_ton:.1f}t로 무거운 편 → 하단 배치가 무게 역전을 방지.",
        ]))
    else:
        reasons.append(_pick(rng, [
            f"중량 {c.weight_ton:.1f}t → Light-Up 원칙상 상대적 상부 Tier 배치 가능.",
            f"{c.weight_ton:.1f}t로 가벼운 편 → 상단에 두어 하부를 고중량용으로 확보.",
        ]))

    # 4) 컬럼 중량(SOLAS) 제약
    reasons.append(_pick(rng, [
        f"해당 컬럼 누적중량 {col_cum_weight:.1f}t로 SOLAS 한계(145t) 이내라 적재 안전.",
        f"행 누적 {col_cum_weight:.1f}t < 145t(SOLAS 컬럼 한계) → 중량 제약 위반 없음.",
    ]))

    # 5) Tier/재취급
    if slot.tier == 0:
        reasons.append("최하단 Tier로 상단 간섭이 없어 재취급(rehandling) 위험 없음.")
    else:
        reasons.append(_pick(rng, [
            "상단 간섭이 없어 재취급(rehandling) 위험 low.",
            "상부 적재 컨테이너와의 간섭이 없어 반출 시 재취급 불필요.",
        ]))

    return reasons


def gen_examples(n_target, engine="greedy", seed=42, id_prefix="AUG"):
    rng = random.Random(seed)
    out = []
    scen_idx = 0
    strat = get_strategy(engine)  # greedy = 어디서나 동작, rl_* 는 서버(체크포인트 필요)
    while len(out) < n_target:
        yard = make_scenario(rng, scen_idx, id_prefix=id_prefix)
        scen_idx += 1
        try:
            plan = strat.plan(yard)
        except Exception:
            continue
        cmap = {c.id: c for c in yard.queue}
        amap = {a.container_id: a for a in plan.assignments}
        # 컬럼(행) 누적중량 계산용: row → [(tier, weight)]
        col = {}
        for a in plan.assignments:
            col.setdefault(a.row, []).append((a.tier, cmap[a.container_id].weight_ton))

        for a in plan.assignments:
            c = cmap[a.container_id]
            slot = Slot(bay=a.bay, row=a.row, tier=a.tier, max_stack_weight=145.0)
            # 같은 행에서 이 컨테이너 tier 이하의 누적중량 (자기 포함)
            col_cum = sum(w for (t, w) in col.get(a.row, []) if t <= a.tier)

            correct_slot = f"BAY{a.bay:02d}-ROW{a.row:02d}-TIER{a.tier:02d}"
            # 오답 후보 합성 (정답이 하부면 상부, 반대도)
            wt = 8 if a.tier <= 2 else 1
            wr = (a.row + 2) % 10
            wrong_slot = f"BAY{a.bay:02d}-ROW{wr:02d}-TIER{wt:02d}"
            cands = [correct_slot, wrong_slot]
            rng.shuffle(cands)

            reasons = build_rationale(rng, c, slot, col_cum)
            output = f"추천 슬롯: {correct_slot}. 근거: " + " ".join(
                f"({i+1}) {r}" for i, r in enumerate(reasons))
            inp = (f"container_id={c.id}, size_type={c.size}{c.type if c.type!='GP' else 'GP'}, "
                   f"weight={c.weight_ton:.1f}t, POD={c.pod}, "
                   f"reefer={str(c.reefer).lower()}, dg={str(c.dg).lower()}, oog=false, "
                   f"candidate_slots={cands}")
            out.append({
                "type": "recommend_with_reason",
                "instruction": INSTRUCTION,
                "input": inp,
                "output": output,
                "meta": {
                    "source": "aug_engine",
                    "engine": engine,
                    "difficulty": "hard" if c.weight_ton >= 20 else ("medium" if c.weight_ton >= 17 else "easy"),
                    "constraints": (["dg"] if c.dg else []) + (["reefer"] if c.reefer else [])
                                   + (["heavy_down"] if c.weight_ton >= 17 else ["light_up"])
                                   + ["pod_grouping", "solas_col_weight", "rehandling"],
                },
            })
            if len(out) >= n_target:
                break
    return out


def main():
    p = argparse.ArgumentParser(description="결정형 SFT 증강 생성 (엔진 ground-truth + 심화 근거)")
    p.add_argument("--n", type=int, default=660, help="생성할 결정형 예시 수 (기본 660 → v1 337과 합쳐 ~1000)")
    p.add_argument("--engine", type=str, default="greedy", help="greedy | rl_bl | rl_sf | rl_ef")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None, help="출력 jsonl 경로 (기본: data/aug/decisions.jsonl)")
    args = p.parse_args()

    repo = os.environ.get("SNCT_BASE_DIR") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    out = args.out or os.path.join(repo, "data", "aug", "decisions.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"Generating {args.n} decision examples via engine={args.engine} ...")
    rows = gen_examples(args.n, engine=args.engine, seed=args.seed)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 간단 통계
    import collections
    diff = collections.Counter(r["meta"]["difficulty"] for r in rows)
    heavy = sum(1 for r in rows if "weight=2" in r["input"])
    print(f"Saved {len(rows)} → {out}")
    print(f"  difficulty: {dict(diff)} | heavy(>=20t)~: {heavy}")


if __name__ == "__main__":
    main()

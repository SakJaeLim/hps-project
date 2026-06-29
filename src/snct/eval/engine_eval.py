"""RL 적재 엔진 정량 평가 하니스 (greedy vs rl_bl/sf/ef).

SLM 평가(eval_metrics.py)와 별개 트랙. 엔진이 만든 '적재 계획의 운영 품질'을
KPI로 채점한다 — compute_plan_kpi(엔진 무관) 재사용:
  · 배정률, Heavy-Down 준수율, POD 역전 위반률, Row 무게편차, DG 제약위반
정량 비교 기준: RL 이 greedy(휴리스틱 하한선)보다 KPI 우위인지 + 하드제약 위반 0.

산출: data/simulated/engine_eval.json (평가 대시보드가 읽음) + 콘솔 표.
"""
import os
import csv
import glob
import json
import argparse
import statistics

from snct.data.provider import get_provider
from snct.engine.base import get_strategy
from snct.engine.metrics import compute_plan_kpi

ENGINE_TO_POLICY = {"rl_bl": "BL", "rl_sf": "SF", "rl_ef": "EF"}


def _load_train_kpi():
    """RL 학습 시점 KPI(kpi.csv)를 정책별로 로드 — 라이브 재평가가 못 드러내는
    BL/SF/EF 실제 성능차(reward·WBI·OSR·PSR·CWVR)를 표시하기 위함. round 최댓값(=가장 복잡) 사용."""
    base = os.environ.get("SNCT_BASE_DIR") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    hits = glob.glob(os.path.join(base, "data", "RL", "*", "*RDB_LPG*", "rdb", "kpi.csv"))
    if not hits:
        return {}
    rows = list(csv.DictReader(open(hits[0], encoding="utf-8-sig")))
    by_pol = {}
    for r in rows:
        pol = r.get("policy")
        try:
            rnd = int(float(r.get("round_id", 0)))
        except Exception:
            rnd = 0
        prev = by_pol.get(pol)
        if prev is None or rnd >= prev["_round"]:
            by_pol[pol] = {
                "_round": rnd,
                "reward": float(r["reward"]) if r.get("reward") else None,
                "wbi": float(r["wbi"]) if r.get("wbi") else None,
                "osr": float(r["osr"]) if r.get("osr") else None,
                "psr": float(r["psr"]) if r.get("psr") else None,
                "cwvr": float(r["cwvr"]) if r.get("cwvr") else None,
            }
    for v in by_pol.values():
        v.pop("_round", None)
    return by_pol

ENGINES = ["greedy", "rl_bl", "rl_sf", "rl_ef"]
VESSELS = ["VESSEL-LV1", "VESSEL-LV2", "VESSEL-LV3", "VESSEL-LV4"]

# 대시보드 표시용 KPI (키, 라벨, 높을수록 좋음)
KPIS = [
    ("assign_rate", "배정률(%)", True),
    ("heavy_down_rate", "Heavy-Down 준수율(%)", True),
    ("pod_violation_rate", "POD 역전 위반률(%)", False),
    ("row_weight_std", "Row 무게편차(t)", False),
    ("dg_violations", "DG 제약위반(건)", False),
]


def run_engine_eval(engines, vessels, out_json=None):
    prov = get_provider("simulated")
    train_kpi = _load_train_kpi()  # 정책별 학습 KPI (BL/SF/EF 실제 차이)
    detail = {}       # engine -> {vessel -> kpi}
    by_engine = {}    # engine -> {metric mean ..., loaded}

    for eng in engines:
        strat = get_strategy(eng)
        loaded = getattr(strat, "model", None) is not None  # RL: 체크포인트 실제 로드 여부
        per = {}
        for v in vessels:
            try:
                yard = prov.get_yard_state(v)
                plan = strat.plan(yard)
                per[v] = compute_plan_kpi(yard, plan)
            except Exception as e:
                print(f"  [{eng}/{v}] ERROR: {e}")
                per[v] = None
        detail[eng] = per

        agg = {}
        valid = [k for k in per.values() if k]
        for key, _label, _hb in KPIS:
            vals = [k[key] for k in valid if key in k]
            agg[key] = round(statistics.mean(vals), 2) if vals else None
        agg["loaded"] = loaded
        agg["is_rl"] = eng.startswith("rl")
        # RL 정책별 학습 KPI 부착 (라이브가 못 드러내는 실제 성능차)
        agg["train_kpi"] = train_kpi.get(ENGINE_TO_POLICY.get(eng))
        by_engine[eng] = agg
        tag = "" if not eng.startswith("rl") else (" [PPO loaded]" if loaded else " [greedy fallback!]")
        print(f"  {eng:8s}{tag}: " + " | ".join(
            f"{lab}={agg[key]}" for key, lab, _ in KPIS))

    summary = {"engines": engines, "vessels": vessels,
               "kpis": [{"key": k, "label": l, "higher_better": hb} for k, l, hb in KPIS],
               "by_engine": by_engine, "detail": detail}

    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved → {out_json}")

    # 합격 판정: RL 이 greedy 대비 핵심 KPI(POD역전·무게편차) 우위인지
    g = by_engine.get("greedy", {})
    print("\n=== 판정 (RL vs greedy, 낮을수록 좋은 KPI) ===")
    for eng in engines:
        if not eng.startswith("rl"):
            continue
        e = by_engine[eng]
        if not e.get("loaded"):
            print(f"  {eng}: ⚠️ PPO 미로드(greedy 폴백) — 판정 불가")
            continue
        better_pod = (e["pod_violation_rate"] or 0) <= (g.get("pod_violation_rate") or 0)
        better_bal = (e["row_weight_std"] or 0) <= (g.get("row_weight_std") or 0)
        verdict = "✅ 우위" if (better_pod and better_bal) else "🟡 일부/열위"
        print(f"  {eng}: POD역전 {e['pod_violation_rate']} vs {g.get('pod_violation_rate')}, "
              f"무게편차 {e['row_weight_std']} vs {g.get('row_weight_std')} → {verdict}")

    # RL 정책별 학습 KPI (라이브가 못 드러내는 실제 차이)
    if train_kpi:
        print("\n=== RL 학습 KPI (정책별 실제 성능차 — reward↑·WBI↑·OSR↓·CWVR↓) ===")
        for pol, k in train_kpi.items():
            print(f"  {pol}: reward={k['reward']} WBI={k['wbi']} OSR={k['osr']} PSR={k['psr']} CWVR={k['cwvr']}")
    return summary


if __name__ == "__main__":
    base_dir = os.environ.get("SNCT_BASE_DIR") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    p = argparse.ArgumentParser(description="RL 적재 엔진 정량 평가 (greedy vs rl_bl/sf/ef)")
    p.add_argument("--engines", nargs="+", default=ENGINES)
    p.add_argument("--vessels", nargs="+", default=VESSELS)
    p.add_argument("--out-json", type=str, default=os.path.join(base_dir, "data", "simulated", "engine_eval.json"))
    args = p.parse_args()
    print("=== RL 엔진 평가 (engine × vessel KPI) ===")
    run_engine_eval(args.engines, args.vessels, out_json=args.out_json)

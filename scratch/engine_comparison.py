"""
적재 계획 엔진 비교 분석 스크립트
greedy / rl_bl / rl_sf / rl_ef  ×  Level 1~4 결과 비교
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict

# ── 한글 폰트 ──────────────────────────────────────────────
for fp in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
    if "malgun" in fp.lower() or "gulim" in fp.lower():
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# ── 엔진 / 레벨 설정 ────────────────────────────────────────
ENGINES   = ["greedy", "rl_bl", "rl_sf", "rl_ef"]
LEVELS    = {
    "LV1 (4R×4T)":  "VESSEL-LV1",
    "LV2 (6R×6T)":  "VESSEL-LV2",
    "LV3 (8R×8T)":  "VESSEL-LV3",
    "LV4 (10R×10T)":"VESSEL-LV4",
}

from snct.data.provider import get_provider
from snct.engine.base    import get_strategy

# ── KPI 계산 ────────────────────────────────────────────────
def compute_kpi(yard, plan):
    """배정 결과로 KPI 계산"""
    n_containers = len(yard.queue)
    n_assigned   = len(plan.assignments)
    cntr_map     = {c.id: c for c in yard.queue}

    # 1. 배정률
    assign_rate  = n_assigned / max(n_containers, 1) * 100

    # 2. Heavy-Down 준수율: 같은 Row에서 아래 Tier에 더 무거운 컨테이너가 있어야 함
    row_tiers = defaultdict(list)
    for a in plan.assignments:
        c = cntr_map.get(a.container_id)
        wt = c.weight_ton if c else 0
        row_tiers[a.row].append((a.tier, wt))

    hd_ok, hd_total = 0, 0
    for row, pairs in row_tiers.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        for i in range(len(pairs_sorted) - 1):
            hd_total += 1
            if pairs_sorted[i][1] >= pairs_sorted[i+1][1]:
                hd_ok += 1
    heavy_down_rate = (hd_ok / hd_total * 100) if hd_total > 0 else 100.0

    # 3. POD 위반률 (같은 Row 내 POD 순서가 역전된 쌍)
    POD_ORDER = {"ROTTERDAM":1,"COLOMBO":2,"SINGAPORE":3,"NINGBO":4,"SHANGHAI":5,"BUSAN":6}
    pod_viol, pod_total = 0, 0
    for row, pairs in row_tiers.items():
        cntr_by_tier = sorted(
            [(a.tier, cntr_map.get(a.container_id)) for a in plan.assignments if a.row == row],
            key=lambda x: x[0]
        )
        for i in range(len(cntr_by_tier)-1):
            if cntr_by_tier[i][1] and cntr_by_tier[i+1][1]:
                pod_total += 1
                p1 = POD_ORDER.get(cntr_by_tier[i][1].pod.upper(), 0)
                p2 = POD_ORDER.get(cntr_by_tier[i+1][1].pod.upper(), 0)
                if p1 < p2:  # 근항(높은번호)이 아래 있어야 올바름
                    pod_viol += 1
    pod_violation_rate = (pod_viol / pod_total * 100) if pod_total > 0 else 0.0

    # 4. 무게 중심 불균형 (Row별 총무게 표준편차)
    row_weights = defaultdict(float)
    for a in plan.assignments:
        c = cntr_map.get(a.container_id)
        if c:
            row_weights[a.row] += c.weight_ton
    wt_vals = list(row_weights.values()) if row_weights else [0]
    cog_std  = float(np.std(wt_vals))

    # 5. 평균 적재 무게
    total_wt = sum(c.weight_ton for c in yard.queue if c.id in {a.container_id for a in plan.assignments})
    avg_wt   = total_wt / max(n_assigned, 1)

    # 6. DG/Reefer 제약 위반 수
    dg_viol = sum(
        1 for a in plan.assignments
        for s in yard.slots
        if s.row == a.row and s.tier == a.tier and s.bay == a.bay
        and cntr_map.get(a.container_id, None)
        and cntr_map[a.container_id].dg and not s.dg_allowed
    )

    return {
        "배정률(%)":           round(assign_rate, 1),
        "Heavy-Down 준수율(%)": round(heavy_down_rate, 1),
        "POD 역전 위반률(%)":   round(pod_violation_rate, 1),
        "Row 무게편차(t)":      round(cog_std, 2),
        "평균 배정무게(t)":     round(avg_wt, 2),
        "DG 제약위반 수":       dg_viol,
        "배정 컨테이너":        n_assigned,
        "미배정 컨테이너":      n_containers - n_assigned,
        "총 컨테이너":          n_containers,
    }


# ── 실험 실행 ────────────────────────────────────────────────
print("=" * 70)
print("  적재 계획 엔진 비교 분석  (greedy / rl_bl / rl_sf / rl_ef)")
print("=" * 70)

results    = {}
row_detail = defaultdict(dict)   # {engine: {level: kpi}}

provider = get_provider("simulated")

for lv_name, vessel_id in LEVELS.items():
    yard = provider.get_yard_state(vessel_id)
    print(f"\n▶ {lv_name}  —  Slots: {len(yard.slots)}, Containers: {len(yard.queue)}")
    for engine_name in ENGINES:
        try:
            strategy = get_strategy(engine_name)
            plan     = strategy.plan(yard)
            kpi      = compute_kpi(yard, plan)
            row_detail[engine_name][lv_name] = kpi
            print(f"   [{engine_name:8s}] 배정률={kpi['배정률(%)']:5.1f}%  "
                  f"HeavyDown={kpi['Heavy-Down 준수율(%)']:5.1f}%  "
                  f"POD위반={kpi['POD 역전 위반률(%)']:4.1f}%  "
                  f"Row편차={kpi['Row 무게편차(t)']:5.2f}t")
        except Exception as e:
            print(f"   [{engine_name:8s}] ERROR: {e}")
            row_detail[engine_name][lv_name] = {}


# ── 결과 테이블 구성 ──────────────────────────────────────────
kpi_keys = ["배정률(%)", "Heavy-Down 준수율(%)", "POD 역전 위반률(%)",
            "Row 무게편차(t)", "평균 배정무게(t)", "DG 제약위반 수",
            "배정 컨테이너", "미배정 컨테이너"]

records = []
for engine in ENGINES:
    for lv_name in LEVELS:
        kpi = row_detail[engine].get(lv_name, {})
        rec = {"엔진": engine, "레벨": lv_name}
        for k in kpi_keys:
            rec[k] = kpi.get(k, None)
        records.append(rec)

df_result = pd.DataFrame(records)
print("\n\n" + "=" * 70)
print("  종합 KPI 테이블")
print("=" * 70)
print(df_result.to_string(index=False))


# ── 시각화 ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("엔진별 적재 계획 KPI 비교 (Greedy vs RL-BL vs RL-SF vs RL-EF)",
             fontsize=15, fontweight="bold", y=1.01)

colors = {"greedy": "#636EFA", "rl_bl": "#EF553B", "rl_sf": "#00CC96", "rl_ef": "#AB63FA"}
x = np.arange(len(LEVELS))
lv_labels = list(LEVELS.keys())
width = 0.2

KPI_PLOTS = [
    ("배정률(%)",               axes[0, 0], True,  "배정률 (높을수록 좋음)"),
    ("Heavy-Down 준수율(%)",     axes[0, 1], True,  "Heavy-Down 준수율 (높을수록 좋음)"),
    ("POD 역전 위반률(%)",       axes[1, 0], False, "POD 역전 위반률 (낮을수록 좋음)"),
    ("Row 무게편차(t)",          axes[1, 1], False, "Row 무게편차 (낮을수록 좋음)"),
]

for (kpi_key, ax, higher_better, title) in KPI_PLOTS:
    for i, engine in enumerate(ENGINES):
        vals = [row_detail[engine].get(lv, {}).get(kpi_key, 0) for lv in lv_labels]
        bars = ax.bar(x + i * width - 1.5 * width, vals, width,
                      label=engine, color=colors[engine], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(lv_labels, rotation=10, fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.35)
    arrow_label = "↑ 좋음" if higher_better else "↓ 좋음"
    ax.set_ylabel(f"{kpi_key}  {arrow_label}", fontsize=9)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "..", "img", "engine_comparison_kpi.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n[저장] KPI 비교 차트 → {out_path}")


# ── 배정 분포 히트맵 (LV4 기준 Row별 무게 분포) ───────────────
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
fig2.suptitle("LV4 (10R×10T) — 엔진별 Row별 무게 분포 히트맵",
              fontsize=13, fontweight="bold")

lv4_yard = provider.get_yard_state("VESSEL-LV4")
lv4_slots = {(s.bay, s.row, s.tier): s for s in lv4_yard.slots}
cntr_map  = {c.id: c for c in lv4_yard.queue}
MAX_ROWS, N_TIERS = 10, 10

for ax, engine_name in zip(axes2, ENGINES):
    wt_grid = np.zeros((MAX_ROWS, N_TIERS))
    try:
        strategy = get_strategy(engine_name)
        plan     = strategy.plan(lv4_yard)
        for a in plan.assignments:
            c = cntr_map.get(a.container_id)
            if c and 1 <= a.row <= MAX_ROWS and 1 <= a.tier <= N_TIERS:
                wt_grid[a.row - 1, a.tier - 1] = c.weight_ton
    except Exception as e:
        pass

    im = ax.imshow(wt_grid, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=25, origin="lower")
    ax.set_title(f"[{engine_name}]\n총배정={int(wt_grid.astype(bool).sum())}개",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Tier (T0→T9)", fontsize=8)
    ax.set_ylabel("Row (R0→R9)", fontsize=8)
    ax.set_xticks(range(N_TIERS)); ax.set_xticklabels([f"T{i}" for i in range(N_TIERS)], fontsize=7)
    ax.set_yticks(range(MAX_ROWS)); ax.set_yticklabels([f"R{i}" for i in range(MAX_ROWS)], fontsize=7)
    plt.colorbar(im, ax=ax, label="무게(t)", fraction=0.046, pad=0.04)

plt.tight_layout()
out2_path = os.path.join(os.path.dirname(__file__), "..", "img", "engine_comparison_heatmap.png")
plt.savefig(out2_path, dpi=150, bbox_inches="tight")
print(f"[저장] 히트맵 비교 차트  → {out2_path}")

# ── CSV 저장 ────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "..", "img", "engine_comparison_kpi.csv")
df_result.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"[저장] KPI CSV           → {csv_path}")
print("\n분석 완료.")

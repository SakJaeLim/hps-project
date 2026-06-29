"""개선된 SFT 데이터셋 조립기 (v2 학습용) — 결정형 2배 + 독립 골든셋.

구성:
  · 결정형(recommend_with_reason): v1 CSV 실배정(337) + 엔진 증강(660) ≈ 997  ← 목표 2배
  · QA/진단/절차: v1 seed + 증강 (커버리지 유지)
  · 독립 골든셋(평가): 학습엔 안 쓴 **다른 시드(777) 시나리오** 결정 10 +
    학습에서 제외한 **held-out seed** QA 10 + 진단 10  → 누설 없는 정직한 평가

산출: data/sft_v2/{train,val,eval_golden}.jsonl  (v1 data/simulated 는 보존)
주의: VESSL 학습 연결은 하지 않음(데이터셋만). 검증 후 별도로 job 수정.
"""
import os
import json
import random
import argparse
import collections

from snct.data import gen_sft as g
from snct.data.gen_sft_aug import gen_examples

SEED = 42


def _split_holdout(seeds, holdout_per_type, rng):
    """타입별로 holdout_per_type 개를 골든셋용으로 분리(학습/증강에서 제외) → 누설 방지."""
    by = collections.defaultdict(list)
    for s in seeds:
        by[s["type"]].append(s)
    held, kept = [], []
    for t, items in by.items():
        rng.shuffle(items)
        k = holdout_per_type.get(t, 0)
        held += items[:k]
        kept += items[k:]
    return held, kept


def main(out_dir, n_aug):
    rng = random.Random(SEED)
    repo = os.environ.get("SNCT_BASE_DIR") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    out_dir = out_dir or os.path.join(repo, "data", "sft_v2")
    os.makedirs(out_dir, exist_ok=True)

    # 1) seed 로드 + 골든셋용 held-out 분리 (학습/증강에서 제외)
    stow = g.load_jsonl(g.STOWAGE_SEED_PATH)
    saf = g.load_jsonl(g.SAFETY_SEED_PATH)
    stow_held, stow_keep = _split_holdout(stow, {"regulation_qa": 5, "violation_diagnosis": 5}, rng)
    saf_held, saf_keep = _split_holdout(saf, {"safety_regulation_qa": 5, "hazard_diagnosis": 5}, rng)
    print(f"Seeds: stowage {len(stow)} (held {len(stow_held)}), safety {len(saf)} (held {len(saf_held)})")

    # 2) 결정형: v1 CSV 실배정(전량) + 엔진 증강
    recs_v1 = g.generate_recommendations_from_csv(g.SLOT_CSV_PATH)
    recs_aug = gen_examples(n_aug, engine="greedy", seed=SEED, id_prefix="AUG")
    print(f"Decisions: v1_csv {len(recs_v1)} + aug {len(recs_aug)} = {len(recs_v1)+len(recs_aug)}")

    # 3) QA/진단: held-out 제외한 seed만 증강 (누설 차단) + 그 raw seed
    augmented = g.augment_seeds_by_paraphrasing(stow_keep, saf_keep)

    # 4) 학습 풀 조립 + 9:1 train/val
    train_pool = recs_v1 + recs_aug + augmented + stow_keep + saf_keep
    rng.shuffle(train_pool)
    n_val = int(len(train_pool) * 0.1)
    val_set = train_pool[:n_val]
    train_set = train_pool[n_val:]

    # 5) 독립 골든셋: 다른 시드(777) 결정 10 + held-out QA/진단
    gold_dec = gen_examples(10, engine="greedy", seed=777, id_prefix="GOLD")
    golden = gold_dec + stow_held + saf_held
    rng.shuffle(golden)

    def save(data, name):
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        dist = dict(collections.Counter(d["type"] for d in data))
        print(f"Saved {name}: {len(data)}  {dist}")

    save(train_set, "train.jsonl")
    save(val_set, "val.jsonl")
    save(golden, "eval_golden.jsonl")

    # 누설 점검: 골든셋 input 이 학습셋에 존재하는지
    train_inputs = {d["input"] for d in train_set + val_set}
    leak = [d for d in golden if d["input"] in train_inputs]
    print(f"Leakage check — golden inputs also in train: {len(leak)} (0이어야 정상)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="개선 SFT 데이터셋 조립 (결정형 2배 + 독립 골든셋)")
    p.add_argument("--out-dir", type=str, default=None, help="기본: data/sft_v2")
    p.add_argument("--n-aug", type=int, default=660, help="엔진 증강 결정형 수 (기본 660)")
    args = p.parse_args()
    main(args.out_dir, args.n_aug)

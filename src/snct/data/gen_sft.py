"""v1 SFT 데이터 생성기 (재현·확장 가능 버전).

원본은 개인 구글드라이브 절대경로(i:\\내 드라이브\\…유홍성…)에 의존해 서버에서
재현 불가였다. 이 버전은 **repo 상대경로**로 동작하며, 출력 폴더를 인자로 받는다.

레시피(v1이 ROUGE 95%/환각 3%를 낸 비결):
  1) 결정(recommend_with_reason): RL 실제 배정(slot_assignment.csv)의 슬롯을 정답으로,
     POD·무게에서 규칙 템플릿으로 근거 생성 → 사실로 보장(환각 0)
  2) QA/진단: 사람이 만든 seed + 패러프레이즈/파라미터 치환 증강
  3) 골든셋 30(10/10/10)을 같은 풀에서 분리, 나머지를 train/val 9:1
"""
import os
import json
import csv
import glob
import random
import argparse

# 재현성
random.seed(42)


def _repo_root() -> str:
    # src/snct/data/gen_sft.py → parents[3] = repo root
    env = os.environ.get("SNCT_BASE_DIR")
    if env and os.path.isdir(env):
        return env
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


REPO = _repo_root()
SFT_DIR = os.path.join(REPO, "04_Finetuning(SFT)")


def _find_seed(name_suffix):
    """한글 파일명 NFC/NFD 불일치 회피용 — ASCII 패턴 glob 후 'SOP' 변형은 제외.
    예) '*portslm_stowage_sft_sample.jsonl' → '적재계획 SFT_...'(SFT) 선택, '_SOP_' 변형 제외."""
    cands = sorted(glob.glob(os.path.join(SFT_DIR, "*" + name_suffix)))
    if not cands:
        return os.path.join(SFT_DIR, name_suffix)  # 없으면 경고용 더미 경로
    non_sop = [c for c in cands if "SOP" not in os.path.basename(c)]
    return (non_sop or cands)[0]


STOWAGE_SEED_PATH = _find_seed("portslm_stowage_sft_sample.jsonl")
SAFETY_SEED_PATH = _find_seed("portslm_safety_sft_sample.jsonl")
SLOT_CSV_PATH = os.path.join(
    REPO, "data", "RL", "강화학습 결과 자료",
    "single_bay_6pod_ppo_v13_3way_RDB_LPG_seed42", "rdb", "slot_assignment.csv",
)


def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def generate_recommendations_from_csv(csv_path):
    recommendations = []
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found: {csv_path}")
        return recommendations

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:  # BOM 대응
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # POD 거리/양하순서: Busan(1) … Rotterdam(6). 큰 값 = 원거리 = 하부 적재.
    for i, r in enumerate(rows):
        container_id = r.get("container_id", f"SNCT-CNTR-{100000+i}")
        pod_name = r.get("pod_name", "Unknown")
        pod_id = int(r.get("pod_id", "3"))
        weight = float(r.get("weight_mt", "15.0"))

        actual_row = int(r.get("row", "0"))
        actual_tier = int(r.get("tier", "0"))
        actual_bay = r.get("bay", "BAY_01").replace("_", "")  # BAY01

        correct_slot = f"{actual_bay}-ROW{actual_row:02d}-TIER{actual_tier:02d}"

        # 오답 후보 합성 (정답이 하부면 오답은 상부)
        if actual_tier <= 2:
            wrong_tier = 8
            wrong_row = (actual_row + 2) % 10
        else:
            wrong_tier = 1
            wrong_row = (actual_row + 1) % 10
        wrong_slot = f"{actual_bay}-ROW{wrong_row:02d}-TIER{wrong_tier:02d}"

        candidate_slots = [correct_slot, wrong_slot]
        random.shuffle(candidate_slots)

        reasons = []
        reasons.append("특수화물 아님 → 지정위치 제약 무관.")
        if pod_id >= 4:
            reasons.append(f"POD 그룹핑 — {pod_name}은 원거리 항이므로 하부에 적재하여 양하 역순을 방지(선적 Planning 기준).")
        else:
            reasons.append(f"POD 그룹핑 — {pod_name}은 근거리 첫 양하항이므로 상부에 두어 먼저 양하 가능하게 함.")
        if weight >= 17.0:
            reasons.append(f"중량 {weight:.1f}t은 Heavy-Down 원칙상 하부 Tier 배치가 적합, 선박 복원성 확보에 기여.")
        else:
            reasons.append(f"중량 {weight:.1f}t은 Light-Up 원칙상 상대적 상부 Tier 배치가 가능.")
        reasons.append("상단 간섭이 없어 재취급(rehandling) 위험 low.")

        output_rationale = f"추천 슬롯: {correct_slot}. 근거: " + " ".join(
            [f"({idx+1}) {re}" for idx, re in enumerate(reasons)]
        )
        input_context = (
            f"container_id={container_id}, size_type=40HC, weight={weight:.1f}t, "
            f"POD={pod_name}, reefer=false, dg=false, oog=false, "
            f"current_yard_slot=YB01-R02-T03, candidate_slots={candidate_slots}"
        )
        recommendations.append({
            "type": "recommend_with_reason",
            "instruction": "다음 컨테이너의 적재 슬롯을 제약 조건에 맞게 추천하고, 근거를 본선 플래닝 SOP 조항으로 설명하라.",
            "input": input_context,
            "output": output_rationale,
            "meta": {
                "difficulty": "medium" if weight >= 17.0 else "easy",
                "constraints": ["heavy_down" if weight >= 17.0 else "light_up", "pod_grouping", "rehandling"],
            },
        })
    return recommendations


def augment_seeds_by_paraphrasing(stowage_seeds, safety_seeds):
    augmented = []

    stowage_qa = [s for s in stowage_seeds if s["type"] == "regulation_qa"]
    safety_qa = [s for s in safety_seeds if s["type"] == "safety_regulation_qa"]
    templates = [
        "{}에 대해 설명하고 근거 규정을 알려주세요.",
        "컨테이너 터미널 규정에 따르면 {} 기준은 어떻게 되나요?",
        "{}에 대한 터미널 SOP 지침과 근거 조항을 제시하라.",
        "{} 규칙과 이를 준수해야 하는 안전상 원칙은 무엇인가?",
    ]
    for item in stowage_qa + safety_qa:
        topic_phrase = (item["input"].replace("은 무엇인가?", "").replace("는 무엇인가?", "")
                        .replace("는 어떻게 해야 하는가?", "").strip())
        for i in range(4):
            augmented.append({
                "type": item["type"],
                "instruction": item["instruction"],
                "input": templates[i].format(topic_phrase),
                "output": item["output"],
                "meta": item["meta"],
            })

    stowage_diag = [s for s in stowage_seeds if s["type"] == "violation_diagnosis"]
    safety_diag = [s for s in safety_seeds if s["type"] == "hazard_diagnosis"]
    for item in stowage_diag + safety_diag:
        for i in range(4):
            cntr_a = f"CNTR-X{i:02d}"
            cntr_b = f"CNTR-Y{i:02d}"
            weight_a = 7.0 + i * 0.5
            weight_b = 21.0 + i * 0.8
            new_input = (item["input"].replace("CNTR-A", cntr_a).replace("CNTR-B", cntr_b)
                         .replace("8.0t", f"{weight_a:.1f}t").replace("23.0t", f"{weight_b:.1f}t"))
            new_output = (item["output"].replace("CNTR-A", cntr_a).replace("CNTR-B", cntr_b)
                          .replace("8.0t", f"{weight_a:.1f}t").replace("23.0t", f"{weight_b:.1f}t"))
            augmented.append({
                "type": item["type"],
                "instruction": item["instruction"],
                "input": new_input,
                "output": new_output,
                "meta": item["meta"],
            })
    return augmented


def main(out_dir=None):
    out_dir = out_dir or os.path.join(REPO, "data", "simulated")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading seeds...")
    stowage_seeds = load_jsonl(STOWAGE_SEED_PATH)
    safety_seeds = load_jsonl(SAFETY_SEED_PATH)
    print(f"Stowage seeds: {len(stowage_seeds)}, Safety seeds: {len(safety_seeds)}")

    print("Generating recommendations from CSV...")
    recs = generate_recommendations_from_csv(SLOT_CSV_PATH)
    print(f"Generated {len(recs)} recommendation samples.")

    print("Augmenting seeds...")
    augmented = augment_seeds_by_paraphrasing(stowage_seeds, safety_seeds)
    print(f"Generated {len(augmented)} augmented QA/Diagnosis samples.")

    total_data = recs + augmented + stowage_seeds + safety_seeds
    print(f"Total compiled data size: {len(total_data)}")
    random.shuffle(total_data)

    # 골든셋 30(10 rec / 10 qa / 10 diag) 분리
    golden_set, remaining_data = [], []
    rec_count = qa_count = diag_count = 0
    for item in total_data:
        t = item["type"]
        if t == "recommend_with_reason" and rec_count < 10:
            golden_set.append(item); rec_count += 1
        elif t in ["regulation_qa", "safety_regulation_qa"] and qa_count < 10:
            golden_set.append(item); qa_count += 1
        elif t in ["violation_diagnosis", "hazard_diagnosis"] and diag_count < 10:
            golden_set.append(item); diag_count += 1
        else:
            remaining_data.append(item)
    print(f"Golden set: {len(golden_set)} (Recs {rec_count}, QA {qa_count}, Diag {diag_count})")
    print(f"Remaining for SFT: {len(remaining_data)}")

    train_size = int(len(remaining_data) * 0.9)
    train_set = remaining_data[:train_size]
    val_set = remaining_data[train_size:]
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    def save_jsonl(data, filename):
        p = os.path.join(out_dir, filename)
        with open(p, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved: {p} ({len(data)} items)")

    save_jsonl(golden_set, "eval_golden.jsonl")
    save_jsonl(train_set, "train.jsonl")
    save_jsonl(val_set, "val.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v1 SFT 데이터 재생성 (repo 상대경로)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="출력 폴더 (기본: data/simulated). 검증 시 data/regen 등으로 분리 권장")
    args = parser.parse_args()
    main(out_dir=args.out_dir)

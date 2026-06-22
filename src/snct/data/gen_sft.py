import os
import json
import csv
import random

# Set random seed for reproducibility
random.seed(42)

# Resolve paths dynamically to support both local workspace and Google Drive fallback
cwd = os.getcwd()
STOWAGE_SEED_PATH = None
SAFETY_SEED_PATH = None
SLOT_CSV_PATH = None
OUT_DIR = None

# If running in local workspace, find files under "유홍성"
if os.path.exists(os.path.join(cwd, "유홍성")):
    def find_local_path(filename):
        for root, dirs, files in os.walk(os.path.join(cwd, "유홍성")):
            for f in files:
                if f.lower() == filename.lower():
                    return os.path.join(root, f)
        return None
    STOWAGE_SEED_PATH = find_local_path("portslm_stowage_sft_sample.jsonl")
    SAFETY_SEED_PATH = find_local_path("portslm_safety_sft_sample.jsonl")
    SLOT_CSV_PATH = find_local_path("slot_assignment.csv")
    OUT_DIR = os.path.join(cwd, "data", "simulated")

# Fallback to Google Drive if local search failed
if not STOWAGE_SEED_PATH or not os.path.exists(STOWAGE_SEED_PATH):
    BASE_DIR = r"i:\내 드라이브\01. AI 프로젝트(석제)\[aSSIST] AI project\01. HPS 프로젝트"
    STOWAGE_SEED_PATH = os.path.join(BASE_DIR, "유홍성", "자료 수집", "SFT 데이터 수집_SLM 파인튜닝", "적재계획 SFT", "portslm_stowage_sft_sample.jsonl")
    SAFETY_SEED_PATH = os.path.join(BASE_DIR, "유홍성", "자료 수집", "SFT 데이터 수집_SLM 파인튜닝", "Safety SFT", "portslm_safety_sft_sample.jsonl")
    SLOT_CSV_PATH = os.path.join(BASE_DIR, "유홍성", "자료 수집", "강화학습 결과 자료", "single_bay_6pod_ppo_v13_3way_RDB_LPG_seed42", "rdb", "slot_assignment.csv")
    OUT_DIR = r"i:\내 드라이브\01. AI 프로젝트(석제)\[aSSIST] AI project\01. HPS 프로젝트\임석제\snct-decision-platform\data\simulated"

os.makedirs(OUT_DIR, exist_ok=True)

def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return data
    with open(path, 'r', encoding='utf-8') as f:
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
        
    # Read rows
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    # Map POD distances (6 = Rotterdam is furthest, 1 = Busan is closest)
    # Distance/order mapping: Busan (1), Shanghai (2), Ningbo (3), Singapore (4), Colombo (5), Rotterdam (6)
    # Larger number = further = must be stacked bottom.
    
    for i, r in enumerate(rows):
        container_id = r.get("container_id", f"CNTR-{100000+i}")
        pod_name = r.get("pod_name", "Unknown")
        pod_id = int(r.get("pod_id", "3"))
        weight = float(r.get("weight_mt", "15.0"))
        
        actual_row = int(r.get("row", "0"))
        actual_tier = int(r.get("tier", "0"))
        actual_bay = r.get("bay", "BAY_01").replace("_", "") # e.g. BAY01
        
        # Format slots
        correct_slot = f"{actual_bay}-ROW{actual_row:02d}-TIER{actual_tier:02d}"
        
        # Generate an incorrect candidate slot
        # If correct is bottom, make incorrect top.
        if actual_tier <= 2:
            wrong_tier = 8
            wrong_row = (actual_row + 2) % 10
        else:
            wrong_tier = 1
            wrong_row = (actual_row + 1) % 10
        wrong_slot = f"{actual_bay}-ROW{wrong_row:02d}-TIER{wrong_tier:02d}"
        
        candidate_slots = [correct_slot, wrong_slot]
        random.shuffle(candidate_slots)
        
        # Rationale logic based on the constraints
        reasons = []
        # 1. Special cargo check
        reasons.append("특수화물 아님 → 지정위치 제약 무관.")
        
        # 2. POD Grouping & Discharge order
        if pod_id >= 4: # Rotterdam (6), Colombo (5), Singapore (4) - Furthest
            reasons.append(f"POD 그룹핑 — {pod_name}은 원거리 항이므로 하부에 적재하여 양하 역순을 방지(선적 Planning 기준).")
        else: # Ningbo (3), Shanghai (2), Busan (1) - Closest
            reasons.append(f"POD 그룹핑 — {pod_name}은 근거리 첫 양하항이므로 상부에 두어 먼저 양하 가능하게 함.")
            
        # 3. Weight planning
        if weight >= 17.0:
            reasons.append(f"중량 {weight:.1f}t은 Heavy-Down 원칙상 하부 Tier 배치가 적합, 선박 복원성 확보에 기여.")
        else:
            reasons.append(f"중량 {weight:.1f}t은 Light-Up 원칙상 상대적 상부 Tier 배치가 가능.")
            
        # 4. Rehandling
        reasons.append("상단 간섭이 없어 재취급(rehandling) 위험 low.")
        
        # Assemble correct rationale
        output_rationale = f"추천 슬롯: {correct_slot}. 근거: " + " ".join([f"({idx+1}) {re}" for idx, re in enumerate(reasons)])
        
        input_context = f"container_id={container_id}, size_type=40HC, weight={weight:.1f}t, POD={pod_name}, reefer=false, dg=false, oog=false, current_yard_slot=YB01-R02-T03, candidate_slots={candidate_slots}"
        
        rec = {
            "type": "recommend_with_reason",
            "instruction": "다음 컨테이너의 적재 슬롯을 제약 조건에 맞게 추천하고, 근거를 본선 플래닝 SOP 조항으로 설명하라.",
            "input": input_context,
            "output": output_rationale,
            "meta": {
                "difficulty": "medium" if weight >= 17.0 else "easy",
                "constraints": ["heavy_down" if weight >= 17.0 else "light_up", "pod_grouping", "rehandling"]
            }
        }
        recommendations.append(rec)
        
    return recommendations

def augment_seeds_by_paraphrasing(stowage_seeds, safety_seeds):
    augmented = []
    
    # 1. Regulation QA Paraphrasing
    stowage_qa = [s for s in stowage_seeds if s["type"] == "regulation_qa"]
    safety_qa = [s for s in safety_seeds if s["type"] == "safety_regulation_qa"]
    
    # Paraphrase templates
    templates = [
        "{}에 대해 설명하고 근거 규정을 알려주세요.",
        "컨테이너 터미널 규정에 따르면 {} 기준은 어떻게 되나요?",
        "{}에 대한 터미널 SOP 지침과 근거 조항을 제시하라.",
        "{} 규칙과 이를 준수해야 하는 안전상 원칙은 무엇인가?"
    ]
    
    for item in stowage_qa + safety_qa:
        topic_phrase = item["input"].replace("은 무엇인가?", "").replace("는 무엇인가?", "").replace("는 어떻게 해야 하는가?", "").strip()
        for i in range(4):
            new_input = templates[i].format(topic_phrase)
            augmented.append({
                "type": item["type"],
                "instruction": item["instruction"],
                "input": new_input,
                "output": item["output"],
                "meta": item["meta"]
            })
            
    # 2. Diagnosis Parameter Substitution
    stowage_diag = [s for s in stowage_seeds if s["type"] == "violation_diagnosis"]
    safety_diag = [s for s in safety_seeds if s["type"] == "hazard_diagnosis"]
    
    for item in stowage_diag + safety_diag:
        # Generate 4 substitutions
        for i in range(4):
            # Substitute container IDs, weights, slots
            old_input = item["input"]
            old_output = item["output"]
            
            cntr_a = f"CNTR-X{i:02d}"
            cntr_b = f"CNTR-Y{i:02d}"
            weight_a = 7.0 + i * 0.5
            weight_b = 21.0 + i * 0.8
            
            new_input = old_input.replace("CNTR-A", cntr_a).replace("CNTR-B", cntr_b)
            new_input = new_input.replace("8.0t", f"{weight_a:.1f}t").replace("23.0t", f"{weight_b:.1f}t")
            
            new_output = old_output.replace("CNTR-A", cntr_a).replace("CNTR-B", cntr_b)
            new_output = new_output.replace("8.0t", f"{weight_a:.1f}t").replace("23.0t", f"{weight_b:.1f}t")
            
            augmented.append({
                "type": item["type"],
                "instruction": item["instruction"],
                "input": new_input,
                "output": new_output,
                "meta": item["meta"]
            })
            
    return augmented

def main():
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
    
    # Combine everything
    total_data = recs + augmented + stowage_seeds + safety_seeds
    print(f"Total compiled data size: {len(total_data)}")
    
    # Shuffle
    random.shuffle(total_data)
    
    # Extract evaluation golden set (30 items: 10 recs, 10 qa, 10 diag)
    golden_set = []
    remaining_data = []
    
    rec_count = 0
    qa_count = 0
    diag_count = 0
    
    for item in total_data:
        t = item["type"]
        if t == "recommend_with_reason" and rec_count < 10:
            golden_set.append(item)
            rec_count += 1
        elif t in ["regulation_qa", "safety_regulation_qa"] and qa_count < 10:
            golden_set.append(item)
            qa_count += 1
        elif t in ["violation_diagnosis", "hazard_diagnosis"] and diag_count < 10:
            golden_set.append(item)
            diag_count += 1
        else:
            remaining_data.append(item)
            
    print(f"Golden set size: {len(golden_set)} (Recs: {rec_count}, QA: {qa_count}, Diag: {diag_count})")
    print(f"Remaining data size for SFT: {len(remaining_data)}")
    
    # Split remaining 90% train, 10% val
    train_size = int(len(remaining_data) * 0.9)
    train_set = remaining_data[:train_size]
    val_set = remaining_data[train_size:]
    
    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")
    
    # Save files
    def save_jsonl(data, filename):
        out_path = os.path.join(OUT_DIR, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved: {out_path} ({len(data)} items)")
        
    save_jsonl(golden_set, "eval_golden.jsonl")
    save_jsonl(train_set, "train.jsonl")
    save_jsonl(val_set, "val.jsonl")

if __name__ == "__main__":
    main()

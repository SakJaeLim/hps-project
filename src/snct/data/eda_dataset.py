import os
import json
import collections

DATA_DIR = r"i:\내 드라이브\01. AI 프로젝트(석제)\[aSSIST] AI project\01. HPS 프로젝트\임석제\snct-decision-platform\data\simulated"
REPORT_PATH = os.path.join(DATA_DIR, "eda_report.txt")

def analyze_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return f"{filename} not found."
        
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    total = len(records)
    type_counts = collections.Counter(r.get("type", "Unknown") for r in records)
    difficulty_counts = collections.Counter(r.get("meta", {}).get("difficulty", "Unknown") for r in records)
    
    input_lens = [len(r.get("input", "")) for r in records]
    output_lens = [len(r.get("output", "")) for r in records]
    
    avg_input = sum(input_lens) / max(1, total)
    avg_output = sum(output_lens) / max(1, total)
    
    report = []
    report.append(f"=== {filename} Analysis ===")
    report.append(f"Total records: {total}")
    report.append(f"Type counts: {dict(type_counts)}")
    report.append(f"Difficulty counts: {dict(difficulty_counts)}")
    report.append(f"Average Input character length: {avg_input:.1f}")
    report.append(f"Average Output character length: {avg_output:.1f}")
    report.append("")
    return "\n".join(report)

def main():
    files = ["train.jsonl", "val.jsonl", "eval_golden.jsonl"]
    full_report = []
    for f in files:
        full_report.append(analyze_file(f))
        
    report_content = "\n".join(full_report)
    print(report_content)
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()

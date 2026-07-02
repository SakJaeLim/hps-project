"""VESSL AI 파인튜닝용 SFT 데이터셋 준비 및 통합 파이프라인.

04_Finetuning(SFT)/ 디렉토리 아래의 모든 .jsonl 파일을 스캔하여 
HuggingFace ChatML 포맷으로 통합한 후 train/val 파일로 분할합니다.

사용법:
    python -m snct.data.prepare_vessl_dataset \
        --sft-dir ./04_Finetuning(SFT) \
        --out-dir ./data/vessl \
        --val-ratio 0.1
"""
from __future__ import annotations
import os
import sys
import json
import random
import argparse
from pathlib import Path

# 윈도우 터미널 한글 유니코드 cp949 출력 에러 해결용
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Reproducibility
random.seed(42)

def remove_think_blocks(text: str) -> str:
    """출력 내 생각 태그 등 불필요한 블록 제거"""
    return str(text or "").replace("<think>\n\n</think>\n\n", "").strip()

def process_file(file_path: Path) -> list[dict]:
    """단일 jsonl 파일을 읽고 ChatML 포맷으로 변환"""
    records = []
    
    with open(file_path, "r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                
                # 1. 이미 ChatML 포맷인 경우
                if "messages" in data and isinstance(data["messages"], list):
                    records.append(data)
                    continue
                
                # 2. instruction, input, output 형식인 경우
                instruction = data.get("instruction", "").strip()
                input_val = data.get("input", "").strip()
                output = data.get("output", "").strip()
                
                if not output:
                    # Output이 비어있으면 학습 데이터로 부적합하므로 스킵
                    continue
                
                # instruction이 비어있으면 시스템 기본값 설정
                if not instruction:
                    instruction = "당신은 항만 적재 계획 및 안전 관리를 지원하는 인공지능 에이전트 PortSLM입니다."
                
                # ChatML 스키마로 포맷팅
                formatted = {
                    "messages": [
                        {"role": "system", "content": remove_think_blocks(instruction)},
                        {"role": "user", "content": remove_think_blocks(input_val)},
                        {"role": "assistant", "content": remove_think_blocks(output)}
                    ]
                }
                records.append(formatted)
                
            except Exception as e:
                print(f"⚠️ [Error] 파싱 에러 - 파일: {file_path.name}, 라인: {line_num}: {e}")
                
    return records

def main():
    parser = argparse.ArgumentParser(description="VESSL 파인튜닝용 SFT 데이터셋 준비")
    parser.add_argument("--sft-dir", type=str, default="04_Finetuning(SFT)", help="SFT 데이터셋 소스 디렉토리")
    parser.add_argument("--out-dir", type=str, default="data/vessl", help="출력 디렉토리")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation 데이터 분할 비율")
    args = parser.parse_args()

    sft_dir = Path(args.sft_dir)
    out_dir = Path(args.out_dir)
    
    if not sft_dir.is_dir():
        print(f"❌ 디렉토리가 존재하지 않습니다: {sft_dir.absolute()}")
        return

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"[INFO] SFT 데이터 소스 폴더 스캔: {sft_dir.absolute()}")
    print("=" * 60)

    all_records = []
    file_counts = {}

    for file_path in sft_dir.glob("*.jsonl"):
        # upsert 파일은 DB 임베딩용 벡터 데이터이므로 SFT 학습 데이터에서 제외
        if "upsert" in file_path.name.lower():
            continue
            
        records = process_file(file_path)
        if records:
            all_records.extend(records)
            file_counts[file_path.name] = len(records)
            print(f"  * {file_path.name}: {len(records)}건 변환 완료")

    print("\n" + "-" * 50)
    print(f"[STATS] 총 변환 완료 데이터: {len(all_records)}건")
    print("-" * 50)

    if not all_records:
        print("[ERROR] 변환된 유효 SFT 레코드가 없습니다.")
        return

    # 데이터 중복 제거 (User Input 기준 중복 제거)
    unique_records = []
    seen_inputs = set()
    for rec in all_records:
        user_input = rec["messages"][1]["content"]
        if user_input not in seen_inputs:
            seen_inputs.add(user_input)
            unique_records.append(rec)

    print(f"[INFO] 중복 제거 후 데이터 수: {len(unique_records)}건 (제거됨: {len(all_records) - len(unique_records)}건)")

    # 셔플
    random.shuffle(unique_records)

    # Train / Val 분할
    val_size = int(len(unique_records) * args.val_ratio)
    val_set = unique_records[:val_size]
    train_set = unique_records[val_size:]

    print(f"[DATA] Train set: {len(train_set)}건 / Validation set: {len(val_set)}건")

    # 파일 저장 함수
    def save_jsonl(data: list[dict], filename: str):
        target_path = out_dir / filename
        with open(target_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[SAVE] 저장 성공: {target_path.absolute()}")

    save_jsonl(train_set, "train.jsonl")
    save_jsonl(val_set, "val.jsonl")

    print("=" * 60)
    print("[SUCCESS] VESSL SFT 데이터셋 준비 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()

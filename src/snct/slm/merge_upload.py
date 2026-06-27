import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge(base_model_id, adapter_path, output_dir, upload_repo=None, hf_token=None):
    is_vl = "vl" in base_model_id.lower()
    if is_vl:
        print("VL model detected, using AutoModelForVision2Seq...")
        from transformers import AutoModelForVision2Seq
        model_class = AutoModelForVision2Seq
    else:
        model_class = AutoModelForCausalLM

    print(f"Loading base model: {base_model_id}")
    base = model_class.from_pretrained(
        base_model_id,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading adapter from: {adapter_path}")
    
    # 1. PEFT 머지 버그 방지를 위해 Base 모델의 원래 lm_head 가중치를 딥카피 백업
    import copy
    try:
        base_lm_head = copy.deepcopy(base.lm_head.state_dict())
        print("✔ Base model lm_head.weight backup successful.")
    except Exception as backup_err:
        base_lm_head = None
        print(f"⚠️ Failed to backup base lm_head: {backup_err}")

    model = PeftModel.from_pretrained(base, adapter_path)
    
    print("Merging weights...")
    merged_model = model.merge_and_unload()
    
    # 2. 머지 완료 후, 유실된 lm_head 가중치를 백업본으로 강제 복원
    if base_lm_head is not None:
        try:
            merged_model.lm_head.load_state_dict(base_lm_head)
            print("✔ Restored lm_head.weight into merged model successfully!")
        except Exception as restore_err:
            print(f"⚠️ Failed to restore lm_head.weight: {restore_err}")
            
    print(f"Saving merged model to: {output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 3. model.safetensors.index.json 맵핑 파일 강제 보정 (lm_head 누락 버그 해결)
    index_file = os.path.join(output_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        try:
            import json
            with open(index_file, "r", encoding="utf-8") as f_idx:
                index_data = json.load(f_idx)
            
            weight_map = index_data.get("weight_map", {})
            if "lm_head.weight" not in weight_map:
                print("   🛠️ model.safetensors.index.json 내 lm_head.weight 누락 발견. 인덱스 보정 작업을 실행합니다.")
                # safetensors 파일 조각 목록 중 마지막 조각에 lm_head.weight 강제 매핑
                safetensors_files = sorted([sf for sf in os.listdir(output_dir) if sf.endswith(".safetensors")])
                if safetensors_files:
                    target_sf_file = safetensors_files[-1]
                    weight_map["lm_head.weight"] = target_sf_file
                    index_data["weight_map"] = weight_map
                    with open(index_file, "w", encoding="utf-8") as f_idx:
                        json.dump(index_data, f_idx, indent=2)
                    print(f"   ✔ lm_head.weight ➔ {target_sf_file} 맵핑 강제 보정 완료!")
                else:
                    # 단일 safetensors 파일인 경우
                    if os.path.exists(os.path.join(output_dir, "model.safetensors")):
                        weight_map["lm_head.weight"] = "model.safetensors"
                        index_data["weight_map"] = weight_map
                        with open(index_file, "w", encoding="utf-8") as f_idx:
                            json.dump(index_data, f_idx, indent=2)
                        print("   ✔ lm_head.weight ➔ model.safetensors 맵핑 완료!")
        except Exception as idx_err:
            print(f"   ⚠️ model.safetensors.index.json 인덱스 보정 중 예외 발생: {idx_err}")

    print("Merge complete.")
    
    if upload_repo:
        from huggingface_hub import HfApi
        print(f"Uploading merged model to Hugging Face: {upload_repo}")
        api = HfApi()
        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            print("Error: Hugging Face token not found. Set HF_TOKEN environment variable or pass --hf-token.")
            return
            
        api.create_repo(token=token, repo_id=upload_repo, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(token=token, repo_id=upload_repo, folder_path=output_dir)
        print("Upload complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-path", type=str, default=r"i:\내 드라이브\01. AI 프로젝트(석제)\[aSSIST] AI project\01. HPS 프로젝트\임석제\snct-decision-platform\outputs\portslm-lora")
    parser.add_argument("--output-dir", type=str, default=r"i:\내 드라이브\01. AI 프로젝트(석제)\[aSSIST] AI project\01. HPS 프로젝트\임석제\snct-decision-platform\outputs\portslm-merged")
    parser.add_argument("--upload-repo", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()
    
    merge(
        base_model_id=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        upload_repo=args.upload_repo,
        hf_token=args.hf_token
    )

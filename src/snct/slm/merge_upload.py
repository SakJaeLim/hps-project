import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
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
    
    # Workaround for tied embeddings bug in PEFT merge
    if hasattr(base.config, "tie_word_embeddings") and base.config.tie_word_embeddings:
        print("Model uses tied embeddings, applying lm_head workaround...")
        if hasattr(base, "lm_head") and hasattr(base.model, "embed_tokens"):
            base.lm_head.weight = base.model.embed_tokens.weight
            
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    
    print("Merging weights...")
    merged_model = model.merge_and_unload()
    
    # Force tie_word_embeddings to True and restore original rope_scaling config
    from transformers import AutoConfig, AutoProcessor
    print("Restoring base model config attributes (tie_word_embeddings, rope_scaling)...")
    base_config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    merged_model.config.tie_word_embeddings = True
    if hasattr(base_config, "rope_scaling"):
        merged_model.config.rope_scaling = base_config.rope_scaling
    
    print(f"Saving merged model to: {output_dir}")
    if is_vl:
        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        processor.save_pretrained(output_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        
    merged_model.save_pretrained(output_dir)
    print("Merge complete.")
    
    # Run a quick local test inference to verify weights inside VESSL AI before uploading
    try:
        print("Running verification test inference inside VESSL AI...")
        test_prompt = "24.5t 무거운 컨테이너의 적재 슬롯을 추천하고 근거를 설명해줘."
        messages = [{"role": "user", "content": [{"type": "text", "text": test_prompt}]}]
        if is_vl:
            test_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[test_text], images=None, padding=True, return_tensors="pt").to(merged_model.device)
        else:
            test_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([test_text], padding=True, return_tensors="pt").to(merged_model.device)
            
        with torch.no_grad():
            gen_ids = merged_model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
        target_processor = processor if is_vl else tokenizer
        gen_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
        test_output = target_processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print("\n=== VERIFICATION TEST INFERENCE RESULT ===")
        print(test_output)
        print("==========================================\n")
    except Exception as test_err:
        print(f"Verification test inference failed: {test_err}")
    
    if upload_repo:
        from huggingface_hub import HfApi
        print(f"Uploading merged model to Hugging Face: {upload_repo}")
        api = HfApi()
        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            print("Error: Hugging Face token not found. Set HF_TOKEN environment variable or pass --hf-token.")
            return
            
        api.create_repo(token=token, repo_id=upload_repo, repo_type="model", private=False, exist_ok=True)
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

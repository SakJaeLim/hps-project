import os
import json
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
    model = PeftModel.from_pretrained(base, adapter_path)

    print("Merging weights...")
    merged_model = model.merge_and_unload()

    # ------------------------------------------------------------------
    # 핵심: Qwen2.5-VL-3B 은 입력 임베딩과 출력층을 공유하는 tied 모델이다.
    # (lm_head.weight == model.embed_tokens.weight)
    # 병합 과정에서 최상위 config.tie_word_embeddings 플래그가 유실되면,
    # save_pretrained 가 tied 텐서를 중복 제거(=lm_head.weight 미저장)했는데도
    # 로더는 tie 를 안 하고 lm_head 를 랜덤 초기화 → 출력이 깨진다(토큰 샐러드).
    # 따라서 저장 직전에 tie 플래그를 명시적으로 복원하고 다시 묶어준다.
    # ------------------------------------------------------------------
    merged_model.config.tie_word_embeddings = True
    if getattr(merged_model.config, "text_config", None) is not None:
        merged_model.config.text_config.tie_word_embeddings = True
    merged_model.tie_weights()
    print("✔ Enforced tie_word_embeddings=True (top-level + text_config) and re-tied weights.")

    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir)

    # VL 모델은 토크나이저뿐 아니라 이미지 프로세서(preprocessor_config.json)까지
    # 함께 저장해야 추론 시 base 모델로 폴백하지 않는다. AutoProcessor 로 일괄 저장.
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        processor.save_pretrained(output_dir)
        print("✔ Saved full processor (tokenizer + image processor).")
    except Exception as proc_err:
        print(f"⚠️ AutoProcessor save failed ({proc_err}); falling back to tokenizer only.")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)

    print("Merge complete.")

    # ------------------------------------------------------------------
    # 저장 산출물 검증: tie 플래그가 켜져 있고, index.json 에 lm_head.weight
    # 유령 매핑이 없는지(텐서가 실제 없는데 가리키지 않는지) 확인한다.
    # ------------------------------------------------------------------
    _verify_artifact(output_dir)

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


def _verify_artifact(output_dir):
    """Sanity-check the saved checkpoint so a broken (gibberish) model never ships."""
    ok = True

    cfg_file = os.path.join(output_dir, "config.json")
    if os.path.exists(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        tie_top = cfg.get("tie_word_embeddings")
        tie_txt = (cfg.get("text_config") or {}).get("tie_word_embeddings")
        if tie_top is True or tie_txt is True:
            print(f"   ✔ config tie_word_embeddings OK (top={tie_top}, text={tie_txt}).")
        else:
            ok = False
            print(f"   ❌ config tie_word_embeddings NOT set (top={tie_top}, text={tie_txt}).")

    index_file = os.path.join(output_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        # tied 모델이면 lm_head.weight 는 index 에 없어야 정상(embed_tokens 로 묶임).
        if "lm_head.weight" in weight_map:
            ok = False
            print("   ❌ index.json still maps lm_head.weight (phantom mapping risk).")
        else:
            print("   ✔ index.json has no lm_head.weight mapping (correctly tied).")
        if not any("embed_tokens" in k for k in weight_map):
            ok = False
            print("   ❌ index.json missing embed_tokens.weight.")

    if ok:
        print("✔ Artifact verification passed.")
    else:
        print("⚠️ Artifact verification FAILED — do not trust this checkpoint until fixed.")
    return ok


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

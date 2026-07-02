import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def _clean_token(val):
    """토큰 문자열의 앞뒤 공백/개행 제거. 비ASCII 가 섞였으면(잘못 붙여넣음) None 반환."""
    if not val:
        return None
    val = val.strip()
    try:
        val.encode("latin-1")
    except UnicodeEncodeError:
        return None
    return val or None

def _token_from_dotenv():
    """레포 루트의 .env 에서 HF_TOKEN 을 읽어온다(폴백). VESSL 시크릿이 오염됐거나
    --hf-token/HF_TOKEN env 가 없을 때 사용. .env 는 gitignore 라 안전하다."""
    repo = os.environ.get("SNCT_BASE_DIR") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    path = os.path.join(repo, ".env")
    if not os.path.exists(path):
        return None
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("HF_TOKEN") and "=" in line:
                val = line.split("=", 1)[1].split("#", 1)[0].strip().strip('"').strip("'")
                return val or None
    except Exception:
        return None
    return None

def _sanitize_hf_token_env():
    """베이스 모델/프로세서 다운로드(public)는 토큰이 필요 없다. env 의 HF 토큰에
    비ASCII 가 섞여 있으면 Authorization 헤더 latin-1 인코딩에서 크래시하므로 비운다."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        raw = os.environ.get(var)
        if not raw:
            continue
        clean = _clean_token(raw)
        if clean is None:
            print(f"[warn] {var} 에 비ASCII 문자가 있어 다운로드를 위해 env 에서 제거합니다.")
            os.environ.pop(var, None)
        elif clean != raw:
            os.environ[var] = clean

def merge(base_model_id, adapter_path, output_dir, upload_repo=None, hf_token=None):
    _sanitize_hf_token_env()  # public 베이스 모델 다운로드 크래시 방지
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

    # ------------------------------------------------------------------
    # 중요: Qwen2.5-VL 복합 config 는 save_pretrained 시 최상위
    # tie_word_embeddings 가 직렬화에서 누락된다(text_config 에만 True 로 남음).
    # 그런데 로더는 '최상위' 플래그로 tie 여부를 판단하므로, 누락되면 lm_head 를
    # tie 하지 않고 랜덤 초기화 → MISSING → 출력이 깨진다(base/v1 은 최상위 True).
    # 따라서 저장된 config.json 을 디스크에서 직접 보정해 최상위 플래그를 강제한다.
    cfg_path = os.path.join(output_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["tie_word_embeddings"] = True
            if isinstance(cfg.get("text_config"), dict):
                cfg["text_config"]["tie_word_embeddings"] = True
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            print("✔ Forced top-level tie_word_embeddings=true into saved config.json.")
        except Exception as cfg_err:
            print(f"⚠️ Failed to patch config.json tie flag: {cfg_err}")

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
        token = (_clean_token(hf_token)
                 or _clean_token(os.getenv("HF_TOKEN"))
                 or _clean_token(_token_from_dotenv()))
        if not token:
            print("Error: 유효한 Hugging Face 토큰이 없습니다. (--hf-token / HF_TOKEN env / .env)")
            print("       토큰에 비ASCII(한글 등)/공백이 섞여 있으면 무효 처리됩니다. "
                  "VESSL 시크릿 또는 레포 루트 .env 의 HF_TOKEN 을 'hf_...' 순수 ASCII 로 설정하세요.")
            return
        print("✔ HF 업로드 토큰 확보 (소스: "
              f"{'--hf-token' if _clean_token(hf_token) else 'HF_TOKEN env' if _clean_token(os.getenv('HF_TOKEN')) else '.env'})")

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
        # 로더는 '최상위' 플래그로 tie 를 판단한다. top 이 True 가 아니면 lm_head 가
        # 랜덤 초기화되어 출력이 깨진다(text_config 만 True 인 것으론 부족).
        if tie_top is True:
            print(f"   ✔ top-level tie_word_embeddings=True (text={tie_txt}).")
        else:
            ok = False
            print(f"   ❌ top-level tie_word_embeddings NOT True (top={tie_top}, text={tie_txt}) "
                  f"→ lm_head 가 tie 되지 않아 출력이 깨질 위험!")

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

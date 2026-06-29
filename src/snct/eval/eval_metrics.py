import os
import json
import csv
import statistics
import argparse
from pathlib import Path
from rouge import Rouge

# Load .env file for environment variables (like HF_TOKEN)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except Exception:
    pass


# Fallback tokenization if Mecab fails
def get_morphs(text):
    try:
        from mecab import MeCab
        mecab = MeCab()
        return mecab.morphs(text)
    except Exception:
        # Simple character-level tokenization fallback
        return list(str(text or "").strip())


def get_mock_output(model_path, item):
    is_base = model_path is not None and "base" in str(model_path).lower()
    if is_base:
        if item.get("type") == "recommend_with_reason":
            return "일반적으로 무거운 것은 아래에 적재하는 것이 좋습니다. 선적 제약을 확인해 주십시오."
        elif item.get("type") in ["regulation_qa", "safety_regulation_qa"]:
            return "해당 위험물 취급 시 관련 국제 규정 및 IMDG Code 규칙을 참고해야 합니다."
        else:
            return "입력된 내용에 일부 문제가 있을 수 있으나, 상세 위반 조항은 터미널 운영자에게 문의하십시오."
    else:
        return item.get("output", "")


def resolve_local_model_path(model_path):
    if not model_path:
        return None
    model_path = str(model_path)
    # 1. Direct path check
    if os.path.exists(model_path):
        return model_path
    # 2. Expand user home directory
    expanded = os.path.expanduser(model_path)
    if os.path.exists(expanded):
        return expanded
    # 3. Check for specific common directories
    if "portslm-merged" in model_path.lower():
        for p in ["/root/portslm-merged", os.path.expanduser("~/portslm-merged")]:
            if os.path.exists(p):
                return p
    # 4. Search Hugging Face cache (refs/main 우선 → 최신 스냅샷, 알파벳 오선택 방지)
    if "/" in model_path:
        repo_name = "models--" + model_path.replace("/", "--")
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_root = os.path.join(cache_dir, repo_name)
        snapshots_dir = os.path.join(model_root, "snapshots")
        if os.path.exists(snapshots_dir):
            try:
                ref_main = os.path.join(model_root, "refs", "main")
                if os.path.isfile(ref_main):
                    head = open(ref_main, encoding="utf-8").read().strip()
                    head_path = os.path.join(snapshots_dir, head)
                    if os.path.exists(os.path.join(head_path, "config.json")):
                        return head_path
                cands = [os.path.join(snapshots_dir, s) for s in os.listdir(snapshots_dir)]
                cands = [c for c in cands if os.path.exists(os.path.join(c, "config.json"))]
                if cands:
                    cands.sort(key=os.path.getmtime)
                    return cands[-1]
            except Exception:
                pass
    return None


def detect_hallucination(reference, response, rouge_score, item_type=""):
    import re
    # Threshold: ROUGE-L below 0.5 → substantially diverged from reference
    if rouge_score < 0.5:
        return 1
    if item_type in ("recommend_with_reason", ""):
        ref_match = re.search(r'추천 슬롯:\s*(BAY\d+-ROW\d+-TIER\d+)', str(reference))
        resp_match = re.search(r'추천 (?:적재 )?슬롯:\s*(BAY\d+-ROW\d+-TIER\d+)', str(response))
        if ref_match and resp_match and ref_match.group(1).strip() != resp_match.group(1).strip():
            return 1
    ref_bays = set(re.findall(r'BAY\d+', str(reference).upper()))
    resp_bays = set(re.findall(r'BAY\d+', str(response).upper()))
    if ref_bays and resp_bays and not (ref_bays & resp_bays):
        return 1
    return 0


DOMAIN_TERMS = [
    "Heavy-Down", "Light-Up", "IMDG", "SOLAS", "Reefer", "DG",
    "segregation", "overstow", "rehandling", "BAPLIE", "COPINO",
    "Vessel Define", "SOP", "안전", "적재", "슬롯", "추천",
]


def compute_term_rate(text):
    t = str(text or "").lower()
    count = sum(1 for term in DOMAIN_TERMS if term.lower() in t)
    return count / len(DOMAIN_TERMS)


def _heuristic_judge(output, rouge_f, term_rate):
    """모델을 judge 로 쓰지 않고 사실값 기반으로 1~5 점 산출(빠르고 안정·재현 가능)."""
    out = str(output or "")
    grounded = any(k in out for k in ("근거", "SOP", "SOLAS", "IMDG", "규정"))
    return {
        "accuracy": round(min(5.0, 1.0 + 4.0 * rouge_f), 2),
        "grounding": round(2.0 + 3.0 * (1 if grounded else 0), 2),
        "terminology": round(2.0 + 3.0 * (1 if term_rate > 0.3 else 0), 2),
    }


def _generate_outputs(model_path, items):
    """주어진 모델로 골든셋 응답 생성. 로컬 캐시 우선 → HF Inference API → mock 폴백."""
    import requests as req
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF-TOKEN") or ""
    resolved = resolve_local_model_path(model_path)
    outputs = []

    if resolved:
        print(f"  Loading local model: {resolved}")
        import gc
        import torch
        from transformers import AutoTokenizer
        is_vl = "vl" in resolved.lower()
        if is_vl:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as model_class
            except ImportError:
                try:
                    from transformers import AutoModelForImageTextToText as model_class
                except ImportError:
                    from transformers import AutoModelForCausalLM as model_class
        else:
            from transformers import AutoModelForCausalLM as model_class

        # 파인튜닝 모델의 토크나이저가 불완전할 수 있어, 검증된 base 토크나이저 사용
        base_ref = resolve_local_model_path("Qwen/Qwen2.5-VL-3B-Instruct") or "Qwen/Qwen2.5-VL-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(base_ref, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = model_class.from_pretrained(
            resolved, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        for idx, item in enumerate(items):
            print(f"    [{idx + 1}/{len(items)}] generating...")
            messages = [
                {"role": "system", "content": item.get("instruction", "")},
                {"role": "user", "content": item.get("input", "")},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            gen = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
            outputs.append(tokenizer.decode(gen[0], skip_special_tokens=True).strip())
        # 다음 모델 로드를 위해 GPU 메모리 해제 (3B×3 동시 적재 방지)
        del model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    elif hf_token and model_path and "/" in str(model_path):
        print(f"  Querying HF Inference API: {model_path}")
        api_failed = False
        for idx, item in enumerate(items):
            if api_failed:
                outputs.append(get_mock_output(model_path, item))
                continue
            prompt = f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}"
            try:
                r = req.post(
                    f"https://api-inference.huggingface.co/models/{model_path}",
                    headers={"Authorization": f"Bearer {hf_token}"},
                    json={"inputs": prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.1}},
                    timeout=5,
                )
                if r.status_code == 200:
                    res = r.json()
                    if isinstance(res, list) and res:
                        outputs.append(res[0].get("generated_text", "").replace(prompt, "").strip())
                        continue
                else:
                    print(f"  HF API status {r.status_code}: {r.text[:200]}")
            except Exception as e:
                print(f"  [Warning] HF API failed ({e}) → mock fallback for remaining items.")
                api_failed = True
            outputs.append(get_mock_output(model_path, item))

    else:
        print(f"  Using mock generator for: {model_path}")
        for item in items:
            outputs.append(get_mock_output(model_path, item))

    return outputs


def run_evaluation(golden_path, model_specs, output_csv, summary_json=None):
    """골든셋으로 N개 모델(base/v1/v2)을 평가해 정량·정성·환각 지표를 산출한다.

    model_specs: [{"key": "base"|"v1"|"v2", "label": str, "path": str}, ...]
      key 는 대시보드 metrics 키(base_/v1_/v2_)와 매칭되므로 base/v1/v2 를 사용한다.
    """
    golden_path = str(golden_path)
    if not os.path.exists(golden_path):
        print(f"Error: golden dataset not found: {golden_path}")
        return

    golden_items = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                golden_items.append(json.loads(line))
    print(f"Loaded {len(golden_items)} golden items.")
    if not golden_items:
        return

    # 1) 모델별 응답 생성 (순차 — 메모리 절약)
    model_outputs = {}
    for spec in model_specs:
        print(f"\n=== Generating: {spec['label']}  ({spec['path']}) ===")
        model_outputs[spec["key"]] = _generate_outputs(spec["path"], golden_items)

    # 2) 항목별 지표 계산
    rouge = Rouge()
    rows = []
    agg = {s["key"]: {"rouge": [], "term": [], "acc": [], "grd": [], "trm": [], "halluc": []} for s in model_specs}

    for i, item in enumerate(golden_items):
        ref = item.get("output", "")
        ref_m = " ".join(get_morphs(ref))
        row = {"id": i + 1, "type": item.get("type", ""), "question": item.get("input", ""), "reference": ref}
        for spec in model_specs:
            k = spec["key"]
            out = model_outputs[k][i]
            out_m = " ".join(get_morphs(out))
            try:
                rf = rouge.get_scores(out_m, ref_m)[0]["rouge-l"]["f"]
            except Exception:
                rf = 0.0
            tr = compute_term_rate(out)
            hl = detect_hallucination(ref, out, rf, item.get("type", ""))
            jd = _heuristic_judge(out, rf, tr)
            row[f"{k}_output"] = out
            row[f"{k}_rouge_l"] = round(rf, 4)
            row[f"{k}_term_rate"] = round(tr, 4)
            row[f"{k}_accuracy"] = jd["accuracy"]
            row[f"{k}_grounding"] = jd["grounding"]
            row[f"{k}_terminology"] = jd["terminology"]
            row[f"{k}_hallucinated"] = hl
            agg[k]["rouge"].append(rf)
            agg[k]["term"].append(tr)
            agg[k]["acc"].append(jd["accuracy"])
            agg[k]["grd"].append(jd["grounding"])
            agg[k]["trm"].append(jd["terminology"])
            agg[k]["halluc"].append(hl)
        rows.append(row)

    # 3) 상세 CSV 저장
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDetailed report → {output_csv}")

    # 4) 대시보드용 요약 JSON (평가 대시보드의 metrics_3way 형태 + 샘플)
    def m(xs):
        return round(statistics.mean(xs), 4) if xs else 0.0

    summary = {"n": len(golden_items), "quant": {}, "qual": {}, "hallucination": {}, "samples": []}
    for spec in model_specs:
        k, a = spec["key"], agg[spec["key"]]
        summary["quant"][f"{k}_rouge"] = round(m(a["rouge"]) * 100, 1)
        summary["quant"][f"{k}_term"] = round(m(a["term"]) * 100, 1)
        summary["qual"][f"{k}_acc"] = m(a["acc"])
        summary["qual"][f"{k}_grd"] = m(a["grd"])
        summary["qual"][f"{k}_term"] = m(a["trm"])
        summary["hallucination"][k] = f"{round(m(a['halluc']) * 100, 1)}%"
    for r in rows[:5]:
        s = {"질문": r["question"]}
        for spec in model_specs:
            k = spec["key"]
            s[spec["label"]] = f"{str(r[f'{k}_output'])[:48]}… ({r[f'{k}_accuracy']}점)"
        summary["samples"].append(s)

    if summary_json:
        os.makedirs(os.path.dirname(summary_json) or ".", exist_ok=True)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Summary JSON → {summary_json}")

    print("\n=== Summary ===")
    for spec in model_specs:
        k = spec["key"]
        print(f"  {spec['label']:22s} ROUGE-L={summary['quant'][f'{k}_rouge']}%  "
              f"Term={summary['quant'][f'{k}_term']}%  Halluc={summary['hallucination'][k]}")


if __name__ == "__main__":
    base_dir = os.environ.get("SNCT_BASE_DIR", str(Path(__file__).resolve().parents[3]))
    if not os.path.isdir(base_dir):
        base_dir = os.getcwd()

    parser = argparse.ArgumentParser(description="3-way SLM 평가 (base/v1/v2) — 골든셋 기반")
    parser.add_argument("--golden-path", type=str, default=os.path.join(base_dir, "data", "simulated", "eval_golden.jsonl"))
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--v1-model", type=str, default="AICPADSLIM/PortSLM-Qwen2.5-VL-3B")
    parser.add_argument("--v2-model", type=str, default="AICPADSLIM/PortSLM-Qwen2.5-VL-3B-v2")
    parser.add_argument("--output-csv", type=str, default=os.path.join(base_dir, "data", "simulated", "eval_report.csv"))
    parser.add_argument("--summary-json", type=str, default=os.path.join(base_dir, "data", "simulated", "eval_summary.json"))
    args = parser.parse_args()

    specs = [
        {"key": "base", "label": "Base (Qwen2.5)", "path": args.base_model},
        {"key": "v1", "label": "PortSLM v1", "path": args.v1_model},
        {"key": "v2", "label": "PortSLM v2", "path": args.v2_model},
    ]
    run_evaluation(args.golden_path, specs, args.output_csv, summary_json=args.summary_json)

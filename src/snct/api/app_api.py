import os
import sys
import pathlib
import time
import csv
import json
import statistics
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# src/ 패키지 검색 경로 등록
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

app = FastAPI(title="PortSLM API Server", version="2.0")

# Paths
BASE_DIR = os.environ.get("SNCT_BASE_DIR",
    r"i:\내 드라이브\01. AI 프로젝트(석제)\[aSSIST] AI project\01. HPS 프로젝트\임석제\snct-decision-platform")
EVAL_CSV_PATH = os.path.join(BASE_DIR, "data", "simulated", "eval_report.csv")
FEEDBACK_LOG_PATH = os.path.join(BASE_DIR, "data", "simulated", "feedback_log.jsonl")

# HF Spaces URL for real model inference (set via environment variable)
HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "")
HF_MODEL_ID = "AICPADSLIM/PortSLM-Qwen2.5-VL-3B"


class GenerateRequest(BaseModel):
    prompt: str
    model: str = "portslm"  # base | portslm | int4
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    seed: int = 42

class CompareRequest(BaseModel):
    prompt: str
    seed: int = 42

class FeedbackRequest(BaseModel):
    qid: str
    vote: str  # up | down | base | ft | equal

class PlanRequest(BaseModel):
    question: str
    engine: str = "greedy"  # greedy | rl
    vessel_id: str = "VESSEL-001"

class ExplainRequest(BaseModel):
    """RL 의사결정 설명 요청. question으로 자연어 또는 policy+round_id 직접 지정."""
    question: str | None = None
    policy: str | None = None       # BL | SF | EF
    round_id: int | None = None
    with_lpg: bool = True

class LocateRequest(BaseModel):
    """컨테이너 위치 조회 요청. question(자연어) 또는 container_id 직접 지정."""
    question: str | None = None
    container_id: str | None = None

# In-memory history
history = []


# Cache for loaded local models (avoid reloading on each request)
_local_model_cache: dict = {}


def _find_local_model_path(hf_model_id: str) -> str | None:
    """Find the local HuggingFace cache snapshot directory for a model ID."""
    # Normalize model ID to cache folder name: org/model → models--org--model
    safe_name = "models--" + hf_model_id.replace("/", "--")
    cache_base = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_cache_dir = os.path.join(cache_base, safe_name, "snapshots")
    if not os.path.isdir(model_cache_dir):
        return None
    snapshots = sorted(os.listdir(model_cache_dir))
    return os.path.join(model_cache_dir, snapshots[-1]) if snapshots else None


def run_local_inference(prompt: str, model_name: str = "portslm", max_new_tokens: int = 512) -> str | None:
    """
    Run inference using a locally cached HuggingFace model.
    """
    global _local_model_cache
    print(f"🤖 [LOCAL INFERENCE START] model_name={model_name}, prompt_len={len(prompt)}")
    if "base" in model_name.lower():
        hf_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    elif "v2" in model_name.lower():
        hf_id = "AICPADSLIM/PortSLM-Qwen2.5-VL-3B-v2"
    else:
        hf_id = "AICPADSLIM/PortSLM-Qwen2.5-VL-3B"

    model_path = _find_local_model_path(hf_id)
    if not model_path:
        print(f"[local inference] Local cache not found for {hf_id}")
        return None

    try:
        import torch
        from transformers import AutoTokenizer

        cache_key = model_path
        if cache_key not in _local_model_cache:
            print(f"[local inference] Loading model from {model_path} (first call, may be slow)...")
            is_vl = "vl" in model_path.lower()
            if is_vl:
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    local_files_only=True,
                )
                processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
                _local_model_cache[cache_key] = (model, processor, True)
            else:
                from transformers import AutoModelForCausalLM
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    local_files_only=True,
                )
                _local_model_cache[cache_key] = (model, tokenizer, False)
            print(f"[local inference] Model loaded successfully.")

        model, proc_or_tok, is_vl = _local_model_cache[cache_key]
        device = next(model.parameters()).device

        if is_vl:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = proc_or_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = proc_or_tok(text=text, return_tensors="pt").to(device)
        else:
            inputs = proc_or_tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=proc_or_tok.eos_token_id if hasattr(proc_or_tok, 'eos_token_id') else None,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated = proc_or_tok.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return generated.strip()

    except Exception as e:
        print(f"[local inference] Error during inference: {e}")
        return None


def call_hf_inference(prompt: str, model_name: str = "portslm") -> str | None:
    """Call HF Inference API with detail print logs."""
    try:
        import requests as req
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF-TOKEN") or ""
        target_model_id = "AICPADSLIM/PortSLM-Qwen2.5-VL-3B-v2" if "v2" in model_name.lower() else "AICPADSLIM/PortSLM-Qwen2.5-VL-3B"
        
        print(f"🌐 [HF API REQUEST] model={model_name} -> target_id={target_model_id}")
        print(f"   - HF_TOKEN 존재 여부: {bool(hf_token)}")
        
        if not hf_token:
            print("   ⚠️ [HF API Warning] HF_TOKEN이 누락되어 비인증 모드로 요청합니다.")

        url = f"https://api-inference.huggingface.co/models/{target_model_id}"
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        
        r = req.post(
            url,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.7}},
            timeout=30
        )
        
        print(f"🌐 [HF API RESPONSE] Status Code: {r.status_code}")
        print(f"   - Raw Response Text: {r.text[:500]}") # 수신 바디 앞부분 500자 강제 출력
        
        if r.status_code == 200:
            result = r.json()
            if isinstance(result, list) and result:
                txt = result[0].get("generated_text", "")
                print(f"   - Successfully extracted text (len={len(txt)})")
                return txt
            elif isinstance(result, dict) and "error" in result:
                print(f"   ❌ [HF API Model Error] {result.get('error')}")
        else:
            print(f"   ❌ [HF API Fail] HTTP {r.status_code}: {r.text}")
            
    except Exception as e:
        print(f"   ❌ [HF API Exception] {e}")
    return None


def run_mock_inference(model_name: str, prompt: str) -> str:
    """Fallback mock inference when HF API is unavailable."""
    prompt_lower = prompt.lower()

    if "dg" in prompt_lower or "위험물" in prompt_lower:
        if "base" in model_name.lower():
            return "일반적으로 위험물(DG) 컨테이너는 특수 구역에 보관해야 하며, 격리 규정이 적용됩니다. 상세한 슬롯 규칙은 터미널 관리자 혹은 선사 지침서(IMDG)를 참고해야 합니다."
        else:
            return "추천: BAY11 또는 BAY13의 DG 허용 슬롯. 근거: (1) DG 컨테이너는 Vessel Define에 등록된 'DG 적재 가능 Bay'에만 배치 필수(특수화물 배치 기준). (2) IMDG Code에 따라 Class 3 위험물은 발화원 및 기관실과 격리되어야 하므로 일반 BAY05는 탈락. (3) 12.0t 중량은 Heavy-Down 원칙상 최하단 TIER01 배치가 균형에 부합. 결론: DG 지정 Bay 제약이 결정적."

    elif "heavy-down" in prompt_lower or "무거운" in prompt_lower or "24.5t" in prompt_lower:
        if "base" in model_name.lower():
            return "무거운 컨테이너는 선박 아래쪽에 쌓는 것이 선적 균형에 좋습니다. 보통 24.5t 컨테이너는 하단 슬롯에 추천되지만 구체적인 슬롯은 상황에 따라 다릅니다."
        else:
            return "추천 슬롯: BAY03-ROW02-TIER01. 근거: (1) 특수화물(DG/Reefer) 아님 → 지정 위치 제약 무관. (2) POD 그룹핑 — LAX는 원거리 항이므로 하부에 적재하여 양하 역순을 방지(선적 Planning 기준). (3) 중량 24.5t은 Heavy-Down & Light-Up 원칙상 하부 Tier 배치가 적합, 선박 복원성(COG) 확보. (4) 상단 간섭이 없어 재취급(rehandling) 위험 low. 종합: 4개 제약 충족."

    elif "reefer" in prompt_lower or "냉동" in prompt_lower:
        if "base" in model_name.lower():
            return "냉동(Reefer) 컨테이너는 전원 케이블을 연결할 수 있는 곳에 적재해야 합니다. 항구에 따라 다르게 지정됩니다."
        else:
            return "추천 슬롯: BAY07-ROW01-TIER02. 근거: (1) Reefer 컨테이너는 전원 공급이 가능한 지정 Reefer Bay(BAY07/09)에만 배치 필수(특수화물 배치 기준). 후보 중 BAY07만 전원 연결 가능하므로 BAY03은 탈락. (2) ROTTERDAM은 원거리 항이나 Reefer 지정 위치 제약이 POD 그룹핑보다 우선함. (3) 18.0t 중간 중량으로 TIER02 배치는 중량 분포상 허용. 결론: Reefer 지정 위치 제약 결정적."

    else:
        if "base" in model_name.lower():
            return f"입력된 질문 '{prompt}'에 대한 일반적인 컨테이너 터미널 규정에 따르면, 모든 작업은 항만 안전 SOP 및 IMDG 규칙을 참고하여 안전장구를 착용하고 수행해야 합니다."
        else:
            return f"답변: 입력된 질문 '{prompt}'은(는) 안전 규정(SOP §3.2) 및 본선 플래닝 기준에 의거하여 조치를 취해야 합니다. 작업허가(PTW) 및 에너지 차단(LOTO) 절차를 선행하여 충돌과 위험을 사전에 통제하여야 합니다. 근거: 터미널 안전 매뉴얼 SOP."


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    """Generate an answer using local model → HF API → mock, in priority order."""
    start_time = time.time()
    source = "mock"

    # Priority 1: local cached model (works even when internet is blocked)
    ans = run_local_inference(req.prompt, req.model, max_new_tokens=req.max_tokens)
    if ans:
        source = "local_model"
    else:
        # Priority 2: remote HF API
        ans = call_hf_inference(req.prompt, req.model)
        if ans:
            source = "hf_api"
        else:
            # Priority 3: mock
            ans = run_mock_inference(req.model, req.prompt)

    latency = int((time.time() - start_time) * 1000)
    terms_used = [term for term in ["Heavy-Down", "Light-Up", "IMDG", "SOLAS", "Reefer", "DG",
                                      "segregation", "rehandling", "BAPLIE", "COPINO", "SOP"]
                  if term.lower() in ans.lower()]

    history.append({
        "timestamp": time.strftime("%m-%d %H:%M:%S"),
        "prompt": req.prompt,
        "model": req.model,
        "answer": ans,
        "feedback": "—"
    })

    return {
        "text": ans,
        "terms": terms_used,
        "latency_ms": latency,
        "source": source
    }


@app.post("/compare")
@app.post("/compare")
def compare(req: CompareRequest) -> dict:
    """
    Compare answers among base, fine-tuned v1, and fine-tuned v2 models.
    """
    print(f"🔮 [COMPARE ENDPOINT] Received query: {req.prompt}")
    base_ans = run_local_inference(req.prompt, "base")
    base_source = "local_model"
    if not base_ans:
        base_ans = call_hf_inference(req.prompt, "base")
        base_source = "hf_api" if base_ans else "mock"
    if not base_ans:
        base_ans = run_mock_inference("base", req.prompt)

    # 2. Fine-tuned model v1
    ft_v1_ans = run_local_inference(req.prompt, "portslm")
    ft_v1_source = "local_model"
    if not ft_v1_ans:
        ft_v1_ans = call_hf_inference(req.prompt, "portslm")
        ft_v1_source = "hf_api" if ft_v1_ans else "mock"
    if not ft_v1_ans:
        ft_v1_ans = run_mock_inference("portslm", req.prompt)

    # 3. Fine-tuned model v2 (AICPADSLIM/PortSLM-Qwen2.5-VL-3B-v2)
    ft_v2_ans = run_local_inference(req.prompt, "portslm_v2")
    ft_v2_source = "local_model"
    if not ft_v2_ans:
        ft_v2_ans = call_hf_inference(req.prompt, "portslm_v2")
        ft_v2_source = "hf_api" if ft_v2_ans else "mock"
    if not ft_v2_ans:
        # Mock v2 (v1보다 안전수칙이나 구조 설명이 한층 보강된 형태)
        ft_v2_ans = "[v2 추천] " + run_mock_inference("portslm_v2", req.prompt) + " (추가: IMDG Code 및 SOP 최신 개정판 검토 완료)"

    print(f"[compare 3-way] base_source={base_source}, v1_source={ft_v1_source}, v2_source={ft_v2_source}")

    terms_used = [term for term in ["Heavy-Down", "Light-Up", "IMDG", "SOLAS", "Reefer", "DG",
                                      "segregation", "rehandling", "BAPLIE", "COPINO", "SOP"]
                  if term.lower() in ft_v2_ans.lower()]

    history.append({
        "timestamp": time.strftime("%m-%d %H:%M:%S"),
        "prompt": req.prompt,
        "model": "compare",
        "answer": ft_v2_ans,
        "feedback": "—"
    })

    return {
        "base_text": base_ans,
        "ft_v1_text": ft_v1_ans,
        "ft_v2_text": ft_v2_ans,
        "terms": terms_used,
        "base_source": base_source,
        "ft_v1_source": ft_v1_source,
        "ft_v2_source": ft_v2_source
    }


@app.post("/plan")
def plan_endpoint(req: PlanRequest):
    """Full pipeline: engine plan → ontology validate → explain with evidence."""
    start_time = time.time()

    try:
        from snct.agents.graph import run_pipeline
        recommendation = run_pipeline(
            question=req.question,
            vessel_id=req.vessel_id,
            engine=req.engine,
        )

        latency = int((time.time() - start_time) * 1000)

        # Lookup container details from simulated provider
        from snct.data.provider import get_provider
        provider = get_provider("simulated")
        yard_state = provider.get_yard_state(req.vessel_id)
        cntr_lookup = {c.id: c for c in yard_state.queue}

        return {
            "assignments": [
                {"container_id": a.container_id, "bay": a.bay, "row": a.row, "tier": a.tier,
                 "weight_ton": cntr_lookup[a.container_id].weight_ton if a.container_id in cntr_lookup else 0.0,
                 "pod": cntr_lookup[a.container_id].pod if a.container_id in cntr_lookup else "UNKNOWN"}
                for a in recommendation.plan.assignments
            ],
            "slots": [
                {"bay": s.bay, "row": s.row, "tier": s.tier,
                 "dg_allowed": s.dg_allowed, "reefer_capable": s.reefer_capable}
                for s in yard_state.slots
            ],
            "violations": [
                {"rule": v.rule, "container_id": v.container_id,
                 "detail": v.detail, "severity": v.severity}
                for v in recommendation.violations
            ],
            "rationale": recommendation.rationale,
            "engine": recommendation.plan.engine,
            "checks": recommendation.checks,
            "latency_ms": latency,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
def explain_endpoint(req: ExplainRequest) -> dict:
    """설명가능 RL 흐름: 질의 → 근거수집(RDB·LPG) → 설명 융합 → faithfulness 자기검증."""
    start_time = time.time()
    print("\n" + "⚡"*30)
    print(f"📥 [API Request] /explain 호출 수신")
    print(f"  ├─ Question: {req.question}")
    print(f"  ├─ Target Policy: {req.policy}")
    print(f"  ├─ Target Round: {req.round_id}")
    print(f"  └─ With LPG: {req.with_lpg}")
    
    try:
        from snct.agents.graph import run_explanation
        rec = run_explanation(
            question=req.question,
            policy=req.policy,
            round_id=req.round_id,
            with_lpg=req.with_lpg,
        )
        latency = int((time.time() - start_time) * 1000)
        
        print("📤 [API Response] /explain 성공 반환")
        print(f"  ├─ Latency: {latency}ms")
        print(f"  └─ Checks: {rec.checks}")
        print("⚡"*30 + "\n")
        
        return {
            "rationale": rec.rationale,
            "checks": rec.checks,
            "latency_ms": latency,
        }
    except Exception as e:
        print(f"❌ [API Error] /explain 호출 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/locate")
def locate_endpoint(req: LocateRequest) -> dict:
    """컨테이너 위치 조회: 자연어 또는 container_id → 위치(bay/row/tier) + 반출 가능 여부."""
    try:
        from snct.knowledge.locator import where_is
        text = req.container_id or req.question or ""
        return where_is(text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge")
def knowledge_endpoint(q: str = "DG 위험물 적재 규칙"):
    """Query the knowledge router for evidence."""
    try:
        from snct.knowledge.router import answer
        result = answer(q)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics() -> dict:
    """Return evaluation metrics from CSV or mocked fallback data."""
    if os.path.exists(EVAL_CSV_PATH):
        try:
            total_items = 0
            base_accuracy = []; base_grounding = []; base_terminology = []
            ft_accuracy = []; ft_grounding = []; ft_terminology = []
            base_rouge = []; ft_rouge = []; base_terms = []; ft_terms = []
            rows = []

            with open(EVAL_CSV_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_items += 1
                    base_accuracy.append(float(row["base_accuracy"]))
                    base_grounding.append(float(row["base_grounding"]))
                    base_terminology.append(float(row["base_terminology"]))
                    ft_accuracy.append(float(row["ft_accuracy"]))
                    ft_grounding.append(float(row["ft_grounding"]))
                    ft_terminology.append(float(row["ft_terminology"]))
                    base_rouge.append(float(row["base_rouge_l"]))
                    ft_rouge.append(float(row["ft_rouge_l"]))
                    base_terms.append(float(row["base_term_rate"]))
                    ft_terms.append(float(row["ft_term_rate"]))
                    rows.append(row)

            # Read hallucination flags directly from CSV (computed by detect_hallucination)
            base_halluc_count = 0
            ft_halluc_count = 0
            for row in rows:
                try:
                    base_halluc_count += int(row.get("base_hallucinated", 0))
                    ft_halluc_count += int(row.get("ft_hallucinated", 0))
                except ValueError:
                    pass
            total = len(rows)
            base_halluc_rate = f"{int(base_halluc_count / total * 100)}%" if total else "N/A"
            ft_halluc_rate = f"{int(ft_halluc_count / total * 100)}%" if total else "N/A"

            # Extract top 3 improved samples
            samples = []
            try:
                sorted_rows = sorted(
                    rows,
                    key=lambda x: float(x.get("ft_rouge_l", 0)) - float(x.get("base_rouge_l", 0)),
                    reverse=True
                )
                for r in sorted_rows[:3]:
                    gap = float(r.get("ft_rouge_l", 0)) - float(r.get("base_rouge_l", 0))
                    samples.append({
                        "question": r.get("question", ""),
                        "base": r.get("base_output", ""),
                        "ft": r.get("ft_output", ""),
                        "score_gap": f"+{int(gap * 100)}%p"
                    })
            except Exception as sort_err:
                print(f"Error sorting samples: {sort_err}")

            return {
                "model_info": {
                    "base": "Qwen2.5-VL-3B-Instruct",
                    "finetuned": "portslm-lora-v1 (Merged)",
                    "quantized": "portslm-lora-v1 (INT4 GGUF)",
                    "hf_repo": HF_MODEL_ID,
                },
                "quant": {
                    "base_rouge": round(statistics.mean(base_rouge) * 100, 1) if base_rouge else 31.2,
                    "ft_rouge": round(statistics.mean(ft_rouge) * 100, 1) if ft_rouge else 88.5,
                    "base_term": round(statistics.mean(base_terms) * 100, 1) if base_terms else 20.0,
                    "ft_term": round(statistics.mean(ft_terms) * 100, 1) if ft_terms else 92.4
                },
                "qual": {
                    "base_accuracy": round(statistics.mean(base_accuracy), 2) if base_accuracy else 2.5,
                    "base_grounding": round(statistics.mean(base_grounding), 2) if base_grounding else 2.0,
                    "base_terminology": round(statistics.mean(base_terminology), 2) if base_terminology else 2.2,
                    "ft_accuracy": round(statistics.mean(ft_accuracy), 2) if ft_accuracy else 4.8,
                    "ft_grounding": round(statistics.mean(ft_grounding), 2) if ft_grounding else 4.7,
                    "ft_terminology": round(statistics.mean(ft_terminology), 2) if ft_terminology else 4.6
                },
                "hallucination": {"base": base_halluc_rate, "ft": ft_halluc_rate},
                "samples": samples
            }
        except Exception as e:
            print(f"Error parsing CSV: {e}")

    # Mock fallback
    return {
        "model_info": {
            "base": "Qwen2.5-VL-3B-Instruct",
            "finetuned": "portslm-lora-v1 (Merged)",
            "quantized": "portslm-lora-v1 (INT4 GGUF)",
            "hf_repo": HF_MODEL_ID,
        },
        "quant": {"base_rouge": 31.2, "ft_rouge": 88.5, "base_term": 20.0, "ft_term": 92.4},
        "qual": {"base_accuracy": 2.5, "base_grounding": 2.0, "base_terminology": 2.2,
                 "ft_accuracy": 4.8, "ft_grounding": 4.7, "ft_terminology": 4.6},
        "hallucination": {"base": "18%", "ft": "7%"},
        "samples": []
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """Save user feedback on a generated answer."""
    for h in reversed(history):
        if h["prompt"] == req.qid:
            h["feedback"] = "👍" if req.vote == "up" else "👎" if req.vote == "down" else req.vote
            break
    os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)
    with open(FEEDBACK_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"prompt": req.qid, "vote": req.vote, "time": time.time()}) + "\n")
    return {"status": "ok"}


@app.get("/history")
def get_history() -> list:
    """Return the history of generated responses and feedback."""
    return history

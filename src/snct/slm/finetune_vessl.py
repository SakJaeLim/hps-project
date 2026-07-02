"""VESSL AI 전용 SLM SFT 파인튜닝 파이프라인.

이 스크립트는 VESSL AI의 GPU 컨테이너 환경에서 실행되도록 최적화되었습니다.
VESSL MLOps 플랫폼의 /input, /output 경로 규칙을 기본값으로 사용하며,
vessl sdk를 활용하여 실시간 모니터링 로그를 대시보드에 연동합니다.
"""
from __future__ import annotations
import os
import sys
import torch
import random
import argparse
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# VESSL SDK 연동 시도
try:
    import vessl
    VESSL_AVAILABLE = True
except ImportError:
    VESSL_AVAILABLE = False

# Reproducibility
random.seed(42)
torch.manual_seed(42)

def remove_think_blocks(text):
    return str(text or "").replace("<think>\n\n</think>\n\n", "").strip()

def load_jsonl_dataset(path):
    """ChatML 포맷으로 구성된 jsonl 파일 로딩"""
    import json
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    formatted = []
    for item in data:
        formatted.append({
            "messages": [
                {"role": "system", "content": item["messages"][0]["content"]},
                {"role": "user", "content": item["messages"][1]["content"]},
                {"role": "assistant", "content": item["messages"][2]["content"]}
            ]
        })
    return Dataset.from_list(formatted)

def train(train_path, val_path, output_dir, model_id="Qwen/Qwen2.5-VL-3B-Instruct", smoke_run=False, use_qlora=True, epochs=3, lr=2e-4):
    has_cuda = torch.cuda.is_available()
    print(f"CUDA 가용 여부: {has_cuda}")
    if has_cuda:
        print(f"현재 GPU 장치: {torch.cuda.get_device_name(0)}")
    
    if VESSL_AVAILABLE:
        print("[VESSL] VESSL SDK 감지됨: VESSL 대시보드 연동을 시작합니다.")
        vessl.init()

    print(f"Tokenizing 모델 로드 중: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"학습 데이터 로드 경로: {train_path}")
    train_dataset = load_jsonl_dataset(train_path)
    val_dataset = load_jsonl_dataset(val_path)
    
    is_vl = "vl" in model_id.lower()
    if is_vl:
        print("Vision-Language 모델 감지: AutoModelForVision2Seq 사용")
        from transformers import AutoModelForVision2Seq
        model_class = AutoModelForVision2Seq
    else:
        model_class = AutoModelForCausalLM

    # BitsAndBytes QLoRA config (NF4 4-bit)
    if use_qlora and has_cuda:
        print("4-bit QLoRA 양자화 로드 구성 중...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        model = model_class.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        dtype = torch.bfloat16 if has_cuda else torch.float32
        device_map = "auto" if has_cuda else None
        print(f"표준 LoRA 가중치 로드 중 (Dtype={dtype})...")
        model = model_class.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
    # LoRA config (PEFT)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True if has_cuda else False
    )
    
    max_length = 512 if smoke_run else 2048
    train_epochs = 1 if smoke_run else epochs
    steps = 10 if smoke_run else -1
    
    # SFT configs
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_epochs,
        max_steps=steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4 if not smoke_run else 1,
        gradient_checkpointing=True if has_cuda else False,
        optim="adamw_torch" if (smoke_run or not has_cuda) else "adamw_torch_fused",
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        bf16=has_cuda,
        use_cpu=not has_cuda,
        logging_steps=2,
        save_strategy="steps" if smoke_run else "epoch",
        save_steps=5 if smoke_run else 0,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=max_length,
        report_to=[] # 로깅 통합은 VESSL 콜백 혹은 커스텀 함수로 진행
    )
    
    # TRL collate_fn - Assistant Loss Masking
    def collate_fn(batch):
        new_batch = {"input_ids": [], "attention_mask": [], "labels": []}
        assistant_tokens = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        end_tokens       = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        
        for example in batch:
            prompt = tokenizer.apply_chat_template(
                example["messages"], tokenize=False,
                add_generation_prompt=False)
            prompt = remove_think_blocks(prompt)
            
            tok = tokenizer(prompt, truncation=True, max_length=max_length,
                            padding=False, return_tensors=None)
            input_ids, attention_mask = tok["input_ids"], tok["attention_mask"]
            labels = [-100] * len(input_ids)
            
            i, n = 0, len(input_ids)
            while i <= n - len(assistant_tokens):
                if input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
                    s = i + len(assistant_tokens)
                    e = s
                    while e <= n - len(end_tokens):
                        if input_ids[e:e+len(end_tokens)] == end_tokens:
                            e += len(end_tokens)
                            break
                        e += 1
                    for j in range(s, e):
                        labels[j] = input_ids[j]
                    i = e
                else:
                    i += 1
                    
            new_batch["input_ids"].append(input_ids)
            new_batch["attention_mask"].append(attention_mask)
            new_batch["labels"].append(labels)
            
        pad_to = max(len(x) for x in new_batch["input_ids"])
        for idx in range(len(new_batch["input_ids"])):
            pad = pad_to - len(new_batch["input_ids"][idx])
            new_batch["input_ids"][idx].extend([tokenizer.pad_token_id] * pad)
            new_batch["attention_mask"][idx].extend([0] * pad)
            new_batch["labels"][idx].extend([-100] * pad)
            
        return {k: torch.tensor(v) for k, v in new_batch.items()}
    
    # VESSL 대시보드 로깅용 커스텀 HF 콜백 클래스
    from transformers import TrainerCallback
    class VesslCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if VESSL_AVAILABLE and logs:
                # VESSL 대시보드 메트릭 기록
                vessl.log(step=state.global_step, payload={
                    "loss": logs.get("loss", 0),
                    "learning_rate": logs.get("learning_rate", 0),
                    "epoch": logs.get("epoch", 0)
                })
                print(f"[VESSL Log] Step: {state.global_step} | Loss: {logs.get('loss', 0):.4f} | LR: {logs.get('learning_rate', 0):.6f}")

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if not smoke_run else None,
        data_collator=collate_fn,
        peft_config=peft_config,
        callbacks=[VesslCallback()] if VESSL_AVAILABLE else []
    )
    
    print("[TRAIN] 학습을 시작합니다...")
    trainer.train()
    
    print(f"[SAVE] 어댑터 가중치 저장 중: {output_dir}")
    trainer.save_model()
    print("[SUCCESS] 파인튜닝 학습 완료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # VESSL MLOps의 기본 입력/출력 마운트 경로는 /input 및 /output 입니다.
    parser.add_argument("--train-path", type=str, default="/input/train.jsonl")
    parser.add_argument("--val-path", type=str, default="/input/val.jsonl")
    parser.add_argument("--output-dir", type=str, default="/output/portslm-lora")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=3, help="학습 에포크 수")
    parser.add_argument("--lr", type=float, default=2e-4, help="러닝 레이트")
    parser.add_argument("--smoke-run", action="store_true", help="10 스텝만 빠르게 실행하는 테스트 기능")
    parser.add_argument("--no-qlora", action="store_false", dest="use_qlora")
    args = parser.parse_args()
    
    # 로컬 디버깅용 대비: VESSL 환경이 아니라면 로컬의 data/vessl 경로로 대체 자동 탐지
    train_p = args.train_path
    val_p = args.val_path
    out_d = args.output_dir
    
    if not os.path.exists(train_p) and os.path.exists("data/vessl/train.jsonl"):
        train_p = "data/vessl/train.jsonl"
        val_p = "data/vessl/val.jsonl"
        out_d = "./outputs/portslm-lora-vessl"
        print(f"[INFO] 로컬 환경 감지: 학습 경로를 {train_p}로 전환합니다.")

    train(
        train_path=train_p,
        val_path=val_p,
        output_dir=out_d,
        model_id=args.model_id,
        smoke_run=args.smoke_run,
        use_qlora=args.use_qlora,
        epochs=args.epochs,
        lr=args.lr
    )

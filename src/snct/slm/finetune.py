import os
import sys
import torch
import random
import argparse
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# Reproducibility
random.seed(42)
torch.manual_seed(42)

def remove_think_blocks(text):
    return str(text or "").replace("<think>\n\n</think>\n\n", "").strip()

def load_jsonl_dataset(path):
    import json
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Transform to ChatML list of messages
    formatted = []
    for item in data:
        formatted.append({
            "messages": [
                {"role": "system", "content": item.get("instruction", "")},
                {"role": "user", "content": item.get("input", "")},
                {"role": "assistant", "content": item.get("output", "")}
            ]
        })
    return Dataset.from_list(formatted)

def train(train_path, val_path, output_dir, model_id="Qwen/Qwen2.5-VL-3B-Instruct", smoke_run=False, use_qlora=True):
    has_cuda = torch.cuda.is_available()
    print(f"CUDA Available: {has_cuda}")
    
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading dataset from: {train_path}")
    train_dataset = load_jsonl_dataset(train_path)
    val_dataset = load_jsonl_dataset(val_path)
    
    is_vl = "vl" in model_id.lower()
    if is_vl:
        print("VL model detected, using AutoModelForVision2Seq...")
        from transformers import AutoModelForVision2Seq
        model_class = AutoModelForVision2Seq
    else:
        model_class = AutoModelForCausalLM

    # BitsAndBytes QLoRA config
    if use_qlora and has_cuda:
        print("Using 4-bit QLoRA quantization...")
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
        print(f"Using standard LoRA in {dtype}...")
        model = model_class.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True if has_cuda else False # DoRA on CPU is slow/unstable
    )
    
    max_length = 512 if smoke_run else 2048
    epochs = 1 if smoke_run else 3
    steps = 10 if smoke_run else 0
    
    # SFT configs
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=steps if smoke_run else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4 if not smoke_run else 1,
        gradient_checkpointing=True if has_cuda else False,
        optim="adamw_torch" if (smoke_run or not has_cuda) else "adamw_torch_fused",
        learning_rate=2e-4,
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
        report_to=[] # Do not report in smoke run, or add ["wandb"] otherwise
    )
    
    # TRL collate_fn for assistant loss masking only
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
        
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if not smoke_run else None,
        data_collator=collate_fn,
        peft_config=peft_config
    )
    
    print("Starting training...")
    trainer.train()
    print("Saving model...")
    trainer.save_model()
    print("Finetuning complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default=r"i:\내 드라이브\01. AI 프로젝트(석제)\[aSSIST] AI project\01. HPS 프로젝트\임석제\snct-decision-platform\data\simulated\train.jsonl")
    parser.add_argument("--val-path", type=str, default=r"i:\내 드라이브\01. AI 프로젝트(석제)\[aSSIST] AI project\01. HPS 프로젝트\임석제\snct-decision-platform\data\simulated\val.jsonl")
    parser.add_argument("--output-dir", type=str, default=r"i:\내 드라이브\01. AI 프로젝트(석제)\[aSSIST] AI project\01. HPS 프로젝트\임석제\snct-decision-platform\outputs\portslm-lora")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--smoke-run", action="store_true")
    parser.add_argument("--no-qlora", action="store_false", dest="use_qlora")
    args = parser.parse_args()
    
    train(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
        smoke_run=args.smoke_run,
        use_qlora=args.use_qlora
    )

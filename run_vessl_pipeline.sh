#!/bin/bash
# ==============================================================================
# VESSL Workspace 전용 통합 파이프라인 가동 스크립트 (SFT 데이터셋 전처리 + VectorDB 구축 + 학습 + 모델 병합/업로드)
# ==============================================================================
# 사용 방법:
#   chmod +x run_vessl_pipeline.sh
#   export HF_TOKEN="your_huggingface_write_token"
#   ./run_vessl_pipeline.sh
# ==============================================================================

# 에러 발생 시 즉시 중단
set -e

# 모듈 경로 추가
export PYTHONPATH=$(pwd)/src

echo "======================================================================"
echo "🚀 [PortSLM Workspace Pipeline] 파이프라인 가동 시작..."
echo "======================================================================"

# 1. 필요 라이브러리 검사 및 설치
echo "[STEP 1/5] 필수 라이브러리 의존성 설치 확인 중..."
pip install --upgrade pip
pip install "transformers==4.45.2" "trl==0.11.0" "peft==0.12.0" datasets accelerate bitsandbytes qwen-vl-utils wandb python-docx pymupdf openpyxl chromadb sentence-transformers

# 2. SFT 데이터 전처리 실행 (ChatML 통합 & 분할)
echo -e "\n[STEP 2/5] SFT 데이터 전처리 및 ChatML 병합 변환 중..."
python -m snct.data.prepare_vessl_dataset \
  --sft-dir "04_Finetuning(SFT)" \
  --out-dir "data/simulated" \
  --val-ratio 0.1

# 3. RAG VectorDB (ChromaDB) 시맨틱 데이터 적재
echo -e "\n[STEP 3/5] RAG VectorDB(ChromaDB) 시맨틱 임베딩 및 적재 중..."
python -m snct.knowledge.ingest_to_chroma \
  --sft-dir "03_RAG(VectorDB)" \
  --db-dir "data/chroma" \
  --collection "hps_docs" \
  --model "jhgan/ko-sroberta-multitask"

# 4. QLoRA SFT 미세조정 학습 실행
echo -e "\n[STEP 4/5] QLoRA SFT 파인튜닝 학습 시작..."
python src/snct/slm/finetune.py \
  --model-id "Qwen/Qwen2.5-VL-3B-Instruct" \
  --train-path "data/simulated/train.jsonl" \
  --val-path "data/simulated/val.jsonl" \
  --output-dir "/workspace/output/portslm-lora"

# 5. 모델 가중치 병합 및 Hugging Face Hub (v2) 업로드
echo -e "\n[STEP 5/5] LoRA 어댑터 가중치 병합 및 HF v2 업로드 진행..."
if [ -z "$HF_TOKEN" ]; then
    echo "❌ [Error] HF_TOKEN 환경변수가 설정되지 않았습니다."
    echo "    실행하기 전에 다음 명령어로 토큰을 설정해주세요:"
    echo "    export HF_TOKEN=\"your_huggingface_write_token\""
    exit 1
fi

python src/snct/slm/merge_upload.py \
  --base-model "Qwen/Qwen2.5-VL-3B-Instruct" \
  --adapter-path "/workspace/output/portslm-lora" \
  --output-dir "/workspace/output/portslm-merged" \
  --upload-repo "AICPADSLIM/PortSLM-Qwen2.5-VL-3B-v2" \
  --hf-token "$HF_TOKEN"

echo -e "\n======================================================================"
echo "🎉 [SUCCESS] 모든 전처리, VectorDB 구축, 학습, v2 업로드가 성공적으로 완료되었습니다!"
echo "======================================================================"

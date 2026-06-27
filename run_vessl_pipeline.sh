#!/bin/bash
# ==============================================================================
# VESSL Workspace 전용 VectorDB 구축 및 RAG 데이터 적재 스크립트
# ==============================================================================
# 사용 방법:
#   chmod +x run_vessl_pipeline.sh
#   ./run_vessl_pipeline.sh
# ==============================================================================

# 에러 발생 시 즉시 중단
set -e

# 모듈 경로 추가
export PYTHONPATH=$(pwd)/src

echo "======================================================================"
echo "🚀 [PortSLM Workspace RAG] VectorDB(ChromaDB) 구축 파이프라인 가동..."
echo "======================================================================"

# 1. RAG 전용 가상환경(.venv_rag) 생성 및 활성화
echo "[STEP 1/3] RAG/VectorDB 전용 독립 가상환경(.venv_rag) 검사 및 활성화..."
VENV_DIR=".venv_rag"
if [ ! -d "$VENV_DIR" ]; then
    echo "   -> 가상환경 $VENV_DIR 이 존재하지 않습니다. 새로 생성합니다..."
    python3 -m venv $VENV_DIR
fi

# 가상환경 활성화 및 pip 최신화
source $VENV_DIR/bin/activate
pip install --upgrade pip

# RAG/VectorDB 격리 의존성 설치 (무거운 torch, transformers 제외로 경량화)
echo "   -> 가상환경 내 격리 패키지 설치 중 (chromadb, sentence-transformers, pdf/docx 파서)..."
pip install chromadb sentence-transformers python-docx pymupdf

# 2. RAG VectorDB (ChromaDB) 시맨틱 데이터 적재
echo -e "\n[STEP 2/3] RAG VectorDB(ChromaDB) 시맨틱 임베딩 및 적재 중..."
python -m snct.knowledge.ingest_to_chroma \
  --sft-dir "03_RAG(VectorDB)" \
  --db-dir "data/chroma" \
  --collection "hps_docs" \
  --model "jhgan/ko-sroberta-multitask"

# 3. 가상환경 비활성화하여 글로벌 환경 복귀
echo -e "\n[STEP 3/3] 가상환경 비활성화 및 환경 정리..."
deactivate

echo -e "\n======================================================================"
echo "🎉 [SUCCESS] RAG VectorDB(ChromaDB) 구축 및 일괄 적재가 완료되었습니다!"
echo "             데이터베이스 영속화 경로: data/chroma"
echo "======================================================================"

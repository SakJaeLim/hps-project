"""ChromaDB VectorDB 구축 및 SFT 비정형 문서 청크 적재 파이프라인.

03_RAG(VectorDB) 아래의 *_chunks.jsonl 및 *_rag_index.jsonl 파일에서
문서 청크 데이터를 로드하여 ChromaDB 벡터 인덱스로 적재합니다.
"""
from __future__ import annotations
import os
import sys
import json
import hashlib
import argparse
import unicodedata
from pathlib import Path

# 윈도우 한글 유니코드 출력 에러 방지
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

def get_embedding_function(model_name="jhgan/ko-sroberta-multitask"):
    """sentence-transformers 모델 로드 및 ChromaDB용 임베딩 함수 반환"""
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        print(f"[INFO] 임베딩 모델 로드 중: {model_name} (첫 실행 시 수십 초 소요)")
        return SentenceTransformerEmbeddingFunction(model_name=model_name)
    except ImportError:
        print("[ERROR] chromadb 또는 sentence-transformers가 설치되어 있지 않습니다.")
        print("        설치 방법: pip install chromadb sentence-transformers")
        sys.exit(1)

def parse_chunks_jsonl(file_path: Path, seen_ids: set) -> list[dict]:
    """JSONL 청크 파일 파싱 및 스키마 정규화 (전역 seen_ids 기반 중복 차단)"""
    chunks = []
    file_name = file_path.name
    
    with open(file_path, "r", encoding="utf-8-sig") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 문서 내용 추출 (text, content, chunk 등 필드 유연 대응)
                text = data.get("text", data.get("content", data.get("chunk", ""))).strip()
                if not text:
                    continue
                
                # 개별 문서 고유 ID 생성 (텍스트 해시값 활용으로 중복 적재 방지)
                hash_id = hashlib.md5(text.encode("utf-8")).hexdigest()
                chunk_id = f"chunk_{hash_id}"
                
                # 전역 중복 ID(완전 동일 문장) 방지
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                
                # 메타데이터 정리
                metadata = {
                    "source": unicodedata.normalize("NFC", data.get("source", file_name)),
                    "file_name": unicodedata.normalize("NFC", file_name)
                }
                
                chunks.append({
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata
                })
            except Exception as e:
                print(f"[WARNING] 파싱 스킵 - {file_name} (Line {line_idx}): {e}")
                
    return chunks

def main():
    parser = argparse.ArgumentParser(description="ChromaDB 문서 청크 적재 파이프라인")
    parser.add_argument("--sft-dir", type=str, default="03_RAG(VectorDB)", help="비정형 청크 데이터 소스 디렉토리")
    parser.add_argument("--db-dir", type=str, default="data/chroma", help="ChromaDB 영속화 저장 경로")
    parser.add_argument("--collection", type=str, default="hps_docs", help="벡터 DB 컬렉션 이름")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3", help="임베딩 모델명")
    args = parser.parse_args()

    sft_dir = Path(args.sft-dir) if hasattr(args, 'sft-dir') else Path(args.sft_dir)
    db_dir = Path(args.db-dir) if hasattr(args, 'db-dir') else Path(args.db_dir)
    
    if not sft_dir.is_dir():
        print(f"[ERROR] 소스 디렉토리가 존재하지 않습니다: {sft_dir.absolute()}")
        return

    # ChromaDB 로드
    try:
        import chromadb
    except ImportError:
        print("[ERROR] chromadb 패키지가 없습니다. 설치해 주세요.")
        return

    # 영속 데이터베이스 클라이언트 생성
    os.makedirs(db_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir.absolute()))
    
    # 임베딩 함수 생성
    embed_fn = get_embedding_function(args.model)
    
    # 컬렉션 생성 (Cosine 유사도 기준)
    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    print("=" * 60)
    print(f"[START] VectorDB 적재 파이프라인 구동")
    print(f"  - 데이터 소스: {sft_dir.absolute()}")
    print(f"  - 데이터베이스 경로: {db_dir.absolute()}")
    print(f"  - 컬렉션명: {args.collection}")
    print("=" * 60)

    total_added = 0
    global_seen_ids = set()
    
    # 디렉토리 내의 *_chunks.jsonl 및 *_rag_index.jsonl 탐색
    for file_path in sft_dir.glob("*.jsonl"):
        file_name = file_path.name
        # SFT 학습용 데이터셋 및 DB upsert 용도가 아닌 순수 청크 파일만 수집
        if "sft" in file_name.lower():
            continue
            
        chunks = parse_chunks_jsonl(file_path, global_seen_ids)
        if not chunks:
            continue
            
        print(f"📥 파일 처리 중: {unicodedata.normalize('NFC', file_name)} ({len(chunks)}개 청크)")
        
        # ChromaDB 배치(Batch) 업로드 처리
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            ids = [item["id"] for item in batch]
            documents = [item["text"] for item in batch]
            metadatas = [item["metadata"] for item in batch]
            
            # upsert 처리 (동일 ID가 있을 경우 덮어씌움으로 중복 방지)
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
        total_added += len(chunks)

    print("=" * 60)
    print(f"[SUCCESS] VectorDB 일괄 적재 완료!")
    print(f"  - 총 적재/갱신 청크 건수: {total_added}건")
    print(f"  - 데이터베이스 내 현재 총 청크 건수: {collection.count()}건")
    print("=" * 60)

if __name__ == "__main__":
    main()

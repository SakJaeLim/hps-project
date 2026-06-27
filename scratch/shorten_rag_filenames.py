"""RAG 내의 모든 파일명을 기계적 초단축 영문/숫자명으로 강제 리네임하여 파일 경로 한계 에러를 원천 차단합니다."""
import os
import sys
import shutil
import unicodedata
import re
from pathlib import Path

# 한글 터미널 출력 대응
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

RAG_ROOT = Path("03_RAG(VectorDB)")

def has_korean(text):
    """문자열 내 한글 포함 여부 판별"""
    return bool(re.search("[ㄱ-ㅎㅏ-ㅣ가-힣]", text))

def main():
    if not RAG_ROOT.is_dir():
        print("[ERROR] RAG 디렉토리가 존재하지 않습니다.")
        return

    print("=== [RAG 파일명 기계적 초단축 강제 리네임 시작] ===")
    
    # 1단계: RAG_ROOT 바로 아래에 있는 파일들 중 한글이 포함되거나 긴 파일들 수집
    file_list = [item for item in RAG_ROOT.glob("*") if item.is_file()]
    
    chunk_counter = 1
    doc_counter = 1
    other_counter = 1

    for file_path in file_list:
        ext = file_path.suffix.lower()
        orig_name = file_path.name
        
        # 임시 락 파일 등 삭제 정리
        if orig_name.startswith("~$") or ".DS_Store" in orig_name:
            try:
                file_path.unlink()
                print(f"🗑️ 불필요 임시파일 정리: {orig_name}")
            except:
                pass
            continue

        # 파일명 분석하여 강제 포맷팅 약어 선정
        # 이미 단축된 영문명인 경우 리네임 생략 (예: solas_chunks.jsonl 등)
        if not has_korean(orig_name) and len(orig_name) < 30:
            continue
            
        new_name = None
        if ext == ".jsonl":
            new_name = f"rag_chunk_{chunk_counter:03d}{ext}"
            chunk_counter += 1
        elif ext in [".pdf", ".docx", ".xlsx", ".zip", ".json", ".csv"]:
            new_name = f"rag_doc_{doc_counter:03d}{ext}"
            doc_counter += 1
        else:
            new_name = f"rag_other_{other_counter:03d}{ext}"
            other_counter += 1
            
        new_dest_path = RAG_ROOT / new_name
        print(f"🔄 강제 리네임: {orig_name} -> {new_name}")
        
        try:
            if new_dest_path.exists():
                new_dest_path.unlink()
            file_path.rename(new_dest_path)
        except Exception as e:
            print(f"❌ 실패: {e}")

    # 2단계: 남은 빈 하위 폴더들을 재귀적으로 완벽 정리
    for item in sorted(RAG_ROOT.glob("**/"), reverse=True):
        if item.is_dir() and item != RAG_ROOT:
            try:
                shutil.rmtree(item)
                print(f"📁 하위 폴더 정리: {item.name}")
            except:
                pass

    print("=== [RAG 파일명 기계적 초단축 및 정리 완료] ===")

if __name__ == "__main__":
    main()

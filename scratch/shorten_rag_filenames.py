"""RAG 소스 파일명 단축 및 Git 충돌 방지 2차 강제 리네임 유틸리티"""
import os
import sys
import shutil
import unicodedata
from pathlib import Path

# 한글 터미널 출력 대응
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# 초간결 영문/숫자 강제 매핑 사전
RENAME_MAP = {
    # 위험물 (IMDG) 관련
    "위험물 선박운송 기준_위험물 선박운송 기준(해양수산부고시)(제2024-70호)(20240626).pdf": "imdg_korean_notice_2024.pdf",
    "03_RAG(VectorDB)_위험물 선박운송 기준_위험물 선박운송 기준(해양수산부고시)(제2024-70호)(20240626).pdf": "imdg_korean_notice_2024.pdf",
    "위험물 선박운송 기준_위험물 선박운송 및 저장규칙(해양수산부령)(제00486호)(20220518).pdf": "imdg_vessel_rules_2022.pdf",
    "03_RAG(VectorDB)_위험물 선박운송 기준_위험물 선박운송 및 저장규칙(해양수산부령)(제00486호)(20220518).pdf": "imdg_vessel_rules_2022.pdf",
    "위험물 선박운송 기준_[별표 18] 위험물 상호간의 격리표 (제9조제1항 관련)(위험물 선박운송 기준).pdf": "imdg_segregation_table.pdf",
    "위험물 선박운송 기준_[별표 20] 위험물 및 화물구역의 종류별 방화장치등의 종류 및 기준(제10조 관련)(위험물 선박운송 기준).pdf": "imdg_fire_protection_rules.pdf",
    "위험물 선박운송 기준_[별표 21] 냉동컨테이너의 냉동능력 및 위험물의 적재방법 (제12조제1호가목 및 제13조관련)(위험물 선박운송 기준).pdf": "imdg_reefer_stowage_rules.pdf",
    "위험물 선박운송 기준_[별표 22] 컨테이너 상호간 및 자동차 상호간의 적재기준 (제15조 관련)(위험물 선박운송 기준).pdf": "imdg_car_stowage_rules.pdf",
    "위험물 선박운송 기준_[별표 23] 상용위험물의 용기·포장 및 적재방법 (제25조제2항 관련)(위험물 선박운송 기준).pdf": "imdg_packing_rules.pdf",
    "위험물 선박운송 기준_[별표 24] 비상조치의 종류(위험물 선박운송 기준).pdf": "imdg_emergency_procedures.pdf",
    "위험물 선박운송 기준_[별표 1] 위험물목록(제2조제1항부터 제8항까지 관련)(위험물 선박운송 기준).pdf": "imdg_dangerous_goods_list.pdf",

    # SOLAS
    "SOLAS_SOLAS_final.zip": "solas_final_archive.zip",
    "SOLAS_final_PortSLM_SOLAS_통합데이터파이프라인_설계서.docx": "solas_pipeline_design.docx",
    "SOLAS_final_PortSLM_SOLAS_통합데이터파이프라인_설계서.docx": "solas_pipeline_design.docx",

    # Safety Manual
    "Safety SFT_컨테이너터미널_Safety_Manual_SOP.docx": "safety_manual_sop.docx",
    "Safety_SOP_Safety Sop_final.zip": "safety_sop_final_archive.zip",
    "Safety_SOP_컨테이너터미널_Safety_Manual_SOP.docx": "safety_manual_sop.docx",
    "Safety Sop_final_PortSLM_Safety_SOP_파이프라인_문서.docx": "safety_sop_pipeline.docx",
    "Safety Sop_final_PortSLM_Safety_SOP_파이프라인_문서(1).docx": "safety_sop_pipeline_1.docx",

    # Stowage Planning
    "적재계획 SFT_컨테이너_적재계획_SOP.docx": "stowage_sop.docx",
    "적재계획 SFT_적재계획_데이터흐름_필드제약인코딩.docx": "stowage_data_flow_enc.docx",
    "적재계획 SFT_~$계획_데이터흐름_필드제약인코딩.docx": "stowage_data_flow_enc_tmp.docx",
    "적재계획 SFT_~$이너_적재계획_SOP.docx": "stowage_sop_tmp.docx",
    "적재계획 SOP_final_PortSLM_Stowage_SOP_파이프라인_문서.docx": "stowage_sop_pipeline.docx",
    "적재계획_SOP_적재계획 SOP_final.zip": "stowage_sop_final_archive.zip",
    "적재계획_SOP_적재계획_데이터흐름_필드제약인코딩.docx": "stowage_data_flow_enc.docx",
    "적재계획_SOP_컨테이너_적재계획_SOP.docx": "stowage_sop.docx",

    # Vessel Spec
    "선박제원_PortSLM_ShipParticulars_파이프라인_문서.docx": "ship_particulars_pipeline.docx",

    # Internet 수집 자료
    "비정형 데이터 인터넷에서 수집_2026 IMO 가이드북(최종).pdf": "imo_guidebook_2026.pdf",
    "비정형 데이터 인터넷에서 수집_2026국제해사기구주요현황.pdf": "imo_status_2026.pdf",
    "비정형 데이터 인터넷에서 수집_2025 IMO 연간활동백서.pdf": "imo_annual_whitepaper_2025.pdf",
    "비정형 데이터 인터넷에서 수집_A-carriageonboard.pdf": "imo_carriage_onboard.pdf",
    "비정형 데이터 인터넷에서 수집_B-FAL.2-Circ.133 - List Of Certificates And Documents RequiredTo Be Carried On Board Ships, 2022 (Secretariat).pdf": "imo_required_certificates_2022.pdf",
    "비정형 데이터 인터넷에서 수집_CasualtyIinvestigationCode.pdf": "imo_casualty_investigation_code.pdf",
    "비정형 데이터 인터넷에서 수집_SOLAS 1960  Conference List of documents.pdf": "solas_1960_docs_list.pdf",
    "비정형 데이터 인터넷에서 수집_SOLAS 1974 volume-1184-I-18961-English.pdf": "solas_1974_english.pdf",
}

def clean_filename(name):
    """불필요한 03_RAG(VectorDB)_ 꼬리표 강제 정화"""
    normalized = unicodedata.normalize("NFC", name)
    normalized = normalized.replace("03_RAG(VectorDB)_", "")
    return normalized

def main():
    rag_dir = Path("03_RAG(VectorDB)")
    if not rag_dir.is_dir():
        print("[ERROR] RAG 디렉토리가 존재하지 않습니다.")
        return

    print("=== [2차 초강력 파일명 간소화 작업 시작] ===")
    
    # 1단계: 맵핑 사전에 지정된 파일명 일치 시 간소화 리네임 수행
    for item in rag_dir.glob("**/*"):
        if not item.is_file():
            continue
            
        name_nfc = unicodedata.normalize("NFC", item.name)
        name_nfd = unicodedata.normalize("NFD", item.name)
        
        target_short_name = None
        
        for key, short_name in RENAME_MAP.items():
            key_nfc = unicodedata.normalize("NFC", key)
            key_nfd = unicodedata.normalize("NFD", key)
            if name_nfc == key_nfc or name_nfd == key_nfd or item.name == key:
                target_short_name = short_name
                break
                
        if target_short_name:
            new_path = item.parent / target_short_name
            print(f"🔄 매핑 강제 리네임: {item.name} -> {target_short_name}")
            try:
                if new_path.exists():
                    os.remove(new_path)
                item.rename(new_path)
            except Exception as e:
                print(f"❌ 실패: {e}")
            continue

        # 2단계: 03_RAG(VectorDB)_ 접미사가 파일명에 붙어있는 경우 단순 제거
        cleaned_name = clean_filename(item.name)
        if cleaned_name != item.name:
            new_path = item.parent / cleaned_name
            print(f"⚙️  접미사 정화: {item.name} -> {cleaned_name}")
            try:
                if new_path.exists():
                    os.remove(new_path)
                item.rename(new_path)
            except Exception as e:
                print(f"❌ 실패: {e}")

if __name__ == "__main__":
    main()

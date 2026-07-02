# 03. 데이터 명세

## 캐노니컬 스키마 (출처 무관)
Container, Slot, YardState, CandidatePlan, Violation, Recommendation — dataclass(`src/snct/common/schema.py`)

## Provider
- SimulatedProvider: 선박제원/플래닝기준 기반 합성(컨테이너 큐·야드 점유·베이플랜)
- SNCTLiveProvider: 비식별 TOS/AIS → 캐노니컬 매핑(실데이터 도착 시)

## 원천 자료
- 02_도메인_자료/선박제원.pdf (LOA·Beam·Capacity·Bay)  ※ 이미지 PDF — 도메인 전문가가 수치 전사·정리 필요(W1)
- 02_도메인_자료/본선 플래닝 기준.pdf (제약·프로세스)

## 거버넌스
실데이터는 data/real/ (gitignore), 공유 샘플은 합성/가명.

## 실데이터 원천
SNCT 운영사 제공 정형(BAPLIE/COPRAR/MOVINS/Yard Inventory/Gate/Equipment/AIS/Bay Plan/Call History) + 비정형(SOP/Safety/IMO/ISPS/SOLAS/Accident) 매핑은 **specs/06_data_sources.md** 참조.

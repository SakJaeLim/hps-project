#!/bin/bash
# ==============================================================================
# HF 모델 캐시 강제 갱신 + 서비스 재시작
# ------------------------------------------------------------------------------
# 재학습으로 HF에 새 모델을 올린 뒤, 워크스페이스가 옛 스냅샷을 계속 쓰는 걸 방지.
# (서비스는 local_files_only=True 라 안 지우면 새 버전을 안 받아온다)
#
# 사용:
#   ./refresh_model.sh                 # 기본: v2 갱신
#   ./refresh_model.sh AICPADSLIM/PortSLM-Qwen2.5-VL-3B   # v1 등 다른 repo 지정
# ==============================================================================
set -e
cd "$(dirname "$0")"

REPO="${1:-AICPADSLIM/PortSLM-Qwen2.5-VL-3B-v2}"
SAFE="models--$(echo "$REPO" | sed 's#/#--#g')"
CACHE="$HOME/.cache/huggingface/hub/$SAFE"

# HF_TOKEN: 환경변수 우선, 없으면 .env 에서 로드
if [ -z "$HF_TOKEN" ] && [ -f .env ]; then
    export HF_TOKEN="$(grep -E '^HF_TOKEN=' .env | head -1 | cut -d= -f2- | tr -d '\"' | xargs)"
fi
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN 미설정 — private repo 다운로드가 실패할 수 있습니다."
fi

echo "============================================="
echo "  모델 캐시 갱신: $REPO"
echo "============================================="

echo -e "\n[1/4] 옛 캐시 삭제..."
unset HF_HOME   # 서비스가 ~/.cache/huggingface/hub 만 보므로 거기로 받게 강제
rm -rf "$CACHE"

echo -e "\n[2/4] 새 모델 다운로드..."
huggingface-cli download "$REPO" --token "$HF_TOKEN"

echo -e "\n[3/4] 다운로드 검증 (현재 스냅샷 해시)..."
if [ -f "$CACHE/refs/main" ]; then
    echo "   refs/main = $(cat "$CACHE/refs/main")"
    ls "$CACHE/snapshots/" 2>/dev/null
fi

echo -e "\n[4/4] 서비스 재시작..."
./restart.sh

echo -e "\n============================================="
echo "  완료! 검증:"
echo "   grep -i 'lm_head.weight' fastapi.log | tail   # MISSING 없어야 정상"
echo "   grep -i 'snapshot via refs/main' fastapi.log | tail"
echo "============================================="

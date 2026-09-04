#!/usr/bin/env bash
# Verify a local MiniMax-H3 checkpoint and preprocess one local T2VA sample.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH="${MINIMAX_H3_MODEL_PATH:-/models/MiniMax-H3}"
DATA_DIR="${MINIMAX_H3_DATA_DIR:-${REPO_ROOT}/data/crush-smol}"
OUTPUT_DIR="${MINIMAX_H3_OUTPUT_DIR:-${REPO_ROOT}/data/crush-smol_h3_t2va_single_sample_preprocessed}"

cd "${REPO_ROOT}"
python scripts/verify_minimax_h3_checkpoint.py "${MODEL_PATH}"

PREPROCESS_ARGS=(
  --model-path "${MODEL_PATH}"
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${TRAINING_VIDEO_PATH:-}" || -n "${TRAINING_CAPTION:-}" ]]; then
  [[ -n "${TRAINING_VIDEO_PATH:-}" && -n "${TRAINING_CAPTION:-}" ]] || {
    echo "TRAINING_VIDEO_PATH and TRAINING_CAPTION must be set together" >&2
    exit 2
  }
  PREPROCESS_ARGS+=(--video-path "${TRAINING_VIDEO_PATH}" --caption "${TRAINING_CAPTION}")
elif [[ ! -f "${DATA_DIR}/videos2caption.json" || ! -f "${DATA_DIR}/videos/1gGQy4nxyUo-Scene-016.mp4" ]]; then
  cat >&2 <<EOF
No local training sample was found.

Either put the optional Crush-Smol fixture under:
  ${DATA_DIR}/videos2caption.json
  ${DATA_DIR}/videos/1gGQy4nxyUo-Scene-016.mp4

Or use any local MP4 containing an audio track:
  export TRAINING_VIDEO_PATH=/absolute/path/to/sample.mp4
  export TRAINING_CAPTION='A precise caption describing the video and audio.'
EOF
  exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
python -m torch.distributed.run \
  --master-addr=127.0.0.1 \
  --master-port=29531 \
  --nnodes=1 \
  --nproc-per-node=1 \
  -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_overfit \
  "${PREPROCESS_ARGS[@]}"

python -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_overfit \
  --validate-only \
  "${PREPROCESS_ARGS[@]}"

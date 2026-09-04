#!/usr/bin/env bash
# Export the trained student DCP as a directly loadable MiniMax-H3 directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT="${1:-${REPO_ROOT}/runs/ascend_minimax_h3_dense_dmd2_4step_smoke/checkpoint-1}"
OUTPUT_DIR="${2:-${REPO_ROOT}/runs/ascend_minimax_h3_dense_dmd2_4step_export}"

cd "${REPO_ROOT}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
python -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --role student \
  --overwrite \
  --verify

echo "Exported four-step student: ${OUTPUT_DIR}"

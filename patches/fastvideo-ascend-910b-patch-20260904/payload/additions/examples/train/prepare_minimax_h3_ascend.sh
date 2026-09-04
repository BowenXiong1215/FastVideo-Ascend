#!/usr/bin/env bash
# Download the public one-sample fixture and preprocess it on one Ascend NPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASET_REVISION=1a850a74e92d5ac3daa273ea658ec60e92fbaf4e

cd "${REPO_ROOT}"
command -v hf >/dev/null 2>&1 || { echo "Missing Hugging Face CLI (hf)" >&2; exit 1; }
[[ -f data/models/MiniMax-H3/model_index.json ]] || hf download MiniMaxAI/MiniMax-H3 --local-dir data/models/MiniMax-H3
hf download wlsaidhi/crush-smol-merged \
    --repo-type dataset \
    --revision "${DATASET_REVISION}" \
    --local-dir data/crush-smol

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_overfit

python -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_overfit --validate-only

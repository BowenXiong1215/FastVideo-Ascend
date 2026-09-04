#!/usr/bin/env bash
# Launch a FastVideo training config with torch_npu/HCCL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${1:?Usage: $0 <config.yaml> [--dotted.key value ...]}"
shift

NUM_NPUS="${NUM_NPUS:-8}"
export NUM_GPUS="${NUM_NPUS}"
if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    ASCEND_RT_VISIBLE_DEVICES="$(seq -s, 0 "$((NUM_NPUS - 1))")"
    export ASCEND_RT_VISIBLE_DEVICES
fi
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

cd "${REPO_ROOT}"
python - <<'PY'
import torch
import torch_npu

assert torch.npu.is_available(), "torch_npu imported, but no Ascend NPU is available"
print(f"torch={torch.__version__} torch_npu={torch_npu.__version__} npus={torch.npu.device_count()}")
PY

exec bash examples/train/run.sh "${CONFIG}" "$@"

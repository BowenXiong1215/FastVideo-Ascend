#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPECTED_TORCH=2.7.1
EXPECTED_TORCH_NPU=2.7.1.post4

check_framework() {
  EXPECTED_TORCH="${EXPECTED_TORCH}" EXPECTED_TORCH_NPU="${EXPECTED_TORCH_NPU}" python - <<'PY'
import os
import sys

import torch
import torch_npu

expected_torch = os.environ["EXPECTED_TORCH"]
expected_torch_npu = os.environ["EXPECTED_TORCH_NPU"]
actual_torch = torch.__version__.split("+")[0]
actual_torch_npu = torch_npu.__version__.split("+")[0]
print(f"torch={torch.__version__} ({torch.__file__})")
print(f"torch_npu={torch_npu.__version__} ({torch_npu.__file__})")
if actual_torch != expected_torch or actual_torch_npu != expected_torch_npu:
    print(
        f"ERROR: expected torch={expected_torch} and torch_npu={expected_torch_npu}; "
        f"got torch={actual_torch} and torch_npu={actual_torch_npu}",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY
}

cd "${REPO_ROOT}"
echo "Checking the base image PyTorch pair before dependency installation"
check_framework

python -m pip install \
  --upgrade-strategy only-if-needed \
  --constraint requirements/ascend-torch-constraints.txt \
  --requirement requirements/ascend-training.txt

python -m pip install -e . --no-deps

echo "Checking that pip preserved the base image PyTorch pair"
check_framework
if ! python -m pip check; then
  echo "WARN: pip check reported an environment conflict; the pinned torch/torch_npu pair is still intact" >&2
fi
echo "FastVideo Ascend dependency installation: PASS"

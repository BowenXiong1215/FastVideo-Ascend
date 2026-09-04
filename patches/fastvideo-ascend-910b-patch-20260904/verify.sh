#!/usr/bin/env bash
set -euo pipefail

PATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="${1:-$(pwd)}"
TARGET_ROOT="$(cd "${TARGET_ROOT}" && pwd)"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

failed=0
while IFS=$'\t' read -r _ expected relative; do
  target="${TARGET_ROOT}/${relative}"
  if ! test -f "${target}" || test "$(sha256_file "${target}")" != "${expected}"; then
    echo "FAILED: ${relative}" >&2
    failed=1
  fi
done < "${PATCH_ROOT}/modified.tsv"

while IFS=$'\t' read -r expected relative; do
  target="${TARGET_ROOT}/${relative}"
  if ! test -f "${target}" || test "$(sha256_file "${target}")" != "${expected}"; then
    echo "FAILED: ${relative}" >&2
    failed=1
  fi
done < "${PATCH_ROOT}/added.tsv"

test "${failed}" -eq 0
echo "FastVideo Ascend 910B Dense DMD2 bring-up tree: PASS"

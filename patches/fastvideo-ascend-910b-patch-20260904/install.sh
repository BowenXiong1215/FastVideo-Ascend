#!/usr/bin/env bash
set -euo pipefail

PATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="${1:-$(pwd)}"
TARGET_ROOT="$(cd "${TARGET_ROOT}" && pwd)"
UPSTREAM_COMMIT=7bb76b5ec99807a66aa3047b901f15019abe0f00

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

require_source_tree() {
  test -f "${TARGET_ROOT}/pyproject.toml" || {
    echo "Not a FastVideo source tree: ${TARGET_ROOT}" >&2
    exit 2
  }
  grep -q 'name = "fastvideo"' "${TARGET_ROOT}/pyproject.toml" || {
    echo "Unexpected pyproject.toml in ${TARGET_ROOT}" >&2
    exit 2
  }
}

validate_bundle() {
  local expected relative source actual
  while IFS=$'\t' read -r _ expected relative; do
    source="${PATCH_ROOT}/payload/replacements/${relative}"
    test -f "${source}" || { echo "Missing replacement payload: ${relative}" >&2; exit 3; }
    actual="$(sha256_file "${source}")"
    test "${actual}" = "${expected}" || { echo "Corrupt replacement payload: ${relative}" >&2; exit 3; }
  done < "${PATCH_ROOT}/modified.tsv"

  while IFS=$'\t' read -r expected relative; do
    source="${PATCH_ROOT}/payload/additions/${relative}"
    test -f "${source}" || { echo "Missing addition payload: ${relative}" >&2; exit 3; }
    actual="$(sha256_file "${source}")"
    test "${actual}" = "${expected}" || { echo "Corrupt addition payload: ${relative}" >&2; exit 3; }
  done < "${PATCH_ROOT}/added.tsv"
}

validate_target() {
  local upstream expected relative target actual
  while IFS=$'\t' read -r upstream expected relative; do
    target="${TARGET_ROOT}/${relative}"
    test -f "${target}" || { echo "Missing upstream file: ${relative}" >&2; exit 4; }
    actual="$(sha256_file "${target}")"
    if test "${actual}" != "${upstream}" && test "${actual}" != "${expected}"; then
      echo "Source version mismatch: ${relative}" >&2
      echo "Expected upstream FastVideo commit ${UPSTREAM_COMMIT}" >&2
      exit 4
    fi
  done < "${PATCH_ROOT}/modified.tsv"

  while IFS=$'\t' read -r expected relative; do
    target="${TARGET_ROOT}/${relative}"
    if test -e "${target}" && test "$(sha256_file "${target}")" != "${expected}"; then
      echo "Existing addition differs: ${relative}" >&2
      exit 4
    fi
  done < "${PATCH_ROOT}/added.tsv"
}

replace_with_sed() {
  local target="$1" replacement="$2" temporary
  temporary="/tmp/fastvideo-ascend-replacement.$$.tmp"
  cp "${replacement}" "${temporary}"
  if sed --version >/dev/null 2>&1; then
    sed -i -e "1r ${temporary}" -e '1,$d' "${target}"
  else
    sed -i '' -e "1r ${temporary}" -e '1,$d' "${target}"
  fi
  rm -f "${temporary}"
}

apply_replacements() {
  local upstream expected relative target actual
  while IFS=$'\t' read -r upstream expected relative; do
    target="${TARGET_ROOT}/${relative}"
    actual="$(sha256_file "${target}")"
    if test "${actual}" = "${upstream}"; then
      replace_with_sed "${target}" "${PATCH_ROOT}/payload/replacements/${relative}"
      echo "sed -i: ${relative}"
    else
      echo "present: ${relative}"
    fi
  done < "${PATCH_ROOT}/modified.tsv"
}

install_additions() {
  local expected relative source target
  while IFS=$'\t' read -r expected relative; do
    source="${PATCH_ROOT}/payload/additions/${relative}"
    target="${TARGET_ROOT}/${relative}"
    mkdir -p "$(dirname "${target}")"
    cp -p "${source}" "${target}"
    echo "added: ${relative}"
  done < "${PATCH_ROOT}/added.tsv"
}

verify_installed_tree() {
  local expected relative target actual failed=0
  while IFS=$'\t' read -r _ expected relative; do
    target="${TARGET_ROOT}/${relative}"
    actual="$(sha256_file "${target}")"
    if test "${actual}" != "${expected}"; then
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
}

require_source_tree
validate_bundle
validate_target
apply_replacements
install_additions
verify_installed_tree

echo "FastVideo Ascend 910B stage-1 patch: PASS"

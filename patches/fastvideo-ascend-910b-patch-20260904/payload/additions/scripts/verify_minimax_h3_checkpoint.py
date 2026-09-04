#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify a local MiniMax-H3 Diffusers checkpoint without loading its tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_COMPONENTS = (
    "audio_scheduler",
    "audio_vae",
    "processor",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer",
    "vae",
)
WEIGHT_COMPONENTS = ("audio_vae", "text_encoder", "transformer", "vae")
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        errors.append(f"invalid JSON: {path}: {error}")
        return None


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(len(LFS_HEADER)) == LFS_HEADER
    except OSError:
        return False


def verify_checkpoint(model_path: Path, check_tensor_headers: bool) -> int:
    model_path = model_path.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if not model_path.is_dir():
        print(f"FAIL: model directory does not exist: {model_path}", file=sys.stderr)
        return 1

    model_index_path = model_path / "model_index.json"
    model_index = load_json(model_index_path, errors) if model_index_path.is_file() else None
    if model_index is None:
        errors.append(f"missing or unreadable Diffusers manifest: {model_index_path}")
    elif not isinstance(model_index, dict):
        errors.append(f"model_index.json must contain a JSON object: {model_index_path}")

    for component in REQUIRED_COMPONENTS:
        component_dir = model_path / component
        if not component_dir.is_dir():
            errors.append(f"missing required component directory: {component}/")
        if isinstance(model_index, dict) and component not in model_index:
            errors.append(f"model_index.json has no {component!r} component")

    json_paths = sorted(model_path.rglob("*.json"))
    for json_path in json_paths:
        if json_path != model_index_path:
            load_json(json_path, errors)

    tensor_paths = sorted(model_path.rglob("*.safetensors"))
    legacy_weight_paths = sorted(model_path.rglob("*.bin")) + sorted(model_path.rglob("*.pth"))
    all_weight_paths = tensor_paths + legacy_weight_paths
    for path in all_weight_paths:
        if path.stat().st_size == 0:
            errors.append(f"empty weight file: {path.relative_to(model_path)}")
        elif is_lfs_pointer(path):
            errors.append(f"Git LFS pointer was not downloaded: {path.relative_to(model_path)}")

    indexed_paths: set[Path] = set()
    index_paths = sorted(model_path.rglob("*.safetensors.index.json"))
    for index_path in index_paths:
        index = load_json(index_path, errors)
        if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
            errors.append(f"missing weight_map object: {index_path.relative_to(model_path)}")
            continue
        for relative_shard in set(index["weight_map"].values()):
            if not isinstance(relative_shard, str) or not relative_shard:
                errors.append(f"invalid shard name in {index_path.relative_to(model_path)}: {relative_shard!r}")
                continue
            shard_path = (index_path.parent / relative_shard).resolve()
            try:
                shard_path.relative_to(model_path)
            except ValueError:
                errors.append(
                    f"shard escapes model directory: {relative_shard!r} in {index_path.relative_to(model_path)}")
                continue
            indexed_paths.add(shard_path)
            if not shard_path.is_file():
                errors.append(f"missing indexed shard: {shard_path.relative_to(model_path)}")
            elif shard_path.stat().st_size == 0:
                errors.append(f"empty indexed shard: {shard_path.relative_to(model_path)}")
            elif is_lfs_pointer(shard_path):
                errors.append(f"indexed shard is only a Git LFS pointer: {shard_path.relative_to(model_path)}")

    for component in WEIGHT_COMPONENTS:
        component_dir = model_path / component
        if component_dir.is_dir() and not any(path.is_file() for path in component_dir.rglob("*.safetensors")):
            errors.append(f"no safetensors weights found under required component: {component}/")

    unindexed_shards = [path for path in tensor_paths if index_paths and path.resolve() not in indexed_paths]
    if unindexed_shards:
        warnings.append(
            f"{len(unindexed_shards)} safetensors file(s) are not referenced by an index "
            "(single-file components are valid)")

    if check_tensor_headers and tensor_paths:
        try:
            from safetensors import safe_open
        except ImportError:
            warnings.append("safetensors is not installed; skipped tensor-header validation")
        else:
            for tensor_path in tensor_paths:
                if tensor_path.stat().st_size == 0 or is_lfs_pointer(tensor_path):
                    continue
                try:
                    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
                        if not list(handle.keys()):
                            errors.append(f"safetensors file has no tensors: {tensor_path.relative_to(model_path)}")
                except Exception as error:  # safetensors exposes several backend exception types
                    errors.append(f"cannot read safetensors header: {tensor_path.relative_to(model_path)}: {error}")

    total_bytes = sum(path.stat().st_size for path in all_weight_paths if path.is_file())
    print(f"Model path:       {model_path}")
    print(f"JSON files:      {len(json_paths)}")
    print(f"Index files:     {len(index_paths)}")
    print(f"Safetensors:     {len(tensor_paths)}")
    print(f"Weight bytes:    {human_size(total_bytes)}")
    print(f"Indexed shards:  {len(indexed_paths)}")
    for warning in warnings:
        print(f"WARN: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"MiniMax-H3 checkpoint verification: FAIL ({len(errors)} error(s))", file=sys.stderr)
        return 1
    print("MiniMax-H3 checkpoint verification: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, help="local MiniMax-H3 Diffusers checkpoint directory")
    parser.add_argument(
        "--skip-tensor-headers",
        action="store_true",
        help="skip safetensors header parsing (file, JSON, index, size, and LFS checks still run)",
    )
    args = parser.parse_args()
    return verify_checkpoint(args.model_path, check_tensor_headers=not args.skip_tensor_headers)


if __name__ == "__main__":
    raise SystemExit(main())

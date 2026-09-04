#!/usr/bin/env python3
"""Fail-fast validation for the Ascend Dense H3 four-step DMD2 bring-up."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

LADDER = [999, 749, 500, 250]
MODEL_TARGET = "fastvideo.train.models.minimax_h3.MiniMaxH3Model"
METHOD_TARGET = (
    "fastvideo.train.methods.distribution_matching.minimax_h3_dmd2."
    "MiniMaxH3DMD2Method"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--check-paths", action="store_true")
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = raw["models"]
    assert set(models) == {"student", "teacher", "critic"}
    assert models["student"]["trainable"] is True
    assert models["teacher"]["trainable"] is False
    assert models["critic"]["trainable"] is True
    for role, config in models.items():
        assert config["_target_"] == MODEL_TARGET, role
        assert config["attention_backend"] == "TORCH_SDPA", role
        if args.check_paths:
            assert Path(config["init_from"]).is_dir(), config["init_from"]

    method = raw["method"]
    assert method["_target_"] == METHOD_TARGET
    assert method["rollout_mode"] == "simulate"
    assert method["dmd_denoising_steps"] == LADDER
    assert float(method["real_score_guidance_scale"]) == 1.0

    training = raw["training"]
    distributed = training["distributed"]
    assert int(distributed["num_gpus"]) == 8
    assert int(distributed["sp_size"]) == 8
    assert int(distributed["hsdp_shard_dim"]) == 8
    assert distributed["fsdp_cpu_offload"] is True
    assert training["data"]["preprocessed_data_type"] == "t2va"
    if args.check_paths:
        assert Path(training["data"]["data_path"]).is_dir(), training["data"]["data_path"]

    print("MiniMax-H3 Dense DMD2 four-step config: PASS")


if __name__ == "__main__":
    main()

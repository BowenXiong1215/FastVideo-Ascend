# SPDX-License-Identifier: Apache-2.0
"""Repair the dtype contract of an already-exported MiniMax-H3 student."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def repair(model_dir: str) -> Path:
    config_path = Path(model_dir).expanduser().resolve() / "transformer" / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing MiniMax-H3 transformer config: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    before = config.get("uniform_parameter_dtype", "<missing>")
    config["uniform_parameter_dtype"] = True

    temporary_path = config_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, config_path)
    print(f"updated: {config_path}")
    print(f"uniform_parameter_dtype: {before!r} -> True")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Diffusers-style exported MiniMax-H3 model directory")
    args = parser.parse_args()
    repair(args.model_dir)


if __name__ == "__main__":
    main()

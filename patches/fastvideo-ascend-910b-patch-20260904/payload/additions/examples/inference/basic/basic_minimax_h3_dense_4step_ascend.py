# SPDX-License-Identifier: Apache-2.0
"""Run an exported Dense MiniMax-H3 student with exactly four DiT forwards."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

try:
    from . import basic_fasth3
except ImportError:
    import basic_fasth3  # type: ignore[no-redef]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = basic_fasth3.build_parser(description=__doc__)
    parser.set_defaults(
        output="outputs/minimax_h3_dense_4step_ascend",
        profile="strict",
        steps=5,
        num_gpus=8,
        repeats=1,
        warmup=False,
        fa4=False,
        h3_fusions=False,
        compile_vae=False,
        parallel_vae=False,
        replicated_dit=False,
        inference_torch_compile=False,
    )
    args = basic_fasth3.validate_args(parser, parser.parse_args(argv))
    args.vsa = False
    args.dense_attention_backend = "TORCH_SDPA"
    if args.steps != 5:
        parser.error("this bring-up launcher requires --steps 5 (four DiT forwards)")
    return args


def main() -> None:
    args = parse_args()
    print("Attention: dense TORCH_SDPA")
    basic_fasth3.run(args)


if __name__ == "__main__":
    main()

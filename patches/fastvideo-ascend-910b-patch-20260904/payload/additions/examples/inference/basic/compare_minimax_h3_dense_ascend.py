# SPDX-License-Identifier: Apache-2.0
"""Compare base MiniMax-H3 with a Dense four-forward DMD2 student on Ascend."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


_E2E_PATTERN = re.compile(r"E2E wall time: ([0-9.]+)s")
_DENOISE_PATTERN = re.compile(r"Denoising time: ([0-9.]+)s")
_OUTPUT_PATTERN = re.compile(r"Output written to: (.+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--student-model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/minimax_h3_dense_comparison")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--base-sigma-points", type=int, default=50)
    parser.add_argument("--student-sigma-points", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run one excluded request for each model before measured requests",
    )
    parser.add_argument(
        "--pin-cpu-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--lazy-module-load",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--h3-sequential-load",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()
    if args.base_sigma_points < 2:
        parser.error("--base-sigma-points must be at least 2")
    if args.student_sigma_points != 5:
        parser.error("the distilled student comparison requires 5 sigma points (4 DiT forwards)")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def _optional_boolean_argument(name: str, value: bool | None) -> list[str]:
    if value is None:
        return []
    return [f"--{name}" if value else f"--no-{name}"]


def _write_run_config(args: argparse.Namespace, role: str, output_dir: Path) -> Path:
    is_base = role == "base"
    config = {
        "role": role,
        "model_path": args.base_model_path if is_base else args.student_model_path,
        "prompt": args.prompt,
        "output": str(output_dir / role),
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "steps": args.base_sigma_points if is_base else args.student_sigma_points,
        "seed": args.seed,
        "repeats": args.repeats,
        "num_gpus": args.num_gpus,
        "warmup": args.warmup,
        "pin_cpu_memory": args.pin_cpu_memory,
        "lazy_module_load": args.lazy_module_load,
        "h3_sequential_load": args.h3_sequential_load,
    }
    path = output_dir / f"{role}_run_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _run_one(config: dict[str, Any]) -> None:
    try:
        from . import basic_fasth3
    except ImportError:
        import basic_fasth3  # type: ignore[no-redef]

    argv = [
        "--model-path",
        str(config["model_path"]),
        "--prompt",
        str(config["prompt"]),
        "--output",
        str(config["output"]),
        "--profile",
        "strict",
        "--height",
        str(config["height"]),
        "--width",
        str(config["width"]),
        "--num-frames",
        str(config["num_frames"]),
        "--steps",
        str(config["steps"]),
        "--seed",
        str(config["seed"]),
        "--repeats",
        str(config["repeats"]),
        "--num-gpus",
        str(config["num_gpus"]),
        "--execution-backend",
        "mp",
        "--no-fa4",
        "--no-h3-fusions",
        "--no-compile-vae",
        "--no-parallel-vae",
        "--no-replicated-dit",
        "--no-inference-torch-compile",
        "--ulysses-a2a",
        "off",
    ]
    argv.extend(_optional_boolean_argument("warmup", bool(config["warmup"])))
    argv.extend(_optional_boolean_argument("pin-cpu-memory", bool(config["pin_cpu_memory"])))
    argv.extend(_optional_boolean_argument("lazy-module-load", config["lazy_module_load"]))
    argv.extend(_optional_boolean_argument("h3-sequential-load", config["h3_sequential_load"]))
    run_args = basic_fasth3.parse_args(argv)
    run_args.vsa = False
    run_args.dense_attention_backend = "TORCH_SDPA"
    print(f"Comparison role: {config['role']}")
    print("Attention: dense TORCH_SDPA")
    basic_fasth3.run(run_args)


def _execute_role(role: str, config_path: Path, output_dir: Path) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--run-config", str(config_path)]
    log_path = output_dir / f"{role}.log"
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{role}] {line}", end="", flush=True)
            log_file.write(line)
            log_file.flush()
            lines.append(line)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{role} inference failed with exit code {return_code}; see {log_path}")

    text = "".join(lines)
    e2e = [float(value) for value in _E2E_PATTERN.findall(text)]
    denoise = [float(value) for value in _DENOISE_PATTERN.findall(text)]
    outputs = [value.strip() for value in _OUTPUT_PATTERN.findall(text)]
    if not e2e:
        raise RuntimeError(f"{role} inference produced no measured E2E timing; see {log_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    forwards = int(config["steps"]) - 1
    median_denoise = statistics.median(denoise) if denoise else None
    return {
        **config,
        "dit_forwards": forwards,
        "e2e_seconds": e2e,
        "median_e2e_seconds": statistics.median(e2e),
        "denoise_seconds": denoise,
        "median_denoise_seconds": median_denoise,
        "median_denoise_seconds_per_forward": (median_denoise / forwards if median_denoise is not None else None),
        "video_paths": outputs,
        "log_path": str(log_path),
    }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _format_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _write_report(output_dir: Path, base: dict[str, Any], student: dict[str, Any]) -> None:
    report = {
        "comparison_contract": {
            "prompt": base["prompt"],
            "seed": base["seed"],
            "height": base["height"],
            "width": base["width"],
            "num_frames": base["num_frames"],
            "attention": "dense TORCH_SDPA",
            "profile": "strict",
            "guidance": "baked into both H3 checkpoints; one forward per denoising interval",
        },
        "base": base,
        "student": student,
        "speedup": {
            "e2e": _ratio(base["median_e2e_seconds"], student["median_e2e_seconds"]),
            "denoising": _ratio(base["median_denoise_seconds"], student["median_denoise_seconds"]),
        },
    }
    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    e2e_speedup = report["speedup"]["e2e"]
    denoise_speedup = report["speedup"]["denoising"]
    markdown = [
        "# MiniMax-H3 Dense strict comparison",
        "",
        f"Prompt: `{base['prompt']}`  ",
        f"Seed: `{base['seed']}`  ",
        f"Shape: `{base['width']}x{base['height']}`, `{base['num_frames']}` frames",
        "",
        "| Model | Sigma points | DiT forwards | Median E2E (s) | Median denoise (s) | Denoise/forward (s) |",
        "|---|---:|---:|---:|---:|---:|",
        (f"| Base H3 | {base['steps']} | {base['dit_forwards']} | "
         f"{_format_seconds(base['median_e2e_seconds'])} | {_format_seconds(base['median_denoise_seconds'])} | "
         f"{_format_seconds(base['median_denoise_seconds_per_forward'])} |"),
        (f"| Dense DMD2 student | {student['steps']} | {student['dit_forwards']} | "
         f"{_format_seconds(student['median_e2e_seconds'])} | "
         f"{_format_seconds(student['median_denoise_seconds'])} | "
         f"{_format_seconds(student['median_denoise_seconds_per_forward'])} |"),
        "",
        f"E2E speedup: `{e2e_speedup:.3f}x`" if e2e_speedup is not None else "E2E speedup: `n/a`",
        (f"Denoising speedup: `{denoise_speedup:.3f}x`"
         if denoise_speedup is not None else "Denoising speedup: `n/a`"),
        "",
        f"Base outputs: `{base['video_paths']}`",
        f"Student outputs: `{student['video_paths']}`",
        "",
    ]
    markdown_path = output_dir / "comparison.md"
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    print(f"Comparison JSON: {json_path.resolve()}")
    print(f"Comparison Markdown: {markdown_path.resolve()}")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--run-config":
        _run_one(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
        return

    args = _parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Running base and student in isolated processes so every model fully releases its workers and memory.")
    base_config = _write_run_config(args, "base", output_dir)
    student_config = _write_run_config(args, "student", output_dir)
    base = _execute_role("base", base_config, output_dir)
    student = _execute_role("student", student_config, output_dir)
    _write_report(output_dir, base, student)


if __name__ == "__main__":
    main()

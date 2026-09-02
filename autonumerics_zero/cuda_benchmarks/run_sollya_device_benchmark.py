#!/usr/bin/env python3
"""Compile and run the BF16 Sollya-vs-current device-side benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "benchmark_bfloat16_sollya.cu"
BIN = ROOT / "benchmark_bfloat16_sollya"
SPLINE_DIR = ROOT.parent / "spline_ops"
ERROR_JSON = ROOT / "analysis_results" / "sollya_device_bf16.json"
OUT_JSON = ROOT / "analysis_results" / "sollya_device_benchmark.json"

RESULT_RE = re.compile(
    r"^RESULT\s+(?P<family>\w+)\s+D(?P<degree>\d+)\s+"
    r"ours_ms=(?P<ours_ms>[0-9.]+)\s+"
    r"sollya_ms=(?P<sollya_ms>[0-9.]+)\s+"
    r"sollya_over_ours=(?P<ratio>[0-9.]+)$"
)


def compile_benchmark() -> None:
    nvcc = (
        shutil.which("nvcc")
        or shutil.which("/usr/local/cuda/bin/nvcc")
        or shutil.which("/usr/local/cuda-13.0/bin/nvcc")
    )
    if nvcc is None:
        raise RuntimeError("Could not locate nvcc. Set PATH or install the CUDA toolkit.")
    cmd = [
        nvcc,
        "-O3",
        "-arch=sm_100",
        "--use_fast_math",
        "-I",
        str(SPLINE_DIR),
        "-o",
        str(BIN),
        str(SRC),
    ]
    subprocess.run(cmd, check=True)


def parse_results(stdout: str) -> dict[str, dict[str, dict[str, float]]]:
    rows: dict[str, dict[str, dict[str, float]]] = {}
    for line in stdout.splitlines():
        match = RESULT_RE.match(line.strip())
        if not match:
            continue
        family = match.group("family")
        degree = f"D{match.group('degree')}"
        rows.setdefault(family, {})[degree] = {
            "ours_ms": float(match.group("ours_ms")),
            "sollya_ms": float(match.group("sollya_ms")),
            "sollya_over_ours": float(match.group("ratio")),
        }
    if not rows:
        raise RuntimeError(f"Failed to parse benchmark output:\n{stdout}")
    return rows


def merge_with_error_data(
    perf_rows: dict[str, dict[str, dict[str, float]]],
    error_data: dict[str, object],
) -> dict[str, object]:
    merged: dict[str, object] = {"families": {}}
    families = error_data["families"]
    for family, by_degree in perf_rows.items():
        merged["families"][family] = {}
        for degree, perf in by_degree.items():
            error_row = families[family][degree]
            merged["families"][family][degree] = {
                **error_row,
                **perf,
                "sollya_speedup_pct": (1.0 - perf["sollya_over_ours"]) * 100.0,
            }
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--cuda-visible-devices", default=None)
    args = parser.parse_args()

    if not args.skip_compile:
        compile_benchmark()

    env = None
    if args.cuda_visible_devices is not None:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    proc = subprocess.run([str(BIN)], check=True, capture_output=True, text=True, env=env)
    perf_rows = parse_results(proc.stdout)
    error_data = json.loads(ERROR_JSON.read_text())
    merged = merge_with_error_data(perf_rows, error_data)
    OUT_JSON.write_text(json.dumps(merged, indent=2) + "\n")

    print(f"Wrote {OUT_JSON}")
    for family in ("sigmoid_fwd", "tanh_fwd", "swish_fwd", "gelu_fwd"):
        if family not in merged["families"]:
            continue
        print(f"{family}:")
        for degree, row in merged["families"][family].items():
            print(
                f"  {degree}: ours_ms={row['ours_ms']:.4f} "
                f"sollya_ms={row['sollya_ms']:.4f} "
                f"ratio={row['sollya_over_ours']:.4f} "
                f"ours_err={row['current_max_error']:.6f} "
                f"sollya_err={row['sollya_max_error']:.6f}"
            )


if __name__ == "__main__":
    main()

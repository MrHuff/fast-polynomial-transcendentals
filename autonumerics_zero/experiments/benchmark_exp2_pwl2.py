#!/usr/bin/env python3
"""Build and run the pure-CUDA two-piece exp2 benchmark.

This intentionally invokes NVCC directly. No Triton-generated kernel is used.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "autonumerics_zero/cuda_benchmarks/benchmark_exp2_pwl2.cu"


def parse_value(value: str):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_record(line: str) -> dict[str, object]:
    fields = {}
    for token in shlex.split(line)[1:]:
        key, value = token.split("=", 1)
        fields[key] = parse_value(value)
    return fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nvcc", type=Path, default=Path("/usr/local/cuda-13.0/bin/nvcc")
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    compile_command = [
        str(args.nvcc),
        "-O3",
        "-std=c++17",
        "-arch=sm_100",
        "-lineinfo",
        str(SOURCE),
    ]
    with tempfile.TemporaryDirectory(prefix="exp2-pwl2-") as temporary_dir:
        binary = Path(temporary_dir) / "benchmark_exp2_pwl2"
        compile_command.extend(["-o", str(binary)])
        subprocess.run(compile_command, cwd=REPO_ROOT, check=True)
        completed = subprocess.run(
            [str(binary)],
            cwd=REPO_ROOT,
            env={
                **__import__("os").environ,
                "CUDA_VISIBLE_DEVICES": args.device,
            },
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )

    print(completed.stdout, end="")
    payload = {
        "source": str(SOURCE.relative_to(REPO_ROOT)),
        "compile_command": compile_command[:-2] + ["-o", "<temporary-binary>"],
        "gpu": None,
        "errors": [],
        "timings": [],
    }
    for line in completed.stdout.splitlines():
        if line.startswith("GPU "):
            payload["gpu"] = line.removeprefix("GPU ")
        elif line.startswith("ERROR "):
            payload["errors"].append(parse_record(line))
        elif line.startswith("RESULT "):
            payload["timings"].append(parse_record(line))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"JSON path={args.json_out}")


if __name__ == "__main__":
    main()

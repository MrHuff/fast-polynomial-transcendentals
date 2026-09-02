#!/usr/bin/env python3
"""Benchmark the report's selected FP16 functions at L2 and HBM endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, os.fspath(THIS_DIR.parent / "spline_ops"))
import spline_ops  # noqa: E402


TensorFn = Callable[[], torch.Tensor]


def parse_sizes(value: str) -> list[int]:
    sizes = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive comma-separated integers")
    return sizes


def git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", os.fspath(REPOSITORY_ROOT), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def time_calls(fn: TensorFn, repetitions: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repetitions):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1_000.0 / repetitions


def warm_up(fn: TensorFn, calls: int) -> None:
    for _ in range(calls):
        fn()
    torch.cuda.synchronize()


def measure_pair(
    native: TensorFn,
    polynomial: TensorFn,
    *,
    warmup: int,
    repetitions: int,
    rounds: int,
) -> dict[str, object]:
    warm_up(native, warmup)
    warm_up(polynomial, warmup)

    native_us: list[float] = []
    polynomial_us: list[float] = []
    orders: list[str] = []
    for round_index in range(rounds):
        native_first = round_index % 2 == 0
        orders.append("native-polynomial" if native_first else "polynomial-native")
        if native_first:
            native_us.append(time_calls(native, repetitions))
            polynomial_us.append(time_calls(polynomial, repetitions))
        else:
            polynomial_us.append(time_calls(polynomial, repetitions))
            native_us.append(time_calls(native, repetitions))

    native_median = statistics.median(native_us)
    polynomial_median = statistics.median(polynomial_us)
    return {
        "round_order": orders,
        "native_us_per_call": native_us,
        "polynomial_us_per_call": polynomial_us,
        "native_median_us": native_median,
        "polynomial_median_us": polynomial_median,
        "speedup_native_over_polynomial": native_median / polynomial_median,
    }


def selected_cases(x: torch.Tensor, grad_out: torch.Tensor) -> list[dict[str, object]]:
    sigmoid_output = torch.sigmoid(x)
    tanh_output = torch.tanh(x)
    torch.cuda.synchronize()

    return [
        {
            "direction": "forward",
            "function": "sigmoid",
            "degree": 3,
            "native": lambda: torch.sigmoid(x),
            "polynomial": lambda: spline_ops.sigmoid_fwd_d3(x),
        },
        {
            "direction": "forward",
            "function": "tanh",
            "degree": 4,
            "native": lambda: torch.tanh(x),
            "polynomial": lambda: spline_ops.tanh_fwd_d4(x),
        },
        {
            "direction": "forward",
            "function": "silu",
            "degree": 3,
            "native": lambda: F.silu(x),
            "polynomial": lambda: spline_ops.swish_fwd_d3(x),
        },
        {
            "direction": "forward",
            "function": "gelu",
            "degree": 5,
            "native": lambda: F.gelu(x),
            "polynomial": lambda: spline_ops.gelu_fwd_d5(x),
        },
        {
            "direction": "backward",
            "function": "sigmoid",
            "degree": 4,
            "native": lambda: torch.ops.aten.sigmoid_backward(grad_out, sigmoid_output),
            "polynomial": lambda: spline_ops.sigmoid_bwd_d4(grad_out, x),
        },
        {
            "direction": "backward",
            "function": "tanh",
            "degree": 4,
            "native": lambda: torch.ops.aten.tanh_backward(grad_out, tanh_output),
            "polynomial": lambda: spline_ops.tanh_bwd_d4(grad_out, x),
        },
        {
            "direction": "backward",
            "function": "silu",
            "degree": 3,
            "native": lambda: torch.ops.aten.silu_backward(grad_out, x),
            "polynomial": lambda: spline_ops.swish_bwd_d3(grad_out, x),
        },
        {
            "direction": "backward",
            "function": "gelu",
            "degree": 5,
            "native": lambda: torch.ops.aten.gelu_backward(grad_out, x),
            "polynomial": lambda: spline_ops.gelu_bwd_d5(grad_out, x),
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("4096,268435456"))
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.repetitions, args.rounds) <= 0:
        parser.error("warmup, repetitions, and rounds must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    properties = torch.cuda.get_device_properties(device)
    driver_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    extension_path = Path(spline_ops.__file__).resolve()
    extension_hash = hashlib.sha256(extension_path.read_bytes()).hexdigest()
    status = git_output("status", "--porcelain")

    artifact: dict[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "selected isolated FP16 polynomial functions",
        "speedup_definition": "native PyTorch median time divided by polynomial median time",
        "protocol": {
            "dtype": "torch.float16",
            "warmup_calls_per_path": args.warmup,
            "timed_calls_per_round": args.repetitions,
            "alternating_rounds": args.rounds,
            "seed": args.seed,
            "sizes": args.sizes,
            "native_sigmoid_tanh_backward_uses_precomputed_forward_output": True,
        },
        "environment": {
            "gpu_name": properties.name,
            "gpu_index": device.index,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "sm_count": properties.multi_processor_count,
            "l2_cache_bytes": properties.L2_cache_size,
            "total_memory_bytes": properties.total_memory,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "repository_commit": git_output("rev-parse", "HEAD"),
            "repository_branch": git_output("branch", "--show-current"),
            "repository_dirty": bool(status),
            "driver_sha256": driver_hash,
            "extension_binary": extension_path.name,
            "extension_binary_sha256": extension_hash,
        },
        "measurements": [],
    }

    with torch.inference_mode():
        for size in args.sizes:
            x = torch.randn(size, device=device, dtype=torch.float16)
            grad_out = torch.ones_like(x)
            for case in selected_cases(x, grad_out):
                native = case.pop("native")
                polynomial = case.pop("polynomial")
                resident_arrays = 2 if case["direction"] == "forward" else 3
                working_set_bytes = size * x.element_size() * resident_arrays
                measurement = {
                    **case,
                    "elements": size,
                    "estimated_working_set_bytes": working_set_bytes,
                    "memory_regime": (
                        "L2 cache"
                        if working_set_bytes <= properties.L2_cache_size
                        else "HBM"
                    ),
                    **measure_pair(
                        native,
                        polynomial,
                        warmup=args.warmup,
                        repetitions=args.repetitions,
                        rounds=args.rounds,
                    ),
                }
                artifact["measurements"].append(measurement)
                print(
                    f"{measurement['direction']:8s} {measurement['function']:7s} "
                    f"D{measurement['degree']} N={size}: "
                    f"{measurement['speedup_native_over_polynomial']:.3f}x"
                )
            del x, grad_out
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

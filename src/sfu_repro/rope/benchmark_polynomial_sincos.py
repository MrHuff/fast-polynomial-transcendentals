#
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified 2026-09-02 for the standalone SFU reproduction package.
#

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

from .polynomial_sincos import (
    SOLLYA_D3_D4,
    UNIFORM_D5_D4,
    UNIFORM_D7_D6,
    polynomial_sincos,
)


TensorFunction = Callable[[torch.Tensor], torch.Tensor]


def make_rope_angles(
    *,
    head_dim: int,
    max_seq_len: int,
    theta: float,
    device: torch.device,
) -> torch.Tensor:
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim must be a positive even integer")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")

    dimensions = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    frequencies = torch.pow(theta, -dimensions / head_dim)
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    return torch.outer(positions, frequencies)


def torch_native_sincos(angles: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.cos(angles), torch.sin(angles)))


def complex_repack(angles: torch.Tensor) -> torch.Tensor:
    table = torch.polar(torch.ones_like(angles), angles)
    return torch.stack((table.real, table.imag))


def polynomial_d3_accurate(angles: torch.Tensor) -> torch.Tensor:
    cos, sin = polynomial_sincos(
        angles,
        polynomial=SOLLYA_D3_D4,
        accurate_reduction=True,
    )
    return torch.stack((cos, sin))


def load_spline_ops():
    try:
        import spline_ops
    except ImportError as exc:
        raise RuntimeError(
            "The CUDA sin/cos benchmark requires spline_ops. Build it with "
            "`python -m pip install -v ./autonumerics_zero/spline_ops`."
        ) from exc
    required = (
        "sincos_native_f32",
        "sincos_d3_d4_f32",
        "sincos_d5_d4_f32",
        "sincos_d7_d6_f32",
    )
    missing = [name for name in required if not hasattr(spline_ops, name)]
    if missing:
        raise RuntimeError(
            "spline_ops is stale and lacks "
            + ", ".join(missing)
            + "; rebuild the extension."
        )
    return spline_ops


def polynomial_d5_accurate(angles: torch.Tensor) -> torch.Tensor:
    cos, sin = polynomial_sincos(
        angles,
        polynomial=UNIFORM_D5_D4,
        accurate_reduction=True,
    )
    return torch.stack((cos, sin))


def polynomial_d5_fast(angles: torch.Tensor) -> torch.Tensor:
    cos, sin = polynomial_sincos(
        angles,
        polynomial=UNIFORM_D5_D4,
        accurate_reduction=False,
    )
    return torch.stack((cos, sin))


def polynomial_d7_accurate(angles: torch.Tensor) -> torch.Tensor:
    cos, sin = polynomial_sincos(
        angles,
        polynomial=UNIFORM_D7_D6,
        accurate_reduction=True,
    )
    return torch.stack((cos, sin))


def polynomial_d7_fast(angles: torch.Tensor) -> torch.Tensor:
    cos, sin = polynomial_sincos(
        angles,
        polynomial=UNIFORM_D7_D6,
        accurate_reduction=False,
    )
    return torch.stack((cos, sin))


def numerical_metrics(
    output: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    output = output.float()
    reference = reference.float()
    cos, sin = output.unbind(0)
    reference_cos, reference_sin = reference.unbind(0)
    phase_error = torch.atan2(
        sin * reference_cos - cos * reference_sin,
        cos * reference_cos + sin * reference_sin,
    )
    error = output - reference
    return {
        "max_abs": error.abs().max().item(),
        "rmse": error.square().mean().sqrt().item(),
        "phase_max_abs_rad": phase_error.abs().max().item(),
        "unit_norm_max_abs": (cos.square() + sin.square() - 1.0).abs().max().item(),
    }


@torch.inference_mode()
def benchmark(
    function: TensorFunction,
    angles: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
    trials: int,
) -> tuple[dict[str, float], torch.Tensor]:
    output = function(angles)
    for _ in range(warmup):
        output = function(angles)

    samples = []
    for _ in range(trials):
        if angles.is_cuda:
            torch.cuda.synchronize(angles.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                output = function(angles)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0 / repeats)
        else:
            start_time = time.perf_counter()
            for _ in range(repeats):
                output = function(angles)
            samples.append((time.perf_counter() - start_time) * 1.0e6 / repeats)

    timing = {
        "microseconds": statistics.median(samples),
        "microseconds_min": min(samples),
        "microseconds_max": max(samples),
    }
    return timing, output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark native and polynomial RoPE sin/cos table generation."
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--theta", type=float, default=500_000.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")

    device = torch.device(args.device)
    angles = make_rope_angles(
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        theta=args.theta,
        device=device,
    )
    reference = complex_repack(angles)
    implementations: list[tuple[str, TensorFunction]] = [
        ("complex_repack", complex_repack),
        ("torch_native_sincos", torch_native_sincos),
        ("polynomial_d3_accurate", polynomial_d3_accurate),
        ("polynomial_d5_accurate", polynomial_d5_accurate),
        ("polynomial_d5_fast", polynomial_d5_fast),
        ("polynomial_d7_accurate", polynomial_d7_accurate),
        ("polynomial_d7_fast", polynomial_d7_fast),
    ]
    if device.type == "cuda":
        spline_ops = load_spline_ops()
        implementations.extend(
            (
                (
                    "cuda_native_sfu",
                    spline_ops.sincos_native_f32,
                ),
                (
                    "cuda_poly_d3_d4",
                    spline_ops.sincos_d3_d4_f32,
                ),
                (
                    "cuda_poly_d5_d4",
                    spline_ops.sincos_d5_d4_f32,
                ),
                (
                    "cuda_poly_d7_d6",
                    spline_ops.sincos_d7_d6_f32,
                ),
            )
        )

    results = []
    for name, function in implementations:
        timing, output = benchmark(
            function,
            angles,
            warmup=args.warmup,
            repeats=args.repeats,
            trials=args.trials,
        )
        result = {
            "implementation": name,
            **timing,
            "million_values_per_second": angles.numel() / timing["microseconds"],
            **numerical_metrics(output, reference),
        }
        results.append(result)

    payload = {
        "configuration": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "head_dim": args.head_dim,
            "max_seq_len": args.max_seq_len,
            "theta": args.theta,
            "angle_count": angles.numel(),
            "angle_max": angles.max().item(),
        },
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

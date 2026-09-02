# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified 2026-09-02 for the standalone SFU reproduction package.

"""Benchmark RoPE-specific FP16 polynomial sin/cos against native SFUs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Callable

import torch

from .benchmark_polynomial_sincos import (
    attest_rope_sources,
    load_spline_ops,
    numerical_metrics,
    rope_result_metadata,
)


TensorCall = Callable[[], torch.Tensor]


@torch.inference_mode()
def benchmark_interleaved(
    implementations: tuple[tuple[str, TensorCall], ...],
    *,
    warmup: int,
    repeats: int,
    trials: int,
) -> tuple[dict[str, dict[str, float]], dict[str, torch.Tensor]]:
    outputs: dict[str, torch.Tensor] = {}
    for _ in range(warmup):
        for name, function in implementations:
            outputs[name] = function()

    samples: dict[str, list[float]] = {name: [] for name, _ in implementations}
    for trial in range(trials):
        ordered = implementations if trial % 2 == 0 else implementations[::-1]
        for name, function in ordered:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                outputs[name] = function()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0 / repeats)

    timings = {
        name: {
            "microseconds": statistics.median(values),
            "microseconds_min": min(values),
            "microseconds_max": max(values),
            "samples_us": values,
        }
        for name, values in samples.items()
    }
    return timings, outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--theta", type=float, default=500_000.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--compute-iterations", type=int, default=64)
    parser.add_argument("--compute-repeats", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-unbound-source",
        action="store_true",
        help="Permit a diagnostic result from dirty or non-local source.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.head_dim <= 0 or args.head_dim % 4:
        raise ValueError("head_dim must be a positive multiple of four")
    if args.sequence_length <= 0 or args.compute_iterations <= 0:
        raise ValueError("sequence length and compute iterations must be positive")

    spline_ops = load_spline_ops()
    repository_state, attestations, source_bound = attest_rope_sources(
        spline_ops=spline_ops,
        allow_unbound_source=args.allow_unbound_source,
    )
    required = (
        "rope_sincos_native_fp16",
        "rope_sincos_fixed_d3_d4_fp16",
        "rope_sincos_fixed_half_turn_d5_d6_fp16",
        "rope_sincos_native_fp16_compute",
        "rope_sincos_fixed_d3_d4_fp16_compute",
        "rope_sincos_fixed_half_turn_d5_d6_fp16_compute",
    )
    missing = [name for name in required if not hasattr(spline_ops, name)]
    if missing:
        raise RuntimeError(
            "spline_ops is stale and lacks "
            + ", ".join(missing)
            + "; rebuild the CUDA extension."
        )

    device = torch.device("cuda")
    dimensions = torch.arange(0, args.head_dim, 2, dtype=torch.float64)
    frequencies = args.theta ** (-dimensions / args.head_dim)
    frequencies_fp32 = frequencies.float().to(device)
    phase_scale = float(1 << 32) / (2.0 * math.pi)
    phase_increments = torch.round(frequencies * phase_scale).to(torch.int32).to(device)
    positions = torch.arange(args.sequence_length, dtype=torch.float64, device=device)[
        :, None
    ]
    angles = positions * frequencies.to(device)[None, :]
    reference = torch.stack((torch.cos(angles), torch.sin(angles)))

    implementations: tuple[tuple[str, TensorCall], ...] = (
        (
            "native_sfu_fp16",
            lambda: spline_ops.rope_sincos_native_fp16(
                frequencies_fp32, args.sequence_length
            ),
        ),
        (
            "q32_quarter_d3_d4_fp16",
            lambda: spline_ops.rope_sincos_fixed_d3_d4_fp16(
                phase_increments, args.sequence_length
            ),
        ),
        (
            "q32_half_d5_d6_fp16_vec4",
            lambda: spline_ops.rope_sincos_fixed_half_turn_d5_d6_fp16(
                phase_increments, args.sequence_length
            ),
        ),
    )
    timings, outputs = benchmark_interleaved(
        implementations,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
    )
    results: list[dict[str, object]] = []
    for name, _ in implementations:
        results.append(
            {
                "implementation": name,
                **timings[name],
                **numerical_metrics(outputs[name], reference),
            }
        )
    native_time = float(results[0]["microseconds"])
    for row in results:
        row["speedup_vs_native_sfu"] = native_time / float(row["microseconds"])

    compute_implementations: tuple[tuple[str, TensorCall], ...] = (
        (
            "native_sfu_fp16",
            lambda: spline_ops.rope_sincos_native_fp16_compute(
                frequencies_fp32,
                args.sequence_length,
                args.compute_iterations,
            ),
        ),
        (
            "q32_quarter_d3_d4_fp16",
            lambda: spline_ops.rope_sincos_fixed_d3_d4_fp16_compute(
                phase_increments,
                args.sequence_length,
                args.compute_iterations,
            ),
        ),
        (
            "q32_half_d5_d6_fp16_vec4",
            lambda: spline_ops.rope_sincos_fixed_half_turn_d5_d6_fp16_compute(
                phase_increments,
                args.sequence_length,
                args.compute_iterations,
            ),
        ),
    )
    compute_timings, _ = benchmark_interleaved(
        compute_implementations,
        warmup=args.warmup,
        repeats=args.compute_repeats,
        trials=args.trials,
    )
    compute_results: list[dict[str, object]] = []
    for name, _ in compute_implementations:
        timing = compute_timings[name]
        pair_count = (
            args.sequence_length * (args.head_dim // 2) * args.compute_iterations
        )
        compute_results.append(
            {
                "implementation": name,
                **timing,
                "billion_sincos_pairs_per_second": (
                    pair_count / float(timing["microseconds"]) / 1_000.0
                ),
            }
        )
    compute_native_time = float(compute_results[0]["microseconds"])
    for row in compute_results:
        row["speedup_vs_native_sfu"] = compute_native_time / float(row["microseconds"])

    trial_order = [
        [
            name
            for name, _ in (
                implementations if trial % 2 == 0 else implementations[::-1]
            )
        ]
        for trial in range(args.trials)
    ]
    compute_trial_order = [
        [
            name
            for name, _ in (
                compute_implementations
                if trial % 2 == 0
                else compute_implementations[::-1]
            )
        ]
        for trial in range(args.trials)
    ]
    payload = {
        **rope_result_metadata(
            "rope-table-and-repeated-evaluator",
            repository_state=repository_state,
            attestations=attestations,
            source_bound=source_bound,
        ),
        "measurement": {
            "device_name": torch.cuda.get_device_name(device),
            "head_dim": args.head_dim,
            "sequence_length": args.sequence_length,
            "theta": args.theta,
            "compute_iterations": args.compute_iterations,
            "warmup": args.warmup,
            "table_repeats": args.repeats,
            "compute_repeats": args.compute_repeats,
            "trials": args.trials,
            "summary_statistic": "median of per-trial means",
            "table_measurement_order": trial_order,
            "compute_measurement_order": compute_trial_order,
        },
        "table_generation": results,
        "compute_saturated": compute_results,
        "results": {
            "table_generation": results,
            "compute_saturated": compute_results,
        },
    }

    print(
        f"{'implementation':>32} {'table us':>10} {'speedup':>9} "
        f"{'max abs':>11} {'phase':>11}"
    )
    for row in results:
        print(
            f"{row['implementation']:>32} {row['microseconds']:10.3f} "
            f"{row['speedup_vs_native_sfu']:9.3f} {row['max_abs']:11.3e} "
            f"{row['phase_max_abs_rad']:11.3e}"
        )
    print("\nCompute-saturated")
    print(f"{'implementation':>32} {'us':>10} {'speedup':>9} {'Gpair/s':>11}")
    for row in compute_results:
        print(
            f"{row['implementation']:>32} {row['microseconds']:10.3f} "
            f"{row['speedup_vs_native_sfu']:9.3f} "
            f"{row['billion_sincos_pairs_per_second']:11.3f}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

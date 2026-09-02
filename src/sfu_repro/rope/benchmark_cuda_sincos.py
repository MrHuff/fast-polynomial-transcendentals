# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified 2026-09-02 for the standalone SFU reproduction package.

"""Compare paired CUDA polynomial sin/cos against the native SFU path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .benchmark_polynomial_sincos import (
    attest_rope_sources,
    benchmark,
    load_spline_ops,
    make_rope_angles,
    numerical_metrics,
    rope_result_metadata,
)


def parse_element_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("element counts must be positive integers")
    return counts


def make_rope_angle_vector(
    element_count: int,
    *,
    head_dim: int,
    theta: float,
    device: torch.device,
) -> torch.Tensor:
    values_per_position = head_dim // 2
    sequence_length = (element_count + values_per_position - 1) // values_per_position
    return (
        make_rope_angles(
            head_dim=head_dim,
            max_seq_len=sequence_length,
            theta=theta,
            device=device,
        )
        .reshape(-1)[:element_count]
        .contiguous()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--element-counts",
        type=parse_element_counts,
        default=parse_element_counts("4096,65536,1048576,16777216,134217728"),
        help="Comma-separated sweep from cache-resident to HBM-resident sizes.",
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--theta", type=float, default=500_000.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--compute-element-count",
        type=int,
        default=262_144,
        help="Element count for the compute-saturated repeated-evaluation control.",
    )
    parser.add_argument(
        "--compute-iterations",
        type=int,
        default=64,
        help="Evaluations per loaded value in the compute-saturated control.",
    )
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
    if args.head_dim <= 0 or args.head_dim % 2:
        raise ValueError("head_dim must be a positive even integer")
    if args.compute_element_count <= 0 or args.compute_iterations <= 0:
        raise ValueError("compute element count and iterations must be positive")

    spline_ops = load_spline_ops()
    repository_state, attestations, source_bound = attest_rope_sources(
        spline_ops=spline_ops,
        allow_unbound_source=args.allow_unbound_source,
    )
    required_apis = (
        "sincos_native_bf16",
        "sincos_d3_d4_bf16",
        "sincos_d3_d4_quarter_turn_bf16",
        "sincos_d3_d4_half_turn_bf16",
        "sincos_d5_d6_half_turn_bf16",
        "sincos_d5_d4_half_turn_fp16_bf16",
        "sincos_native_fp16",
        "sincos_d3_d4_quarter_turn_fp16",
        "sincos_d3_d4_half_turn_fp16",
        "sincos_d5_d4_half_turn_fp16",
        "sincos_d5_d6_half_turn_fp16",
        "sincos_d7_d6_half_turn_fp16",
        "sincos_native_compute_f32",
        "sincos_d3_d4_compute_f32",
        "sincos_d3_d4_cycle_compute_f32",
        "sincos_d3_d4_magic_bias_compute_f32",
        "sincos_d5_d4_half_turn_ls_compute_f32",
        "sincos_d5_d4_half_turn_sollya_compute_f32",
        "sincos_d5_d4_half_turn_sollya_fast_compute_f32",
        "sincos_d5_d4_compute_f32",
        "sincos_d7_d6_compute_f32",
        "sincos_native_bf16_compute",
        "sincos_d3_d4_bf16_compute",
        "sincos_d3_d4_quarter_turn_bf16_compute",
        "sincos_d3_d4_half_turn_bf16_compute",
        "sincos_d5_d6_half_turn_bf16_compute",
        "sincos_d5_d4_half_turn_fp16_bf16_compute",
        "sincos_native_fp16_compute",
        "sincos_d3_d4_quarter_turn_fp16_compute",
        "sincos_d3_d4_half_turn_fp16_compute",
        "sincos_d5_d4_half_turn_fp16_compute",
        "sincos_d5_d6_half_turn_fp16_compute",
        "sincos_d7_d6_half_turn_fp16_compute",
    )
    missing_apis = [name for name in required_apis if not hasattr(spline_ops, name)]
    if missing_apis:
        raise RuntimeError(
            "spline_ops is stale and lacks "
            + ", ".join(missing_apis)
            + "; rebuild the CUDA extension."
        )
    implementations = (
        ("native_sfu", spline_ops.sincos_native_f32),
        ("poly_d3_d4", spline_ops.sincos_d3_d4_f32),
        ("poly_d3_d4_cycle", spline_ops.sincos_d3_d4_cycle_f32),
        ("poly_d3_d4_magic_bias", spline_ops.sincos_d3_d4_magic_bias_f32),
        (
            "half_turn_d5_d4_ls",
            spline_ops.sincos_d5_d4_half_turn_ls_f32,
        ),
        (
            "half_turn_d5_d4_sollya",
            spline_ops.sincos_d5_d4_half_turn_sollya_f32,
        ),
        (
            "half_turn_d5_d4_sollya_fast",
            spline_ops.sincos_d5_d4_half_turn_sollya_fast_f32,
        ),
        ("poly_d5_d4", spline_ops.sincos_d5_d4_f32),
        ("poly_d7_d6", spline_ops.sincos_d7_d6_f32),
    )
    bf16_implementations = (
        ("native_sfu_bf16", spline_ops.sincos_native_bf16),
        ("poly_d3_d4_bf16", spline_ops.sincos_d3_d4_bf16),
        (
            "quarter_turn_d3_d4_packed_bf16",
            spline_ops.sincos_d3_d4_quarter_turn_bf16,
        ),
        (
            "half_turn_d3_d4_packed_bf16",
            spline_ops.sincos_d3_d4_half_turn_bf16,
        ),
        (
            "half_turn_d5_d6_packed_bf16",
            spline_ops.sincos_d5_d6_half_turn_bf16,
        ),
        (
            "half_turn_d5_d4_fp16_bf16",
            spline_ops.sincos_d5_d4_half_turn_fp16_bf16,
        ),
    )
    fp16_implementations = (
        ("native_sfu_fp16", spline_ops.sincos_native_fp16),
        (
            "quarter_turn_d3_d4_packed_fp16",
            spline_ops.sincos_d3_d4_quarter_turn_fp16,
        ),
        (
            "half_turn_d3_d4_packed_fp16",
            spline_ops.sincos_d3_d4_half_turn_fp16,
        ),
        (
            "half_turn_d5_d4_packed_fp16",
            spline_ops.sincos_d5_d4_half_turn_fp16,
        ),
        (
            "half_turn_d5_d6_packed_fp16",
            spline_ops.sincos_d5_d6_half_turn_fp16,
        ),
        (
            "half_turn_d7_d6_packed_fp16",
            spline_ops.sincos_d7_d6_half_turn_fp16,
        ),
    )
    device = torch.device("cuda")
    results: list[dict[str, object]] = []

    for element_count in args.element_counts:
        angles = make_rope_angle_vector(
            element_count,
            head_dim=args.head_dim,
            theta=args.theta,
            device=device,
        )
        reference = torch.stack((torch.cos(angles), torch.sin(angles)))
        size_results: list[dict[str, object]] = []
        for name, function in implementations:
            timing, output = benchmark(
                function,
                angles,
                warmup=args.warmup,
                repeats=args.repeats,
                trials=args.trials,
            )
            size_results.append(
                {
                    "implementation": name,
                    **timing,
                    "million_values_per_second": (
                        element_count / timing["microseconds"]
                    ),
                    **numerical_metrics(output, reference),
                }
            )

        native_time = float(size_results[0]["microseconds"])
        for row in size_results:
            row["speedup_vs_native_sfu"] = native_time / float(row["microseconds"])
        bf16_size_results: list[dict[str, object]] = []
        for name, function in bf16_implementations:
            timing, output = benchmark(
                function,
                angles,
                warmup=args.warmup,
                repeats=args.repeats,
                trials=args.trials,
            )
            bf16_size_results.append(
                {
                    "implementation": name,
                    **timing,
                    "million_values_per_second": (
                        element_count / timing["microseconds"]
                    ),
                    **numerical_metrics(output.float(), reference),
                }
            )
        native_bf16_time = float(bf16_size_results[0]["microseconds"])
        for row in bf16_size_results:
            row["speedup_vs_native_sfu_bf16"] = native_bf16_time / float(
                row["microseconds"]
            )
        fp16_size_results: list[dict[str, object]] = []
        for name, function in fp16_implementations:
            timing, output = benchmark(
                function,
                angles,
                warmup=args.warmup,
                repeats=args.repeats,
                trials=args.trials,
            )
            fp16_size_results.append(
                {
                    "implementation": name,
                    **timing,
                    "million_values_per_second": (
                        element_count / timing["microseconds"]
                    ),
                    **numerical_metrics(output.float(), reference),
                }
            )
        native_fp16_time = float(fp16_size_results[0]["microseconds"])
        for row in fp16_size_results:
            row["speedup_vs_native_sfu_fp16"] = native_fp16_time / float(
                row["microseconds"]
            )
        results.append(
            {
                "element_count": element_count,
                "working_set_mib": element_count * 12 / (1024 * 1024),
                "working_set_mib_bf16": element_count * 8 / (1024 * 1024),
                "angle_max": angles.max().item(),
                "implementations": size_results,
                "bf16_implementations": bf16_size_results,
                "fp16_implementations": fp16_size_results,
            }
        )
        del angles, reference

    compute_angles = make_rope_angle_vector(
        args.compute_element_count,
        head_dim=args.head_dim,
        theta=args.theta,
        device=device,
    )
    compute_implementations = (
        ("native_sfu", spline_ops.sincos_native_compute_f32),
        ("poly_d3_d4", spline_ops.sincos_d3_d4_compute_f32),
        ("poly_d3_d4_cycle", spline_ops.sincos_d3_d4_cycle_compute_f32),
        (
            "poly_d3_d4_magic_bias",
            spline_ops.sincos_d3_d4_magic_bias_compute_f32,
        ),
        (
            "half_turn_d5_d4_ls",
            spline_ops.sincos_d5_d4_half_turn_ls_compute_f32,
        ),
        (
            "half_turn_d5_d4_sollya",
            spline_ops.sincos_d5_d4_half_turn_sollya_compute_f32,
        ),
        (
            "half_turn_d5_d4_sollya_fast",
            spline_ops.sincos_d5_d4_half_turn_sollya_fast_compute_f32,
        ),
        ("poly_d5_d4", spline_ops.sincos_d5_d4_compute_f32),
        ("poly_d7_d6", spline_ops.sincos_d7_d6_compute_f32),
    )
    compute_results: list[dict[str, object]] = []
    for name, function in compute_implementations:
        timing, _ = benchmark(
            lambda values, function=function: function(values, args.compute_iterations),
            compute_angles,
            warmup=args.warmup,
            repeats=args.repeats,
            trials=args.trials,
        )
        evaluation_count = args.compute_element_count * args.compute_iterations
        compute_results.append(
            {
                "implementation": name,
                **timing,
                "billion_pairs_per_second": (
                    evaluation_count / timing["microseconds"] / 1_000.0
                ),
            }
        )
    compute_native_time = float(compute_results[0]["microseconds"])
    for row in compute_results:
        row["speedup_vs_native_sfu"] = compute_native_time / float(row["microseconds"])
    compute_bf16_implementations = (
        ("native_sfu_bf16", spline_ops.sincos_native_bf16_compute),
        ("poly_d3_d4_bf16", spline_ops.sincos_d3_d4_bf16_compute),
        (
            "quarter_turn_d3_d4_packed_bf16",
            spline_ops.sincos_d3_d4_quarter_turn_bf16_compute,
        ),
        (
            "half_turn_d3_d4_packed_bf16",
            spline_ops.sincos_d3_d4_half_turn_bf16_compute,
        ),
        (
            "half_turn_d5_d6_packed_bf16",
            spline_ops.sincos_d5_d6_half_turn_bf16_compute,
        ),
        (
            "half_turn_d5_d4_fp16_bf16",
            spline_ops.sincos_d5_d4_half_turn_fp16_bf16_compute,
        ),
    )
    compute_bf16_results: list[dict[str, object]] = []
    for name, function in compute_bf16_implementations:
        timing, _ = benchmark(
            lambda values, function=function: function(values, args.compute_iterations),
            compute_angles,
            warmup=args.warmup,
            repeats=args.repeats,
            trials=args.trials,
        )
        evaluation_count = args.compute_element_count * args.compute_iterations
        compute_bf16_results.append(
            {
                "implementation": name,
                **timing,
                "billion_pairs_per_second": (
                    evaluation_count / timing["microseconds"] / 1_000.0
                ),
            }
        )
    compute_native_bf16_time = float(compute_bf16_results[0]["microseconds"])
    for row in compute_bf16_results:
        row["speedup_vs_native_sfu_bf16"] = compute_native_bf16_time / float(
            row["microseconds"]
        )
    compute_fp16_implementations = (
        ("native_sfu_fp16", spline_ops.sincos_native_fp16_compute),
        (
            "quarter_turn_d3_d4_packed_fp16",
            spline_ops.sincos_d3_d4_quarter_turn_fp16_compute,
        ),
        (
            "half_turn_d3_d4_packed_fp16",
            spline_ops.sincos_d3_d4_half_turn_fp16_compute,
        ),
        (
            "half_turn_d5_d4_packed_fp16",
            spline_ops.sincos_d5_d4_half_turn_fp16_compute,
        ),
        (
            "half_turn_d5_d6_packed_fp16",
            spline_ops.sincos_d5_d6_half_turn_fp16_compute,
        ),
        (
            "half_turn_d7_d6_packed_fp16",
            spline_ops.sincos_d7_d6_half_turn_fp16_compute,
        ),
    )
    compute_fp16_results: list[dict[str, object]] = []
    for name, function in compute_fp16_implementations:
        timing, _ = benchmark(
            lambda values, function=function: function(values, args.compute_iterations),
            compute_angles,
            warmup=args.warmup,
            repeats=args.repeats,
            trials=args.trials,
        )
        evaluation_count = args.compute_element_count * args.compute_iterations
        compute_fp16_results.append(
            {
                "implementation": name,
                **timing,
                "billion_pairs_per_second": (
                    evaluation_count / timing["microseconds"] / 1_000.0
                ),
            }
        )
    compute_native_fp16_time = float(compute_fp16_results[0]["microseconds"])
    for row in compute_fp16_results:
        row["speedup_vs_native_sfu_fp16"] = compute_native_fp16_time / float(
            row["microseconds"]
        )

    payload = {
        **rope_result_metadata(
            "rope-cache-hbm-and-repeated-evaluation",
            repository_state=repository_state,
            attestations=attestations,
            source_bound=source_bound,
        ),
        "measurement": {
            "device_name": torch.cuda.get_device_name(device),
            "head_dim": args.head_dim,
            "theta": args.theta,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "trials": args.trials,
            "compute_element_count": args.compute_element_count,
            "compute_iterations": args.compute_iterations,
            "summary_statistic": "median of per-trial means",
            "size_sweep_order": list(args.element_counts),
            "implementation_order": [name for name, _ in implementations],
            "bf16_implementation_order": [name for name, _ in bf16_implementations],
            "fp16_implementation_order": [name for name, _ in fp16_implementations],
            "compute_implementation_order": [
                name for name, _ in compute_implementations
            ],
            "compute_bf16_implementation_order": [
                name for name, _ in compute_bf16_implementations
            ],
            "compute_fp16_implementation_order": [
                name for name, _ in compute_fp16_implementations
            ],
        },
        "results": results,
        "compute_saturated": compute_results,
        "compute_saturated_bf16": compute_bf16_results,
        "compute_saturated_fp16": compute_fp16_results,
    }
    print(
        f"{'elements':>12} {'working MiB':>12} {'implementation':>26} "
        f"{'us':>10} {'speedup':>10} {'max error':>12}"
    )
    for size in results:
        for row in size["implementations"]:
            print(
                f"{size['element_count']:12d} {size['working_set_mib']:12.2f} "
                f"{row['implementation']:>26} {row['microseconds']:10.3f} "
                f"{row['speedup_vs_native_sfu']:10.3f} {row['max_abs']:12.3e}"
            )
        for row in size["bf16_implementations"]:
            print(
                f"{size['element_count']:12d} {size['working_set_mib_bf16']:12.2f} "
                f"{row['implementation']:>26} {row['microseconds']:10.3f} "
                f"{row['speedup_vs_native_sfu_bf16']:10.3f} "
                f"{row['max_abs']:12.3e}"
            )
        for row in size["fp16_implementations"]:
            print(
                f"{size['element_count']:12d} {size['working_set_mib_bf16']:12.2f} "
                f"{row['implementation']:>26} {row['microseconds']:10.3f} "
                f"{row['speedup_vs_native_sfu_fp16']:10.3f} "
                f"{row['max_abs']:12.3e}"
            )
    print(
        f"\nCompute-saturated control: {args.compute_element_count:,} values x "
        f"{args.compute_iterations} evaluations"
    )
    print(f"{'implementation':>26} {'us':>10} {'speedup':>10} " f"{'Gpair/s':>12}")
    for row in compute_results:
        print(
            f"{row['implementation']:>26} {row['microseconds']:10.3f} "
            f"{row['speedup_vs_native_sfu']:10.3f} "
            f"{row['billion_pairs_per_second']:12.3f}"
        )
    print("\nCompute-saturated BF16-output control")
    print(f"{'implementation':>26} {'us':>10} {'speedup':>10} " f"{'Gpair/s':>12}")
    for row in compute_bf16_results:
        print(
            f"{row['implementation']:>26} {row['microseconds']:10.3f} "
            f"{row['speedup_vs_native_sfu_bf16']:10.3f} "
            f"{row['billion_pairs_per_second']:12.3f}"
        )
    print("\nCompute-saturated FP16-output control")
    print(f"{'implementation':>26} {'us':>10} {'speedup':>10} " f"{'Gpair/s':>12}")
    for row in compute_fp16_results:
        print(
            f"{row['implementation']:>26} {row['microseconds']:10.3f} "
            f"{row['speedup_vs_native_sfu_fp16']:10.3f} "
            f"{row['billion_pairs_per_second']:12.3f}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

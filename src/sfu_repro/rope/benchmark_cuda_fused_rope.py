# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified 2026-09-02 for the standalone SFU reproduction package.

"""Benchmark fused FP16 Q/K rotary embedding implementations on CUDA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch

from .benchmark_cuda_rope_sincos import benchmark_interleaved
from .benchmark_polynomial_sincos import (
    attest_rope_sources,
    load_spline_ops,
    rope_result_metadata,
)


RopeOutput = tuple[torch.Tensor, torch.Tensor]
RopeCall = Callable[[], RopeOutput]


def parse_head_configs(value: str) -> tuple[tuple[int, int], ...]:
    configs: list[tuple[int, int]] = []
    for item in value.split(","):
        try:
            q_heads, k_heads = (int(part) for part in item.split(":"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "head configs must use q_heads:k_heads pairs"
            ) from exc
        if q_heads <= 0 or k_heads <= 0:
            raise argparse.ArgumentTypeError("head counts must be positive")
        configs.append((q_heads, k_heads))
    if not configs:
        raise argparse.ArgumentTypeError("at least one head config is required")
    return tuple(configs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--theta", type=float, default=500_000.0)
    parser.add_argument(
        "--head-configs",
        type=parse_head_configs,
        default=parse_head_configs("1:1,8:2,32:8"),
        help="comma-separated q_heads:k_heads pairs",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-unbound-source",
        action="store_true",
        help="Permit a diagnostic result from dirty or non-local source.",
    )
    return parser.parse_args()


def rope_reference(
    values: torch.Tensor,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    batch_size, sequence_length, head_count, head_dim = values.shape
    positions = torch.arange(
        sequence_length,
        dtype=torch.float64,
        device=values.device,
    )
    angles = positions[:, None] * frequencies[None, :]
    cos = torch.cos(angles).float()[None, :, None, :]
    sin = torch.sin(angles).float()[None, :, None, :]
    pairs = values.float().reshape(
        batch_size,
        sequence_length,
        head_count,
        head_dim // 2,
        2,
    )
    even = pairs[..., 0]
    odd = pairs[..., 1]
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        dim=-1,
    ).flatten(-2)
    return rotated.transpose(1, 2).contiguous()


def output_metrics(
    output: RopeOutput,
    reference: RopeOutput,
) -> dict[str, float]:
    squared_error = 0.0
    element_count = 0
    max_abs = 0.0
    for actual, expected in zip(output, reference, strict=True):
        delta = actual.float() - expected
        max_abs = max(max_abs, float(delta.abs().max().item()))
        squared_error += float(torch.sum(delta.square()).item())
        element_count += delta.numel()
    return {
        "max_abs": max_abs,
        "rmse": math.sqrt(squared_error / element_count),
    }


@torch.inference_mode()
def benchmark_config(
    *,
    spline_ops,
    batch_size: int,
    sequence_length: int,
    q_head_count: int,
    k_head_count: int,
    head_dim: int,
    frequencies: torch.Tensor,
    frequencies_fp32: torch.Tensor,
    phase_increments: torch.Tensor,
    rope_table: torch.Tensor,
    warmup: int,
    repeats: int,
    trials: int,
    seed: int,
) -> list[dict[str, object]]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn(
        batch_size,
        sequence_length,
        q_head_count,
        head_dim,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn(
        batch_size,
        sequence_length,
        k_head_count,
        head_dim,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    reference = (
        rope_reference(q, frequencies),
        rope_reference(k, frequencies),
    )
    implementations: tuple[tuple[str, RopeCall], ...] = (
        (
            "cached_table_fp16",
            lambda: tuple(spline_ops.rope_apply_cached_fp16(q, k, rope_table)),
        ),
        (
            "native_sfu_on_the_fly_fp16",
            lambda: tuple(spline_ops.rope_apply_native_fp16(q, k, frequencies_fp32)),
        ),
        (
            "q32_half_d5_d6_on_the_fly_fp16",
            lambda: tuple(
                spline_ops.rope_apply_fixed_half_turn_d5_d6_fp16(
                    q,
                    k,
                    phase_increments,
                )
            ),
        ),
    )
    timings, outputs = benchmark_interleaved(
        implementations,
        warmup=warmup,
        repeats=repeats,
        trials=trials,
    )
    cached_time = timings["cached_table_fp16"]["microseconds"]
    native_time = timings["native_sfu_on_the_fly_fp16"]["microseconds"]
    qk_bytes = 2 * (q.numel() + k.numel()) * q.element_size()
    rows: list[dict[str, object]] = []
    for name, _ in implementations:
        microseconds = timings[name]["microseconds"]
        rows.append(
            {
                "implementation": name,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "q_head_count": q_head_count,
                "k_head_count": k_head_count,
                "head_dim": head_dim,
                **timings[name],
                "speedup_vs_cached": cached_time / microseconds,
                "speedup_vs_native_sfu": native_time / microseconds,
                "effective_qk_gb_per_s": qk_bytes / microseconds / 1_000.0,
                **output_metrics(outputs[name], reference),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size <= 0 or args.sequence_length <= 0:
        raise ValueError("batch size and sequence length must be positive")
    if args.head_dim <= 0 or args.head_dim % 8:
        raise ValueError("head_dim must be a positive multiple of eight")

    spline_ops = load_spline_ops()
    repository_state, attestations, source_bound = attest_rope_sources(
        spline_ops=spline_ops,
        allow_unbound_source=args.allow_unbound_source,
    )
    required = (
        "rope_apply_cached_fp16",
        "rope_apply_native_fp16",
        "rope_apply_fixed_half_turn_d5_d6_fp16",
    )
    missing = [name for name in required if not hasattr(spline_ops, name)]
    if missing:
        raise RuntimeError(
            "spline_ops is stale and lacks "
            + ", ".join(missing)
            + "; rebuild the CUDA extension."
        )

    dimensions = torch.arange(0, args.head_dim, 2, dtype=torch.float64)
    frequencies = (args.theta ** (-dimensions / args.head_dim)).to(device="cuda")
    frequencies_fp32 = frequencies.float()
    phase_scale = float(1 << 32) / (2.0 * math.pi)
    phase_increments = torch.round(frequencies * phase_scale).to(torch.int32)
    rope_table = spline_ops.rope_sincos_native_fp16(
        frequencies_fp32,
        args.sequence_length,
    )

    results: list[dict[str, object]] = []
    for index, (q_head_count, k_head_count) in enumerate(args.head_configs):
        results.extend(
            benchmark_config(
                spline_ops=spline_ops,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                q_head_count=q_head_count,
                k_head_count=k_head_count,
                head_dim=args.head_dim,
                frequencies=frequencies,
                frequencies_fp32=frequencies_fp32,
                phase_increments=phase_increments,
                rope_table=rope_table,
                warmup=args.warmup,
                repeats=args.repeats,
                trials=args.trials,
                seed=args.seed + index,
            )
        )

    print(
        f"{'QH:KH':>7} {'implementation':>38} {'us':>9} "
        f"{'vs SFU':>8} {'vs cache':>9} {'GB/s':>9} {'max abs':>11}"
    )
    for row in results:
        heads = f"{row['q_head_count']}:{row['k_head_count']}"
        print(
            f"{heads:>7} {row['implementation']:>38} "
            f"{row['microseconds']:9.3f} "
            f"{row['speedup_vs_native_sfu']:8.3f} "
            f"{row['speedup_vs_cached']:9.3f} "
            f"{row['effective_qk_gb_per_s']:9.1f} "
            f"{row['max_abs']:11.3e}"
        )

    measurement_order = [
        [
            "cached_table_fp16",
            "native_sfu_on_the_fly_fp16",
            "q32_half_d5_d6_on_the_fly_fp16",
        ][:: 1 if trial % 2 == 0 else -1]
        for trial in range(args.trials)
    ]
    payload = {
        **rope_result_metadata(
            "rope-fused-integration",
            repository_state=repository_state,
            attestations=attestations,
            source_bound=source_bound,
        ),
        "measurement": {
            "device_name": torch.cuda.get_device_name(),
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "head_dim": args.head_dim,
            "theta": args.theta,
            "head_configs": args.head_configs,
            "cached_table_generation_included": False,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "trials": args.trials,
            "seed": args.seed,
            "summary_statistic": "median of per-trial means",
            "measurement_order_per_head_config": measurement_order,
        },
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

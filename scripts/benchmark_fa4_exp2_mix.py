#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
#
# Modified in 2026 for the standalone paper artifact: this script now uses the
# standalone FA4 config and contains no training-package or path assumptions.
"""Benchmark mixed SFU/software exp2 schedules in causal FA4 softmax.

The default geometry matches one Llama 8B attention layer. Variant syntax is
``name=backend:forward_frequency:backward_frequency``. Zero is all SFU; a
positive frequency N routes four of every N lanes to the selected software
backend. ``auto`` is valid only for the forward frequency and retains FA4's
geometry-dependent route.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from sfu_repro.fa4 import (
    DEFAULT_EXP2_VARIANTS,
    Exp2Variant,
    build_fa4_module,
    parse_exp2_frequency,
    parse_exp2_variants,
    softmax_exp2_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Compatibility aliases keep the original benchmark's import-level API small.
Variant = Exp2Variant
parse_frequency = parse_exp2_frequency
parse_variants = parse_exp2_variants


def git_revision(repository: Path = REPOSITORY_ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_worktree_state(
    repository: Path = REPOSITORY_ROOT,
) -> tuple[bool | None, int | None]:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        entries = tuple(entry for entry in status.split("\0") if entry)
        untracked_files = sum(entry.startswith("?? ") for entry in entries)
        return bool(entries), untracked_files
    except (OSError, subprocess.CalledProcessError):
        return None, None


def flash_attention_revision(repository: Path = REPOSITORY_ROOT) -> str | None:
    revision = git_revision(repository / "flash-attention")
    return revision if len(revision) == 40 else None


def build_module(
    variant: Exp2Variant, *, head_dim: int, sequence_length: int
) -> torch.nn.Module:
    return (
        build_fa4_module(
            softmax_exp2_config(variant),
            head_dim=head_dim,
            sequence_length=sequence_length,
        )
        .cuda()
        .to(torch.bfloat16)
    )


def elapsed_samples(
    callables: dict[str, Callable[[], object]],
    *,
    warmup: int,
    iterations: int,
    rounds: int,
) -> tuple[dict[str, list[float]], list[list[str]]]:
    names = tuple(callables)
    for name in names:
        for _ in range(warmup):
            callables[name]()
        torch.cuda.synchronize()

    samples = {name: [] for name in names}
    measurement_order = []
    for round_index in range(rounds):
        offset = round_index % len(names)
        round_order = list(names[offset:] + names[:offset])
        if round_index % 2:
            round_order.reverse()
        measurement_order.append(round_order)
        for name in round_order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                callables[name]()
            end.record()
            end.synchronize()
            samples[name].append(float(start.elapsed_time(end)) / iterations)
    return samples, measurement_order


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    return (
        (actual_f32 - reference_f32).norm() / reference_f32.norm().clamp_min(1e-30)
    ).item()


def relative_mae(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    return (
        (actual_f32 - reference_f32).abs().mean()
        / reference_f32.abs().mean().clamp_min(1e-30)
    ).item()


def cosine(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        actual.float().flatten(), reference.float().flatten(), dim=0
    ).item()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--variants", default=DEFAULT_EXP2_VARIANTS)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "fa4-exp2-mix.json",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    variants = parse_exp2_variants(args.variants)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("this benchmark requires an SM100 or newer CUDA GPU")
    if args.warmup < 0 or args.iterations < 1 or args.rounds < 1:
        raise ValueError(
            "warmup must be non-negative; iterations and rounds must be positive"
        )

    torch.manual_seed(args.seed)
    q_shape = (
        args.batch_size,
        args.heads,
        args.sequence_length,
        args.head_dim,
    )
    kv_shape = (
        args.batch_size,
        args.kv_heads,
        args.sequence_length,
        args.head_dim,
    )
    q = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
    modules = {
        variant.name: build_module(
            variant,
            head_dim=args.head_dim,
            sequence_length=args.sequence_length,
        )
        for variant in variants
    }

    outputs: dict[str, torch.Tensor] = {}
    gradients: dict[str, tuple[torch.Tensor, ...]] = {}
    for variant in variants:
        module = modules[variant.name]
        local_inputs = tuple(
            tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v)
        )
        output = module(*local_inputs)
        outputs[variant.name] = output.detach()
        if not args.forward_only:
            gradients[variant.name] = tuple(
                gradient.detach()
                for gradient in torch.autograd.grad(output, local_inputs, dout)
            )
        torch.cuda.synchronize()

    forward_callables = {
        variant.name: (lambda module=modules[variant.name]: module(q, k, v))
        for variant in variants
    }
    with torch.no_grad():
        forward_samples, forward_order = elapsed_samples(
            forward_callables,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
        )

    forward_backward_samples = None
    forward_backward_order = None
    if not args.forward_only:
        forward_backward_callables = {}
        for variant in variants:
            module = modules[variant.name]
            q_bwd = q.detach().clone().requires_grad_(True)
            k_bwd = k.detach().clone().requires_grad_(True)
            v_bwd = v.detach().clone().requires_grad_(True)

            def forward_backward(
                module=module,
                q_bwd=q_bwd,
                k_bwd=k_bwd,
                v_bwd=v_bwd,
            ):
                output = module(q_bwd, k_bwd, v_bwd)
                torch.autograd.backward(output, dout, inputs=(q_bwd, k_bwd, v_bwd))
                q_bwd.grad = None
                k_bwd.grad = None
                v_bwd.grad = None

            forward_backward_callables[variant.name] = forward_backward
        forward_backward_samples, forward_backward_order = elapsed_samples(
            forward_backward_callables,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
        )

    reference_name = variants[0].name
    reference_forward_samples = forward_samples[reference_name]
    reference_forward_backward_samples = (
        forward_backward_samples[reference_name]
        if forward_backward_samples is not None
        else None
    )
    rows = []
    for variant in variants:
        name = variant.name
        compared_outputs = ((outputs[name], outputs[reference_name]),)
        compared_gradients = (
            tuple(zip(gradients[name], gradients[reference_name])) if gradients else ()
        )
        compared_tensors = compared_outputs + compared_gradients
        forward_ms = statistics.median(forward_samples[name])
        forward_backward_ms = (
            statistics.median(forward_backward_samples[name])
            if forward_backward_samples is not None
            else None
        )
        rows.append(
            {
                **asdict(variant),
                "nominal_forward_polynomial_fraction": (
                    variant.nominal_forward_polynomial_fraction
                ),
                "nominal_backward_polynomial_fraction": (
                    variant.nominal_backward_polynomial_fraction
                ),
                "forward_ms": forward_ms,
                "forward_samples_ms": forward_samples[name],
                "forward_paired_speedup": statistics.median(
                    reference / actual
                    for reference, actual in zip(
                        reference_forward_samples, forward_samples[name]
                    )
                ),
                "forward_backward_ms": forward_backward_ms,
                "forward_backward_samples_ms": (
                    forward_backward_samples[name]
                    if forward_backward_samples is not None
                    else None
                ),
                "forward_backward_paired_speedup": (
                    statistics.median(
                        reference / actual
                        for reference, actual in zip(
                            reference_forward_backward_samples,
                            forward_backward_samples[name],
                        )
                    )
                    if reference_forward_backward_samples is not None
                    else None
                ),
                "output_relative_l2": relative_l2(
                    outputs[name], outputs[reference_name]
                ),
                "output_relative_mae": relative_mae(
                    outputs[name], outputs[reference_name]
                ),
                "max_tensor_relative_l2": max(
                    relative_l2(actual, reference)
                    for actual, reference in compared_tensors
                ),
                "max_tensor_relative_mae": max(
                    relative_mae(actual, reference)
                    for actual, reference in compared_tensors
                ),
                "min_tensor_cosine": min(
                    cosine(actual, reference) for actual, reference in compared_tensors
                ),
            }
        )

    properties = torch.cuda.get_device_properties(0)
    dirty, untracked_files = git_worktree_state()
    return {
        "schema_version": 1,
        "experiment": {
            "id": "fa4-exp2-mix",
            "reference_variant": reference_name,
            "provenance_class": "new-measurement",
        },
        "source": {
            "repository": "MrHuff/fast-polynomial-transcendentals",
            "revision": (revision if len(revision := git_revision()) == 40 else None),
            "dirty": dirty,
            "untracked_files": untracked_files,
            "external_components": {
                "flash-attention": flash_attention_revision(),
            },
        },
        "environment": {
            "gpu": properties.name,
            "compute_capability": ".".join(
                str(part) for part in torch.cuda.get_device_capability(0)
            ),
            "total_memory_bytes": properties.total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "measurement": {
            "dtype": "bfloat16",
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "rounds": args.rounds,
            "summary_statistic": "median of per-round means",
            "forward_measurement_order": forward_order,
            "forward_backward_measurement_order": forward_backward_order,
            "geometry": {
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "query_heads": args.heads,
                "kv_heads": args.kv_heads,
                "head_dim": args.head_dim,
            },
        },
        "results": rows,
    }


def print_results(rows: list[dict[str, Any]]) -> None:
    print(
        "variant          fwd_emu bwd_emu  fwd_ms  fwd_x  fwd+bwd_ms  "
        "step_x  out_rel_l2  max_rel_l2  min_cos"
    )
    for row in rows:
        forward_backward_ms = row["forward_backward_ms"]
        step_speedup = row["forward_backward_paired_speedup"]
        step_speedup_text = "-" if step_speedup is None else f"{step_speedup:.3f}x"
        forward_fraction = row["nominal_forward_polynomial_fraction"]
        forward_fraction_text = (
            "auto" if forward_fraction is None else f"{100 * forward_fraction:.1f}%"
        )
        print(
            f"{row['name']:<16} "
            f"{forward_fraction_text:>7} "
            f"{100 * row['nominal_backward_polynomial_fraction']:6.1f}% "
            f"{row['forward_ms']:7.4f} "
            f"{row['forward_paired_speedup']:6.3f}x "
            f"{('-' if forward_backward_ms is None else f'{forward_backward_ms:.4f}'):>10} "
            f"{step_speedup_text:>7} "
            f"{row['output_relative_l2']:11.4e} "
            f"{row['max_tensor_relative_l2']:11.4e} "
            f"{row['min_tensor_cosine']:9.7f}"
        )


def main() -> int:
    args = get_parser().parse_args()
    payload = run(args)
    print_results(payload["results"])
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nJSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
#
# Modified in 2026 for the standalone paper artifact: training configuration,
# trainer imports, and sibling-repository path manipulation were removed.
"""Run matched B1--B4 forward/backward component probes on CUDA.

B1 compares one-kernel SiLU evaluators. B2 compares native SFU tanh with the
device D4 fit and analytical derivative at the full FA4 boundary. B3 compares
native SFU sigmoid attention with the direct device D3 forward/coupled D4
derivative at sequence length 4096. B4 compares one-kernel native and D3
polynomial SwiGLU. Measurement order rotates each round.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from sfu_repro.activations import load_spline_ops
from sfu_repro.fa4 import (
    FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH,
    FA4AttentionWrapper,
    b2_component_configs,
    b3_component_configs,
    resolve_fa4_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
Call = Callable[[], object]


def rotated_order(labels: Sequence[str], round_index: int) -> list[str]:
    if not labels:
        raise ValueError("at least one label is required")
    offset = round_index % len(labels)
    return list(labels[offset:]) + list(labels[:offset])


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_worktree_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def measure_callables(
    callables: dict[str, Call],
    *,
    warmup: int,
    iterations: int,
    rounds: int,
) -> dict[str, Any]:
    labels = tuple(callables)
    for label in labels:
        for _ in range(warmup):
            callables[label]()
        torch.cuda.synchronize()

    samples_us = {label: [] for label in labels}
    orders: list[list[str]] = []
    for round_index in range(rounds):
        order = rotated_order(labels, round_index)
        orders.append(order)
        for label in order:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                callables[label]()
            stop.record()
            stop.synchronize()
            samples_us[label].append(
                float(start.elapsed_time(stop)) * 1000.0 / iterations
            )

    medians_us = {
        label: statistics.median(samples) for label, samples in samples_us.items()
    }
    result: dict[str, Any] = {
        "samples_us": samples_us,
        "median_us": medians_us,
        "measurement_order": orders,
    }
    if "native" in medians_us and "polynomial" in medians_us:
        result["speedup"] = medians_us["native"] / medians_us["polynomial"]
    return result


def tensor_error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    absolute = (actual_f32 - reference_f32).abs()
    return {
        "max_abs": absolute.max().item(),
        "mean_abs": absolute.mean().item(),
        "relative_mae": (
            absolute.mean() / reference_f32.abs().mean().clamp_min(1e-12)
        ).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            actual_f32.flatten(), reference_f32.flatten(), dim=0
        ).item(),
    }


def benchmark_b1(
    *, elements: int, warmup: int, iterations: int, rounds: int, seed: int
) -> dict[str, Any]:
    spline_ops = load_spline_ops()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(elements, device="cuda", dtype=torch.bfloat16, generator=generator)
    grad_out = torch.randn(
        elements, device="cuda", dtype=torch.bfloat16, generator=generator
    )

    native_y = spline_ops.swish_fwd_variant(x, 0, 0)
    polynomial_y = spline_ops.swish_fwd_variant(x, 3, 0)
    native_grad = spline_ops.swish_bwd_variant(grad_out, x, 0, 0)
    polynomial_grad = spline_ops.swish_bwd_variant(grad_out, x, 3, 0)
    return {
        "case": "B1",
        "operation": "SiLU evaluator",
        "scope": "one-kernel BF16 activation",
        "polynomial": "D3 current fit",
        "shape": {"elements": elements},
        "forward": measure_callables(
            {
                "native": lambda: spline_ops.swish_fwd_variant(x, 0, 0),
                "polynomial": lambda: spline_ops.swish_fwd_variant(x, 3, 0),
            },
            warmup=warmup,
            iterations=iterations,
            rounds=rounds,
        ),
        "backward": measure_callables(
            {
                "native": lambda: spline_ops.swish_bwd_variant(grad_out, x, 0, 0),
                "polynomial": lambda: spline_ops.swish_bwd_variant(grad_out, x, 3, 0),
            },
            warmup=warmup,
            iterations=iterations,
            rounds=rounds,
        ),
        "numerics": {
            "forward": tensor_error(polynomial_y, native_y),
            "backward": tensor_error(polynomial_grad, native_grad),
        },
    }


def build_fa4_modules(
    configs: dict[str, Any], *, sequence_length: int, head_dim: int
) -> dict[str, torch.nn.Module]:
    return {
        label: FA4AttentionWrapper(
            resolve_fa4_config(config, sequence_length=sequence_length),
            head_dim=head_dim,
        )
        .cuda()
        .to(torch.bfloat16)
        for label, config in configs.items()
    }


def benchmark_fa4_pair(
    *,
    case: str,
    operation: str,
    polynomial: str,
    configs: dict[str, Any],
    configuration: dict[str, Any],
    sequence_length: int,
    warmup: int,
    iterations: int,
    rounds: int,
    seed: int,
) -> dict[str, Any]:
    modules = build_fa4_modules(configs, sequence_length=sequence_length, head_dim=128)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        1,
        32,
        sequence_length,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    k = torch.randn(
        1,
        8,
        sequence_length,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    v = torch.randn(
        1,
        8,
        sequence_length,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    dout = torch.randn(
        q.shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )

    outputs: dict[str, torch.Tensor] = {}
    gradients: dict[str, tuple[torch.Tensor, ...]] = {}
    backward_calls: dict[str, Call] = {}
    for label, module in modules.items():
        local_inputs = tuple(
            tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v)
        )
        output = module(*local_inputs)
        outputs[label] = output.detach()
        gradients[label] = tuple(
            gradient.detach()
            for gradient in torch.autograd.grad(
                output, local_inputs, dout, retain_graph=True
            )
        )

        def backward(output=output, local_inputs=local_inputs):
            return torch.autograd.grad(output, local_inputs, dout, retain_graph=True)

        backward_calls[label] = backward

    compared = {
        "output": (outputs["polynomial"], outputs["native"]),
        "dq": (gradients["polynomial"][0], gradients["native"][0]),
        "dk": (gradients["polynomial"][1], gradients["native"][1]),
        "dv": (gradients["polynomial"][2], gradients["native"][2]),
    }
    errors = {
        name: tensor_error(actual, reference)
        for name, (actual, reference) in compared.items()
    }
    with torch.no_grad():
        forward = measure_callables(
            {
                label: (lambda module=module: module(q, k, v))
                for label, module in modules.items()
            },
            warmup=warmup,
            iterations=iterations,
            rounds=rounds,
        )
    backward = measure_callables(
        backward_calls, warmup=warmup, iterations=iterations, rounds=rounds
    )
    return {
        "case": case,
        "operation": operation,
        "scope": "complete causal grouped-query attention",
        "polynomial": polynomial,
        "shape": {
            "batch": 1,
            "query_heads": 32,
            "kv_heads": 8,
            "sequence_length": sequence_length,
            "head_dim": 128,
        },
        "configuration": configuration,
        "forward": forward,
        "backward": backward,
        "numerics": {
            "tensors": errors,
            "max_tensor_relative_mae": max(
                error["relative_mae"] for error in errors.values()
            ),
            "min_tensor_cosine": min(error["cosine"] for error in errors.values()),
        },
    }


def benchmark_b2(
    *, sequence_length: int, warmup: int, iterations: int, rounds: int, seed: int
) -> dict[str, Any]:
    return benchmark_fa4_pair(
        case="B2",
        operation="tanh softcap inside FA4",
        polynomial="device D4 current fit, analytical derivative",
        configs=b2_component_configs(),
        configuration={
            "native": "SFU tanh",
            "polynomial": "device D4 analytical",
            "softcap": 30.0,
        },
        sequence_length=sequence_length,
        warmup=warmup,
        iterations=iterations,
        rounds=rounds,
        seed=seed,
    )


def benchmark_b3(
    *, sequence_length: int, warmup: int, iterations: int, rounds: int, seed: int
) -> dict[str, Any]:
    configs = b3_component_configs(sequence_length=sequence_length)
    return benchmark_fa4_pair(
        case="B3",
        operation="direct D3/D4 FlashSigmoid inside FA4",
        polynomial="direct device D3 forward and coupled D4 derivative",
        configs=configs,
        configuration={
            "native": "SFU sigmoid",
            "polynomial": "direct device D3/D4",
            "sigmoid_bias": "-log(sequence_length)",
            "sigmoid_qk_norm": True,
        },
        sequence_length=sequence_length,
        warmup=warmup,
        iterations=iterations,
        rounds=rounds,
        seed=seed,
    )


def benchmark_b4(
    *,
    rows: int,
    hidden_size: int,
    warmup: int,
    iterations: int,
    rounds: int,
    seed: int,
) -> dict[str, Any]:
    spline_ops = load_spline_ops()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (rows, hidden_size)
    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    up = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    grad_out = torch.randn(
        shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )

    def native_forward():
        return spline_ops.swish_mul_fwd_variant(gate, up, 0, 0)

    def native_backward():
        return spline_ops.swish_mul_bwd_variant(grad_out, gate, up, 0, 0)

    native_y = native_forward()
    polynomial_y = spline_ops.swish_mul_fwd_variant(gate, up, 3, 0)
    native_grads = native_backward()
    polynomial_grads = spline_ops.swish_mul_bwd_variant(grad_out, gate, up, 3, 0)
    return {
        "case": "B4",
        "operation": "routed-expert SwiGLU activation",
        "scope": "same-layout one-kernel BF16 native SFU versus polynomial",
        "polynomial": "D3 current fit",
        "shape": {"rows": rows, "hidden_size": hidden_size},
        "forward": measure_callables(
            {
                "native": native_forward,
                "polynomial": lambda: spline_ops.swish_mul_fwd_variant(gate, up, 3, 0),
            },
            warmup=warmup,
            iterations=iterations,
            rounds=rounds,
        ),
        "backward": measure_callables(
            {
                "native": native_backward,
                "polynomial": lambda: spline_ops.swish_mul_bwd_variant(
                    grad_out, gate, up, 3, 0
                ),
            },
            warmup=warmup,
            iterations=iterations,
            rounds=rounds,
        ),
        "numerics": {
            "forward": tensor_error(polynomial_y, native_y),
            "grad_gate": tensor_error(polynomial_grads[0], native_grads[0]),
            "grad_up": tensor_error(polynomial_grads[1], native_grads[1]),
        },
    }


def parse_cases(value: str) -> list[str]:
    cases = [item.strip().lower() for item in value.split(",") if item.strip()]
    supported = {"b1", "b2", "b3", "b4"}
    unknown = set(cases) - supported
    if not cases or unknown:
        raise argparse.ArgumentTypeError(
            f"cases must select from {sorted(supported)}; " f"unknown={sorted(unknown)}"
        )
    return cases


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=parse_cases, default=parse_cases("b1,b2,b3,b4"))
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--b1-elements", type=int, default=4096 * 14336)
    parser.add_argument("--b2-sequence-length", type=int, default=4096)
    parser.add_argument(
        "--b3-sequence-length",
        type=int,
        default=FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH,
        help="must remain 4096 for the fitted direct D3/D4 kernel",
    )
    parser.add_argument("--b4-rows", type=int, default=6 * 4096)
    parser.add_argument("--b4-hidden-size", type=int, default=1280)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "component-probes.json",
    )
    return parser


def print_results(rows: Sequence[dict[str, Any]]) -> None:
    print(
        "case  operation                              native fwd  poly fwd  "
        "fwd speedup  native bwd  poly bwd  bwd speedup"
    )
    print("-" * 116)
    for row in rows:
        forward = row["forward"]
        backward = row["backward"]
        print(
            f"{row['case']:<5} {row['operation'][:38]:<38} "
            f"{forward['median_us']['native']:10.3f} "
            f"{forward['median_us']['polynomial']:9.3f} "
            f"{forward['speedup']:11.3f}x "
            f"{backward['median_us']['native']:10.3f} "
            f"{backward['median_us']['polynomial']:8.3f} "
            f"{backward['speedup']:11.3f}x"
        )


def main() -> int:
    args = get_parser().parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("component probes require an SM100 or newer CUDA GPU")
    if args.warmup < 0 or args.iterations < 1 or args.rounds < 1:
        raise ValueError(
            "warmup must be non-negative; iterations and rounds must be positive"
        )
    if "b3" in args.cases and (
        args.b3_sequence_length != FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH
    ):
        raise ValueError(
            "B3 direct D3/D4 requires --b3-sequence-length="
            f"{FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH}"
        )

    runners: dict[str, Callable[[], dict[str, Any]]] = {
        "b1": lambda: benchmark_b1(
            elements=args.b1_elements,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
            seed=args.seed,
        ),
        "b2": lambda: benchmark_b2(
            sequence_length=args.b2_sequence_length,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
            seed=args.seed,
        ),
        "b3": lambda: benchmark_b3(
            sequence_length=args.b3_sequence_length,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
            seed=args.seed,
        ),
        "b4": lambda: benchmark_b4(
            rows=args.b4_rows,
            hidden_size=args.b4_hidden_size,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
            seed=args.seed,
        ),
    }
    rows = []
    for case in args.cases:
        rows.append(runners[case]())
        gc.collect()
        torch.cuda.empty_cache()

    properties = torch.cuda.get_device_properties(0)
    payload = {
        "schema_version": 1,
        "experiment": {
            "id": "b1-b4-component-probes",
            "selected_cases": args.cases,
            "provenance_class": "new-measurement",
        },
        "source": {
            "repository": "MrHuff/fast-polynomial-transcendentals",
            "revision": (revision if len(revision := git_revision()) == 40 else None),
            "dirty": git_worktree_dirty(),
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
            "timed_iterations_per_variant": args.iterations * args.rounds,
            "summary_statistic": "median of per-round means",
            "order_policy": "rotate native/polynomial order each round",
        },
        "results": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print_results(rows)
    print(f"\nJSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

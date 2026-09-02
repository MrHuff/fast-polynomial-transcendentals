#
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified 2026-09-02 for the standalone SFU reproduction package.
#

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from .benchmark_polynomial_sincos import (
    attest_rope_sources,
    rope_result_metadata,
)


def reduce_to_quarter_turn(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    work = np.asarray(angles, dtype=np.float64)
    quadrant = np.rint(work * (2.0 / math.pi)).astype(np.int64)
    reduced = work - quadrant * (math.pi / 2.0)
    return reduced, quadrant & 3


def reduce_to_half_turn(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce to [-pi/2, pi/2] and return the shared sin/cos sign parity."""

    work = np.asarray(angles, dtype=np.float64)
    half_turn = np.rint(work / math.pi).astype(np.int64)
    reduced = work - half_turn * math.pi
    return reduced, half_turn & 1


def rope_angles(
    *,
    head_dim: int,
    max_seq_len: int,
    theta: float,
) -> np.ndarray:
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim must be a positive even integer")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")

    dimensions = np.arange(0, head_dim, 2, dtype=np.float64)
    frequencies = 1.0 / np.power(theta, dimensions / head_dim)
    positions = np.arange(max_seq_len, dtype=np.float64)
    return np.multiply.outer(positions, frequencies)


def fit_parity_pair(
    reduced: np.ndarray,
    *,
    sin_terms: int,
    cos_terms: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sin_terms < 1 or cos_terms < 1:
        raise ValueError("sin_terms and cos_terms must be positive")

    reduced = np.asarray(reduced, dtype=np.float64).reshape(-1)
    squared = reduced * reduced
    sin_basis = np.stack(
        [reduced * np.power(squared, index) for index in range(sin_terms)],
        axis=1,
    )
    cos_basis = np.stack(
        [np.power(squared, index) for index in range(cos_terms)],
        axis=1,
    )
    sin_coefficients = np.linalg.lstsq(
        sin_basis,
        np.sin(reduced),
        rcond=None,
    )[0]
    cos_coefficients = np.linalg.lstsq(
        cos_basis,
        np.cos(reduced),
        rcond=None,
    )[0]
    return sin_coefficients, cos_coefficients


def fit_full_pair(
    reduced: np.ndarray,
    *,
    sin_degree: int,
    cos_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit unrestricted polynomials for an asymmetric reduced-angle measure."""

    if sin_degree < 0 or cos_degree < 0:
        raise ValueError("sin_degree and cos_degree must be non-negative")

    reduced = np.asarray(reduced, dtype=np.float64).reshape(-1)
    sin_basis = np.vander(reduced, N=sin_degree + 1, increasing=True)
    cos_basis = np.vander(reduced, N=cos_degree + 1, increasing=True)
    sin_coefficients = np.linalg.lstsq(
        sin_basis,
        np.sin(reduced),
        rcond=None,
    )[0]
    cos_coefficients = np.linalg.lstsq(
        cos_basis,
        np.cos(reduced),
        rcond=None,
    )[0]
    return sin_coefficients, cos_coefficients


def _run_sollya_sparse_fit(
    expression: str,
    monomials: list[int],
    *,
    interval_max: float,
    precision_bits: int,
) -> np.ndarray:
    monomial_argument = "[|" + ",".join(str(value) for value in monomials) + "|]"
    format_argument = "[|" + ",".join(str(precision_bits) for _ in monomials) + "|]"
    interval_min = "1b-20" if min(monomials) > 0 else "0"
    commands = [
        "verbosity = 0;",
        (
            f"fit = fpminimax({expression}, {monomial_argument}, "
            f"{format_argument}, [{interval_min};{interval_max:.17g}], absolute);"
        ),
    ]
    commands.extend(f"print(coeff(fit,{degree}));" for degree in monomials)
    commands.append("quit;")
    completed = subprocess.run(
        ["sollya", "--flush", "--noprompt"],
        input="\n".join(commands) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    coefficients: list[float] = []
    for line in completed.stdout.splitlines():
        try:
            coefficients.append(float(line.strip()))
        except ValueError:
            continue
    if len(coefficients) != len(monomials):
        raise RuntimeError(
            "Sollya did not return the expected coefficient count: "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        )
    return np.asarray(coefficients, dtype=np.float64)


def fit_sollya_pair(
    *,
    sin_terms: int,
    cos_terms: int,
    interval_max: float,
    coefficient_dtype: str,
    basis: str = "parity",
) -> tuple[np.ndarray, np.ndarray]:
    """Run Sollya fpminimax with the same monomials used by the CUDA kernel."""

    precision_bits = {
        "float32": 24,
        "float16": 11,
        "bfloat16": 8,
    }[coefficient_dtype]
    if basis == "parity":
        sin_monomials = list(range(1, 2 * sin_terms, 2))
        cos_monomials = list(range(0, 2 * cos_terms, 2))
    elif basis == "full":
        sin_monomials = list(range(2 * sin_terms))
        cos_monomials = list(range(2 * cos_terms - 1))
    else:
        raise ValueError(f"unsupported basis: {basis}")
    return (
        _run_sollya_sparse_fit(
            "sin(x)",
            sin_monomials,
            interval_max=interval_max,
            precision_bits=precision_bits,
        ),
        _run_sollya_sparse_fit(
            "cos(x)",
            cos_monomials,
            interval_max=interval_max,
            precision_bits=precision_bits,
        ),
    )


def quantize_coefficients(
    coefficients: np.ndarray,
    dtype: str,
) -> np.ndarray:
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    return (
        torch.tensor(coefficients, dtype=torch_dtype)
        .to(torch.float32)
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def _horner(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    value = np.full_like(x, coefficients[-1])
    for coefficient in coefficients[-2::-1]:
        value = value * x + coefficient
    return value


def evaluate_pair(
    angles: np.ndarray,
    sin_coefficients: np.ndarray,
    cos_coefficients: np.ndarray,
    *,
    basis: str = "parity",
    reduction: str = "quarter-turn",
) -> tuple[np.ndarray, np.ndarray]:
    if reduction == "quarter-turn":
        reduced, region = reduce_to_quarter_turn(angles)
    elif reduction == "half-turn":
        reduced, region = reduce_to_half_turn(angles)
    else:
        raise ValueError(f"unsupported reduction: {reduction}")
    if basis == "parity":
        squared = reduced * reduced
        sin_reduced = reduced * _horner(squared, sin_coefficients)
        cos_reduced = _horner(squared, cos_coefficients)
    elif basis == "full":
        sin_reduced = _horner(reduced, sin_coefficients)
        cos_reduced = _horner(reduced, cos_coefficients)
    else:
        raise ValueError(f"unsupported basis: {basis}")

    if reduction == "half-turn":
        negative = region.astype(bool)
        sin = np.where(negative, -sin_reduced, sin_reduced)
        cos = np.where(negative, -cos_reduced, cos_reduced)
        return cos, sin

    swap = (region & 1).astype(bool)
    sin_magnitude = np.where(swap, cos_reduced, sin_reduced)
    cos_magnitude = np.where(swap, sin_reduced, cos_reduced)
    sin = np.where((region & 2).astype(bool), -sin_magnitude, sin_magnitude)
    cos = np.where(((region + 1) & 2).astype(bool), -cos_magnitude, cos_magnitude)
    return cos, sin


def error_metrics(
    angles: np.ndarray,
    sin_coefficients: np.ndarray,
    cos_coefficients: np.ndarray,
    *,
    basis: str = "parity",
    reduction: str = "quarter-turn",
) -> dict[str, float]:
    cos, sin = evaluate_pair(
        angles,
        sin_coefficients,
        cos_coefficients,
        basis=basis,
        reduction=reduction,
    )
    reference_cos = np.cos(angles)
    reference_sin = np.sin(angles)
    phase_error = np.arctan2(
        sin * reference_cos - cos * reference_sin,
        cos * reference_cos + sin * reference_sin,
    )
    norm_error = cos * cos + sin * sin - 1.0
    return {
        "sin_max_abs": float(np.max(np.abs(sin - reference_sin))),
        "sin_rmse": float(np.sqrt(np.mean(np.square(sin - reference_sin)))),
        "cos_max_abs": float(np.max(np.abs(cos - reference_cos))),
        "cos_rmse": float(np.sqrt(np.mean(np.square(cos - reference_cos)))),
        "phase_max_abs_rad": float(np.max(np.abs(phase_error))),
        "phase_rmse_rad": float(np.sqrt(np.mean(np.square(phase_error)))),
        "unit_norm_max_abs": float(np.max(np.abs(norm_error))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a paired range-reduced sin/cos polynomial for RoPE."
    )
    parser.add_argument("--basis", choices=("parity", "full"), default="parity")
    parser.add_argument(
        "--reduction",
        choices=("quarter-turn", "half-turn"),
        default="quarter-turn",
    )
    parser.add_argument(
        "--fit-method",
        choices=("least-squares", "sollya"),
        default="least-squares",
    )
    parser.add_argument("--weighting", choices=("uniform", "rope"), default="uniform")
    parser.add_argument("--sin-terms", type=int, default=4)
    parser.add_argument("--cos-terms", type=int, default=4)
    parser.add_argument("--uniform-samples", type=int, default=131_073)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--theta", type=float, default=500_000.0)
    parser.add_argument(
        "--coefficient-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-unbound-source",
        action="store_true",
        help="Permit a diagnostic fit result from a dirty source checkout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interval_max = math.pi / 4.0 if args.reduction == "quarter-turn" else math.pi / 2.0
    uniform_angles = np.linspace(
        -interval_max,
        interval_max,
        args.uniform_samples,
        dtype=np.float64,
    )
    full_rope_angles = rope_angles(
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        theta=args.theta,
    )
    reducer = (
        reduce_to_quarter_turn
        if args.reduction == "quarter-turn"
        else reduce_to_half_turn
    )
    reduced_rope_angles, _ = reducer(full_rope_angles)
    fit_arguments = (
        uniform_angles if args.weighting == "uniform" else reduced_rope_angles
    )

    sin_degree = 2 * args.sin_terms - 1
    cos_degree = 2 * (args.cos_terms - 1)
    if args.fit_method == "sollya":
        if args.weighting != "uniform":
            raise ValueError("Sollya fpminimax only supports uniform weighting")
        sin_coefficients, cos_coefficients = fit_sollya_pair(
            sin_terms=args.sin_terms,
            cos_terms=args.cos_terms,
            interval_max=interval_max,
            coefficient_dtype=args.coefficient_dtype,
            basis=args.basis,
        )
        sin_form = "r * P(r^2)" if args.basis == "parity" else "P(r)"
        cos_form = "Q(r^2)" if args.basis == "parity" else "Q(r)"
    elif args.basis == "parity":
        sin_coefficients, cos_coefficients = fit_parity_pair(
            fit_arguments,
            sin_terms=args.sin_terms,
            cos_terms=args.cos_terms,
        )
        sin_form = "r * P(r^2)"
        cos_form = "Q(r^2)"
    else:
        sin_coefficients, cos_coefficients = fit_full_pair(
            fit_arguments,
            sin_degree=sin_degree,
            cos_degree=cos_degree,
        )
        sin_form = "P(r)"
        cos_form = "Q(r)"
    sin_coefficients = quantize_coefficients(
        sin_coefficients,
        args.coefficient_dtype,
    )
    cos_coefficients = quantize_coefficients(
        cos_coefficients,
        args.coefficient_dtype,
    )

    repository_state, attestations, source_bound = attest_rope_sources(
        spline_ops=None,
        allow_unbound_source=args.allow_unbound_source,
    )
    result = {
        **rope_result_metadata(
            "rope-polynomial-fitting",
            repository_state=repository_state,
            attestations=attestations,
            source_bound=source_bound,
        ),
        "basis": {
            "name": args.basis,
            "sin": sin_form,
            "cos": cos_form,
            "sin_degree": sin_degree,
            "cos_degree": cos_degree,
            "coefficient_count": len(sin_coefficients) + len(cos_coefficients),
        },
        "fit": {
            "method": args.fit_method,
            "reduction": args.reduction,
            "weighting": args.weighting,
            "coefficient_dtype": args.coefficient_dtype,
            "head_dim": args.head_dim,
            "max_seq_len": args.max_seq_len,
            "theta": args.theta,
        },
        "measurement": {
            "fit_method": args.fit_method,
            "reduction": args.reduction,
            "weighting": args.weighting,
            "coefficient_dtype": args.coefficient_dtype,
            "summary_statistic": "deterministic fit and numerical error sweep",
        },
        "sin_coefficients": sin_coefficients.tolist(),
        "cos_coefficients": cos_coefficients.tolist(),
        "uniform_metrics": error_metrics(
            uniform_angles,
            sin_coefficients,
            cos_coefficients,
            basis=args.basis,
            reduction=args.reduction,
        ),
        "rope_metrics": error_metrics(
            full_rope_angles,
            sin_coefficients,
            cos_coefficients,
            basis=args.basis,
            reduction=args.reduction,
        ),
        "reduced_rope_distribution": {
            "mean": float(np.mean(reduced_rope_angles)),
            "std": float(np.std(reduced_rope_angles)),
            "positive_fraction": float(np.mean(reduced_rope_angles > 0.0)),
        },
    }
    result["results"] = {
        "sin_coefficients": result["sin_coefficients"],
        "cos_coefficients": result["cos_coefficients"],
        "uniform_metrics": result["uniform_metrics"],
        "rope_metrics": result["rope_metrics"],
        "reduced_rope_distribution": result["reduced_rope_distribution"],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

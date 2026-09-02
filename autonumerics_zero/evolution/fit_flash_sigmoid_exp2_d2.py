#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# Modified in 2026 for the standalone fast-polynomial-transcendentals release.
"""Fit the two-FMA range-reduced polynomial used by FlashSigmoid.

The FlashSigmoid bias is ``-log(sequence_length)``. For normalized QK scores,
the useful target is therefore the low-probability distribution
``sigmoid(score - log(n))`` rather than a centered sigmoid. The fitted D2
polynomial approximates the range-reduced exp2 mantissa and absorbs the small
denominator correction for the configured sequence length. The fitter reports
the algebraic ``p * (1 - p)`` error for a D2 probability recomputation; the
runtime can independently select a D3 probability recomputation for backward.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import differential_evolution, minimize

try:
    from .fit_provenance import bind_fit_payload, build_fit_provenance
except ImportError:  # Direct script execution.
    from fit_provenance import bind_fit_payload, build_fit_provenance


GENERIC_EXP2_D2 = (1.0, 0.6657850742340088, 0.33010703325271606)


@dataclass(frozen=True)
class FitMetrics:
    forward_relative_l1: float
    gradient_relative_l1: float
    central_max_relative: float
    endpoint_jump: float
    minimum_mantissa: float


class FlashSigmoidD2Fit:
    def __init__(self, sequence_length: int, score_sigma: float) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if score_sigma <= 0.0:
            raise ValueError("score_sigma must be positive")

        self.sequence_length = sequence_length
        self.scores = np.linspace(-7.0, 7.0, 28_001, dtype=np.float64)
        self.weights = np.exp(
            -0.5 * np.square(self.scores / score_sigma), dtype=np.float64
        )
        biased_scores = self.scores - math.log(sequence_length)
        self.expected = 1.0 / (1.0 + np.exp(-biased_scores))
        self.expected_gradient = self.expected * (1.0 - self.expected)

        exp2_input = self.scores * math.log2(math.e)
        self.exponents = np.floor(exp2_input).astype(np.int64)
        self.fractions = exp2_input - self.exponents
        self.scales = (
            np.ldexp(np.ones_like(self.scores), self.exponents) / sequence_length
        )

    @staticmethod
    def round_coefficients(parameters: np.ndarray) -> np.ndarray:
        return np.asarray(parameters, dtype=np.float32).astype(np.float64)

    def probability(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c0, c1, c2 = self.round_coefficients(parameters)
        mantissa = c0 + self.fractions * (c1 + self.fractions * c2)
        return self.scales * mantissa, mantissa

    def metrics(self, parameters: np.ndarray) -> FitMetrics:
        probability, mantissa = self.probability(parameters)
        gradient = probability * (1.0 - probability)
        forward_relative_l1 = np.sum(
            self.weights * np.abs(probability - self.expected)
        ) / np.sum(self.weights * self.expected)
        gradient_relative_l1 = np.sum(
            self.weights * np.abs(gradient - self.expected_gradient)
        ) / np.sum(self.weights * self.expected_gradient)

        central = np.abs(self.scores) <= 5.0
        central_max_relative = np.max(
            np.abs(probability[central] - self.expected[central])
            / self.expected[central]
        )
        c0, c1, c2 = self.round_coefficients(parameters)
        endpoint_jump = abs((c0 + c1 + c2) - 2.0 * c0)
        return FitMetrics(
            forward_relative_l1=float(forward_relative_l1),
            gradient_relative_l1=float(gradient_relative_l1),
            central_max_relative=float(central_max_relative),
            endpoint_jump=float(endpoint_jump),
            minimum_mantissa=float(np.min(mantissa)),
        )

    def objective(self, parameters: np.ndarray) -> float:
        metrics = self.metrics(parameters)
        negativity = max(0.0, -metrics.minimum_mantissa)
        return (
            metrics.forward_relative_l1
            + metrics.gradient_relative_l1
            + 0.01 * metrics.central_max_relative
            + 0.01 * metrics.endpoint_jump
            + 1e6 * negativity
        )


def float32_hex(value: float) -> str:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return f"0x{bits:08X}"


def fit(args: argparse.Namespace) -> tuple[np.ndarray, FitMetrics]:
    problem = FlashSigmoidD2Fit(args.sequence_length, args.score_sigma)
    initial = np.asarray(GENERIC_EXP2_D2, dtype=np.float64)
    result = differential_evolution(
        problem.objective,
        bounds=((0.995, 1.005), (0.60, 0.75), (0.25, 0.40)),
        x0=initial,
        seed=args.seed,
        popsize=20,
        maxiter=args.maxiter,
        polish=False,
        workers=1,
        updating="immediate",
        tol=1e-11,
    )
    result = minimize(
        problem.objective,
        result.x,
        method="Nelder-Mead",
        options={"maxiter": 5_000, "xatol": 1e-12, "fatol": 1e-12},
    )
    coefficients = problem.round_coefficients(result.x)
    return coefficients, problem.metrics(coefficients)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--score-sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional caller-selected path for coefficients and fit metrics.",
    )
    return parser.parse_args(argv)


def result_document(
    args: argparse.Namespace,
    coefficients: np.ndarray,
    metrics: FitMetrics,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "fit": "flash-sigmoid-sequence-specific-exp2-d2",
        "sequence_length": args.sequence_length,
        "score_sigma": args.score_sigma,
        "seed": args.seed,
        "maxiter": args.maxiter,
        "coefficients_float32": [float(value) for value in coefficients],
        "ptx_hex_high_to_low": [float32_hex(value) for value in reversed(coefficients)],
        "metrics": asdict(metrics),
    }


def main(argv: list[str] | None = None) -> None:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    provenance = (
        build_fit_provenance(
            script=Path(__file__),
            arguments=raw_arguments,
            distributions=("numpy", "scipy"),
        )
        if args.json_out is not None
        else None
    )

    coefficients, metrics = fit(args)
    print("coefficients:", tuple(float(value) for value in coefficients))
    print("ptx_hex:", tuple(float32_hex(value) for value in reversed(coefficients)))
    print("forward_relative_l1:", f"{metrics.forward_relative_l1:.9g}")
    print("gradient_relative_l1:", f"{metrics.gradient_relative_l1:.9g}")
    print("central_max_relative:", f"{metrics.central_max_relative:.9g}")
    print("endpoint_jump:", f"{metrics.endpoint_jump:.9g}")
    print("minimum_mantissa:", f"{metrics.minimum_mantissa:.9g}")
    if args.json_out is not None:
        document = result_document(args, coefficients, metrics)
        bind_fit_payload(document, provenance)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(document, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote: {args.json_out}")


if __name__ == "__main__":
    main()

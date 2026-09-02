#!/usr/bin/env python3
"""Fit low-order exp2 approximations that preserve online-softmax endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def golden_section_minimize(fn, lo: float, hi: float, iterations: int = 100) -> float:
    ratio = (5.0**0.5 - 1.0) / 2.0
    left = hi - ratio * (hi - lo)
    right = lo + ratio * (hi - lo)
    f_left = fn(left)
    f_right = fn(right)
    for _ in range(iterations):
        if f_left < f_right:
            hi, right, f_right = right, left, f_left
            left = hi - ratio * (hi - lo)
            f_left = fn(left)
        else:
            lo, left, f_left = left, right, f_right
            right = lo + ratio * (hi - lo)
            f_right = fn(right)
    return (lo + hi) / 2.0


def error_metrics(approx: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = approx - target
    relative_error = error / target
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "max_rel_error": float(np.max(np.abs(relative_error))),
        "rms_error": float(np.sqrt(np.mean(error * error))),
        "rms_rel_error": float(np.sqrt(np.mean(relative_error * relative_error))),
    }


def fit_quadratic(x: np.ndarray, target: np.ndarray) -> dict[str, object]:
    # p(x) = 1 + a*x + (1-a)*x^2, so p(0)=1 and p(1)=2 exactly.
    base = 1.0 + x * x - target
    direction = x - x * x
    least_squares_a = float(-np.dot(base, direction) / np.dot(direction, direction))

    def max_relative_error(a: float) -> float:
        approx = 1.0 + a * x + (1.0 - a) * x * x
        return float(np.max(np.abs(approx / target - 1.0)))

    minimax_a = golden_section_minimize(max_relative_error, 0.5, 1.0)
    result: dict[str, object] = {}
    for name, a in (("least_squares", least_squares_a), ("sampled_minimax", minimax_a)):
        approx = 1.0 + a * x + (1.0 - a) * x * x
        result[name] = {
            "c1": a,
            "c2": 1.0 - a,
            **error_metrics(approx, target),
        }
    return result


def fit_two_piece_linear(x: np.ndarray, target: np.ndarray) -> dict[str, object]:
    # p(x) = 1 + a*x + 2*(1-a)*max(x-0.5, 0), preserving both endpoints.
    hinge = np.maximum(x - 0.5, 0.0)
    base = 1.0 + 2.0 * hinge - target
    direction = x - 2.0 * hinge
    least_squares_a = float(-np.dot(base, direction) / np.dot(direction, direction))

    def evaluate(a: float) -> np.ndarray:
        return 1.0 + a * x + 2.0 * (1.0 - a) * hinge

    def max_relative_error(a: float) -> float:
        return float(np.max(np.abs(evaluate(a) / target - 1.0)))

    minimax_a = golden_section_minimize(max_relative_error, 0.5, 1.0)
    result: dict[str, object] = {}
    for name, a in (("least_squares", least_squares_a), ("sampled_minimax", minimax_a)):
        result[name] = {
            "c1": a,
            "c1_delta": 2.0 * (1.0 - a),
            **error_metrics(evaluate(a), target),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_000_001)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("--samples must be at least 3")

    x = np.linspace(0.0, 1.0, args.samples, dtype=np.float64)
    target = np.exp2(x)
    result = {
        "domain": [0.0, 1.0],
        "samples": args.samples,
        "constraints": {"p(0)": 1.0, "p(1)": 2.0},
        "quadratic": fit_quadratic(x, target),
        "two_piece_linear": fit_two_piece_linear(x, target),
        "note": "Sampled minimax values are dense-grid estimates, not formal certificates.",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

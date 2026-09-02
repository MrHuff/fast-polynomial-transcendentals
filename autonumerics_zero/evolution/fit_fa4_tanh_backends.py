#!/usr/bin/env python3
"""
Refit FA4 softcap tanh forward polynomials for both deployed targets.

Targets:
  - cute: Float32 Horner evaluation in FA4 score_mod
  - device: BF16x2 handwritten device evaluation (simulated via torch.bfloat16)

The output is a JSON file with the chosen Li/Lc sweep point, max/mean error,
and the exact evaluation semantics used for each backend/degree pair.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from constrained_ls_fitter import dual_constrained_ls_fit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "cuda_benchmarks" / "analysis_results" / "fa4_tanh_backend_fits.json"


def tanh_target(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _bf16_scalar(val: float) -> float:
    return torch.tensor([val], dtype=torch.float32).to(torch.bfloat16).item()


def to_bf16_list(coeffs: list[float] | np.ndarray) -> list[float]:
    return [_bf16_scalar(float(c)) for c in coeffs]


def eval_tanh_odd_cute_f32(x_eval: np.ndarray, coeffs: list[float], clamp: float) -> np.ndarray:
    """FA4 CuTe score_mod semantics: f32 Horner on clamped |x|, then min(..., 1)."""
    t = np.minimum(np.abs(x_eval).astype(np.float32), np.float32(clamp))
    coeffs32 = np.asarray(coeffs, dtype=np.float32)
    h = np.float32(coeffs32[-1]) * t + np.float32(coeffs32[-2])
    for idx in range(len(coeffs32) - 3, 0, -1):
        h = h * t + np.float32(coeffs32[idx])
    h = h * t
    h = np.minimum(h, np.float32(1.0))
    return h.astype(np.float64)


def eval_tanh_odd_device_bf16(x_eval: np.ndarray, coeffs: list[float], clamp: float) -> np.ndarray:
    """Handwritten BF16x2 semantics: BF16 Horner on clamped |x|, then min(..., 1)."""
    t = torch.tensor(np.abs(x_eval), dtype=torch.float64).to(torch.bfloat16)
    t = torch.minimum(t, torch.tensor(clamp, dtype=torch.bfloat16))
    coeffs_bf16 = [torch.tensor(c, dtype=torch.bfloat16) for c in coeffs]
    h = (t * coeffs_bf16[-1]).to(torch.bfloat16)
    h = (h + coeffs_bf16[-2]).to(torch.bfloat16)
    for idx in range(len(coeffs_bf16) - 3, 0, -1):
        h = (t * h).to(torch.bfloat16)
        h = (h + coeffs_bf16[idx]).to(torch.bfloat16)
    h = (t * h).to(torch.bfloat16)
    h = torch.minimum(h, torch.tensor(1.0, dtype=torch.bfloat16))
    return h.to(torch.float64).numpy()


def sweep_backend(
    *,
    backend: str,
    degrees: list[int],
    li_range: tuple[float, float],
    lc_range: tuple[float, float],
    step: float,
    eval_range: float,
    eval_points: int,
) -> dict[str, dict[str, object]]:
    li_vals = np.arange(li_range[0], li_range[1] + step / 2, step)
    lc_vals = np.arange(lc_range[0], lc_range[1] + step / 2, step)
    x_eval = np.linspace(0.0, eval_range, eval_points, dtype=np.float64)
    y_true = tanh_target(x_eval)

    if backend == "cute":
        evaluator = eval_tanh_odd_cute_f32
        semantics = {
            "coefficients": "float32",
            "input": "abs(x) clamped to Lc in float32",
            "evaluation": "odd Horner in float32, result *= t, result=min(result, 1.0)",
        }
    elif backend == "device":
        evaluator = eval_tanh_odd_device_bf16
        semantics = {
            "coefficients": "bfloat16",
            "input": "abs(x) clamped to Lc in bfloat16",
            "evaluation": "odd Horner in bfloat16, result *= t, result=min(result, 1.0)",
        }
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    best: dict[str, dict[str, object]] = {}
    for degree in degrees:
        best_key = f"d{degree}"
        best[best_key] = {"max_error": float("inf")}
        for li in li_vals:
            coeffs = dual_constrained_ls_fit(tanh_target, (0.0, float(li)), degree)
            if backend == "device":
                runtime_coeffs = to_bf16_list(coeffs.tolist())
            else:
                runtime_coeffs = [float(np.float32(c)) for c in coeffs.tolist()]
            for lc in lc_vals:
                if lc > li + 1.0:
                    continue
                y_pred = evaluator(x_eval, runtime_coeffs, float(lc))
                abs_err = np.abs(y_true - y_pred)
                max_err = float(abs_err.max())
                if max_err >= float(best[best_key]["max_error"]):
                    continue
                best[best_key] = {
                    "degree": degree,
                    "Li": float(li),
                    "Lc": float(lc),
                    "max_error": max_err,
                    "mean_error": float(abs_err.mean()),
                    "coeffs_runtime": runtime_coeffs,
                    "coeffs_fit_f64": [float(c) for c in coeffs.tolist()],
                    "semantics": semantics,
                }
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degrees", default="3,4,5,6")
    parser.add_argument("--li-min", type=float, default=2.0)
    parser.add_argument("--li-max", type=float, default=5.0)
    parser.add_argument("--lc-min", type=float, default=2.0)
    parser.add_argument("--lc-max", type=float, default=5.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--eval-range", type=float, default=12.0)
    parser.add_argument("--eval-points", type=int, default=8000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    degrees = [int(tok.strip()) for tok in args.degrees.split(",") if tok.strip()]
    results = {
        "function": "tanh_fwd_odd",
        "targets": {
            "cute": sweep_backend(
                backend="cute",
                degrees=degrees,
                li_range=(args.li_min, args.li_max),
                lc_range=(args.lc_min, args.lc_max),
                step=args.step,
                eval_range=args.eval_range,
                eval_points=args.eval_points,
            ),
            "device": sweep_backend(
                backend="device",
                degrees=degrees,
                li_range=(args.li_min, args.li_max),
                lc_range=(args.lc_min, args.lc_max),
                step=args.step,
                eval_range=args.eval_range,
                eval_points=args.eval_points,
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)

    for backend, backend_results in results["targets"].items():
        print(f"\n{backend.upper()} target")
        for key, row in backend_results.items():
            print(
                f"  {key.upper()}: Li={row['Li']:.2f} Lc={row['Lc']:.2f} "
                f"max_err={row['max_error']:.6f} mean_err={row['mean_error']:.6f}"
            )
            print(f"    coeffs={row['coeffs_runtime']}")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()

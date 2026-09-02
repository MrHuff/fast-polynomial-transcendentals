#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified in 2026 for the standalone fast-polynomial-transcendentals release.

"""
Fit D3-D6 ODD/EVEN coefficients for ALL activation functions using the
dual-constrained LS fitter with 2D sweep over (L_interp, L_clamp).

This preserves the historical FP16 coefficient-search surrogate. NumPy rounds
the multiply and add separately at each Horner stage; the deployed packed CUDA
path uses fused ``__hfma2`` operations and must be measured independently.

Usage:
    python fit_all_degrees.py
"""
import argparse
import os
import sys
import numpy as np
import struct
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from constrained_ls_fitter import (
    dual_constrained_ls_fit,
    eval_poly,
    to_fp16_hex,
    max_error,
    sigmoid,
    sigmoid_grad,
    tanh_grad,
    swish_grad,
)
from fit_provenance import bind_fit_payload, build_fit_provenance


# Target functions for ODD decomposition
def sigmoid_odd(x):
    return sigmoid(x) - 0.5


def tanh_func(x):
    return np.tanh(x)


def swish_grad_odd(x):
    return swish_grad(x) - 0.5


def fp16_quantize(val):
    fp16 = np.float16(val)
    h = struct.pack("<e", fp16).hex()
    hex_str = f"0x{h[2:4]}{h[0:2]}".upper()
    return float(fp16), hex_str


def eval_horner_fp16_vec(t_f64, coeffs_fp16, is_odd=True):
    """
    Vectorized replay of the historical split-rounded FP16 search surrogate.

    Multiplication and addition are rounded separately. This does not emulate
    the single rounding of a packed ``__hfma2`` fused multiply-add. Uses NumPy
    arrays for speed.

    For ODD (c0=0):
      h = ((cn*t + c_{n-1})*t + ... + c2)*t + c1
      result = h * t
    For EVEN:
      result = ((cn*t + c_{n-1})*t + ... + c1)*t + c0
    """
    t = t_f64.astype(np.float16)
    n = len(coeffs_fp16) - 1  # degree

    if is_odd:
        # Start: h = cn*t + c_{n-1}
        h = (t * np.float16(coeffs_fp16[n])).astype(np.float16)
        h = (h + np.float16(coeffs_fp16[n - 1])).astype(np.float16)
        # Continue: h = h*t + c_i, from i=n-2 down to 1
        for i in range(n - 2, 0, -1):
            h = (t * h).astype(np.float16)
            h = (h + np.float16(coeffs_fp16[i])).astype(np.float16)
        # Final: result = h * t
        result = (t * h).astype(np.float16)
    else:
        # Start: r = cn*t + c_{n-1}
        r = (t * np.float16(coeffs_fp16[n])).astype(np.float16)
        r = (r + np.float16(coeffs_fp16[n - 1])).astype(np.float16)
        # Continue: r = r*t + c_i, from i=n-2 down to 0
        for i in range(n - 2, -1, -1):
            r = (t * r).astype(np.float16)
            r = (r + np.float16(coeffs_fp16[i])).astype(np.float16)
        result = r

    return result.astype(np.float64)


def sweep_2d_and_fit(
    func, name, li_range, lc_range, degrees, is_odd=True, step=0.25, eval_range=12.0
):
    """2D grid search with FP16 Horner simulation."""
    li_vals = np.arange(li_range[0], li_range[1] + step / 2, step)
    lc_vals = np.arange(lc_range[0], lc_range[1] + step / 2, step)

    x_eval = np.linspace(0, eval_range, 8000)
    y_true = func(x_eval)

    best = {d: {"err": float("inf")} for d in degrees}

    print(
        f"\nSweeping {name}: Li=[{li_range[0]}-{li_range[1]}], "
        f"Lc=[{lc_range[0]}-{lc_range[1]}], step={step}"
    )

    for Li in li_vals:
        for deg in degrees:
            coeffs = dual_constrained_ls_fit(func, (0.0, Li), deg)
            fp16_coeffs = [float(np.float16(c)) for c in coeffs]

            for Lc in lc_vals:
                if Lc > Li + 1.0:
                    continue

                t_clamped = np.minimum(x_eval, Lc)
                y_pred = eval_horner_fp16_vec(t_clamped, fp16_coeffs, is_odd=is_odd)
                err = np.max(np.abs(y_true - y_pred))

                if err < best[deg]["err"]:
                    best[deg] = {
                        "err": float(err),
                        "Li": float(Li),
                        "Lc": float(Lc),
                        "coeffs_f64": coeffs.tolist(),
                        "coeffs_fp16": fp16_coeffs,
                        "coeffs_hex": to_fp16_hex(coeffs),
                    }

    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")

    results = {}
    for deg in degrees:
        b = best[deg]
        print(
            f"  D{deg}: Li={b['Li']:.2f}, Lc={b['Lc']:.2f}, "
            f"MaxErr_fp16_horner={b['err']:.6f}"
        )
        print(f"    hex={b['coeffs_hex']}")
        for i, (c64, cfp16) in enumerate(zip(b["coeffs_f64"], b["coeffs_fp16"])):
            _, hex_str = fp16_quantize(c64)
            print(f"    c{i} = {cfp16:.10f}  ({hex_str})")
        results[f"d{deg}"] = b

    return results


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "cuda_benchmarks"
    / "analysis_results"
    / "all_degree_coefficients.json"
)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Result path. The default preserves the historical checked-in "
            "artifact location; use a results/ or outputs/ path for a rerun."
        ),
    )
    return parser


def main(argv=None):
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    args = get_parser().parse_args(argv)
    provenance = build_fit_provenance(
        script=Path(__file__),
        arguments=raw_arguments,
        source_files=[Path(__file__).with_name("constrained_ls_fitter.py")],
        distributions=("numpy", "scipy"),
    )
    all_results = {}

    all_results["sigmoid_fwd_odd"] = sweep_2d_and_fit(
        sigmoid_odd,
        "SIGMOID FWD ODD: h(|x|) = sigmoid(|x|)-0.5",
        li_range=(3.0, 8.0),
        lc_range=(3.0, 8.0),
        degrees=[3, 4, 5, 6],
        is_odd=True,
    )

    all_results["tanh_fwd_odd"] = sweep_2d_and_fit(
        tanh_func,
        "TANH FWD ODD: h(|x|) = tanh(|x|)",
        li_range=(2.0, 5.0),
        lc_range=(2.0, 5.0),
        degrees=[3, 4, 5, 6],
        is_odd=True,
    )

    all_results["sigmoid_bwd_even"] = sweep_2d_and_fit(
        sigmoid_grad,
        "SIGMOID BWD EVEN: sigmoid'(|x|)",
        li_range=(4.0, 10.0),
        lc_range=(4.0, 10.0),
        degrees=[3, 4, 5, 6],
        is_odd=False,
    )

    all_results["tanh_bwd_even"] = sweep_2d_and_fit(
        tanh_grad,
        "TANH BWD EVEN: tanh'(|x|)",
        li_range=(2.0, 6.0),
        lc_range=(2.0, 6.0),
        degrees=[3, 4, 5, 6],
        is_odd=False,
    )

    all_results["swish_bwd_odd"] = sweep_2d_and_fit(
        swish_grad_odd,
        "SWISH BWD ODD: h(|x|) = swish'(|x|)-0.5",
        li_range=(4.0, 10.0),
        lc_range=(4.0, 10.0),
        degrees=[3, 4, 5, 6],
        is_odd=True,
    )

    bind_fit_payload(all_results, provenance)

    out_path = args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

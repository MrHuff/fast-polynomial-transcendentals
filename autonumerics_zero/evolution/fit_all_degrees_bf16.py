#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified in 2026 for the standalone fast-polynomial-transcendentals release.

"""
Fit D3-D6 ODD/EVEN coefficients for ALL activation functions using the
dual-constrained LS fitter with 2D sweep over (L_interp, L_clamp).

BF16 VERSION: reproduces the historical coefficient-search replay.  It rounds
the multiply and add separately to BF16.  The deployed CUDA path uses packed
``__hfma2`` instructions, so final device accuracy must be measured with the
compiled kernel rather than inferred from this search surrogate.

Usage:
    python fit_all_degrees_bf16.py
"""
import argparse
import os
import sys
import numpy as np
import struct
import json
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from constrained_ls_fitter import (
    dual_constrained_ls_fit,
    eval_poly,
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


def bf16_quantize(val):
    """Quantize a float to BF16 and return (float_val, hex_str)."""
    t = torch.tensor([val], dtype=torch.float32).to(torch.bfloat16)
    bits = t.view(torch.int16).item() & 0xFFFF
    return t.item(), f"0x{bits:04X}"


def to_bf16_hex(coeffs):
    """Convert float coefficients to BF16 hex strings."""
    return [bf16_quantize(c)[1] for c in coeffs]


def eval_horner_bf16_vec(t_f64, coeffs_fp64, is_odd=True):
    """
    Vectorized replay of the historical two-round BF16 search surrogate.

    Multiplication and addition are rounded separately.  This deliberately
    preserves the fit-selection procedure that produced the checked-in
    coefficients; it does not emulate the single rounding of ``__hfma2``.
    Uses torch tensors for BF16 support (NumPy lacks native BF16).

    For ODD (c0=0):
      h = ((cn*t + c_{n-1})*t + ... + c2)*t + c1
      result = h * t
    For EVEN:
      result = ((cn*t + c_{n-1})*t + ... + c1)*t + c0
    """
    # Convert to torch bf16 for simulation
    t = torch.tensor(t_f64, dtype=torch.float64).to(torch.bfloat16)
    coeffs_bf16 = [
        float(torch.tensor([c], dtype=torch.float64).to(torch.bfloat16).item())
        for c in coeffs_fp64
    ]

    n = len(coeffs_bf16) - 1  # degree

    if is_odd:
        # Start: h = cn*t + c_{n-1}
        h = (t * torch.tensor(coeffs_bf16[n], dtype=torch.bfloat16)).to(torch.bfloat16)
        h = (h + torch.tensor(coeffs_bf16[n - 1], dtype=torch.bfloat16)).to(
            torch.bfloat16
        )
        # Continue: h = h*t + c_i, from i=n-2 down to 1
        for i in range(n - 2, 0, -1):
            h = (t * h).to(torch.bfloat16)
            h = (h + torch.tensor(coeffs_bf16[i], dtype=torch.bfloat16)).to(
                torch.bfloat16
            )
        # Final: result = h * t
        result = (t * h).to(torch.bfloat16)
    else:
        # Start: r = cn*t + c_{n-1}
        r = (t * torch.tensor(coeffs_bf16[n], dtype=torch.bfloat16)).to(torch.bfloat16)
        r = (r + torch.tensor(coeffs_bf16[n - 1], dtype=torch.bfloat16)).to(
            torch.bfloat16
        )
        # Continue: r = r*t + c_i, from i=n-2 down to 0
        for i in range(n - 2, -1, -1):
            r = (t * r).to(torch.bfloat16)
            r = (r + torch.tensor(coeffs_bf16[i], dtype=torch.bfloat16)).to(
                torch.bfloat16
            )
        result = r

    return result.to(torch.float64).numpy()


def sweep_2d_and_fit(
    func, name, li_range, lc_range, degrees, is_odd=True, step=0.25, eval_range=12.0
):
    """2D grid search with BF16 Horner simulation."""
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
            bf16_coeffs = [
                float(torch.tensor([c], dtype=torch.float64).to(torch.bfloat16).item())
                for c in coeffs
            ]

            for Lc in lc_vals:
                if Lc > Li + 1.0:
                    continue

                t_clamped = np.minimum(x_eval, Lc)
                y_pred = eval_horner_bf16_vec(t_clamped, coeffs, is_odd=is_odd)
                err = np.max(np.abs(y_true - y_pred))

                if err < best[deg]["err"]:
                    best[deg] = {
                        "err": float(err),
                        "Li": float(Li),
                        "Lc": float(Lc),
                        "coeffs_f64": coeffs.tolist(),
                        "coeffs_bf16": bf16_coeffs,
                        "coeffs_hex": to_bf16_hex(coeffs),
                    }

    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")

    results = {}
    for deg in degrees:
        b = best[deg]
        print(
            f"  D{deg}: Li={b['Li']:.2f}, Lc={b['Lc']:.2f}, "
            f"MaxErr_bf16_horner={b['err']:.6f}"
        )
        print(f"    hex={b['coeffs_hex']}")
        for i, (c64, cbf16) in enumerate(zip(b["coeffs_f64"], b["coeffs_bf16"])):
            _, hex_str = bf16_quantize(c64)
            print(f"    c{i} = {cbf16:.10f}  ({hex_str})")
        results[f"d{deg}"] = b

    return results


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "cuda_benchmarks"
    / "analysis_results"
    / "all_degree_coefficients_bf16.json"
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
        distributions=("numpy", "scipy", "torch"),
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

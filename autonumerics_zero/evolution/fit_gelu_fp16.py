#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the historical FP16 ERF/GELU D3--D6 coefficient search.

The deployed BF16 ERF/GELU structs were mechanically initialized from these
FP16-derived coefficients.  This program preserves the original two-dimensional
fit/clamp sweep and split-rounding FP16 replay, while writing only the requested
JSON output.  It does not rewrite generated headers or checked-in evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import erf as scipy_erf


# Reuse the exact historical sweep implementation and fitting dependency.  The
# standalone tree keeps both in this directory instead of the former
# experiments/fused_gemm package.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from fit_all_degrees import sweep_2d_and_fit  # noqa: E402
from fit_provenance import bind_fit_payload, build_fit_provenance  # noqa: E402


def erf_func(x: np.ndarray) -> np.ndarray:
    """Error function, represented as an odd polynomial."""

    return scipy_erf(x)


def gaussian_cdf_odd(x: np.ndarray) -> np.ndarray:
    """Odd component of the standard Gaussian cumulative distribution."""

    return 0.5 * scipy_erf(x / np.sqrt(2.0))


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU used by the historical fitter."""

    return x * 0.5 * (1.0 + scipy_erf(x / np.sqrt(2.0)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """Exact derivative of GELU used by the historical fitter."""

    phi = np.exp(-(x**2) / 2.0) / np.sqrt(2.0 * np.pi)
    cdf = 0.5 * (1.0 + scipy_erf(x / np.sqrt(2.0)))
    return cdf + x * phi


def gelu_grad_odd(x: np.ndarray) -> np.ndarray:
    """Odd component of the GELU derivative."""

    return gelu_grad(x) - 0.5


def verify_symmetries() -> None:
    x = np.linspace(-5.0, 5.0, 10_000)
    if not np.allclose(erf_func(x), -erf_func(-x), atol=1e-12):
        raise RuntimeError("erf odd symmetry check failed")
    if not np.allclose(gaussian_cdf_odd(x), -gaussian_cdf_odd(-x), atol=1e-12):
        raise RuntimeError("Gaussian-CDF odd symmetry check failed")
    if not np.allclose(gelu(x) - gelu(-x), x, atol=1e-10):
        raise RuntimeError("GELU forward identity check failed")
    if not np.allclose(gelu_grad(x) + gelu_grad(-x), 1.0, atol=1e-10):
        raise RuntimeError("GELU derivative identity check failed")


def fit_all() -> dict[str, object]:
    """Run the unchanged historical ranges, degree sweep, and grid sizes."""

    verify_symmetries()
    return {
        "erf_fwd_odd": sweep_2d_and_fit(
            erf_func,
            "ERF FWD ODD: erf(|x|)",
            li_range=(2.0, 5.0),
            lc_range=(2.0, 5.0),
            degrees=[3, 4, 5, 6],
            is_odd=True,
        ),
        "gelu_fwd_odd": sweep_2d_and_fit(
            gaussian_cdf_odd,
            "GELU FWD ODD: h(|x|) = 0.5*erf(|x|/sqrt(2))",
            li_range=(3.0, 8.0),
            lc_range=(3.0, 8.0),
            degrees=[3, 4, 5, 6],
            is_odd=True,
        ),
        "gelu_bwd_odd": sweep_2d_and_fit(
            gelu_grad_odd,
            "GELU BWD ODD: h(|x|) = GELU'(|x|) - 0.5",
            li_range=(3.0, 8.0),
            lc_range=(3.0, 8.0),
            degrees=[3, 4, 5, 6],
            is_odd=True,
        ),
    }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("outputs/gelu_coefficients_fp16.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    args = get_parser().parse_args(argv)
    provenance = build_fit_provenance(
        script=Path(__file__),
        arguments=raw_arguments,
        source_files=[
            SCRIPT_DIRECTORY / "fit_all_degrees.py",
            SCRIPT_DIRECTORY / "constrained_ls_fitter.py",
        ],
        distributions=("numpy", "scipy"),
    )
    results = fit_all()
    bind_fit_payload(results, provenance)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

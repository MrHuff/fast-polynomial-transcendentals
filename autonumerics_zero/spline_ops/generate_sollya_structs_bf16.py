#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified in 2026 for the standalone fast-polynomial-transcendentals release.
"""Generate Sollya-based BF16 spline structs plus comparison metadata.

This script:
1. Parses the shipped BF16 spline header to recover the current runtime clamps
   and coefficients.
2. Uses Sollya `fpminimax` to fit matching polynomial structures at the same
   degree and clamp.
3. Writes a new BF16 device header with Sollya coefficients.
4. Emits a JSON file with current-vs-Sollya coefficients and a host-side
   coefficient-control error calculation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit(
        "This script requires PyTorch for BF16 coefficient rounding."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
EVOLUTION_ROOT = ROOT / "autonumerics_zero/evolution"
if str(EVOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVOLUTION_ROOT))

from fit_provenance import (  # noqa: E402
    bind_fit_payload,
    build_fit_provenance,
    sha256_file,
)


CURRENT_HEADER = Path(__file__).resolve().with_name("spline_structs_odd_bf16.cuh")
SOLLYA_HEADER = Path(__file__).resolve().with_name("spline_structs_sollya_bf16.cuh")
OUT_JSON = (
    ROOT
    / "autonumerics_zero"
    / "cuda_benchmarks"
    / "analysis_results"
    / "sollya_device_bf16.json"
)
ERROR_GRID_POINTS = 20_001
ERROR_MEASUREMENT = {
    "metric": "maximum absolute error",
    "evaluation": "host NumPy real-arithmetic Horner reconstruction",
    "grid": ("20,001 uniformly spaced points on each row's closed interval [-Lc, Lc]"),
    "current_coefficients": (
        "decimal literals parsed from the deployed CUDA header and evaluated "
        "directly without BF16 pre-rounding"
    ),
    "sollya_coefficients": (
        "Sollya fpminimax coefficients constrained to 8-bit precision and cast "
        "to BF16 before host evaluation"
    ),
    "intermediate_rounding": (
        "none; target-precision intermediate rounding is not replayed"
    ),
    "device_measurement": (
        "not a device measurement; this is a host-side coefficient control"
    ),
}


def bf16_round(value: float) -> float:
    return float(torch.tensor(value, dtype=torch.float32).to(torch.bfloat16).float())


def sigmoid(xs: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-xs))


def swish(xs: np.ndarray) -> np.ndarray:
    return xs * sigmoid(xs)


def swish_grad(xs: np.ndarray) -> np.ndarray:
    s = sigmoid(xs)
    return s * (1.0 + xs * (1.0 - s))


def gelu(xs: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * xs * (1.0 + erf(xs / math.sqrt(2.0)))


def gelu_grad(xs: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(xs / math.sqrt(2.0))) + xs / math.sqrt(
        2.0 * math.pi
    ) * np.exp(-0.5 * xs * xs)


def eval_centered_odd(
    xs: np.ndarray, coeffs: list[float], clamp: float, offset: float = 0.5
) -> np.ndarray:
    xs = np.asarray(xs)
    t = np.minimum(np.abs(xs), clamp)
    poly = np.zeros_like(t)
    for coeff in reversed(coeffs):
        poly = poly * t + coeff
    return offset + np.sign(xs) * t * poly


def eval_odd_factorized(
    xs: np.ndarray, coeffs: list[float], clamp: float
) -> np.ndarray:
    xs = np.asarray(xs)
    t = np.minimum(np.abs(xs), clamp)
    poly = np.zeros_like(t)
    for coeff in reversed(coeffs):
        poly = poly * t + coeff
    y = np.minimum(t * poly, 1.0)
    return np.copysign(y, xs)


def eval_even(xs: np.ndarray, coeffs: list[float], clamp: float) -> np.ndarray:
    xs = np.asarray(xs)
    t = np.minimum(np.abs(xs), clamp)
    poly = np.zeros_like(t)
    for coeff in reversed(coeffs):
        poly = poly * t + coeff
    return poly


def eval_gelu_forward(xs: np.ndarray, coeffs: list[float], clamp: float) -> np.ndarray:
    xs = np.asarray(xs)
    t = np.minimum(np.abs(xs), clamp)
    poly = np.zeros_like(t)
    for coeff in reversed(coeffs):
        poly = poly * t + coeff
    return xs * (0.5 + np.sign(xs) * t * poly)


def eval_swish_composed(
    xs: np.ndarray, sigmoid_coeffs: list[float], clamp: float
) -> np.ndarray:
    return xs * eval_centered_odd(xs, sigmoid_coeffs, clamp, offset=0.5)


@dataclass(frozen=True)
class StructSpec:
    display_name: str
    current_name_tpl: str
    sollya_name_tpl: str
    degrees: tuple[int, ...]
    expr: Callable[[int], str]
    monomials: Callable[[int], list[int]]
    kind: str
    target: Callable[[np.ndarray], np.ndarray]


BASE_SPECS = (
    StructSpec(
        display_name="sigmoid_fwd",
        current_name_tpl="SIGMOID_FWD_D{degree}_ODD_BF16",
        sollya_name_tpl="SIGMOID_FWD_D{degree}_ODD_SOLLYA_BF16",
        degrees=(3, 4, 5, 6),
        expr=lambda degree: "(1/(1+exp(-x))) - 0.5",
        monomials=lambda degree: list(range(1, degree + 1)),
        kind="centered_odd",
        target=sigmoid,
    ),
    StructSpec(
        display_name="tanh_fwd",
        current_name_tpl="TANH_FWD_D{degree}_ODD_BF16",
        sollya_name_tpl="TANH_FWD_D{degree}_ODD_SOLLYA_BF16",
        degrees=(3, 4, 5, 6),
        expr=lambda degree: "tanh(x)",
        monomials=lambda degree: list(range(1, degree + 1)),
        kind="odd_factorized",
        target=np.tanh,
    ),
    StructSpec(
        display_name="sigmoid_bwd",
        current_name_tpl="SIGMOID_BWD_D{degree}_EVEN_BF16",
        sollya_name_tpl="SIGMOID_BWD_D{degree}_EVEN_SOLLYA_BF16",
        degrees=(3, 4, 5, 6),
        expr=lambda degree: "(1/(1+exp(-x)))*(1-(1/(1+exp(-x))))",
        monomials=lambda degree: list(range(0, degree + 1)),
        kind="even",
        target=lambda xs: sigmoid(xs) * (1.0 - sigmoid(xs)),
    ),
    StructSpec(
        display_name="tanh_bwd",
        current_name_tpl="TANH_BWD_D{degree}_EVEN_BF16",
        sollya_name_tpl="TANH_BWD_D{degree}_EVEN_SOLLYA_BF16",
        degrees=(3, 4, 5, 6),
        expr=lambda degree: "1-(tanh(x)^2)",
        monomials=lambda degree: list(range(0, degree + 1)),
        kind="even",
        target=lambda xs: 1.0 - np.tanh(xs) ** 2,
    ),
    StructSpec(
        display_name="swish_bwd",
        current_name_tpl="SWISH_BWD_D{degree}_ODD_BF16",
        sollya_name_tpl="SWISH_BWD_D{degree}_ODD_SOLLYA_BF16",
        degrees=(3, 4, 5, 6),
        expr=lambda degree: "((1/(1+exp(-x)))*(1+x*(1-(1/(1+exp(-x)))))) - 0.5",
        monomials=lambda degree: list(range(1, degree + 1)),
        kind="centered_odd",
        target=swish_grad,
    ),
    StructSpec(
        display_name="gelu_fwd",
        current_name_tpl="GELU_FWD_D{degree}_ODD_BF16",
        sollya_name_tpl="GELU_FWD_D{degree}_ODD_SOLLYA_BF16",
        degrees=(3, 4, 5, 6),
        expr=lambda degree: "(0.5*x*(1+erf(x/sqrt(2)))) - 0.5*x",
        monomials=lambda degree: list(range(2, degree + 2)),
        kind="gelu_forward",
        target=gelu,
    ),
    StructSpec(
        display_name="gelu_bwd",
        current_name_tpl="GELU_BWD_D{degree}_ODD_BF16",
        sollya_name_tpl="GELU_BWD_D{degree}_ODD_SOLLYA_BF16",
        degrees=(3, 4, 5, 6),
        expr=lambda degree: "(0.5*(1+erf(x/sqrt(2))) + x/sqrt(2*pi)*exp(-(x^2)/2)) - 0.5",
        monomials=lambda degree: list(range(1, degree + 1)),
        kind="centered_odd",
        target=gelu_grad,
    ),
)


SWISH_FWD_DEGREES = (3, 4, 5, 6)


def parse_current_struct(
    name: str,
    text: str,
    source: Path = CURRENT_HEADER,
) -> dict[str, object]:
    pattern = re.compile(rf"struct {re.escape(name)} \{{(?P<body>.*?)\n\}};", re.DOTALL)
    match = pattern.search(text)
    if match is None:
        raise KeyError(f"Could not find struct {name} in {source}")
    body = match.group("body")
    clamp_match = re.search(
        r"__hmin2\(abs_val, __float2bfloat162_rn\(([-+0-9.eE]+)f\)\)", body
    )
    if clamp_match is None:
        raise ValueError(f"Could not parse clamp for {name}")
    coeff_matches = re.findall(
        r"c(\d+) = __float2bfloat162_rn\(([-+0-9.eE]+)f\);", body
    )
    if not coeff_matches:
        raise ValueError(f"Could not parse coefficients for {name}")
    max_idx = max(int(idx_str) for idx_str, _ in coeff_matches)
    coeffs = [0.0] * (max_idx + 1)
    for idx_str, value_str in coeff_matches:
        coeffs[int(idx_str)] = float(value_str)
    return {"clamp": float(clamp_match.group(1)), "coeffs": coeffs}


def run_sollya(expr: str, monomials: list[int], clamp: float) -> list[float]:
    lower = "1b-20" if min(monomials) > 0 else "0"
    monomial_arg = "[|" + ",".join(str(m) for m in monomials) + "|]"
    format_arg = "[|" + ",".join("8" for _ in monomials) + "|]"
    script_lines = [
        f"f = fpminimax({expr}, {monomial_arg}, {format_arg}, [{lower};{clamp}], absolute);"
    ]
    for monomial in monomials:
        script_lines.append(f"print(coeff(f,{monomial}));")
    script_lines.append("quit;")
    proc = subprocess.run(
        ["sollya", "--flush", "--noprompt"],
        input="\n".join(script_lines) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    coeffs = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            coeffs.append(bf16_round(float(line)))
        except ValueError:
            continue
    if len(coeffs) != len(monomials):
        raise RuntimeError(
            f"Unexpected Sollya output for {expr}: stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    return coeffs


def sollya_version() -> str | None:
    completed = subprocess.run(
        ["sollya", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    rendered = " ".join((completed.stdout or completed.stderr).split())
    return rendered or None


def max_error(
    kind: str,
    coeffs: list[float],
    clamp: float,
    target: Callable[[np.ndarray], np.ndarray],
) -> float:
    xs = np.linspace(-clamp, clamp, ERROR_GRID_POINTS)
    truth = target(xs)
    if kind == "centered_odd":
        approx = eval_centered_odd(xs, coeffs, clamp)
    elif kind == "odd_factorized":
        approx = eval_odd_factorized(xs, coeffs, clamp)
    elif kind == "even":
        approx = eval_even(xs, coeffs, clamp)
    elif kind == "gelu_forward":
        approx = eval_gelu_forward(xs, coeffs, clamp)
    elif kind == "swish_composed":
        approx = eval_swish_composed(xs, coeffs, clamp)
    else:  # pragma: no cover - developer error
        raise ValueError(f"Unknown kind {kind}")
    return float(np.max(np.abs(approx - truth)))


def format_coeff(value: float) -> str:
    return f"{value:.10f}f"


def emit_centered_odd_struct(
    name: str, coeffs: list[float], clamp: float, max_err: float
) -> str:
    lines = [
        f"struct {name} {{",
        f"    // Clamp={clamp:.4f}, max_err={max_err:.6f} (Sollya BF16, centered odd)",
        "    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {",
        "        unsigned int sign_mask = 0x80008000;",
        "        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);",
        "        unsigned int signs = input_bits & sign_mask;",
        "        unsigned int abs_bits = input_bits & ~sign_mask;",
        "        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);",
        f"        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn({clamp:.10f}f));",
        "",
    ]
    for idx, coeff in enumerate(coeffs, start=1):
        lines.append(
            f"        __nv_bfloat162 c{idx} = __float2bfloat162_rn({format_coeff(coeff)});"
        )
    lines += [
        "",
        f"        __nv_bfloat162 h = c{len(coeffs)};",
    ]
    for idx in range(len(coeffs) - 2, -1, -1):
        lines.append(f"        h = __hfma2(t, h, c{idx + 1});")
    lines += [
        "        h = __hmul2(t, h);",
        "",
        "        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);",
        "        h_bits ^= signs;",
        "        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);",
        "        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);",
        "    }",
        "};",
        "",
    ]
    return "\n".join(lines)


def emit_odd_factorized_struct(
    name: str, coeffs: list[float], clamp: float, max_err: float
) -> str:
    lines = [
        f"struct {name} {{",
        f"    // Clamp={clamp:.4f}, max_err={max_err:.6f} (Sollya BF16, odd factorized)",
        "    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {",
        "        unsigned int sign_mask = 0x80008000;",
        "        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);",
        "        unsigned int signs = input_bits & sign_mask;",
        "        unsigned int abs_bits = input_bits & ~sign_mask;",
        "        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);",
        f"        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn({clamp:.10f}f));",
        "",
    ]
    for idx, coeff in enumerate(coeffs, start=1):
        lines.append(
            f"        __nv_bfloat162 c{idx} = __float2bfloat162_rn({format_coeff(coeff)});"
        )
    lines += [
        "",
        f"        __nv_bfloat162 h = c{len(coeffs)};",
    ]
    for idx in range(len(coeffs) - 2, -1, -1):
        lines.append(f"        h = __hfma2(t, h, c{idx + 1});")
    lines += [
        "        h = __hmul2(t, h);",
        "        h = __hmin2(h, __float2bfloat162_rn(1.0f));",
        "",
        "        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);",
        "        h_bits ^= signs;",
        "        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);",
        "        return h_signed;",
        "    }",
        "};",
        "",
    ]
    return "\n".join(lines)


def emit_even_struct(
    name: str, coeffs: list[float], clamp: float, max_err: float
) -> str:
    lines = [
        f"struct {name} {{",
        f"    // Clamp={clamp:.4f}, max_err={max_err:.6f} (Sollya BF16, even)",
        "    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {",
        "        unsigned int sign_mask = 0x80008000;",
        "        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);",
        "        unsigned int abs_bits = input_bits & ~sign_mask;",
        "        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);",
        f"        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn({clamp:.10f}f));",
        "",
    ]
    for idx, coeff in enumerate(coeffs):
        lines.append(
            f"        __nv_bfloat162 c{idx} = __float2bfloat162_rn({format_coeff(coeff)});"
        )
    lines += [
        "",
        f"        __nv_bfloat162 r = c{len(coeffs) - 1};",
    ]
    for idx in range(len(coeffs) - 2, -1, -1):
        lines.append(f"        r = __hfma2(t, r, c{idx});")
    lines += [
        "        return r;",
        "    }",
        "};",
        "",
    ]
    return "\n".join(lines)


def emit_gelu_forward_struct(
    name: str, coeffs: list[float], clamp: float, max_err: float
) -> str:
    lines = [
        f"struct {name} {{",
        f"    // Clamp={clamp:.4f}, max_err={max_err:.6f} (Sollya BF16, direct GeLU)",
        "    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {",
        "        unsigned int sign_mask = 0x80008000;",
        "        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);",
        "        unsigned int signs = input_bits & sign_mask;",
        "        unsigned int abs_bits = input_bits & ~sign_mask;",
        "        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);",
        f"        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn({clamp:.10f}f));",
        "",
    ]
    for idx, coeff in enumerate(coeffs, start=1):
        lines.append(
            f"        __nv_bfloat162 c{idx} = __float2bfloat162_rn({format_coeff(coeff)});"
        )
    lines += [
        "",
        f"        __nv_bfloat162 h = c{len(coeffs)};",
    ]
    for idx in range(len(coeffs) - 2, -1, -1):
        lines.append(f"        h = __hfma2(t, h, c{idx + 1});")
    lines += [
        "        h = __hmul2(t, h);",
        "",
        "        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);",
        "        h_bits ^= signs;",
        "        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);",
        "        __nv_bfloat162 phi = __hadd2(__float2bfloat162_rn(0.5f), h_signed);",
        "        return __hmul2(val, phi);",
        "    }",
        "};",
        "",
    ]
    return "\n".join(lines)


def emit_swish_composed_struct(name: str, dependency: str) -> str:
    return "\n".join(
        [
            f"struct {name} {{",
            "    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {",
            f"        return __hmul2(val, {dependency}::evaluate(val));",
            "    }",
            "};",
            "",
        ]
    )


def generate(
    *,
    current_header: Path = CURRENT_HEADER,
    sollya_header: Path = SOLLYA_HEADER,
    out_json: Path = OUT_JSON,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    current_text = current_header.read_text(encoding="utf-8")
    results: dict[str, object] = {
        "measurement": dict(ERROR_MEASUREMENT),
        "families": {},
    }
    emitted_sections = [
        "// spline_structs_sollya_bf16.cuh — BF16 Sollya fpminimax activation structs",
        "// AUTO-GENERATED with the same runtime clamps and evaluation shapes as spline_structs_odd_bf16.cuh",
        "#pragma once",
        "#include <cuda_bf16.h>",
        "",
    ]

    parsed_current: dict[str, dict[str, object]] = {}
    for spec in BASE_SPECS:
        results["families"][spec.display_name] = {}
        emitted_sections.append(
            "// ============================================================================="
        )
        emitted_sections.append(f"// {spec.display_name} — Sollya BF16 variants")
        emitted_sections.append(
            "// ============================================================================="
        )
        emitted_sections.append("")
        for degree in spec.degrees:
            current_name = spec.current_name_tpl.format(degree=degree)
            parsed = parse_current_struct(current_name, current_text, current_header)
            parsed_current[current_name] = parsed
            clamp = float(parsed["clamp"])
            current_coeffs = [float(v) for v in parsed["coeffs"]]
            monomials = spec.monomials(degree)
            sollya_coeffs = run_sollya(spec.expr(degree), monomials, clamp)

            if spec.kind == "centered_odd":
                current_eval_coeffs = current_coeffs[1:]
                current_err = max_error(
                    spec.kind, current_eval_coeffs, clamp, spec.target
                )
                sollya_err = max_error(spec.kind, sollya_coeffs, clamp, spec.target)
                emitter = emit_centered_odd_struct
            elif spec.kind == "odd_factorized":
                current_eval_coeffs = current_coeffs[1:]
                current_err = max_error(
                    spec.kind, current_eval_coeffs, clamp, spec.target
                )
                sollya_err = max_error(spec.kind, sollya_coeffs, clamp, spec.target)
                emitter = emit_odd_factorized_struct
            elif spec.kind == "even":
                current_eval_coeffs = current_coeffs
                current_err = max_error(
                    spec.kind, current_eval_coeffs, clamp, spec.target
                )
                sollya_err = max_error(spec.kind, sollya_coeffs, clamp, spec.target)
                emitter = emit_even_struct
            elif spec.kind == "gelu_forward":
                current_eval_coeffs = current_coeffs[1:]
                current_err = max_error(
                    spec.kind, current_eval_coeffs, clamp, spec.target
                )
                sollya_err = max_error(spec.kind, sollya_coeffs, clamp, spec.target)
                emitter = emit_gelu_forward_struct
            else:  # pragma: no cover - developer error
                raise ValueError(f"Unknown kind {spec.kind}")

            sollya_name = spec.sollya_name_tpl.format(degree=degree)
            emitted_sections.append(
                emitter(sollya_name, sollya_coeffs, clamp, sollya_err)
            )
            results["families"][spec.display_name][f"D{degree}"] = {
                "current_struct": current_name,
                "sollya_struct": sollya_name,
                "shape": spec.kind,
                "clamp": clamp,
                "current_coeffs": current_eval_coeffs,
                "sollya_coeffs": sollya_coeffs,
                "current_max_error": current_err,
                "sollya_max_error": sollya_err,
            }

    emitted_sections.append(
        "// ============================================================================="
    )
    emitted_sections.append("// swish_fwd — Sollya-composed BF16 variants")
    emitted_sections.append(
        "// ============================================================================="
    )
    emitted_sections.append("")
    results["families"]["swish_fwd"] = {}
    for degree in SWISH_FWD_DEGREES:
        current_sig = results["families"]["sigmoid_fwd"][f"D{degree}"]
        clamp = float(current_sig["clamp"])
        current_err = max_error(
            "swish_composed", current_sig["current_coeffs"], clamp, swish
        )
        sollya_err = max_error(
            "swish_composed", current_sig["sollya_coeffs"], clamp, swish
        )
        current_name = f"SWISH_FWD_D{degree}_ODD_BF16"
        sollya_name = f"SWISH_FWD_D{degree}_ODD_SOLLYA_BF16"
        emitted_sections.append(
            emit_swish_composed_struct(
                sollya_name,
                f"SIGMOID_FWD_D{degree}_ODD_SOLLYA_BF16",
            )
        )
        results["families"]["swish_fwd"][f"D{degree}"] = {
            "current_struct": current_name,
            "sollya_struct": sollya_name,
            "shape": "swish_composed",
            "clamp": clamp,
            "current_coeffs": current_sig["current_coeffs"],
            "sollya_coeffs": current_sig["sollya_coeffs"],
            "current_max_error": current_err,
            "sollya_max_error": sollya_err,
        }

    sollya_header.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    sollya_header.write_text("\n".join(emitted_sections) + "\n", encoding="utf-8")
    if provenance is not None:
        provenance["generated_header_sha256"] = sha256_file(sollya_header)
        bind_fit_payload(results, provenance)
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-header", type=Path, default=CURRENT_HEADER)
    parser.add_argument("--header-out", type=Path, default=SOLLYA_HEADER)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    return parser


def main(argv: list[str] | None = None) -> None:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    args = get_parser().parse_args(argv)
    provenance = build_fit_provenance(
        script=Path(__file__),
        arguments=raw_arguments,
        source_files=[args.current_header],
        distributions=("numpy", "torch"),
    )
    environment = provenance["environment"]
    if not isinstance(environment, dict):  # pragma: no cover - helper contract
        raise RuntimeError("fit provenance environment must be an object")
    environment["sollya"] = sollya_version()
    results = generate(
        current_header=args.current_header,
        sollya_header=args.header_out,
        out_json=args.json_out,
        provenance=provenance,
    )
    print(f"Wrote {args.header_out}")
    print(f"Wrote {args.json_out}")
    for family in ("sigmoid_fwd", "tanh_fwd", "swish_fwd", "gelu_fwd"):
        print(f"{family}:")
        for degree, row in results["families"][family].items():
            print(
                f"  {degree}: ours={row['current_max_error']:.6f} "
                f"sollya={row['sollya_max_error']:.6f}"
            )


if __name__ == "__main__":
    main()

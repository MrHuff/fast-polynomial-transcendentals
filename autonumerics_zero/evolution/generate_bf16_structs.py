#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified in 2026 for the standalone fast-polynomial-transcendentals release.

"""
Generate a CUDA header with BF16-optimized activation coefficients.

The sigmoid, tanh, and SiLU families come from the caller-selected BF16 fit
JSON. ERF and GELU are copied byte-for-byte at the struct level from a
caller-selected reviewed fallback header because this fitter does not produce
those families.

Usage:
    python3 generate_bf16_structs.py \
      --bf16-input outputs/all_degree_coefficients_bf16.json \
      --output outputs/spline_structs_odd_bf16.generated.cuh
"""
import argparse
import json
from pathlib import Path
import platform
import re
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sfu_repro.source_attestation import (  # noqa: E402
    git_state,
    safe_command,
    sha256_file,
)

try:  # noqa: E402
    from .fit_provenance import (
        fit_output_is_source_bound,
        numerical_payload_sha256,
    )
except ImportError:  # Direct script execution.
    from fit_provenance import fit_output_is_source_bound, numerical_payload_sha256


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bf16-input",
        type=Path,
        default=(
            REPOSITORY_ROOT / "autonumerics_zero/cuda_benchmarks/analysis_results/"
            "all_degree_coefficients_bf16.json"
        ),
        help="BF16 fitting JSON used for the generated activation families.",
    )
    parser.add_argument(
        "--fallback-header",
        type=Path,
        default=(
            REPOSITORY_ROOT / "autonumerics_zero/spline_ops/spline_structs_odd_bf16.cuh"
        ),
        help="Reviewed header from which ERF and GELU structs are copied.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/spline_structs_odd_bf16.generated.cuh"),
        help="Caller-selected generated header; never defaults to a tracked source.",
    )
    parser.add_argument(
        "--receipt-out",
        type=Path,
        help="Optional provenance receipt path (defaults beside --output).",
    )
    parser.add_argument(
        "--allow-unbound-source",
        action="store_true",
        help="Permit generation from a dirty or unversioned checkout.",
    )
    return parser


args = get_parser().parse_args()
bf16_path = args.bf16_input.resolve()
fallback_header = args.fallback_header.resolve()
out_path = args.output.resolve()
receipt_path = (
    args.receipt_out.resolve()
    if args.receipt_out is not None
    else out_path.with_suffix(out_path.suffix + ".provenance.json")
)
repository_state = git_state(REPOSITORY_ROOT)
repository_bound = bool(
    repository_state.get("revision") is not None
    and repository_state.get("dirty") is False
)
if not bf16_path.is_file() or not fallback_header.is_file():
    raise RuntimeError("the BF16 input and fallback header must be regular files")
if out_path in {bf16_path, fallback_header} or receipt_path in {
    bf16_path,
    fallback_header,
    out_path,
}:
    raise RuntimeError("outputs must not replace an input or each other")

with bf16_path.open(encoding="utf-8") as stream:
    bf16_data = json.load(stream)

fit_payload_digest = numerical_payload_sha256(bf16_data)
fit_bound = fit_output_is_source_bound(
    bf16_data,
    expected_script="fit_all_degrees_bf16.py",
    repository_state=repository_state,
)
source_bound = repository_bound and fit_bound
if not source_bound and not args.allow_unbound_source:
    raise RuntimeError(
        "the repository or BF16 fit provenance is unbound; generate the fit in "
        "this clean checkout or use --allow-unbound-source for a diagnostic "
        "generated artifact"
    )


def gen_odd_struct(name, coeffs, Lc, Li, err, degree):
    """Generate ODD struct: f(x) = 0.5 + sign(x)*h(|x|), h is odd poly."""
    c = coeffs  # c0=0, c1, c2, ..., cn
    lines = []
    lines.append(f"struct {name} {{")
    lines.append(f"    // Li={Li:.2f}, Lc={Lc:.2f}, Err={err:.6f} (BF16-optimized)")
    lines.append(
        f"    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {{"
    )
    lines.append(f"        unsigned int sign_mask = 0x80008000;")
    lines.append(
        f"        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);"
    )
    lines.append(f"        unsigned int signs = input_bits & sign_mask;")
    lines.append(f"        unsigned int abs_bits = input_bits & ~sign_mask;")
    lines.append(
        f"        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);"
    )
    lines.append(
        f"        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn({Lc}f));"
    )
    lines.append(f"")

    # Emit coefficient constants (skip c0=0 for ODD)
    for i in range(1, degree + 1):
        lines.append(
            f"        __nv_bfloat162 c{i} = __float2bfloat162_rn({c[i]:.10f}f);"
        )

    lines.append(f"")

    # Horner: h = cn, then h = h*t + c_{n-1}, ..., h = h*t + c1, result = h*t
    lines.append(f"        __nv_bfloat162 h = __hfma2(t, c{degree}, c{degree-1});")
    for i in range(degree - 2, 0, -1):
        lines.append(f"        h = __hfma2(t, h, c{i});")
    lines.append(f"        h = __hmul2(t, h);")

    lines.append(f"")
    lines.append(f"        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);")
    lines.append(f"        h_bits ^= signs;")
    lines.append(
        f"        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);"
    )
    lines.append(f"        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);")
    lines.append(f"    }}")
    lines.append(f"}};")
    return "\n".join(lines)


def gen_tanh_odd_struct(name, coeffs, Lc, Li, err, degree):
    """Generate ODD struct for tanh: tanh(x) = sign(x)*h(|x|)."""
    c = coeffs
    lines = []
    lines.append(f"struct {name} {{")
    lines.append(f"    // Li={Li:.2f}, Lc={Lc:.2f}, Err={err:.6f} (BF16-optimized)")
    lines.append(
        f"    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {{"
    )
    lines.append(f"        unsigned int sign_mask = 0x80008000;")
    lines.append(
        f"        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);"
    )
    lines.append(f"        unsigned int signs = input_bits & sign_mask;")
    lines.append(f"        unsigned int abs_bits = input_bits & ~sign_mask;")
    lines.append(
        f"        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);"
    )
    lines.append(
        f"        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn({Lc}f));"
    )
    lines.append(f"")

    for i in range(1, degree + 1):
        lines.append(
            f"        __nv_bfloat162 c{i} = __float2bfloat162_rn({c[i]:.10f}f);"
        )

    lines.append(f"")
    lines.append(f"        __nv_bfloat162 h = __hfma2(t, c{degree}, c{degree-1});")
    for i in range(degree - 2, 0, -1):
        lines.append(f"        h = __hfma2(t, h, c{i});")
    lines.append(f"        h = __hmul2(t, h);")

    # Clamp to [-1, 1] (tanh output range)
    lines.append(f"        h = __hmin2(h, __float2bfloat162_rn(1.0f));")

    lines.append(f"")
    lines.append(f"        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);")
    lines.append(f"        h_bits ^= signs;")
    lines.append(
        f"        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);"
    )
    lines.append(f"        return h_signed;")
    lines.append(f"    }}")
    lines.append(f"}};")
    return "\n".join(lines)


def gen_even_struct(name, coeffs, Lc, Li, err, degree, asymptotic_val="0.0f"):
    """Generate EVEN struct: f(|x|) evaluated via even poly, clamped at asymptote."""
    c = coeffs
    lines = []
    lines.append(f"struct {name} {{")
    lines.append(f"    // Li={Li:.2f}, Lc={Lc:.2f}, Err={err:.6f} (BF16-optimized)")
    lines.append(
        f"    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {{"
    )
    lines.append(f"        unsigned int sign_mask = 0x80008000;")
    lines.append(
        f"        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);"
    )
    lines.append(f"        unsigned int abs_bits = input_bits & ~sign_mask;")
    lines.append(
        f"        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);"
    )
    lines.append(
        f"        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn({Lc}f));"
    )
    lines.append(f"")

    for i in range(degree, -1, -1):
        lines.append(
            f"        __nv_bfloat162 c{i} = __float2bfloat162_rn({c[i]:.10f}f);"
        )

    lines.append(f"")
    # Horner for EVEN: r = cn*t + c_{n-1}, then r = r*t + c_{n-2}, ..., r = r*t + c0
    lines.append(f"        __nv_bfloat162 r = __hfma2(t, c{degree}, c{degree-1});")
    for i in range(degree - 2, -1, -1):
        lines.append(f"        r = __hfma2(t, r, c{i});")

    lines.append(f"        return r;")
    lines.append(f"    }}")
    lines.append(f"}};")
    return "\n".join(lines)


def gen_swish_bwd_odd_struct(name, coeffs, Lc, Li, err, degree):
    """Generate ODD struct for swish': swish'(x) = 0.5 + sign(x)*h(|x|)."""
    c = coeffs
    lines = []
    lines.append(f"struct {name} {{")
    lines.append(f"    // Li={Li:.2f}, Lc={Lc:.2f}, Err={err:.6f} (BF16-optimized)")
    lines.append(
        f"    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {{"
    )
    lines.append(f"        unsigned int sign_mask = 0x80008000;")
    lines.append(
        f"        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);"
    )
    lines.append(f"        unsigned int signs = input_bits & sign_mask;")
    lines.append(f"        unsigned int abs_bits = input_bits & ~sign_mask;")
    lines.append(
        f"        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);"
    )
    lines.append(
        f"        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn({Lc}f));"
    )
    lines.append(f"")

    for i in range(1, degree + 1):
        lines.append(
            f"        __nv_bfloat162 c{i} = __float2bfloat162_rn({c[i]:.10f}f);"
        )

    lines.append(f"")
    lines.append(f"        __nv_bfloat162 h = __hfma2(t, c{degree}, c{degree-1});")
    for i in range(degree - 2, 0, -1):
        lines.append(f"        h = __hfma2(t, h, c{i});")
    lines.append(f"        h = __hmul2(t, h);")

    lines.append(f"")
    lines.append(f"        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);")
    lines.append(f"        h_bits ^= signs;")
    lines.append(
        f"        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);"
    )
    lines.append(f"        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);")
    lines.append(f"    }}")
    lines.append(f"}};")
    return "\n".join(lines)


def get_coeffs(data, func_key, deg):
    """Extract coefficient info from the JSON data."""
    dk = f"d{deg}"
    v = data[func_key][dk]
    return v["coeffs_bf16"], v["Lc"], v["Li"], v["err"]


# ==========================================================================
# Build the file
# ==========================================================================

output = []
output.append(
    "// spline_structs_odd_bf16.cuh — D3-D6 ODD/EVEN activation structs (BFloat16)"
)
output.append(
    "// AUTO-GENERATED with BF16-optimized coefficients via fit_all_degrees_bf16.py"
)
output.append("// Uses __nv_bfloat162 vectorized type for 2-wide BF16 operations.")
output.append("#pragma once")
output.append("#include <cuda_bf16.h>")
output.append("")

# -------  SIGMOID FWD  -------
output.append(
    "// ============================================================================="
)
output.append("// SIGMOID FWD — ODD: sigmoid(x) = 0.5 + sign(x)*h(|x|)")
output.append(
    "// ============================================================================="
)
output.append("")

for deg in [3, 4, 5, 6]:
    c, Lc, Li, err = get_coeffs(bf16_data, "sigmoid_fwd_odd", deg)
    output.append(gen_odd_struct(f"SIGMOID_FWD_D{deg}_ODD_BF16", c, Lc, Li, err, deg))
    output.append("")

# -------  TANH FWD  -------
output.append(
    "// ============================================================================="
)
output.append("// TANH FWD — ODD: tanh(x) = sign(x)*h(|x|)")
output.append(
    "// ============================================================================="
)
output.append("")

for deg in [3, 4, 5, 6]:
    c, Lc, Li, err = get_coeffs(bf16_data, "tanh_fwd_odd", deg)
    output.append(gen_tanh_odd_struct(f"TANH_FWD_D{deg}_ODD_BF16", c, Lc, Li, err, deg))
    output.append("")

# -------  SWISH FWD (composed from sigmoid) -------
output.append(
    "// ============================================================================="
)
output.append("// SWISH FWD — ODD: swish(x) = x * sigmoid(x)")
output.append("// Composed: uses sigmoid spline internally")
output.append(
    "// ============================================================================="
)
output.append("")

# Swish FWD uses sigmoid FWD internally: swish(x) = x * sigmoid(x)
# We generate these for each degree using the sigmoid coefficients
for deg in [3, 4, 5, 6]:
    c, Lc, Li, err = get_coeffs(bf16_data, "sigmoid_fwd_odd", deg)
    lines = []
    lines.append(f"struct SWISH_FWD_D{deg}_ODD_BF16 {{")
    lines.append(
        f"    // Composed: swish(x) = x * sigmoid(x), uses SIGMOID_FWD_D{deg}_ODD_BF16"
    )
    lines.append(
        f"    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {{"
    )
    lines.append(
        f"        return __hmul2(val, SIGMOID_FWD_D{deg}_ODD_BF16::evaluate(val));"
    )
    lines.append(f"    }}")
    lines.append(f"}};")
    output.append("\n".join(lines))
    output.append("")

# -------  SIGMOID BWD (EVEN)  -------
output.append(
    "// ============================================================================="
)
output.append("// SIGMOID BWD — EVEN: sigmoid'(|x|)")
output.append(
    "// ============================================================================="
)
output.append("")

for deg in [3, 4, 5, 6]:
    c, Lc, Li, err = get_coeffs(bf16_data, "sigmoid_bwd_even", deg)
    output.append(gen_even_struct(f"SIGMOID_BWD_D{deg}_EVEN_BF16", c, Lc, Li, err, deg))
    output.append("")

# -------  TANH BWD (EVEN)  -------
output.append(
    "// ============================================================================="
)
output.append("// TANH BWD — EVEN: tanh'(|x|) = 1 - tanh(|x|)^2")
output.append(
    "// ============================================================================="
)
output.append("")

for deg in [3, 4, 5, 6]:
    c, Lc, Li, err = get_coeffs(bf16_data, "tanh_bwd_even", deg)
    output.append(gen_even_struct(f"TANH_BWD_D{deg}_EVEN_BF16", c, Lc, Li, err, deg))
    output.append("")

# -------  SWISH BWD (ODD)  -------
output.append(
    "// ============================================================================="
)
output.append("// SWISH BWD — ODD: swish'(x) = 0.5 + sign(x)*h(|x|)")
output.append(
    "// ============================================================================="
)
output.append("")

for deg in [3, 4, 5, 6]:
    c, Lc, Li, err = get_coeffs(bf16_data, "swish_bwd_odd", deg)
    output.append(
        gen_swish_bwd_odd_struct(f"SWISH_BWD_D{deg}_ODD_BF16", c, Lc, Li, err, deg)
    )
    output.append("")

# -------  ERF FWD (ODD)  -------
# ERF uses the FP16-derived coefficients (no BF16 fitter for these yet)
output.append(
    "// ============================================================================="
)
output.append("// ERF FWD — ODD: erf(x) = sign(x)*h(|x|)")
output.append("// NOTE: Using FP16-derived coefficients (no BF16 fitter yet)")
output.append(
    "// ============================================================================="
)
output.append("")

# Read the FP16 ERF coefficients from the existing struct file
# Since we don't have ERF in the fitter, we'll use the same coefficients
# as the mechanical conversion from FP16
# ERF is basically 2*sigmoid(sqrt(2)*x) - 1, approximated as tanh-like odd poly
# Let's just keep the original mechanically converted ERF structs

# Read the reviewed fallback file and extract ERF/GELU sections.
original = fallback_header.read_text(encoding="utf-8")

# Extract ERF structs


def extract_fallback_struct(name):
    match = re.search(rf"struct {re.escape(name)}.*?\}};", original, re.DOTALL)
    if match is None:
        raise RuntimeError(f"fallback header does not contain required struct {name}")
    return match.group(0)


for deg in [3, 4, 5, 6]:
    output.append(extract_fallback_struct(f"ERF_FWD_D{deg}_ODD_BF16"))
    output.append("")

# -------  GELU FWD (ODD)  -------
output.append(
    "// ============================================================================="
)
output.append("// GELU FWD — ODD: gelu(x) = 0.5*x*(1 + erf(x/sqrt(2)))")
output.append("// NOTE: Using FP16-derived coefficients (no BF16 fitter yet)")
output.append(
    "// ============================================================================="
)
output.append("")

for deg in [3, 4, 5, 6]:
    output.append(extract_fallback_struct(f"GELU_FWD_D{deg}_ODD_BF16"))
    output.append("")

# -------  GELU BWD (ODD)  -------
output.append(
    "// ============================================================================="
)
output.append("// GELU BWD — ODD")
output.append("// NOTE: Using FP16-derived coefficients (no BF16 fitter yet)")
output.append(
    "// ============================================================================="
)
output.append("")

for deg in [3, 4, 5, 6]:
    output.append(extract_fallback_struct(f"GELU_BWD_D{deg}_ODD_BF16"))
    output.append("")


# -------  ALGEBRAIC BACKWARD  -------
output.append(
    "// ============================================================================="
)
output.append("// ALGEBRAIC BACKWARD PASSES")
output.append(
    "// ============================================================================="
)
output.append("")

# Sigmoid BWD Algebraic: y*(1-y) where y = sigmoid(x)
output.append(
    """struct SIGMOID_BWD_ALGEBRAIC_BF16 {
    static __device__ __forceinline__ __nv_bfloat162 eval_grad(__nv_bfloat162 x, __nv_bfloat162) {
        __nv_bfloat162 y = SIGMOID_FWD_D3_ODD_BF16::evaluate(x);
        __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
        return __hmul2(y, __hsub2(one, y));
    }
};"""
)
output.append("")

# Tanh BWD Algebraic: 1 - tanh(x)^2
output.append(
    """struct TANH_BWD_ALGEBRAIC_BF16 {
    static __device__ __forceinline__ __nv_bfloat162 eval_grad(__nv_bfloat162 x, __nv_bfloat162) {
        __nv_bfloat162 y = TANH_FWD_D3_ODD_BF16::evaluate(x);
        __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
        return __hsub2(one, __hmul2(y, y));
    }
};"""
)
output.append("")

# -------  BACKWARD COMPATIBILITY ALIASES  -------
output.append(
    "// ============================================================================="
)
output.append("// BACKWARD COMPATIBILITY ALIASES")
output.append(
    "// ============================================================================="
)
output.append("#ifndef SPLINE_STRUCTS_NO_ALIASES")
output.append("using SIGMOID_N2_D3_ODD_BF16 = SIGMOID_FWD_D3_ODD_BF16;")
output.append("using SPLINE_TANH_FWD_D3_BF16 = TANH_FWD_D3_ODD_BF16;")
output.append("using SPLINE_SIGMOID_GRAD_D4_BF16 = SIGMOID_BWD_D4_EVEN_BF16;")
output.append("using SPLINE_TANH_GRAD_D4_BF16 = TANH_BWD_D4_EVEN_BF16;")
output.append("using SPLINE_SWISH_GRAD_D3_ODD_BF16 = SWISH_BWD_D3_ODD_BF16;")
output.append("using SPLINE_SWISH_GRAD_D4_ODD_BF16 = SWISH_BWD_D4_ODD_BF16;")
output.append("using SPLINE_SWISH_GRAD_D5_ODD_BF16 = SWISH_BWD_D5_ODD_BF16;")
output.append("using SPLINE_SWISH_GRAD_D6_ODD_BF16 = SWISH_BWD_D6_ODD_BF16;")
output.append("using SWISH_FWD_D3_FUSED_ODD_BF16 = SWISH_FWD_D3_ODD_BF16;")
output.append("#endif // SPLINE_STRUCTS_NO_ALIASES")
output.append("")
output.append("")

# Write the generated header and a sidecar that binds every transformation
# input. The caller promotes the reviewed generated header into source in a
# separate, explicit step.
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(output) + "\n", encoding="utf-8")

input_hashes = {
    "bf16_fit_json": sha256_file(bf16_path),
    "fallback_header": sha256_file(fallback_header),
    "generator": sha256_file(Path(__file__).resolve()),
}
generated_header_sha256 = sha256_file(out_path)
if any(digest is None for digest in input_hashes.values()) or (
    generated_header_sha256 is None
):
    raise RuntimeError("could not hash every header-generation input and output")
receipt = {
    "schema_version": 1,
    "artifact_type": "bf16_activation_header_generation_receipt",
    "provenance_class": (
        "source-bound-generated-artifact"
        if source_bound
        else "diagnostic-unbound-source"
    ),
    "source": {
        "repository": "https://github.com/MrHuff/fast-polynomial-transcendentals",
        **repository_state,
        "input_sha256": input_hashes,
    },
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
    },
    "artifact": {
        "path": out_path.name,
        "sha256": generated_header_sha256,
    },
    "command": safe_command([Path(sys.executable).name, *sys.argv], REPOSITORY_ROOT),
    "transformation": {
        "bf16_fit_source_bound": fit_bound,
        "bf16_fit_numerical_payload_sha256": fit_payload_digest,
        "generated_families": [
            "sigmoid_forward",
            "tanh_forward",
            "silu_forward_composed_from_sigmoid",
            "sigmoid_backward",
            "tanh_backward",
            "silu_backward",
        ],
        "copied_fallback_families": [
            "erf_forward",
            "gelu_forward",
            "gelu_backward",
        ],
    },
}
receipt_path.parent.mkdir(parents=True, exist_ok=True)
receipt_path.write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

line_count = len(output)
print(f"Generated {out_path}: {line_count} lines")
print(f"Wrote provenance receipt {receipt_path}")

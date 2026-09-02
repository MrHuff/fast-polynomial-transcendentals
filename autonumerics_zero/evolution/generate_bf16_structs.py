#!/usr/bin/env python3
"""
Generate spline_structs_odd_bf16.cuh with BF16-optimized coefficients.

Reads the BF16 coefficient JSON and produces the complete CUDA header file.
For functions not covered by the BF16 fitter (Swish FWD, ERF, GELU),
falls back to the FP16 coefficients (mechanically converted).

Usage:
    python3 generate_bf16_structs.py
"""
import json
import os

# Load BF16 coefficients
bf16_path = os.path.join(os.path.dirname(__file__), "..", "cuda_benchmarks",
                         "analysis_results", "all_degree_coefficients_bf16.json")
bf16_data = json.load(open(bf16_path))

# For functions not in BF16 fitter, load FP16 coefficients
fp16_path = os.path.join(os.path.dirname(__file__), "..", "cuda_benchmarks",
                         "analysis_results", "all_degree_coefficients.json")
fp16_data = json.load(open(fp16_path))


def gen_odd_struct(name, coeffs, Lc, Li, err, degree):
    """Generate ODD struct: f(x) = 0.5 + sign(x)*h(|x|), h is odd poly."""
    c = coeffs  # c0=0, c1, c2, ..., cn
    lines = []
    lines.append(f"struct {name} {{")
    lines.append(f"    // Li={Li:.2f}, Lc={Lc:.2f}, Err={err:.6f} (BF16-optimized)")
    lines.append(f"    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {{")
    lines.append(f"        unsigned int sign_mask = 0x80008000;")
    lines.append(f"        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);")
    lines.append(f"        unsigned int signs = input_bits & sign_mask;")
    lines.append(f"        unsigned int abs_bits = input_bits & ~sign_mask;")
    lines.append(f"        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);")
    lines.append(f"        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn({Lc}f));")
    lines.append(f"")

    # Emit coefficient constants (skip c0=0 for ODD)
    for i in range(1, degree + 1):
        lines.append(f"        __nv_bfloat162 c{i} = __float2bfloat162_rn({c[i]:.10f}f);")

    lines.append(f"")

    # Horner: h = cn, then h = h*t + c_{n-1}, ..., h = h*t + c1, result = h*t
    lines.append(f"        __nv_bfloat162 h = __hfma2(t, c{degree}, c{degree-1});")
    for i in range(degree-2, 0, -1):
        lines.append(f"        h = __hfma2(t, h, c{i});")
    lines.append(f"        h = __hmul2(t, h);")

    lines.append(f"")
    lines.append(f"        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);")
    lines.append(f"        h_bits ^= signs;")
    lines.append(f"        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);")
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
    lines.append(f"    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {{")
    lines.append(f"        unsigned int sign_mask = 0x80008000;")
    lines.append(f"        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);")
    lines.append(f"        unsigned int signs = input_bits & sign_mask;")
    lines.append(f"        unsigned int abs_bits = input_bits & ~sign_mask;")
    lines.append(f"        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);")
    lines.append(f"        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn({Lc}f));")
    lines.append(f"")

    for i in range(1, degree + 1):
        lines.append(f"        __nv_bfloat162 c{i} = __float2bfloat162_rn({c[i]:.10f}f);")

    lines.append(f"")
    lines.append(f"        __nv_bfloat162 h = __hfma2(t, c{degree}, c{degree-1});")
    for i in range(degree-2, 0, -1):
        lines.append(f"        h = __hfma2(t, h, c{i});")
    lines.append(f"        h = __hmul2(t, h);")

    # Clamp to [-1, 1] (tanh output range)
    lines.append(f"        h = __hmin2(h, __float2bfloat162_rn(1.0f));")

    lines.append(f"")
    lines.append(f"        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);")
    lines.append(f"        h_bits ^= signs;")
    lines.append(f"        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);")
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
    lines.append(f"    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {{")
    lines.append(f"        unsigned int sign_mask = 0x80008000;")
    lines.append(f"        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);")
    lines.append(f"        unsigned int abs_bits = input_bits & ~sign_mask;")
    lines.append(f"        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);")
    lines.append(f"        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn({Lc}f));")
    lines.append(f"")

    for i in range(degree, -1, -1):
        lines.append(f"        __nv_bfloat162 c{i} = __float2bfloat162_rn({c[i]:.10f}f);")

    lines.append(f"")
    # Horner for EVEN: r = cn*t + c_{n-1}, then r = r*t + c_{n-2}, ..., r = r*t + c0
    lines.append(f"        __nv_bfloat162 r = __hfma2(t, c{degree}, c{degree-1});")
    for i in range(degree-2, -1, -1):
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
    lines.append(f"    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {{")
    lines.append(f"        unsigned int sign_mask = 0x80008000;")
    lines.append(f"        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);")
    lines.append(f"        unsigned int signs = input_bits & sign_mask;")
    lines.append(f"        unsigned int abs_bits = input_bits & ~sign_mask;")
    lines.append(f"        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);")
    lines.append(f"        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn({Lc}f));")
    lines.append(f"")

    for i in range(1, degree + 1):
        lines.append(f"        __nv_bfloat162 c{i} = __float2bfloat162_rn({c[i]:.10f}f);")

    lines.append(f"")
    lines.append(f"        __nv_bfloat162 h = __hfma2(t, c{degree}, c{degree-1});")
    for i in range(degree-2, 0, -1):
        lines.append(f"        h = __hfma2(t, h, c{i});")
    lines.append(f"        h = __hmul2(t, h);")

    lines.append(f"")
    lines.append(f"        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);")
    lines.append(f"        h_bits ^= signs;")
    lines.append(f"        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);")
    lines.append(f"        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);")
    lines.append(f"    }}")
    lines.append(f"}};")
    return "\n".join(lines)


def get_coeffs(data, func_key, deg):
    """Extract coefficient info from the JSON data."""
    dk = f"d{deg}"
    v = data[func_key][dk]
    return v['coeffs_bf16'], v['Lc'], v['Li'], v['err']


# ==========================================================================
# Build the file
# ==========================================================================

output = []
output.append("// spline_structs_odd_bf16.cuh — D3-D6 ODD/EVEN activation structs (BFloat16)")
output.append("// AUTO-GENERATED with BF16-optimized coefficients via fit_all_degrees_bf16.py")
output.append("// Uses __nv_bfloat162 vectorized type for 2-wide BF16 operations.")
output.append("#pragma once")
output.append("#include <cuda_bf16.h>")
output.append("")

# -------  SIGMOID FWD  -------
output.append("// =============================================================================")
output.append("// SIGMOID FWD — ODD: sigmoid(x) = 0.5 + sign(x)*h(|x|)")
output.append("// =============================================================================")
output.append("")

for deg in [3, 4, 5, 6]:
    c, Lc, Li, err = get_coeffs(bf16_data, 'sigmoid_fwd_odd', deg)
    output.append(gen_odd_struct(f"SIGMOID_FWD_D{deg}_ODD_BF16", c, Lc, Li, err, deg))
    output.append("")

# -------  TANH FWD  -------
output.append("// =============================================================================")
output.append("// TANH FWD — ODD: tanh(x) = sign(x)*h(|x|)")
output.append("// =============================================================================")
output.append("")

for deg in [3, 4, 5, 6]:
    c, Lc, Li, err = get_coeffs(bf16_data, 'tanh_fwd_odd', deg)
    output.append(gen_tanh_odd_struct(f"TANH_FWD_D{deg}_ODD_BF16", c, Lc, Li, err, deg))
    output.append("")

# -------  SWISH FWD (composed from sigmoid) -------
output.append("// =============================================================================")
output.append("// SWISH FWD — ODD: swish(x) = x * sigmoid(x)")
output.append("// Composed: uses sigmoid spline internally")
output.append("// =============================================================================")
output.append("")

# Swish FWD uses sigmoid FWD internally: swish(x) = x * sigmoid(x)
# We generate these for each degree using the sigmoid coefficients
for deg in [3, 4, 5, 6]:
    c, Lc, Li, err = get_coeffs(bf16_data, 'sigmoid_fwd_odd', deg)
    lines = []
    lines.append(f"struct SWISH_FWD_D{deg}_ODD_BF16 {{")
    lines.append(f"    // Composed: swish(x) = x * sigmoid(x), uses SIGMOID_FWD_D{deg}_ODD_BF16")
    lines.append(f"    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {{")
    lines.append(f"        return __hmul2(val, SIGMOID_FWD_D{deg}_ODD_BF16::evaluate(val));")
    lines.append(f"    }}")
    lines.append(f"}};")
    output.append("\n".join(lines))
    output.append("")

# -------  SIGMOID BWD (EVEN)  -------
output.append("// =============================================================================")
output.append("// SIGMOID BWD — EVEN: sigmoid'(|x|)")
output.append("// =============================================================================")
output.append("")

for deg in [3, 4, 5, 6]:
    c, Lc, Li, err = get_coeffs(bf16_data, 'sigmoid_bwd_even', deg)
    output.append(gen_even_struct(f"SIGMOID_BWD_D{deg}_EVEN_BF16", c, Lc, Li, err, deg))
    output.append("")

# -------  TANH BWD (EVEN)  -------
output.append("// =============================================================================")
output.append("// TANH BWD — EVEN: tanh'(|x|) = 1 - tanh(|x|)^2")
output.append("// =============================================================================")
output.append("")

for deg in [3, 4, 5, 6]:
    c, Lc, Li, err = get_coeffs(bf16_data, 'tanh_bwd_even', deg)
    output.append(gen_even_struct(f"TANH_BWD_D{deg}_EVEN_BF16", c, Lc, Li, err, deg))
    output.append("")

# -------  SWISH BWD (ODD)  -------
output.append("// =============================================================================")
output.append("// SWISH BWD — ODD: swish'(x) = 0.5 + sign(x)*h(|x|)")
output.append("// =============================================================================")
output.append("")

for deg in [3, 4, 5, 6]:
    c, Lc, Li, err = get_coeffs(bf16_data, 'swish_bwd_odd', deg)
    output.append(gen_swish_bwd_odd_struct(f"SWISH_BWD_D{deg}_ODD_BF16", c, Lc, Li, err, deg))
    output.append("")

# -------  ERF FWD (ODD)  -------
# ERF uses the FP16-derived coefficients (no BF16 fitter for these yet)
output.append("// =============================================================================")
output.append("// ERF FWD — ODD: erf(x) = sign(x)*h(|x|)")
output.append("// NOTE: Using FP16-derived coefficients (no BF16 fitter yet)")
output.append("// =============================================================================")
output.append("")

# Read the FP16 ERF coefficients from the existing struct file
# Since we don't have ERF in the fitter, we'll use the same coefficients
# as the mechanical conversion from FP16
# ERF is basically 2*sigmoid(sqrt(2)*x) - 1, approximated as tanh-like odd poly
# Let's just keep the original mechanically converted ERF structs

# Read original file and extract ERF/GELU sections
with open(os.path.join(os.path.dirname(__file__), "..", "spline_ops", "spline_structs_odd_bf16.cuh")) as f:
    original = f.read()

# Extract ERF structs
import re
for deg in [3, 4, 5, 6]:
    pattern = f"struct ERF_FWD_D{deg}_ODD_BF16.*?\\}};"
    match = re.search(pattern, original, re.DOTALL)
    if match:
        output.append(match.group(0))
        output.append("")

# -------  GELU FWD (ODD)  -------
output.append("// =============================================================================")
output.append("// GELU FWD — ODD: gelu(x) = 0.5*x*(1 + erf(x/sqrt(2)))")
output.append("// NOTE: Using FP16-derived coefficients (no BF16 fitter yet)")
output.append("// =============================================================================")
output.append("")

for deg in [3, 4, 5, 6]:
    pattern = f"struct GELU_FWD_D{deg}_ODD_BF16.*?\\}};"
    match = re.search(pattern, original, re.DOTALL)
    if match:
        output.append(match.group(0))
        output.append("")

# -------  GELU BWD (ODD)  -------
output.append("// =============================================================================")
output.append("// GELU BWD — ODD")
output.append("// NOTE: Using FP16-derived coefficients (no BF16 fitter yet)")
output.append("// =============================================================================")
output.append("")

for deg in [3, 4, 5, 6]:
    pattern = f"struct GELU_BWD_D{deg}_ODD_BF16.*?\\}};"
    match = re.search(pattern, original, re.DOTALL)
    if match:
        output.append(match.group(0))
        output.append("")


# -------  ALGEBRAIC BACKWARD  -------
output.append("// =============================================================================")
output.append("// ALGEBRAIC BACKWARD PASSES")
output.append("// =============================================================================")
output.append("")

# Sigmoid BWD Algebraic: y*(1-y) where y = sigmoid(x)
output.append("""struct SIGMOID_BWD_ALGEBRAIC_BF16 {
    static __device__ __forceinline__ __nv_bfloat162 eval_grad(__nv_bfloat162 x, __nv_bfloat162) {
        __nv_bfloat162 y = SIGMOID_FWD_D3_ODD_BF16::evaluate(x);
        __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
        return __hmul2(y, __hsub2(one, y));
    }
};""")
output.append("")

# Tanh BWD Algebraic: 1 - tanh(x)^2
output.append("""struct TANH_BWD_ALGEBRAIC_BF16 {
    static __device__ __forceinline__ __nv_bfloat162 eval_grad(__nv_bfloat162 x, __nv_bfloat162) {
        __nv_bfloat162 y = TANH_FWD_D3_ODD_BF16::evaluate(x);
        __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
        return __hsub2(one, __hmul2(y, y));
    }
};""")
output.append("")

# -------  BACKWARD COMPATIBILITY ALIASES  -------
output.append("// =============================================================================")
output.append("// BACKWARD COMPATIBILITY ALIASES")
output.append("// =============================================================================")
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

# Write output
out_path = os.path.join(os.path.dirname(__file__), "..", "spline_ops", "spline_structs_odd_bf16.cuh")
with open(out_path, 'w') as f:
    f.write("\n".join(output))

line_count = len(output)
print(f"Generated {out_path}: {line_count} lines")

#!/usr/bin/env python3
"""
Dual-Constrained Least Squares Polynomial Fitter

Fits polynomial P(x) on [0, L] with two endpoint constraints:
  P(0) = func(0)   (exact value at origin)
  P(L) = func(L)   (exact value at boundary)

This produces smooth polynomials that naturally decay to the correct
asymptotic values, enabling branchless CUDA kernels (no saturation mask needed).

Usage as library:
    from constrained_ls_fitter import dual_constrained_ls_fit, eval_poly
    from constrained_ls_fitter import sigmoid_grad, tanh_grad, swish_grad

Usage as script:
    python constrained_ls_fitter.py
"""

import numpy as np
import struct


# ============================================================================
# Target functions (activation gradients)
# ============================================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_grad(x):
    return 1 - np.tanh(x)**2

def swish_grad(x):
    s = sigmoid(x)
    return s * (1 + x * (1 - s))


# ============================================================================
# Core fitting
# ============================================================================

def dual_constrained_ls_fit(func, domain, degree, num_samples=2000):
    """
    Fit polynomial of given degree to func on domain [0, L].

    Constraints:
      1. P(0) = func(0)
      2. P(L) = func(L)
    Remaining coefficients chosen to minimize MSE via least squares.

    Method: Eliminate c0 and c1 by substitution, solve for c2..cn.
      P(x) = c0 + x*(func(L)-c0)/L + sum(c_i * (x^i - x*L^(i-1)), i=2..n)
    """
    t_min, t_max = domain
    L = t_max

    # Chebyshev nodes for better conditioning
    k = np.arange(num_samples)
    cheb = np.cos((2*k + 1) * np.pi / (2 * num_samples))
    x = 0.5 * (t_min + t_max) + 0.5 * (t_max - t_min) * cheb
    x = np.sort(x)
    y = func(x)

    # Endpoint values
    c0 = func(0.0)
    y_L = func(L)
    slope = (y_L - c0) / L

    # Degree 1: fully determined by two constraints
    if degree <= 1:
        return np.array([c0, slope])

    # Residual after subtracting the linear interpolant
    y_target = y - (c0 + x * slope)

    # Build design matrix for c2..cn
    # term_i(x) = x^i - x*L^(i-1)  [vanishes at x=0 and x=L]
    V = np.column_stack([x**i - x * L**(i - 1) for i in range(2, degree + 1)])

    # Solve least squares
    coeffs_higher, _, _, _ = np.linalg.lstsq(V, y_target, rcond=None)

    # Recover c1 = slope - sum(c_i * L^(i-1))
    c1 = slope - sum(c * L**(i + 1) for i, c in enumerate(coeffs_higher))

    return np.array([c0, c1] + list(coeffs_higher))


# ============================================================================
# Utility functions
# ============================================================================

def eval_poly(x, coeffs):
    """Evaluate polynomial with coefficients [c0, c1, ..., cn]."""
    return np.vander(x, len(coeffs), increasing=True) @ coeffs


def to_fp16_hex(coeffs):
    """Convert float coefficients to FP16 hex strings."""
    result = []
    for c in coeffs:
        fp16 = np.float16(c)
        h = struct.pack('<e', fp16).hex()
        result.append(f'0x{h[2:4]}{h[0:2]}'.upper())
    return result


def max_error(func, domain, coeffs, n_points=5000):
    """Compute max absolute error of polynomial fit on domain."""
    x = np.linspace(domain[0], domain[1], n_points)
    return np.max(np.abs(func(x) - eval_poly(x, coeffs)))


# ============================================================================
# Main: example usage
# ============================================================================

if __name__ == "__main__":
    configs = [
        ("Sigmoid_grad", sigmoid_grad, [5.5, 6.5, 7.0], [4, 5, 6]),
        ("Tanh_grad",    tanh_grad,    [2.7, 3.2, 3.5], [4, 5, 6]),
        ("Swish_grad",   swish_grad,   [5.0, 6.0],      [4, 5]),
    ]

    for name, func, L_values, degrees in configs:
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")

        for L in L_values:
            for deg in degrees:
                coeffs = dual_constrained_ls_fit(func, (0.0, L), deg)
                err = max_error(func, (0.0, L), coeffs)
                hex_codes = to_fp16_hex(coeffs)

                print(f"  D{deg} L={L:.1f}: MaxErr={err:.6f}  hex={hex_codes}")

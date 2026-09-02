#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified in 2026 for the standalone fast-polynomial-transcendentals release.

"""
Least Squares Polynomial Fitter - Alternative to Lagrange interpolation

Lagrange interpolation FORCES exact matches at nodes, causing oscillation.
Least squares minimizes OVERALL error without forcing exact matches.

For gradient functions that decay to 0, we can also add endpoint constraints.
"""

import numpy as np
from scipy.optimize import minimize
import struct


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh_grad(x):
    return 1 - np.tanh(x) ** 2


def swish_grad(x):
    s = sigmoid(x)
    return s * (1 + x * (1 - s))


class LeastSquaresPolynomialFitter:
    """
    Fits polynomial using least squares with optional endpoint constraints.

    Unlike Lagrange interpolation which forces exact node matches,
    this minimizes integrated error over the domain.
    """

    def __init__(self, func, domain, degree, num_samples=1000, weights=None):
        self.func = func
        self.domain = domain
        self.degree = degree
        self.num_samples = num_samples
        self.weights = weights
        self.coeffs = None

    def fit(self, endpoint_value=None, endpoint_weight=100.0):
        """
        Fit polynomial to function.

        Args:
            endpoint_value: If set, add soft constraint for f(domain[1]) = endpoint_value
            endpoint_weight: Weight for endpoint constraint
        """
        t_min, t_max = self.domain

        # Sample points - use Chebyshev distribution to reduce edge effects
        k = np.arange(self.num_samples)
        cheb_points = np.cos((2 * k + 1) * np.pi / (2 * self.num_samples))
        x = 0.5 * (t_min + t_max) + 0.5 * (t_max - t_min) * cheb_points
        x = np.sort(x)

        y = self.func(x)

        # Build Vandermonde matrix
        V = np.vander(x, self.degree + 1, increasing=True)

        # Weights for samples (optional)
        if self.weights is not None:
            W = np.diag(self.weights(x))
            V = W @ V
            y = W @ y

        # Solve least squares: minimize ||V @ c - y||^2
        if endpoint_value is not None:
            # Add soft constraint: p(t_max) = endpoint_value
            V_end = np.vander([t_max], self.degree + 1, increasing=True)
            V = np.vstack([V, endpoint_weight * V_end])
            y = np.append(y, endpoint_weight * endpoint_value)

        self.coeffs, residuals, rank, s = np.linalg.lstsq(V, y, rcond=None)
        return self.coeffs

    def evaluate(self, x):
        """Evaluate fitted polynomial at x."""
        V = np.vander(x, self.degree + 1, increasing=True)
        return V @ self.coeffs

    def get_fp16_hex(self):
        """Convert coefficients to FP16 hex format."""
        hex_codes = []
        for c in self.coeffs:
            fp16 = np.float16(c)
            h = struct.pack("<e", fp16).hex()
            hex_codes.append(f"0x{h[2:4]}{h[0:2]}".upper())
        return hex_codes

    def analyze(self, num_test=10000):
        """Compute fit statistics."""
        x = np.linspace(self.domain[0], self.domain[1], num_test)
        y_true = self.func(x)
        y_pred = self.evaluate(x)

        error = np.abs(y_true - y_pred)
        return {
            "max_error": np.max(error),
            "mean_error": np.mean(error),
            "rms_error": np.sqrt(np.mean(error**2)),
        }


def compare_methods():
    """Compare Least Squares vs Minimax (Lagrange) fitting."""

    configs = [
        ("Sigmoid gradient", sigmoid_grad, (0, 7), 5, 0.0),
        ("Tanh gradient", tanh_grad, (0, 4), 5, 0.0),
        ("Swish gradient", swish_grad, (0, 7.5), 5, None),  # asymptotes to 1, not 0
    ]

    print("=" * 80)
    print("LEAST SQUARES POLYNOMIAL FITS")
    print("=" * 80)

    for name, func, domain, degree, endpoint in configs:
        print(f"\n{name} on [{domain[0]}, {domain[1]}], degree {degree}")
        print("-" * 60)

        # 1. Standard least squares
        fitter_ls = LeastSquaresPolynomialFitter(func, domain, degree)
        fitter_ls.fit()
        stats_ls = fitter_ls.analyze()

        # 2. Least squares with endpoint constraint
        if endpoint is not None:
            fitter_constrained = LeastSquaresPolynomialFitter(func, domain, degree)
            fitter_constrained.fit(endpoint_value=endpoint, endpoint_weight=50.0)
            stats_constrained = fitter_constrained.analyze()

        # 3. Weighted least squares (lower weight at edges)
        def edge_weights(x):
            # Lower weight near edges
            t_min, t_max = domain
            center = (t_min + t_max) / 2
            halfwidth = (t_max - t_min) / 2
            normalized = (x - center) / halfwidth  # [-1, 1]
            return 1.0 - 0.5 * normalized**2  # Lower weight at edges

        fitter_weighted = LeastSquaresPolynomialFitter(
            func, domain, degree, weights=edge_weights
        )
        fitter_weighted.fit()
        stats_weighted = fitter_weighted.analyze()

        print(f"  Standard LS:    max_err={stats_ls['max_error']:.6f}")
        if endpoint is not None:
            print(f"  Endpoint const: max_err={stats_constrained['max_error']:.6f}")
        print(f"  Weighted LS:    max_err={stats_weighted['max_error']:.6f}")

        # Print coefficients for best method
        print(f"\n  Coefficients (standard LS):")
        print(f"    FP32: {fitter_ls.coeffs}")
        print(f"    FP16 hex: {fitter_ls.get_fp16_hex()}")

        # Evaluate at boundary to check smoothness
        t_max = domain[1]
        poly_val = fitter_ls.evaluate(np.array([t_max]))[0]
        true_val = func(t_max)
        print(f"\n  At boundary x={t_max}:")
        print(f"    Polynomial: {poly_val:.6f}")
        print(f"    True value: {true_val:.6f}")
        print(f"    Error: {abs(poly_val - true_val):.6f}")

    print("\n" + "=" * 80)
    print("Analysis complete!")


if __name__ == "__main__":
    compare_methods()

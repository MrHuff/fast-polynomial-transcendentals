# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified in 2026 for the standalone fast-polynomial-transcendentals release.

import numpy as np
import math
from polynomial_fitter import PolynomialFitter, PolynomialAnalyzer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    """Derivative of sigmoid: sigma'(x) = sigma(x) * (1 - sigma(x))"""
    s = sigmoid(x)
    return s * (1 - s)


def swish(x):
    return x * sigmoid(x)


def swish_grad(x):
    """Derivative of swish: swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    = sigmoid(x) * (1 + x * (1 - sigmoid(x)))"""
    s = sigmoid(x)
    return s * (1 + x * (1 - s))


def exp_func(x):
    return np.exp(x)


def exp_grad(x):
    """Derivative of exp: exp'(x) = exp(x)"""
    return np.exp(x)


def pow2_func(x):
    return np.power(2, x)


def pow2_grad(x):
    """Derivative of 2^x: (2^x)' = ln(2) * 2^x"""
    return np.log(2) * np.power(2, x)


def tanh_func(x):
    return np.tanh(x)


def tanh_grad(x):
    """Derivative of tanh: tanh'(x) = 1 - tanh(x)^2"""
    return 1 - np.tanh(x) ** 2


def analyze_fit(
    name, func, domain, degrees, intervals, weight_func=None, sampling="uniform"
):
    print(
        f"\n--- Analyzing {name} (N={intervals}, Deg={degrees}, sampling={sampling}) ---"
    )
    fitter = PolynomialFitter(
        func,
        domain=domain,
        degrees=degrees,
        num_intervals=intervals,
        weight_func=weight_func,
        sampling=sampling,
    )
    fitter.fit()

    clean_name = (
        name.replace(" ", "_").replace("[", "").replace("]", "").replace(",", "-")
    )
    suffix = ""
    if sampling != "uniform":
        suffix += f"_{sampling}"
    if weight_func is not None:
        suffix += "_weighted"

    # We can fetch analyzer stats directly or implement a method?
    # analyze_fit in previous code was calling PolynomialAnalyzer.
    # Let's assume PolynomialAnalyzer signature is compatible or check it.
    analyzer = PolynomialAnalyzer(
        fitter, f"{clean_name}_N{intervals}_D{degrees}{suffix}"
    )
    analyzer.analyze(save_dir="analysis_results")


# Weight functions for derivative fitting
def center_weight(x, sigma=1.0):
    """Gaussian-like weight that upweights the center (where derivatives peak)."""
    return np.exp(-0.5 * (x / sigma) ** 2)


def inv_tail_weight(x, eps=0.01):
    """Weight that suppresses flat tails: w(x) = 1 / (1 + |x|^2)."""
    return 1.0 / (1.0 + x * x)


def peak_weight(x, sigma=1.5, scale=10.0):
    """Weight function that emphasizes the center (x=0)."""
    return 1.0 + scale * np.exp(-0.5 * (x / sigma) ** 2)


def tail_weight(x, scale=2.0):
    """Weight function that emphasizes the tails (large |x|)."""
    return 1.0 + scale * np.abs(x)


if __name__ == "__main__":
    configs = [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (4, 1), (4, 2)]

    print("=" * 80)
    print("Running Analysis for Requested Configurations...")
    print("=" * 80)

    # All functions and their derivatives
    functions = [
        # (name, func, domain)
        ("Pow2 [0,1]", pow2_func, (0.0, 1.0)),
        ("Pow2_grad [0,1]", pow2_grad, (0.0, 1.0)),
        ("Exp [0,1]", exp_func, (0.0, 1.0)),
        ("Exp_grad [0,1]", exp_grad, (0.0, 1.0)),
        ("Sigmoid [-6,6]", sigmoid, (-6.0, 6.0)),
        ("Sigmoid_grad [-6,6]", sigmoid_grad, (-6.0, 6.0)),
        ("Swish [-6,6]", swish, (-6.0, 6.0)),
        ("Swish_grad [-6,6]", swish_grad, (-6.0, 6.0)),
        ("Swish [-5.5,5.5]", swish, (-5.5, 5.5)),
        ("Swish_grad [-5.5,5.5]", swish_grad, (-5.5, 5.5)),
        ("Tanh [-4, 4]", tanh_func, (-4.0, 4.0)),
        ("Tanh_grad [-4, 4]", tanh_grad, (-4.0, 4.0)),
        ("Tanh [-3, 3]", tanh_func, (-3.0, 3.0)),
        ("Tanh_grad [-3, 3]", tanh_grad, (-3.0, 3.0)),
        ("Tanh [-2.5, 2.5]", tanh_func, (-2.5, 2.5)),
        ("Tanh_grad [-2.5, 2.5]", tanh_grad, (-2.5, 2.5)),
    ]

    """
    for intervals, degrees in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: N={intervals}, Deg={degrees}")
        print(f"{'='*60}")
        for name, func, domain in functions:
            analyze_fit(name, func, domain, degrees=degrees, intervals=intervals)
    """

    # =========================================================================
    # IMPROVED DERIVATIVE FITTING
    # Compare: uniform vs chebyshev, unweighted vs weighted, narrower domains
    # =========================================================================
    print(f"\n{'='*80}")
    print("IMPROVED DERIVATIVE FITTING EXPERIMENTS")
    print(f"{'='*80}")

    # Test functions
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    tanh = np.tanh

    # Derivatives
    def sigmoid_grad(x):
        s = sigmoid(x)
        return s * (1 - s)

    def tanh_grad(x):
        t = tanh(x)
        return 1 - t**2

    def swish_grad(x):
        s = sigmoid(x)
        return s + x * s * (1 - s)

    improved_configs = []  # Initialize as empty list

    # (name, func, domain, intervals, degree, kwargs_dict)
    # Baseline - SKIPPING
    # ("Sigmoid_grad [-6,6]",   sigmoid_grad, (-6.0, 6.0), 2, 3, {}),
    # ... (skipping others) ...

    # =================================================================================
    # MEGA BENCHMARK: D3/D4/D5 x Uniform/Chebyshev/Weighted
    # =================================================================================

    # --- Base Specs ---
    # Functions to test:
    # 1. Sigmoid Symmetric [0, 6]
    # 2. Tanh Symmetric [0, 3] (Tanh has sharper knee, so [0,3] is standard)
    # 3. Swish Positive [0, 6]

    # Strategies:
    # A. Uniform: Simple equal spacing.
    # B. Chebyshev: Standard Chebyshev nodes (unweighted). "sampling": "chebyshev", "weight_func": None
    # C. Weighted Opt: Chebyshev + Peak Weighting.

    # helper to generate configs
    for metric_name, func, domain, strategies in [
        (
            "Sigmoid_grad_sym",
            sigmoid_grad,
            (0.0, 6.0),
            [
                ("uniform", {"sampling": "uniform", "weight_func": None}),
                ("chebyshev", {"sampling": "chebyshev", "weight_func": None}),
                (
                    "weighted_opt",
                    {
                        "sampling": "chebyshev",
                        "weight_func": lambda x: peak_weight(x, sigma=1.5, scale=10.0),
                    },
                ),
            ],
        ),
        (
            "Tanh_grad_sym",
            tanh_grad,
            (0.0, 3.0),
            [
                ("uniform", {"sampling": "uniform", "weight_func": None}),
                ("chebyshev", {"sampling": "chebyshev", "weight_func": None}),
                (
                    "weighted_opt",
                    {
                        "sampling": "chebyshev",
                        "weight_func": lambda x: peak_weight(x, sigma=1.2, scale=10.0),
                    },
                ),
            ],
        ),
        (
            "Swish_grad_pos",
            swish_grad,
            (0.0, 6.0),
            [
                ("uniform", {"sampling": "uniform", "weight_func": None}),
                ("chebyshev", {"sampling": "chebyshev", "weight_func": None}),
                (
                    "weighted_opt",
                    {
                        "sampling": "chebyshev",
                        "weight_func": lambda x: peak_weight(x, sigma=3.0, scale=10.0),
                    },
                ),
            ],
        ),
    ]:
        for deg in [3, 4, 5]:
            for strat_name, strat_kwargs in strategies:
                conf_name = f"{metric_name}_D{deg}_{strat_name}"
                improved_configs.append((conf_name, func, domain, 1, deg, strat_kwargs))

    # =================================================================================
    # GRID OPTIMIZATION (Domain Tuning)
    # =================================================================================
    # Does changing the domain bounds improve fit quality significantly?
    # Sigmoid is effectively 0 at 6... but maybe 5 is enough? or 7 needs fitting?

    # We test D4 Weighted Opt on different domains
    for d_bound in [4.0, 5.0, 6.0, 7.0, 8.0]:
        improved_configs.append(
            (
                f"Sigmoid_grad_sym_D4_Grid_{d_bound}",
                sigmoid_grad,
                (0.0, d_bound),
                1,
                4,
                {
                    "sampling": "chebyshev",
                    "weight_func": lambda x: peak_weight(x, sigma=1.5, scale=10.0),
                },
            )
        )

    # Tanh bounds
    for d_bound in [2.5, 3.0, 3.5, 4.0]:
        improved_configs.append(
            (
                f"Tanh_grad_sym_D4_Grid_{d_bound}",
                tanh_grad,
                (0.0, d_bound),
                1,
                4,
                {
                    "sampling": "chebyshev",
                    "weight_func": lambda x: peak_weight(x, sigma=1.2, scale=10.0),
                },
            )
        )

    # =================================================================================
    # DEEP DIVE: Minimax / Lagrange Optimized Node Fitting
    # =================================================================================
    # User Request: "just use lagrange polynomials where we select nodes such that the max error is minimized"
    # Enforcing boundary constraints (fix_boundaries=True).

    try:
        from evolution.minimax_fitter import MinimaxPolynomialFitter
    except ImportError:
        from minimax_fitter import MinimaxPolynomialFitter

    deep_dive_configs = [
        # (name, func, domain, intervals, degree, kwargs_dict)
        # Note: MinimaxFitter takes 'fix_boundaries' in init, not strictly kwargs of fit()
        # But our analyze_fit wrapper needs adaptation or we instantiate directly.
        (
            "Sigmoid_grad_sym_D3_Minimax",
            sigmoid_grad,
            (0.0, 6.0),
            1,
            3,
            {"fix_boundaries": True},
        ),
        (
            "Tanh_grad_sym_D3_Minimax",
            tanh_grad,
            (0.0, 3.0),
            1,
            3,
            {"fix_boundaries": True},
        ),
        (
            "Swish_grad_pos_D3_Minimax",
            swish_grad,
            (0.0, 6.0),
            1,
            3,
            {"fix_boundaries": True},
        ),
        (
            "Sigmoid_grad_sym_D4_Minimax",
            sigmoid_grad,
            (0.0, 6.0),
            1,
            4,
            {"fix_boundaries": True},
        ),
        (
            "Tanh_grad_sym_D4_Minimax",
            tanh_grad,
            (0.0, 3.0),
            1,
            4,
            {"fix_boundaries": True},
        ),
        (
            "Swish_grad_pos_D4_Minimax",
            swish_grad,
            (0.0, 6.0),
            1,
            4,
            {"fix_boundaries": True},
        ),
        (
            "Sigmoid_grad_sym_D5_Minimax",
            sigmoid_grad,
            (0.0, 6.0),
            1,
            5,
            {"fix_boundaries": True},
        ),
        (
            "Tanh_grad_sym_D5_Minimax",
            tanh_grad,
            (0.0, 3.0),
            1,
            5,
            {"fix_boundaries": True},
        ),
        (
            "Swish_grad_pos_D5_Minimax",
            swish_grad,
            (0.0, 6.0),
            1,
            5,
            {"fix_boundaries": True},
        ),
        (
            "Sigmoid_grad_sym_D3_Minimax_extended_range",
            sigmoid_grad,
            (0.0, 7.0),
            1,
            3,
            {"fix_boundaries": True},
        ),
        (
            "Tanh_grad_sym_D3_Minimax_extended_range",
            tanh_grad,
            (0.0, 4.0),
            1,
            3,
            {"fix_boundaries": True},
        ),
        (
            "Swish_grad_pos_D3_Minimax_extended_range",
            swish_grad,
            (0.0, 7.0),
            1,
            3,
            {"fix_boundaries": True},
        ),
        (
            "Sigmoid_grad_sym_D4_Minimax_extended_range",
            sigmoid_grad,
            (0.0, 7.0),
            1,
            4,
            {"fix_boundaries": True},
        ),
        (
            "Tanh_grad_sym_D4_Minimax_extended_range",
            tanh_grad,
            (0.0, 4.0),
            1,
            4,
            {"fix_boundaries": True},
        ),
        (
            "Swish_grad_pos_D4_Minimax_extended_range",
            swish_grad,
            (0.0, 7.0),
            1,
            4,
            {"fix_boundaries": True},
        ),
        (
            "Sigmoid_grad_sym_D5_Minimax_extended_range",
            sigmoid_grad,
            (0.0, 7.0),
            1,
            5,
            {"fix_boundaries": True},
        ),
        (
            "Tanh_grad_sym_D5_Minimax_extended_range",
            tanh_grad,
            (0.0, 4.0),
            1,
            5,
            {"fix_boundaries": True},
        ),
        (
            "Swish_grad_pos_D5_Minimax_extended_range",
            swish_grad,
            (0.0, 7.5),
            1,
            5,
            {"fix_boundaries": True},
        ),
        # Ultra-extended range fits - wider domains for better boundary behavior
        (
            "Sigmoid_grad_sym_D5_Minimax_ultra_extended",
            sigmoid_grad,
            (0.0, 10.0),
            1,
            5,
            {"fix_boundaries": True},
        ),
        (
            "Tanh_grad_sym_D5_Minimax_ultra_extended",
            tanh_grad,
            (0.0, 5.0),
            1,
            5,
            {"fix_boundaries": True},
        ),
        (
            "Swish_grad_pos_D5_Minimax_ultra_extended",
            swish_grad,
            (0.0, 10.0),
            1,
            5,
            {"fix_boundaries": True},
        ),
    ]

    for name, func, domain, intervals, degree, kwargs in deep_dive_configs:
        print(f"\n--- Analyzing {name} (Minimax Optimized) ---")
        # Instantiate MinimaxFitter directly
        fitter = MinimaxPolynomialFitter(
            func, domain=domain, degrees=degree, num_intervals=intervals, **kwargs
        )
        fitter.fit()
        analyzer = PolynomialAnalyzer(fitter, name)
        analyzer.analyze(save_dir="analysis_results/deep_dive")

    # =================================================================================
    # LEAST SQUARES FITTING (Better than Lagrange - no oscillation)
    # =================================================================================
    # LS minimizes overall error without forcing exact node matches.
    # This avoids the oscillation issues seen with Lagrange/Minimax interpolation.

    try:
        from evolution.least_squares_fitter import LeastSquaresPolynomialFitter
    except ImportError:
        from least_squares_fitter import LeastSquaresPolynomialFitter

    ls_backward_configs = [
        # (name, func, domain, degree)
        # Sigmoid gradient - symmetric, so fit on [0, domain] and use |x|
        ("Sigmoid_grad_D3_LS", sigmoid_grad, (0.0, 7.0), 3),
        ("Sigmoid_grad_D4_LS", sigmoid_grad, (0.0, 7.0), 4),
        ("Sigmoid_grad_D5_LS", sigmoid_grad, (0.0, 7.0), 5),
        # Tanh gradient - symmetric
        ("Tanh_grad_D3_LS", tanh_grad, (0.0, 4.0), 3),
        ("Tanh_grad_D4_LS", tanh_grad, (0.0, 4.0), 4),
        ("Tanh_grad_D5_LS", tanh_grad, (0.0, 4.0), 5),
        # Swish gradient - asymmetric (need to handle differently)
        ("Swish_grad_D3_LS", swish_grad, (0.0, 7.5), 3),
        ("Swish_grad_D4_LS", swish_grad, (0.0, 7.5), 4),
        ("Swish_grad_D5_LS", swish_grad, (0.0, 7.5), 5),
    ]

    for name, func, domain, degree in ls_backward_configs:
        print(f"\n--- Analyzing {name} (Least Squares) ---")
        fitter = LeastSquaresPolynomialFitter(func, domain, degree)
        fitter.fit()
        stats = fitter.analyze()

        # Save results in same format as other fitters
        import json
        import struct

        result = {
            "name": name,
            "domain": list(domain),
            "degree": degree,
            "method": "least_squares",
            "coeffs": {
                "fp32": {
                    "values": [list(fitter.coeffs)],
                },
                "fp16": {
                    "hex": [fitter.get_fp16_hex()],
                },
            },
            "metrics": {"fp32": stats},
        }

        os.makedirs("analysis_results/least_squares", exist_ok=True)
        with open(f"analysis_results/least_squares/{name}_stats.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: analysis_results/least_squares/{name}_stats.json")
        print(f"  Max error: {stats['max_error']:.6f}")
        print(f"  FP16 hex: {fitter.get_fp16_hex()}")

    # =================================================================================
    # FORWARD PASS MINIMAX (User Request: "Check D=2, D=3, D=4")
    # =================================================================================
    # Functions: Sigmoid [0,6], Tanh [0,3], Swish [0,6]
    # Check if D=2 is enough.

    forward_pass_configs = []
    # Sigmoid Forward
    for deg in [2, 3, 4]:
        forward_pass_configs.append(
            (
                f"Sigmoid_fwd_sym_D{deg}_Minimax",
                sigmoid,
                (0.0, 6.0),
                1,
                deg,
                {"fix_boundaries": True},
            )
        )

    # Tanh Forward
    for deg in [2, 3, 4]:
        forward_pass_configs.append(
            (
                f"Tanh_fwd_sym_D{deg}_Minimax",
                tanh,
                (0.0, 3.0),
                1,
                deg,
                {"fix_boundaries": True},
            )
        )

    # Swish Forward
    # Swish is x * sigmoid(x). [0, 6]
    for deg in [2, 3, 4]:
        forward_pass_configs.append(
            (
                f"Swish_fwd_pos_D{deg}_Minimax",
                swish,
                (0.0, 6.0),
                1,
                deg,
                {"fix_boundaries": True},
            )
        )

    for name, func, domain, intervals, degree, kwargs in forward_pass_configs:
        print(f"\n--- Analyzing {name} (Minimax Optimized) ---")
        try:
            fitter = MinimaxPolynomialFitter(
                func, domain=domain, degrees=degree, num_intervals=intervals, **kwargs
            )
            fitter.fit()
            analyzer = PolynomialAnalyzer(fitter, name)
            analyzer.analyze(save_dir="analysis_results/forward_pass")
        except Exception as e:
            print(f"Error fitting {name}: {e}")

# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified in 2026 for the standalone fast-polynomial-transcendentals release.

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class MinimaxPolynomialFitter:
    """
    Fits piecewise polynomials using Lagrange interpolation with optimized node placement
    to minimize the L-infinity (max absolute) error.

    This is effectively a numerical implementation of the Remez-like exchange algorithm logic,
    but using generic optimization to handle arbitrary constraints (like fixed boundaries).
    """

    def __init__(
        self,
        func,
        domain=(0.0, 1.0),
        degrees=None,
        num_intervals=1,
        fix_boundaries=True,
        method="nelder-mead",
    ):
        """
        Args:
            func: Callable python function to fit
            domain: Tuple (min, max)
            degrees: Int or separate degrees per interval
            num_intervals: Number of intervals
            fix_boundaries: If True, forces the polynomial to exactly interpolate
                          the interval endpoints (guarantees C0 continuity).
            method: 'nelder-mead', 'simulated_annealing', 'brute_force'
        """
        self.func = func
        self.domain = domain
        self.num_intervals = num_intervals
        self.fix_boundaries = fix_boundaries
        self.method = method

        if isinstance(degrees, int):
            self.degrees = [degrees] * num_intervals
        else:
            self.degrees = degrees

        self.coeffs = []  # stored as [c0, c1, ... cn] (low to high)
        self.knot_points = np.linspace(domain[0], domain[1], num_intervals + 1)

    def fit(self, num_dense_samples=10000):
        self.coeffs = []

        from scipy.optimize import dual_annealing, brute

        for i in range(self.num_intervals):
            t_min = self.knot_points[i]
            t_max = self.knot_points[i + 1]
            deg = self.degrees[i]

            # We need deg+1 nodes for unique interpolation of degree deg
            num_nodes = deg + 1

            # Initial guess: Chebyshev nodes mapped to [t_min, t_max]
            k = np.arange(num_nodes)
            cheb_nodes = np.cos((2 * k + 1) * np.pi / (2 * num_nodes))  # [-1, 1]
            cheb_nodes = 0.5 * (t_min + t_max) + 0.5 * (t_max - t_min) * cheb_nodes
            cheb_nodes = np.sort(cheb_nodes)

            if self.fix_boundaries:
                # We fix 0 and N-1. Optimizing indices 1..N-2
                if deg < 1:
                    nodes = np.array([0.5 * (t_min + t_max)])
                    free_indices = []
                else:
                    # Gauss-Lobatto for initial guess usually good
                    k_lob = np.arange(num_nodes)
                    lob_nodes = np.cos(k_lob * np.pi / (num_nodes - 1))
                    lob_nodes = (
                        0.5 * (t_min + t_max) + 0.5 * (t_max - t_min) * lob_nodes
                    )
                    nodes = np.sort(lob_nodes)

                    free_indices = list(range(1, num_nodes - 1))
            else:
                nodes = cheb_nodes
                free_indices = list(range(num_nodes))

            # Dense grid for error evaluation
            dense_grid = np.linspace(t_min, t_max, num_dense_samples)
            dense_y = self.func(dense_grid)

            if len(free_indices) > 0:
                initial_params = nodes[free_indices]

                # Wrapper for objective
                def objective_wrapper(free_params):
                    # 1. Bounds check (Interval) - handled by optimizer bounds usually, but good for safety
                    if np.any(free_params <= t_min) or np.any(free_params >= t_max):
                        return 1e9

                    # 2. Construct full node set
                    current_nodes = nodes.copy()
                    current_nodes[free_indices] = free_params

                    # 3. Order check (Sorted)
                    if np.any(np.diff(current_nodes) <= 1e-7):
                        return 1e9

                    # 4. Compute Max Error
                    try:
                        y_nodes = self.func(current_nodes)
                        V = np.vander(current_nodes, deg + 1, increasing=True)
                        c = np.linalg.solve(V, y_nodes)

                        # Evaluate on dense grid
                        V_dense = np.vander(dense_grid, deg + 1, increasing=True)
                        y_pred = V_dense @ c

                        return np.max(np.abs(dense_y - y_pred))
                    except np.linalg.LinAlgError:
                        return 1e9

                # Define Bounds for all free params
                bounds = [(t_min + 1e-5, t_max - 1e-5) for _ in free_indices]

                if self.method == "simulated_annealing":
                    # Dual Annealing (combine SA with local search)
                    # We need to ensure parameters are sorted though?
                    # Sim Annealing might pick anything.
                    # The objective function penalizes non-sorted, but search space is n-dimensional cube.
                    # Only a slice of the cube (x1 < x2 < x3) is valid.
                    # Let's trust the penalty or re-sort parameters inside objective?
                    # Re-sorting inside objective makes the mapping params->nodes ambiguous for optimizer?
                    # No, if we sort inside, the optimizer sees a symmetric function?
                    # Actually, if we just optimization P1, P2, P3... we can interpret them as the sorted nodes.
                    # So let's sort free_params inside objective! This removes the 'ordering constraint' as a hard barrier
                    # and makes the whole hypercube valid.

                    def objective_sorted(free_params):
                        sorted_params = np.sort(free_params)
                        # Explicitly check bounds again after sort?
                        # If input is in bounds, sorted is in bounds.
                        return objective_wrapper(sorted_params)

                    res = dual_annealing(objective_sorted, bounds, maxiter=5000)
                    best_internal = np.sort(res.x)

                elif self.method == "brute_force":
                    # Scipy brute
                    # Might be slow if Ns=40
                    # For D4 (3 free nodes), Ns=20 => 8000 evals. Feasible.

                    def objective_sorted_brute(free_params):
                        sorted_params = np.sort(free_params)
                        return objective_wrapper(sorted_params)

                    res = brute(
                        objective_sorted_brute,
                        bounds,
                        Ns=20,
                        full_output=True,
                        finish=None,
                    )
                    best_internal = np.sort(res[0])

                else:  # Default Nelder-Mead
                    res = minimize(
                        objective_wrapper,
                        initial_params,
                        method="Nelder-Mead",
                        tol=1e-5,
                        options={"maxiter": 5000},
                    )
                    best_internal = res.x

                nodes[free_indices] = best_internal

            # Final Coefficients
            y_nodes = self.func(nodes)
            V = np.vander(nodes, deg + 1, increasing=True)
            coeffs = np.linalg.solve(V, y_nodes)
            self.coeffs.append(coeffs)

        return self.coeffs

    # Helper method compatible with PolynomialFitter
    def generate_c_code(self, struct_name="POLY_FIT"):
        # Borrow from PolynomialFitter logic?
        # Reuse logic from string matching or just replicate
        max_deg = max(self.degrees)
        code = f"struct {struct_name} {{\n"
        code += "    static __device__ __forceinline__ float get_coeff(int interval_idx, int degree_idx) {\n"
        for d in range(max_deg + 1):
            code += f"        if (degree_idx == {d}) {{\n"
            code += "            switch(interval_idx) {\n"
            for i in range(self.num_intervals):
                if d <= self.degrees[i]:
                    val = self.coeffs[i][d]
                else:
                    val = 0.0
                code += f"                case {i}: return {val:.8f}f;\n"
            code += "                default: return 0.0f;\n"
            code += "            }\n"
            code += "        }\n"
        code += "        return 0.0f;\n"
        code += "    }\n\n"
        code += "    static __device__ __forceinline__ float eval(float x, int interval_idx) {\n"
        code += "        float res = 0.0f;\n"
        code += f"        #pragma unroll\n"
        code += f"        for (int d = {max_deg}; d >= 0; --d) {{\n"
        code += f"            float c = get_coeff(interval_idx, d);\n"
        code += f"            res = fmaf(x, res, c);\n"
        code += "        }\n"
        code += "        return res;\n"
        code += "    }\n"
        code += "};\n"
        return code

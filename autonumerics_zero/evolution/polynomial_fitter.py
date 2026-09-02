# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified in 2026 for the standalone fast-polynomial-transcendentals release.

import numpy as np
import inspect


class PolynomialFitter:
    """
    Fits piecewise polynomials to a given function with optional C0 continuity constraints.
    """

    def __init__(
        self,
        func,
        domain=(0.0, 1.0),
        degrees=None,
        num_intervals=1,
        continuity_c0=True,
        weight_func=None,
        sampling="uniform",
    ):
        """
        Args:
            func: Callable python function to fit (e.g. np.exp, or lambda x: ...)
            domain: Tuple (min, max)
            degrees: Int (uniform degree) or List of ints (per interval)
            num_intervals: Number of uniform intervals
            continuity_c0: Whether to enforce continuity at boundaries
            weight_func: Optional callable w(x) -> float for Weighted Least Squares.
            sampling: 'uniform' or 'chebyshev' for node distribution.
        """
        self.func = func
        self.domain = domain
        self.num_intervals = num_intervals
        self.continuity_c0 = continuity_c0
        self.weight_func = weight_func
        self.sampling = sampling

        if isinstance(degrees, int):
            self.degrees = [degrees] * num_intervals
        else:
            self.degrees = degrees
            assert len(self.degrees) == num_intervals

        self.coeffs = []  # List of lists: [[c_deg, ..., c0], ...]
        self.knot_points = np.linspace(domain[0], domain[1], num_intervals + 1)

    def fit(self, num_samples_per_interval=10000):
        # Total coefficients
        total_coeffs = sum([d + 1 for d in self.degrees])

        # Build A matrix (Least Squares Data Fitting) and b vector
        # Block diagonal structure for independent fitting, but we will merge into one big matrix
        # to apply global constraints.

        # We need enough samples to form a valid system.

        data_rows = []
        data_rhs = []

        # Helper to map (interval_idx, coeff_idx) -> global_col_idx
        def get_col_idx(interval_idx, coeff_degree_idx):
            offset = 0
            for i in range(interval_idx):
                offset += self.degrees[i] + 1
            # Coefficients are stored [c_deg, c_deg-1, ..., c0]
            # Vandermonde usually is [x^n, x^n-1... 1] or [1, x, ... x^n]
            # We will generate [x^deg, x^deg-1, ..., 1] to match C++ (Horner-ish usually prefers High->Low storage or Low->High?)
            # C++ code: get_coeff(idx, degree_idx).
            # If degree_idx==0 (C0), degree_idx==1 (C1*x).
            # So let's store as [C0, C1, C2...].
            # Then Vandermonde row is [1, x, x^2...]
            return offset + coeff_degree_idx

        # 1. Build Least Squares Equations
        current_row = 0
        A_blocks = []
        b_blocks = []

        all_x = []
        all_y = []
        all_w = []  # Weights

        for i in range(self.num_intervals):
            t_min = self.knot_points[i]
            t_max = self.knot_points[i + 1]
            deg = self.degrees[i]

            # Sample points
            if self.sampling == "chebyshev":
                # Chebyshev nodes on [t_min, t_max]
                # x_k = 0.5 * (a + b) + 0.5 * (b - a) * cos((2k+1)/(2n) * pi)
                # We need num_samples_per_interval nodes
                k = np.arange(num_samples_per_interval)
                nodes = np.cos((2 * k + 1) * np.pi / (2 * num_samples_per_interval))
                x_samples = 0.5 * (t_min + t_max) + 0.5 * (t_max - t_min) * nodes
                # Sort for plotting/consistency
                x_samples = np.sort(x_samples)
            else:
                x_samples = np.linspace(t_min, t_max, num_samples_per_interval)

            y_samples = self.func(x_samples)

            if self.weight_func:
                w_samples = self.weight_func(x_samples)
            else:
                w_samples = np.ones_like(x_samples)

            all_x.append(x_samples)
            all_y.append(y_samples)
            all_w.append(w_samples)

        # Global Least Squares Matrix
        # Number of samples total
        total_samples = self.num_intervals * num_samples_per_interval

        A_ls = np.zeros((total_samples, total_coeffs))
        b_ls = np.zeros(total_samples)
        # Weight matrix (diagonal), let's just scale rows of A and b by sqrt(w)
        # This is equivalent to min || W^0.5 (Ax - b) ||^2

        row_offset = 0
        for i in range(self.num_intervals):
            x_vals = all_x[i]
            y_vals = all_y[i]
            w_vals = all_w[i]
            deg = self.degrees[i]

            # Fill A matrix for this block
            # Powers: 0 to deg
            # Matrix shape: (samples, deg+1)
            # V[k, p] = x_vals[k] ^ p
            V = np.vander(x_vals, deg + 1, increasing=True)

            # Apply Weights
            # W is diagonal matrix of weights. sqrt(W) * A
            sqrt_w = np.sqrt(w_vals).reshape(-1, 1)
            V_weighted = V * sqrt_w
            y_weighted = y_vals * sqrt_w.flatten()

            # Place V into global A_ls
            col_start = get_col_idx(i, 0)
            col_end = col_start + deg + 1

            A_ls[
                row_offset : row_offset + num_samples_per_interval, col_start:col_end
            ] = V_weighted
            b_ls[row_offset : row_offset + num_samples_per_interval] = y_weighted

            row_offset += num_samples_per_interval

        # 2. Build Constraint Matrix (Continuity)
        # For each internal knot (i=1 to N-1), Poly_i(knot) == Poly_{i-1}(knot)
        num_internal_knots = self.num_intervals - 1
        if self.continuity_c0 and num_internal_knots > 0:
            C_matrix = np.zeros((num_internal_knots, total_coeffs))
            d_vector = np.zeros(num_internal_knots)  # = 0

            for k in range(num_internal_knots):
                knot_x = self.knot_points[k + 1]

                # Left Polynomial (Interval k) at knot_x
                left_idx = k
                left_deg = self.degrees[left_idx]

                # Right Polynomial (Interval k+1) at knot_x
                right_idx = k + 1
                right_deg = self.degrees[right_idx]

                # Constraint: Sum(C_left * x^p) - Sum(C_right * x^p) = 0

                # Left Coeffs (+1)
                col_start_left = get_col_idx(left_idx, 0)
                for p in range(left_deg + 1):
                    C_matrix[k, col_start_left + p] = knot_x**p

                # Right Coeffs (-1)
                col_start_right = get_col_idx(right_idx, 0)
                for p in range(right_deg + 1):
                    C_matrix[k, col_start_right + p] = -(knot_x**p)

            # Solve Constrained Least Squares
            # Minimize ||Ax - b||^2 s.t. Cx = d
            # Using Lagrange Multipliers:
            # [[2A.T A,   C.T]    [[x]    [[2A.T b]
            #  [C,        0  ]] *  [lambda] = [d]

            # Actually simpler KKT form:
            # [[A.T A, C.T], [C, 0]] [x, lambda]^T = [A.T b, d]^T

            ATA = A_ls.T @ A_ls
            ATb = A_ls.T @ b_ls

            KKT_top = np.hstack([ATA, C_matrix.T])
            KKT_bot = np.hstack(
                [C_matrix, np.zeros((num_internal_knots, num_internal_knots))]
            )
            KKT = np.vstack([KKT_top, KKT_bot])

            rhs = np.concatenate([ATb, d_vector])

            # Solve
            sol = np.linalg.solve(KKT, rhs)
            x_sol = sol[:total_coeffs]

        else:
            # Unconstrained
            x_sol, _, _, _ = np.linalg.lstsq(A_ls, b_ls, rcond=None)

        # Parse coefficients back
        self.coeffs = []
        curr_idx = 0
        for d in self.degrees:
            # We stored as [C0, C1, C2...] (Increasing power)
            # But usually we display/store for Horner as we fit?
            # Let's keep them as [C0, C1...] here, but generate_c_code can handle ordering.
            seg_coeffs = x_sol[curr_idx : curr_idx + d + 1]
            self.coeffs.append(seg_coeffs)
            curr_idx += d + 1

        return self.coeffs

    def generate_c_code(self, struct_name="POLY_FIT"):
        code = f"struct {struct_name} {{\n"

        # 1. get_coeff
        code += "    static __device__ __forceinline__ float get_coeff(int interval_idx, int degree_idx) {\n"

        # Max degree
        max_deg = max(self.degrees)
        for d in range(max_deg + 1):
            code += f"        if (degree_idx == {d}) {{\n"
            code += "            switch(interval_idx) {\n"
            for i in range(self.num_intervals):
                # Check if this interval has this degree
                if d <= self.degrees[i]:
                    val = self.coeffs[i][d]  # Stored as C0, C1... so index match degree
                else:
                    val = 0.0
                code += f"                case {i}: return {val:.8f}f;\n"
            code += "                default: return 0.0f;\n"
            code += "            }\n"
            code += "        }\n"

        code += "        return 0.0f;\n"
        code += "    }\n\n"

        # 2. eval_poly (Generic)
        code += "    static __device__ __forceinline__ float eval(float x, int interval_idx) {\n"
        code += "        float res = 0.0f;\n"
        # Horner's method: C_n * x^n + ... -> (((C_n * x) + C_{n-1}) * x + ...)
        # Loop HIGH to LOW
        code += f"        #pragma unroll\n"
        code += f"        for (int d = {max_deg}; d >= 0; --d) {{\n"
        code += f"            float c = get_coeff(interval_idx, d);\n"
        code += f"            // fma(a, b, c) = a*b + c\n"
        code += f"            res = fmaf(x, res, c);\n"  # x * res + c
        code += "        }\n"
        code += "        return res;\n"
        code += "    }\n"
        code += "};\n"
        return code


# -----------------------------------------------------------------------------
# Analysis & Visualization Extensions
# -----------------------------------------------------------------------------

import torch
import matplotlib.pyplot as plt
import os
import json
import struct


def float_to_hex(f):
    """Converts a python float (FP32) to its IEEE 754 hex representation."""
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def quantize_to_fp16(val_f32):
    """Quantizes a float to FP16 and returns (float_val, hex_str)."""
    t = torch.tensor([val_f32], dtype=torch.float32)
    t_fp16 = t.to(torch.float16)
    val_fp16 = (
        t_fp16.item()
    )  # This is the float value as python sees it (promoted back to f32 usually but rounded)

    # Bit representation:
    # Cast to int16 (reinterpret)
    # torch doesn't have direct bit_cast for half in older versions, doing via numpy/struct
    # Actually simpler: t_fp16.view(torch.int16).item() works if t_fp16 is on CPU
    bits = t_fp16.view(torch.int16).item() & 0xFFFF
    return val_fp16, f"0x{bits:04X}"


def quantize_to_bf16(val_f32):
    """Quantizes a float to BF16 and returns (float_val, hex_str)."""
    t = torch.tensor([val_f32], dtype=torch.float32)
    t_bf16 = t.to(torch.bfloat16)
    val_bf16 = t_bf16.item()

    # Bit representation
    bits = t_bf16.view(torch.int16).item() & 0xFFFF
    return val_bf16, f"0x{bits:04X}"


def compute_ulp_error(true_y, pred_y):
    """Computes the error in Units in Last Place (ULP) for FP32."""
    # ULP(x) is machine epsilon * 2^exponent
    # Approx: |true - pred| / spacing(true)
    # Numpy spacing(x) returns distance to next representable float
    spacing = np.spacing(true_y)
    # Handle 0 spacing (if true_y is subnormal or 0)
    spacing[spacing == 0] = np.finfo(float).tiny
    ulps = np.abs(true_y - pred_y) / spacing
    return np.mean(ulps), np.max(ulps)


class PolynomialAnalyzer:
    def __init__(self, fitter, func_name):
        self.fitter = fitter
        self.func_name = func_name
        self.coeffs = fitter.coeffs
        self.coeffs_flat = [c for interval in self.coeffs for c in interval]

    def analyze(self, save_dir="analysis_results"):
        os.makedirs(save_dir, exist_ok=True)

        # 1. Quantize Coefficients
        stats = {
            "name": self.func_name,
            "domain": self.fitter.domain,
            "intervals": self.fitter.num_intervals,
            "degrees": self.fitter.degrees,
            "coeffs": {},
        }

        # FP32
        stats["coeffs"]["fp32"] = {"values": [], "hex": []}
        for interval_coeffs in self.coeffs:
            curr_vals = []
            curr_hex = []
            for c in interval_coeffs:
                curr_vals.append(float(c))
                curr_hex.append(float_to_hex(c))
            stats["coeffs"]["fp32"]["values"].append(curr_vals)
            stats["coeffs"]["fp32"]["hex"].append(curr_hex)

        # FP16
        stats["coeffs"]["fp16"] = {"values": [], "hex": []}
        coeffs_fp16 = []  # For evaluation
        for interval_coeffs in self.coeffs:
            curr_vals = []
            curr_hex = []
            row_fp16 = []
            for c in interval_coeffs:
                val, h = quantize_to_fp16(c)
                curr_vals.append(val)
                curr_hex.append(h)
                row_fp16.append(val)
            stats["coeffs"]["fp16"]["values"].append(curr_vals)
            stats["coeffs"]["fp16"]["hex"].append(curr_hex)
            coeffs_fp16.append(row_fp16)

        # BF16
        stats["coeffs"]["bf16"] = {"values": [], "hex": []}
        coeffs_bf16 = []
        for interval_coeffs in self.coeffs:
            curr_vals = []
            curr_hex = []
            row_bf16 = []
            for c in interval_coeffs:
                val, h = quantize_to_bf16(c)
                curr_vals.append(val)
                curr_hex.append(h)
                row_bf16.append(val)
            stats["coeffs"]["bf16"]["values"].append(curr_vals)
            stats["coeffs"]["bf16"]["hex"].append(curr_hex)
            coeffs_bf16.append(row_bf16)

        # 2. Evaluate & Metrics
        test_x = np.linspace(self.fitter.domain[0], self.fitter.domain[1], 100000)
        true_y = self.fitter.func(test_x)

        results = {}
        for prec, c_list in [
            ("fp32", self.coeffs),
            ("fp16", coeffs_fp16),
            ("bf16", coeffs_bf16),
        ]:
            pred_y = self._evaluate_spline(test_x, c_list)

            # MSE
            mse = np.mean((true_y - pred_y) ** 2)

            # Rel Error (Handle small values)
            safe_true = np.where(np.abs(true_y) < 1e-7, 1e-7, true_y)
            rel_err = np.abs((true_y - pred_y) / safe_true)
            max_rel = np.max(rel_err)
            avg_rel = np.mean(rel_err)

            # Max Abs
            abs_err = np.abs(true_y - pred_y)
            max_abs = np.max(abs_err)

            # ULP
            mean_ulp, max_ulp = compute_ulp_error(true_y, pred_y)

            results[prec] = {
                "mse": float(mse),
                "max_abs_error": float(max_abs),
                "max_rel_error": float(max_rel),
                "avg_rel_error": float(avg_rel),
                "mean_ulp": float(mean_ulp),
                "max_ulp": float(max_ulp),
            }

            # Save for plotting
            stats[f"pred_{prec}"] = (
                pred_y.tolist()
            )  # Might be large, maybe don't save to JSON?
            # Actually, let's keep JSON specificstats and save plot separately.

        stats["metrics"] = results

        # Save JSON
        json_path = os.path.join(save_dir, f"{self.func_name}_stats.json")
        # Remove large arrays before saving json
        save_stats = {k: v for k, v in stats.items() if not k.startswith("pred_")}
        with open(json_path, "w") as f:
            json.dump(save_stats, f, indent=2)

        print(f"Saved stats to {json_path}")

        # 3. Plotting
        plt.figure(figsize=(12, 10))

        # Top: Function
        plt.subplot(2, 1, 1)
        plt.plot(test_x, true_y, "k-", label="Ground Truth", linewidth=2, alpha=0.5)
        plt.plot(
            test_x,
            self._evaluate_spline(test_x, self.coeffs),
            "b--",
            label=f'FP32 Fit (MaxErr={results["fp32"]["max_abs_error"]:.2e})',
        )
        plt.plot(
            test_x,
            self._evaluate_spline(test_x, coeffs_fp16),
            "r:",
            label=f'FP16 Coeffs (MaxErr={results["fp16"]["max_abs_error"]:.2e})',
        )
        plt.plot(
            test_x,
            self._evaluate_spline(test_x, coeffs_bf16),
            "g:",
            label=f'BF16 Coeffs (MaxErr={results["bf16"]["max_abs_error"]:.2e})',
        )
        plt.title(
            f"Fit: {self.func_name} (N={self.fitter.num_intervals}, Deg={self.fitter.degrees[0]})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Bottom: Error (Log Scale)
        plt.subplot(2, 1, 2)
        plt.semilogy(
            test_x,
            np.abs(true_y - self._evaluate_spline(test_x, self.coeffs)),
            "b-",
            label="FP32 Error",
            alpha=0.7,
        )
        plt.semilogy(
            test_x,
            np.abs(true_y - self._evaluate_spline(test_x, coeffs_fp16)),
            "r-",
            label="FP16 Error",
            alpha=0.7,
        )
        plt.semilogy(
            test_x,
            np.abs(true_y - self._evaluate_spline(test_x, coeffs_bf16)),
            "g-",
            label="BF16 Error",
            alpha=0.7,
        )
        plt.ylabel("Absolute Error (Log)")
        plt.xlabel("Input")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = os.path.join(save_dir, f"{self.func_name}_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")

    def _evaluate_spline(self, x_arr, coeffs):
        y_pred = np.zeros_like(x_arr)
        # Assuming uniform intervals for now as per Fitter
        domain = self.fitter.domain
        num_intervals = self.fitter.num_intervals

        # Vectorized evaluation? Hard with varying degrees strictly speaking
        # But our fitter usually has uniform degrees
        # Doing element-wise for correctness matching C++ logic exactly

        # Optimized for python speed slightly:
        interval_len = (domain[1] - domain[0]) / num_intervals

        # Vectorize interval finding
        indices = np.floor((x_arr - domain[0]) / interval_len).astype(int)
        indices = np.clip(indices, 0, num_intervals - 1)

        # Evaluate
        # Group by interval to vectorize poly eval
        for i in range(num_intervals):
            mask = indices == i
            if not np.any(mask):
                continue

            x_local = x_arr[mask]
            c_list = coeffs[i]

            # Horner
            val = np.zeros_like(x_local)
            # stored C0, C1... (Low to High power)
            # Horner expects High to Low: (...(Cn*x + Cn-1)*x...)
            for d in range(len(c_list) - 1, -1, -1):
                val = val * x_local + c_list[d]

            y_pred[mask] = val

        return y_pred


# Example Usage Test
if __name__ == "__main__":
    # Fit exp(x) on [0, 1] with 2 intervals, degree 2
    fitter = PolynomialFitter(
        np.exp, domain=(0.0, 1.0), degrees=2, num_intervals=2, continuity_c0=True
    )
    fitter.fit()

    analyzer = PolynomialAnalyzer(fitter, "Exp_N2_D2")
    analyzer.analyze()

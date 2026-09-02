"""
Odd-Transform Refitting: Fit sigmoid(x)-0.5, swish'(x)-0.5, and swish forward
using multiple methods and domain ranges to find optimal coefficients.

Usage:
    python fit_odd_transforms.py
"""
import numpy as np
import json
import os
import sys

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(__file__))

from polynomial_fitter import PolynomialFitter, PolynomialAnalyzer, quantize_to_fp16

try:
    from minimax_fitter import MinimaxPolynomialFitter
except ImportError:
    MinimaxPolynomialFitter = None

try:
    from least_squares_fitter import LeastSquaresPolynomialFitter
except ImportError:
    LeastSquaresPolynomialFitter = None

try:
    from constrained_ls_fitter import ConstrainedLSFitter
except ImportError:
    ConstrainedLSFitter = None


# =============================================================================
# Target Functions
# =============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_odd(x):
    """sigmoid(x) - 0.5: ODD function. sigmoid(x) = 0.5 + sigmoid_odd(x)."""
    return sigmoid(x) - 0.5

def swish_grad(x):
    """swish'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))"""
    s = sigmoid(x)
    return s * (1 + x * (1 - s))

def swish_grad_odd(x):
    """swish'(x) - 0.5: ODD function. swish'(x) = 0.5 + swish_grad_odd(x)."""
    return swish_grad(x) - 0.5

def swish(x):
    return x * sigmoid(x)

def tanh_func(x):
    return np.tanh(x)

def sigmoid_grad(x):
    """sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)). EVEN function."""
    s = sigmoid(x)
    return s * (1 - s)

def tanh_grad(x):
    """tanh'(x) = 1 - tanh(x)^2. EVEN function."""
    return 1.0 - np.tanh(x)**2


# =============================================================================
# Fitting Helpers
# =============================================================================

def coeffs_to_fp16_hex(coeffs):
    """Convert coefficient list to FP16 hex strings."""
    result = []
    for c in coeffs:
        fp16_val, hex_str = quantize_to_fp16(c)
        result.append(hex_str)
    return result

def evaluate_polynomial(x, coeffs):
    """Evaluate polynomial: coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ..."""
    result = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        result += c * x**i
    return result

def measure_fit_quality(func, domain, coeffs, n_samples=100000):
    """Measure fit quality metrics."""
    x = np.linspace(domain[0], domain[1], n_samples)
    y_true = func(x)
    y_pred = evaluate_polynomial(x, coeffs)

    abs_err = np.abs(y_true - y_pred)
    return {
        "max_error": float(np.max(abs_err)),
        "mean_error": float(np.mean(abs_err)),
        "mse": float(np.mean((y_true - y_pred)**2)),
    }

def fit_minimax(func, domain, degree, fix_boundaries=True):
    """Fit using minimax optimization."""
    if MinimaxPolynomialFitter is None:
        return None, None
    fitter = MinimaxPolynomialFitter(func, domain=domain, degrees=degree,
                                     num_intervals=1, fix_boundaries=fix_boundaries)
    fitter.fit()
    coeffs = fitter.coeffs[0]  # Single interval
    metrics = measure_fit_quality(func, domain, coeffs)
    return coeffs, metrics

def fit_least_squares(func, domain, degree):
    """Fit using least squares."""
    if LeastSquaresPolynomialFitter is None:
        return None, None
    fitter = LeastSquaresPolynomialFitter(func, domain, degree)
    fitter.fit()
    coeffs = list(fitter.coeffs)
    metrics = measure_fit_quality(func, domain, coeffs)
    return coeffs, metrics

def fit_polynomial(func, domain, degree, sampling='uniform', weight_func=None):
    """Fit using standard polynomial fitter."""
    fitter = PolynomialFitter(func, domain=domain, degrees=degree,
                              num_intervals=1, sampling=sampling, weight_func=weight_func)
    fitter.fit()
    coeffs = fitter.coeffs[0]
    metrics = measure_fit_quality(func, domain, coeffs)
    return coeffs, metrics


# =============================================================================
# Weight functions
# =============================================================================

def peak_weight(x, sigma=1.5, scale=10.0):
    return 1.0 + scale * np.exp(-0.5 * (x / sigma)**2)


# =============================================================================
# Main: Run all fits
# =============================================================================

def run_all_fits():
    save_dir = os.path.join(os.path.dirname(__file__), "..", "cuda_benchmarks", "analysis_results", "odd_transform")
    os.makedirs(save_dir, exist_ok=True)

    all_results = {}

    # =========================================================================
    # 1. SIGMOID ODD: sigmoid(x) - 0.5 on [0, L]
    # =========================================================================
    print("=" * 80)
    print("SIGMOID ODD: fitting sigmoid(x) - 0.5")
    print("=" * 80)

    for deg in [3, 4, 5]:
        for bound in [5.0, 5.5, 6.0, 6.5, 7.0, 8.0]:
            domain = (0.0, bound)
            name = f"sigmoid_odd_D{deg}_L{bound}"
            print(f"\n--- {name} ---")

            results = {"name": name, "degree": deg, "domain": list(domain), "fits": {}}

            # Minimax
            try:
                coeffs, metrics = fit_minimax(sigmoid_odd, domain, deg, fix_boundaries=True)
                if coeffs is not None:
                    fp16_hex = coeffs_to_fp16_hex(coeffs)
                    results["fits"]["minimax"] = {
                        "coeffs_fp32": [float(c) for c in coeffs],
                        "coeffs_fp16_hex": fp16_hex,
                        "metrics": metrics,
                    }
                    print(f"  Minimax:  max_err={metrics['max_error']:.6f}  coeffs={[f'{c:.6f}' for c in coeffs]}")
                    print(f"            fp16_hex={fp16_hex}")
            except Exception as e:
                print(f"  Minimax FAILED: {e}")

            # Least Squares
            try:
                coeffs, metrics = fit_least_squares(sigmoid_odd, domain, deg)
                if coeffs is not None:
                    fp16_hex = coeffs_to_fp16_hex(coeffs)
                    results["fits"]["least_squares"] = {
                        "coeffs_fp32": [float(c) for c in coeffs],
                        "coeffs_fp16_hex": fp16_hex,
                        "metrics": metrics,
                    }
                    print(f"  LS:       max_err={metrics['max_error']:.6f}  coeffs={[f'{c:.6f}' for c in coeffs]}")
            except Exception as e:
                print(f"  LS FAILED: {e}")

            # Chebyshev
            try:
                coeffs, metrics = fit_polynomial(sigmoid_odd, domain, deg, sampling='chebyshev')
                if coeffs is not None:
                    fp16_hex = coeffs_to_fp16_hex(coeffs)
                    results["fits"]["chebyshev"] = {
                        "coeffs_fp32": [float(c) for c in coeffs],
                        "coeffs_fp16_hex": fp16_hex,
                        "metrics": metrics,
                    }
                    print(f"  Cheby:    max_err={metrics['max_error']:.6f}  coeffs={[f'{c:.6f}' for c in coeffs]}")
            except Exception as e:
                print(f"  Cheby FAILED: {e}")

            all_results[name] = results

    # =========================================================================
    # 2. SWISH GRAD ODD: swish'(x) - 0.5 on [0, L]
    # =========================================================================
    print("\n" + "=" * 80)
    print("SWISH GRAD ODD: fitting swish'(x) - 0.5")
    print("=" * 80)

    for deg in [3, 4, 5]:
        for bound in [5.0, 6.0, 7.0, 7.5, 8.0, 10.0]:
            domain = (0.0, bound)
            name = f"swish_grad_odd_D{deg}_L{bound}"
            print(f"\n--- {name} ---")

            results = {"name": name, "degree": deg, "domain": list(domain), "fits": {}}

            # Minimax
            try:
                coeffs, metrics = fit_minimax(swish_grad_odd, domain, deg, fix_boundaries=True)
                if coeffs is not None:
                    fp16_hex = coeffs_to_fp16_hex(coeffs)
                    results["fits"]["minimax"] = {
                        "coeffs_fp32": [float(c) for c in coeffs],
                        "coeffs_fp16_hex": fp16_hex,
                        "metrics": metrics,
                    }
                    print(f"  Minimax:  max_err={metrics['max_error']:.6f}  coeffs={[f'{c:.6f}' for c in coeffs]}")
                    print(f"            fp16_hex={fp16_hex}")
            except Exception as e:
                print(f"  Minimax FAILED: {e}")

            # Least Squares
            try:
                coeffs, metrics = fit_least_squares(swish_grad_odd, domain, deg)
                if coeffs is not None:
                    fp16_hex = coeffs_to_fp16_hex(coeffs)
                    results["fits"]["least_squares"] = {
                        "coeffs_fp32": [float(c) for c in coeffs],
                        "coeffs_fp16_hex": fp16_hex,
                        "metrics": metrics,
                    }
                    print(f"  LS:       max_err={metrics['max_error']:.6f}  coeffs={[f'{c:.6f}' for c in coeffs]}")
            except Exception as e:
                print(f"  LS FAILED: {e}")

            all_results[name] = results

    # =========================================================================
    # 3. SIGMOID FORWARD ODD: refit sigmoid(x)-0.5 with more methods
    #    (to get better coefficients than ABS_OPT)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SIGMOID FWD ODD: Improved fitting for sigmoid(x) - 0.5")
    print("=" * 80)

    # The key question: can we get better coefficients for the ODD sigmoid D3?
    # Current ABS_OPT uses: c1=0x3493(0.286), c2=0xAB32(-0.056), c3=0x1BA8(0.0037)
    # These were from fitted f(x) = c0 + c1*x + c2*x^2 + c3*x^3 with c0=0.5
    # Now refit h(x) = sigmoid(x) - 0.5 directly (c0 should be ~0)

    for deg in [3]:
        for bound in [5.0, 5.5, 6.0, 6.5, 7.0]:
            domain = (0.0, bound)
            name = f"sigmoid_fwd_odd_D{deg}_L{bound}"
            print(f"\n--- {name} ---")

            results = {"name": name, "degree": deg, "domain": list(domain), "fits": {}}

            # Minimax (boundary constrained)
            try:
                coeffs, metrics = fit_minimax(sigmoid_odd, domain, deg, fix_boundaries=True)
                if coeffs is not None:
                    fp16_hex = coeffs_to_fp16_hex(coeffs)
                    results["fits"]["minimax_constrained"] = {
                        "coeffs_fp32": [float(c) for c in coeffs],
                        "coeffs_fp16_hex": fp16_hex,
                        "metrics": metrics,
                    }
                    print(f"  Minimax(const): max_err={metrics['max_error']:.6f}")
                    print(f"    coeffs: {[f'{c:.8f}' for c in coeffs]}")
                    print(f"    fp16:   {fp16_hex}")
            except Exception as e:
                print(f"  Minimax(const) FAILED: {e}")

            # Minimax (unconstrained)
            try:
                coeffs, metrics = fit_minimax(sigmoid_odd, domain, deg, fix_boundaries=False)
                if coeffs is not None:
                    fp16_hex = coeffs_to_fp16_hex(coeffs)
                    results["fits"]["minimax_free"] = {
                        "coeffs_fp32": [float(c) for c in coeffs],
                        "coeffs_fp16_hex": fp16_hex,
                        "metrics": metrics,
                    }
                    print(f"  Minimax(free):  max_err={metrics['max_error']:.6f}")
                    print(f"    coeffs: {[f'{c:.8f}' for c in coeffs]}")
                    print(f"    fp16:   {fp16_hex}")
            except Exception as e:
                print(f"  Minimax(free) FAILED: {e}")

            # Weighted Chebyshev
            try:
                coeffs, metrics = fit_polynomial(sigmoid_odd, domain, deg,
                    sampling='chebyshev', weight_func=lambda x: peak_weight(x, sigma=2.0, scale=5.0))
                if coeffs is not None:
                    fp16_hex = coeffs_to_fp16_hex(coeffs)
                    results["fits"]["weighted_cheby"] = {
                        "coeffs_fp32": [float(c) for c in coeffs],
                        "coeffs_fp16_hex": fp16_hex,
                        "metrics": metrics,
                    }
                    print(f"  Weighted Cheby: max_err={metrics['max_error']:.6f}")
                    print(f"    coeffs: {[f'{c:.8f}' for c in coeffs]}")
            except Exception as e:
                print(f"  Weighted Cheby FAILED: {e}")

            all_results[name] = results

    # =========================================================================
    # 4. SWISH FORWARD: try fitting swish(|x|) on [0, L]
    # =========================================================================
    print("\n" + "=" * 80)
    print("SWISH FORWARD: fitting swish(|x|) on [0, L]")
    print("swish(x) = swish(|x|) + min(x, 0)")
    print("=" * 80)

    # swish(x) for x >= 0 is just x*sigmoid(x), which grows unbounded
    # We fit on [0, L] where L is chosen so swish(L) ≈ L (sigmoid(L) ≈ 1)
    for deg in [3, 4]:
        for bound in [5.0, 6.0, 7.0, 8.0]:
            domain = (0.0, bound)
            name = f"swish_fwd_D{deg}_L{bound}"
            print(f"\n--- {name} ---")

            results = {"name": name, "degree": deg, "domain": list(domain), "fits": {}}

            try:
                coeffs, metrics = fit_minimax(swish, domain, deg, fix_boundaries=True)
                if coeffs is not None:
                    fp16_hex = coeffs_to_fp16_hex(coeffs)
                    results["fits"]["minimax"] = {
                        "coeffs_fp32": [float(c) for c in coeffs],
                        "coeffs_fp16_hex": fp16_hex,
                        "metrics": metrics,
                    }
                    print(f"  Minimax:  max_err={metrics['max_error']:.6f}")
                    print(f"    coeffs: {[f'{c:.8f}' for c in coeffs]}")
                    print(f"    fp16:   {fp16_hex}")
            except Exception as e:
                print(f"  Minimax FAILED: {e}")

            all_results[name] = results

    # =========================================================================
    # Save all results
    # =========================================================================
    output_path = os.path.join(save_dir, "all_odd_transform_fits.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved to: {output_path}")

    # =========================================================================
    # Summary: Best fits per category
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Best fits per category")
    print("=" * 80)

    categories = {
        "sigmoid_fwd_odd": [],
        "sigmoid_odd": [],
        "swish_grad_odd": [],
        "swish_fwd": [],
    }

    for name, result in all_results.items():
        for cat_prefix in categories:
            if name.startswith(cat_prefix):
                for method, fit_data in result.get("fits", {}).items():
                    categories[cat_prefix].append({
                        "name": name,
                        "method": method,
                        "max_error": fit_data["metrics"]["max_error"],
                        "coeffs": fit_data.get("coeffs_fp32"),
                        "fp16_hex": fit_data.get("coeffs_fp16_hex"),
                    })

    for cat_name, fits in categories.items():
        if not fits:
            continue
        fits.sort(key=lambda x: x["max_error"])
        print(f"\n--- {cat_name} (top 5 by max_error) ---")
        for i, f in enumerate(fits[:5]):
            print(f"  {i+1}. {f['name']} ({f['method']}): max_err={f['max_error']:.6f}")
            if f.get('fp16_hex'):
                print(f"     fp16_hex: {f['fp16_hex']}")


if __name__ == "__main__":
    run_all_fits()

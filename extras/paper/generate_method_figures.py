# Copyright 2026 Robert Hu
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the manuscript's two approximation-method figures offline.

The generator consumes only the checked-in BF16 coefficient sweep. It never
contacts an experiment service and writes only to caller-selected locations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .artifact_utils import write_receipt
except ImportError:  # Direct script execution.
    from artifact_utils import write_receipt


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COEFFICIENTS = (
    REPOSITORY_ROOT
    / "autonumerics_zero/cuda_benchmarks/analysis_results/all_degree_coefficients_bf16.json"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs/paper/method"

CORAL = "#FF6F79"
CORAL_DARK = "#C84553"
INK = "#1F2021"
MUTED = "#53595E"
LINE = "#D7DADD"
PAPER = "#FAF8F9"
TEAL = "#16858C"
GOLD = "#B77A19"
BLUE = "#3977A8"


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.0,
            "axes.edgecolor": LINE,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": LINE,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "legend.fontsize": 6.6,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def style_axis(axis: plt.Axes, panel: str) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.03,
        0.05,
        panel,
        transform=axis.transAxes,
        color=CORAL_DARK,
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.5},
        zorder=10,
    )


def save_figure(figure: plt.Figure, path: Path) -> None:
    kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.03}
    if path.suffix.lower() == ".pdf":
        kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
    elif path.suffix.lower() == ".png":
        kwargs["dpi"] = 200
    else:
        raise ValueError("figure format must be PDF or PNG")
    figure.savefig(path, **kwargs)
    plt.close(figure)


def make_symmetry_figure(path: Path) -> None:
    x = np.linspace(-6.0, 6.0, 1601)
    sigma = sigmoid(x)
    centered_sigma = sigma - 0.5
    silu = x * sigma
    silu_residual = silu - 0.5 * x
    silu_grad = sigma * (1.0 + x * (1.0 - sigma))
    centered_grad = silu_grad - 0.5

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    for axis in axes:
        axis.axvspan(0.0, 6.0, color=PAPER, zorder=0)
        axis.axvline(0.0, color=LINE, linewidth=0.8, zorder=1)
        axis.set_xlim(-6.0, 6.0)
        axis.set_xlabel("x")

    axis = axes[0]
    axis.plot(x, sigma, color=INK, linewidth=1.5, label=r"$\sigma(x)$")
    axis.plot(
        x,
        centered_sigma,
        color=CORAL_DARK,
        linewidth=1.8,
        label=r"$g_\sigma(x)=\sigma(x)-1/2$",
    )
    axis.axhline(0.0, color=LINE, linewidth=0.8)
    axis.set_ylim(-0.56, 1.04)
    axis.set_title("Center sigmoid", pad=12)
    axis.text(2.9, -0.43, "fit this half", color=CORAL_DARK, ha="center")
    axis.legend(loc="upper left", frameon=False)
    style_axis(axis, "(a)")

    axis = axes[1]
    axis.plot(x, silu, color=INK, linewidth=1.5, label=r"$s(x)$")
    axis.plot(x, 0.5 * x, color=MUTED, linewidth=1.1, linestyle="--", label=r"$x/2$")
    axis.plot(
        x,
        silu_residual,
        color=CORAL_DARK,
        linewidth=1.8,
        label=r"$s(x)-x/2$",
    )
    axis.axhline(0.0, color=LINE, linewidth=0.8)
    axis.set_ylim(-3.4, 6.2)
    axis.set_title("Expose SiLU's even residual", pad=12)
    axis.text(
        0.5, 0.12, r"$r_s(-x)=r_s(x)$", transform=axis.transAxes, color=CORAL_DARK
    )
    axis.legend(loc="upper left", frameon=False)
    style_axis(axis, "(b)")

    axis = axes[2]
    axis.plot(x, silu_grad, color=INK, linewidth=1.5, label=r"$s'(x)$")
    axis.plot(
        x,
        centered_grad,
        color=CORAL_DARK,
        linewidth=1.8,
        label=r"$s'(x)-1/2$",
    )
    axis.axhline(0.0, color=LINE, linewidth=0.8)
    axis.set_ylim(-0.65, 1.18)
    axis.set_title("Center the SiLU derivative", pad=12)
    axis.text(
        0.5,
        0.12,
        "centered derivative is odd",
        transform=axis.transAxes,
        color=CORAL_DARK,
    )
    axis.legend(loc="upper left", frameon=False)
    style_axis(axis, "(c)")
    figure.tight_layout(w_pad=1.05)
    save_figure(figure, path)


def bf16_round(values: np.ndarray | float) -> np.ndarray:
    """Round float32 values to BF16 with round-to-nearest-even."""
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    finite = (bits & np.uint32(0x7F800000)) != np.uint32(0x7F800000)
    rounded = bits.copy()
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded[finite] = (bits[finite] + bias[finite]) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def constrained_fit(
    interval: float, degree: int = 4, samples: int = 2000
) -> np.ndarray:
    indices = np.arange(samples)
    chebyshev = np.cos((2 * indices + 1) * np.pi / (2 * samples))
    x = np.sort(0.5 * interval * (1.0 + chebyshev))
    target = sigmoid(x) - 0.5
    endpoint = float(sigmoid(np.asarray(interval)) - 0.5)
    slope = endpoint / interval
    residual = target - slope * x
    design = np.column_stack(
        [x**power - x * interval ** (power - 1) for power in range(2, degree + 1)]
    )
    higher = np.linalg.lstsq(design, residual, rcond=None)[0]
    linear = slope - sum(
        coefficient * interval ** (index + 1)
        for index, coefficient in enumerate(higher)
    )
    return np.asarray([0.0, linear, *higher], dtype=np.float64)


def evaluate_real_horner(
    x: np.ndarray, coefficients: np.ndarray, clamp: float
) -> np.ndarray:
    t = np.minimum(np.abs(x), clamp)
    accumulator = np.zeros_like(t, dtype=np.float64)
    for coefficient in coefficients[:0:-1]:
        accumulator = accumulator * t + coefficient
    return t * accumulator


def evaluate_bf16_horner(
    x: np.ndarray,
    coefficients: np.ndarray,
    clamp: float,
    *,
    fused: bool,
) -> np.ndarray:
    t = bf16_round(np.minimum(np.abs(x), clamp))
    rounded_coefficients = bf16_round(coefficients)
    accumulator = np.full_like(t, rounded_coefficients[-1])
    for coefficient in rounded_coefficients[-2:0:-1]:
        if fused:
            accumulator = bf16_round(accumulator * t + coefficient)
        else:
            accumulator = bf16_round(bf16_round(accumulator * t) + coefficient)
    return bf16_round(accumulator * t).astype(np.float64)


def search_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_intervals = np.arange(3.0, 8.0001, 0.25)
    clamps = np.arange(3.0, 8.0001, 0.25)
    x = np.linspace(0.0, 12.0, 8000)
    target = sigmoid(x) - 0.5
    errors = np.full((len(clamps), len(fit_intervals)), np.nan)
    for column, fit_interval in enumerate(fit_intervals):
        coefficients = constrained_fit(fit_interval)
        for row, clamp in enumerate(clamps):
            if clamp > fit_interval + 1.0:
                continue
            prediction = evaluate_bf16_horner(x, coefficients, clamp, fused=False)
            errors[row, column] = np.max(np.abs(prediction - target)) * 1e3
    return fit_intervals, clamps, errors


def selected_sigmoid_candidate(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    document = json.loads(path.read_text())
    try:
        candidate = document["sigmoid_fwd_odd"]["d4"]
        return (
            np.asarray(candidate["coeffs_f64"], dtype=np.float64),
            np.asarray(candidate["coeffs_bf16"], dtype=np.float64),
            float(candidate["Li"]),
            float(candidate["Lc"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "coefficient input lacks the selected sigmoid_fwd_odd.d4 fit"
        ) from error


def make_fitting_figure(coefficients_path: Path, path: Path) -> None:
    coefficients_f64, coefficients_bf16, fit_interval, clamp = (
        selected_sigmoid_candidate(coefficients_path)
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))

    axis = axes[0]
    coordinate = np.linspace(0.0, 1.0, 500)
    for power, color in zip((2, 3, 4), (CORAL_DARK, TEAL, BLUE), strict=True):
        axis.plot(
            coordinate,
            coordinate**power - coordinate,
            color=color,
            linewidth=1.6,
            label=rf"$u^{power}-u$",
        )
    axis.scatter([0.0, 1.0], [0.0, 0.0], color=INK, s=13, zorder=4)
    axis.axhline(0.0, color=LINE, linewidth=0.8)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.52, 0.08)
    axis.set_xlabel(r"normalized coordinate $u=x/L_i$")
    axis.set_ylabel(r"free basis $\phi_j(u)$")
    axis.set_title("Endpoint-null basis", pad=12)
    axis.legend(
        loc="upper center", frameon=False, ncol=3, columnspacing=0.7, handlelength=1.4
    )
    style_axis(axis, "(a)")

    axis = axes[1]
    fit_intervals, clamps, errors = search_surface()
    masked = np.ma.masked_invalid(errors)
    heatmap = plt.get_cmap("viridis_r").copy()
    heatmap.set_bad(color="#F0F1F2")
    image = axis.imshow(
        masked,
        origin="lower",
        aspect="auto",
        extent=(
            fit_intervals[0] - 0.125,
            fit_intervals[-1] + 0.125,
            clamps[0] - 0.125,
            clamps[-1] + 0.125,
        ),
        cmap=heatmap,
        vmin=6.0,
        vmax=30.0,
        interpolation="nearest",
    )
    boundary = np.linspace(3.0, 7.0, 100)
    axis.plot(boundary, boundary + 1.0, color=MUTED, linewidth=1.0, linestyle="--")
    axis.scatter(
        [fit_interval],
        [clamp],
        marker="*",
        s=70,
        color=CORAL,
        edgecolor=INK,
        linewidth=0.55,
        zorder=5,
        label="selected D4",
    )
    axis.set_xlim(3.0, 8.0)
    axis.set_ylim(3.0, 8.0)
    axis.set_xlabel(r"fit interval $L_i$")
    axis.set_ylabel(r"runtime clamp $L_c$")
    axis.set_title("Search fit and clamp separately", pad=12)
    axis.grid(False)
    axis.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=LINE)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.048, pad=0.025)
    colorbar.set_label(r"split-BF16 max error ($\times10^{-3}$)", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6.2)
    style_axis(axis, "(b)")

    axis = axes[2]
    x = np.linspace(0.0, 6.0, 1601)
    target = sigmoid(x) - 0.5
    curves = (
        (
            "FP64 fit, real Horner",
            evaluate_real_horner(x, coefficients_f64, clamp),
            MUTED,
            ":",
            1.2,
        ),
        (
            "BF16 coeffs, real Horner",
            evaluate_real_horner(x, coefficients_bf16, clamp),
            BLUE,
            "-.",
            1.2,
        ),
        (
            "split-BF16 search replay",
            evaluate_bf16_horner(x, coefficients_bf16, clamp, fused=False),
            GOLD,
            "--",
            1.35,
        ),
        (
            "packed-FMA replay",
            evaluate_bf16_horner(x, coefficients_bf16, clamp, fused=True),
            CORAL_DARK,
            "-",
            1.65,
        ),
    )
    for label, prediction, color, linestyle, width in curves:
        error = np.maximum(np.abs(prediction - target), 1e-6)
        axis.plot(
            x, error, label=label, color=color, linestyle=linestyle, linewidth=width
        )
    axis.axvline(clamp, color=LINE, linewidth=0.9)
    axis.text(
        clamp - 0.08,
        1.2e-5,
        r"$L_c$",
        rotation=90,
        va="bottom",
        ha="right",
        color=MUTED,
    )
    axis.set_yscale("log")
    axis.set_xlim(0.0, 6.0)
    axis.set_ylim(1e-6, 1.5e-2)
    axis.set_xlabel("x")
    axis.set_ylabel("absolute error")
    axis.set_title("Replay the deployed schedule", pad=12)
    axis.legend(loc="lower right", frameon=False, handlelength=1.6)
    style_axis(axis, "(c)")

    figure.tight_layout(w_pad=0.8)
    save_figure(figure, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf")
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Receipt path (default: OUTPUT_DIR/method_figures.receipt.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.coefficients.is_file():
        raise FileNotFoundError(args.coefficients)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "." + args.format
    symmetry_path = args.output_dir / ("symmetry_reduction" + suffix)
    fitting_path = args.output_dir / ("constrained_fit_replay" + suffix)
    receipt_path = args.receipt or args.output_dir / "method_figures.receipt.json"

    configure_plotting()
    make_symmetry_figure(symmetry_path)
    make_fitting_figure(args.coefficients, fitting_path)
    write_receipt(
        receipt_path,
        artifact_type="paper-method-figures",
        generator=Path(__file__),
        inputs=[(args.coefficients, "BF16 coefficient sweep")],
        outputs=[
            (symmetry_path, "symmetry reduction figure"),
            (fitting_path, "constrained-fit replay figure"),
        ],
        parameters={"format": args.format, "search_samples": 8000, "fit_nodes": 2000},
        packages=("matplotlib", "numpy"),
        notes=(
            "Figures are deterministic offline reconstructions from the supplied coefficients.",
            "The packed-FMA curve is a software rounding replay, not a new GPU measurement.",
        ),
    )
    print(f"Wrote {symmetry_path}")
    print(f"Wrote {fitting_path}")
    print(f"Wrote {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright 2026 Robert Hu
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the six BF16 kernel-accuracy figures without network access.

This workflow evaluates the compiled CUDA extension and therefore requires a
CUDA GPU. It is offline in the provenance sense: it reads no remote service,
model, or dataset. Outputs and the receipt are always caller-selected.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .artifact_utils import write_receipt
except ImportError:  # Direct script execution.
    from artifact_utils import write_receipt


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTENSION_DIR = REPOSITORY_ROOT / "autonumerics_zero/spline_ops"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs/paper/accuracy"
DEGREES = (3, 4, 5, 6)


def load_extension(extension_dir: Path) -> ModuleType:
    """Import ``spline_ops`` from an explicitly selected source directory."""
    if not extension_dir.is_dir():
        raise FileNotFoundError(extension_dir)
    selected_root = extension_dir.resolve()
    sys.path.insert(0, str(selected_root))
    try:
        module = importlib.import_module("spline_ops")
    finally:
        sys.path.pop(0)
    origin_value = getattr(module, "__file__", None)
    if not origin_value:
        raise RuntimeError("loaded spline_ops module has no filesystem origin")
    origin = Path(origin_value).resolve()
    if not origin.is_relative_to(selected_root):
        raise RuntimeError(
            "loaded spline_ops module does not come from --extension-dir: "
            f"{origin.name}"
        )
    return module


def compute_errors(
    spline_fn: object, reference_fn: object, x: object, torch: ModuleType
):
    with torch.no_grad():
        spline = spline_fn(x)
        reference = reference_fn(x)
    torch.cuda.synchronize()
    error = (spline.float() - reference.float()).abs()
    return spline.float().cpu().numpy(), error.cpu().numpy()


def plot_function_degrees(
    function_name: str,
    direction: str,
    x: np.ndarray,
    reference: np.ndarray,
    degree_data: list[tuple[int, np.ndarray, np.ndarray]],
    output: Path,
) -> None:
    figure, (values_axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(4.25, 3.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1]},
    )
    values_axis.plot(x, reference, "k-", linewidth=2, label="Reference", alpha=0.4)
    colors = ("#e74c3c", "#e67e22", "#2ecc71", "#3498db")
    for color, (degree, values, error) in zip(colors, degree_data, strict=True):
        values_axis.plot(
            x, values, color=color, linewidth=1.2, label=f"D{degree}", alpha=0.8
        )
        error_axis.plot(
            x,
            error,
            color=color,
            linewidth=1,
            label=f"D{degree} (max={np.max(error):.4f})",
            alpha=0.8,
        )
    value_label = "Function value" if direction == "FWD" else "Derivative value"
    values_axis.set_ylabel(value_label, fontsize=8.5)
    values_axis.legend(loc="best", fontsize=7.5, ncol=2)
    values_axis.tick_params(labelsize=7.5)
    values_axis.grid(True, alpha=0.3)
    error_axis.set_xlabel("BF16-representable input", fontsize=8.5)
    error_axis.set_ylabel("Absolute error", fontsize=8.5)
    error_axis.legend(loc="best", fontsize=7.5, ncol=2)
    error_axis.tick_params(labelsize=7.5)
    error_axis.grid(True, alpha=0.3)
    error_axis.set_yscale("log")
    figure.tight_layout()
    save_args: dict[str, object] = {"bbox_inches": "tight"}
    if output.suffix == ".png":
        save_args["dpi"] = 200
    else:
        save_args["metadata"] = {"CreationDate": None, "ModDate": None}
    figure.savefig(output, **save_args)
    plt.close(figure)


def generate(
    extension: ModuleType,
    output_dir: Path,
    *,
    output_format: str,
    samples: int,
    torch: ModuleType,
) -> list[Path]:
    if samples < 2:
        raise ValueError("samples must be at least two")
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required for kernel-accuracy replay")
    output_dir.mkdir(parents=True, exist_ok=True)
    x = torch.linspace(-15, 15, samples, device="cuda", dtype=torch.bfloat16)
    x_numpy = x.float().cpu().numpy()
    grad_output = torch.ones_like(x)
    outputs: list[Path] = []

    specifications = (
        ("Sigmoid", "sigmoid", torch.sigmoid, lambda z: torch.sigmoid(z.float())),
        ("Tanh", "tanh", torch.tanh, lambda z: torch.tanh(z.float())),
        (
            "Swish",
            "swish",
            lambda z: z * torch.sigmoid(z),
            lambda z: z.float() * torch.sigmoid(z.float()),
        ),
    )
    for display_name, kernel_name, _, forward_reference in specifications:
        reference = forward_reference(x).cpu().numpy()
        degree_data = []
        for degree in DEGREES:
            function = getattr(extension, f"{kernel_name}_fwd_d{degree}")
            values, error = compute_errors(function, forward_reference, x, torch)
            degree_data.append((degree, values, error))
        output = output_dir / f"{kernel_name}_fwd_accuracy.{output_format}"
        plot_function_degrees(
            display_name, "FWD", x_numpy, reference, degree_data, output
        )
        outputs.append(output)

        if kernel_name == "sigmoid":
            sigmoid = torch.sigmoid(x.float())
            backward_reference = sigmoid * (1 - sigmoid)
        elif kernel_name == "tanh":
            backward_reference = 1 - torch.tanh(x.float()) ** 2
        else:
            sigmoid = torch.sigmoid(x.float())
            backward_reference = sigmoid * (1 + x.float() * (1 - sigmoid))
        reference = backward_reference.cpu().numpy()
        degree_data = []
        for degree in DEGREES:
            function = getattr(extension, f"{kernel_name}_bwd_d{degree}")
            values = function(grad_output, x)
            torch.cuda.synchronize()
            values_numpy = values.float().cpu().numpy()
            degree_data.append((degree, values_numpy, np.abs(values_numpy - reference)))
        output = output_dir / f"{kernel_name}_bwd_accuracy.{output_format}"
        plot_function_degrees(
            display_name, "BWD", x_numpy, reference, degree_data, output
        )
        outputs.append(output)
    return outputs


def extension_inputs(
    extension_dir: Path, extension: ModuleType
) -> list[tuple[Path, str]]:
    inputs: list[tuple[Path, str]] = []
    for name in (
        "spline_kernels_bf16.cu",
        "spline_ops.cpp",
        "spline_structs_odd_bf16.cuh",
    ):
        candidate = extension_dir / name
        if candidate.is_file():
            inputs.append((candidate, "extension source"))
    binary = Path(str(getattr(extension, "__file__", "")))
    if binary.is_file():
        inputs.append((binary, "loaded extension binary"))
    return inputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-dir", type=Path, default=DEFAULT_EXTENSION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf")
    parser.add_argument("--samples", type=int, default=30_000)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Receipt path (default: OUTPUT_DIR/accuracy_figures.receipt.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for kernel-accuracy replay") from error
    extension = load_extension(args.extension_dir)
    outputs = generate(
        extension,
        args.output_dir,
        output_format=args.format,
        samples=args.samples,
        torch=torch,
    )
    receipt = args.receipt or args.output_dir / "accuracy_figures.receipt.json"
    write_receipt(
        receipt,
        artifact_type="paper-kernel-accuracy-figures",
        generator=Path(__file__),
        inputs=extension_inputs(args.extension_dir, extension),
        outputs=[(path, "kernel accuracy figure") for path in outputs],
        parameters={
            "degrees": list(DEGREES),
            "domain": [-15, 15],
            "dtype": "bfloat16",
            "format": args.format,
            "samples": args.samples,
        },
        packages=("matplotlib", "numpy", "torch"),
        notes=(
            "This is a fresh CUDA measurement of the supplied extension, not a claim of bit-identical historical replay.",
            "References are evaluated in FP32 at BF16-representable input coordinates.",
        ),
    )
    for output in outputs:
        print(f"Wrote {output}")
    print(f"Wrote {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

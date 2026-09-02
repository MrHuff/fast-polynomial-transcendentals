# Copyright 2026 Robert Hu
# SPDX-License-Identifier: Apache-2.0
"""Generate an auxiliary Sollya sensitivity control for D3--D6 sweeps.

This is not the manuscript Table 2 producer. Four workflow differences explain
the selected-cell divergences: sigmoid D3 uses an earlier sweep row and clamp
4.75 rather than the later deployed fit and clamp 6.0; the later tanh D4 device
refit is absent from the sweep; the auxiliary Sollya side fits the SiLU
residual directly while the retained table composes both columns from their
sigmoid coefficients; and GELU/GELU' use the FP16 sweep and an 11-bit Sollya
budget here rather than the table's deployed BF16 header and 8-bit/BF16
control.

The retained table also has separate semantic review items: asymmetric BF16
handling of current and Sollya coefficients, host NumPy rather than device
error, and incomplete endpoint-constrained least-squares lineage. See
``FUNCTION_TABLE_REVIEW.md``.

Within this auxiliary control, degree, monomials, coefficient precision,
clamp, and algebraic reconstruction are held fixed. Sollya must already be
installed; this script performs no downloads.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from .artifact_utils import write_receipt
except ImportError:  # Direct script execution.
    from artifact_utils import write_receipt


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPOSITORY_ROOT / "autonumerics_zero/cuda_benchmarks/analysis_results"
DEFAULT_BF16 = ANALYSIS_DIR / "all_degree_coefficients_bf16.json"
DEFAULT_FP16 = ANALYSIS_DIR / "all_degree_coefficients.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs/paper/sollya_comparison.json"


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def swish(values: np.ndarray) -> np.ndarray:
    return values * sigmoid(values)


def swish_grad(values: np.ndarray) -> np.ndarray:
    probability = sigmoid(values)
    return probability * (1.0 + values * (1.0 - probability))


def gelu(values: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * values * (1.0 + erf(values / math.sqrt(2.0)))


def gelu_grad(values: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(values / math.sqrt(2.0))) + values / math.sqrt(
        2.0 * math.pi
    ) * np.exp(-0.5 * values * values)


def horner(values: np.ndarray, coefficients: list[float]) -> np.ndarray:
    result = np.zeros_like(values)
    for coefficient in reversed(coefficients):
        result = result * values + coefficient
    return result


def eval_odd_factorized(
    values: np.ndarray, coefficients: list[float], clamp: float
) -> np.ndarray:
    magnitude = np.minimum(np.abs(values), clamp)
    output = np.minimum(magnitude * horner(magnitude, coefficients), 1.0)
    return np.copysign(output, values)


def eval_centered_odd(
    values: np.ndarray, coefficients: list[float], clamp: float, offset: float
) -> np.ndarray:
    magnitude = np.minimum(np.abs(values), clamp)
    return offset + np.sign(values) * magnitude * horner(magnitude, coefficients)


def eval_even(
    values: np.ndarray, coefficients: list[float], clamp: float
) -> np.ndarray:
    return horner(np.minimum(np.abs(values), clamp), coefficients)


def eval_swish_forward(
    values: np.ndarray, coefficients: list[float], clamp: float
) -> np.ndarray:
    magnitude = np.minimum(np.abs(values), clamp)
    return values * (
        0.5 + np.sign(values) * magnitude * horner(magnitude, coefficients)
    )


@dataclass(frozen=True)
class FamilySpec:
    display_name: str
    source: str
    source_key: str
    split: str
    precision_bits: int
    expression: str
    monomial_start: int
    highest_degree_offset: int
    coefficient_key: str
    drop_constant: bool
    evaluator: Callable[..., np.ndarray]
    target: Callable[[np.ndarray], np.ndarray]
    offset: float | None = None

    def monomials(self, degree: int) -> list[int]:
        return list(range(self.monomial_start, degree + self.highest_degree_offset + 1))


def build_specs() -> tuple[FamilySpec, ...]:
    return (
        FamilySpec(
            "sigmoid",
            "bf16",
            "sigmoid_fwd_odd",
            "forward",
            8,
            "(1/(1+exp(-x))) - 0.5",
            1,
            0,
            "coeffs_bf16",
            True,
            eval_centered_odd,
            sigmoid,
            0.5,
        ),
        FamilySpec(
            "tanh",
            "bf16",
            "tanh_fwd_odd",
            "forward",
            8,
            "tanh(x)",
            1,
            0,
            "coeffs_bf16",
            True,
            eval_odd_factorized,
            np.tanh,
        ),
        FamilySpec(
            "swish",
            "bf16",
            "sigmoid_fwd_odd",
            "forward",
            8,
            "(x/(1+exp(-x))) - 0.5*x",
            2,
            1,
            "coeffs_bf16",
            True,
            eval_swish_forward,
            swish,
        ),
        FamilySpec(
            "gelu",
            "fp16",
            "gelu_fwd_odd",
            "forward",
            11,
            "(0.5*x*(1+erf(x/sqrt(2)))) - 0.5*x",
            2,
            1,
            "coeffs_fp16",
            True,
            eval_swish_forward,
            gelu,
        ),
        FamilySpec(
            "sigmoid'",
            "bf16",
            "sigmoid_bwd_even",
            "backward",
            8,
            "(1/(1+exp(-x)))*(1-(1/(1+exp(-x))))",
            0,
            0,
            "coeffs_bf16",
            False,
            eval_even,
            lambda values: sigmoid(values) * (1.0 - sigmoid(values)),
        ),
        FamilySpec(
            "tanh'",
            "bf16",
            "tanh_bwd_even",
            "backward",
            8,
            "1-(tanh(x)^2)",
            0,
            0,
            "coeffs_bf16",
            False,
            eval_even,
            lambda values: 1.0 - np.tanh(values) ** 2,
        ),
        FamilySpec(
            "swish'",
            "bf16",
            "swish_bwd_odd",
            "backward",
            8,
            "((1/(1+exp(-x)))*(1+x*(1-(1/(1+exp(-x)))))) - 0.5",
            1,
            0,
            "coeffs_bf16",
            True,
            eval_centered_odd,
            swish_grad,
            0.5,
        ),
        FamilySpec(
            "gelu'",
            "fp16",
            "gelu_bwd_odd",
            "backward",
            11,
            "(0.5*(1+erf(x/sqrt(2))) + x/sqrt(2*pi)*exp(-(x^2)/2)) - 0.5",
            1,
            0,
            "coeffs_fp16",
            True,
            eval_centered_odd,
            gelu_grad,
            0.5,
        ),
    )


def run_sollya(
    executable: str,
    expression: str,
    monomials: list[int],
    precision_bits: int,
    clamp: float,
) -> list[float]:
    monomial_arg = "[|" + ",".join(str(value) for value in monomials) + "|]"
    format_arg = "[|" + ",".join(str(precision_bits) for _ in monomials) + "|]"
    lower = "1b-20" if min(monomials) > 0 else "0"
    commands = [
        f"f = fpminimax({expression}, {monomial_arg}, {format_arg}, [{lower};{clamp}], absolute);"
    ]
    commands.extend(f"print(coeff(f,{degree}));" for degree in monomials)
    commands.append("quit;")
    completed = subprocess.run(
        [executable, "--flush", "--noprompt"],
        input="\n".join(commands) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    coefficients = []
    for line in completed.stdout.splitlines():
        try:
            coefficients.append(float(line.strip()))
        except ValueError:
            continue
    if len(coefficients) != len(monomials):
        raise RuntimeError(
            f"unexpected Sollya coefficient count: expected {len(monomials)}, got {len(coefficients)}"
        )
    return coefficients


def max_error(
    spec: FamilySpec,
    coefficients: list[float],
    clamp: float,
    samples: int,
) -> float:
    values = np.linspace(-clamp, clamp, samples)
    if spec.offset is None:
        approximation = spec.evaluator(values, coefficients, clamp)
    else:
        approximation = spec.evaluator(values, coefficients, clamp, spec.offset)
    return float(np.max(np.abs(approximation - spec.target(values))))


def generate(
    bf16_path: Path,
    fp16_path: Path,
    *,
    sollya: str,
    samples: int,
) -> dict[str, object]:
    if samples < 2:
        raise ValueError("samples must be at least two")
    bf16 = json.loads(bf16_path.read_text())
    fp16 = json.loads(fp16_path.read_text())
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        "forward": {},
        "backward": {},
    }
    for spec in build_specs():
        source = bf16 if spec.source == "bf16" else fp16
        family_rows = results[spec.split].setdefault(spec.display_name, {})
        for degree in (3, 4, 5, 6):
            try:
                row = source[spec.source_key][f"d{degree}"]
                coefficients = [float(value) for value in row[spec.coefficient_key]]
                if spec.drop_constant:
                    coefficients = coefficients[1:]
                clamp = float(row["Lc"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"coefficient input lacks {spec.source_key}.d{degree}"
                ) from error
            sollya_coefficients = run_sollya(
                sollya,
                spec.expression,
                spec.monomials(degree),
                spec.precision_bits,
                clamp,
            )
            family_rows[f"D{degree}"] = {
                "clamp": clamp,
                "precision_bits": spec.precision_bits,
                "ours": max_error(spec, coefficients, clamp, samples),
                "sollya": max_error(spec, sollya_coefficients, clamp, samples),
            }
    return {
        "schema_version": 1,
        "artifact_type": "same-form-sollya-coefficient-control",
        "scope": "auxiliary sensitivity analysis; not manuscript Table 2 evidence",
        "measurement": {
            "error_metric": "maximum absolute error",
            "evaluation": "real arithmetic",
            "grid": "uniform closed interval [-clamp, clamp]",
            "samples": samples,
        },
        "results": results,
    }


def latex_table(split: str, data: dict[str, object]) -> str:
    rows = []
    for family, degrees in data["results"][split].items():
        cells = []
        for degree in ("D3", "D4", "D5", "D6"):
            entry = degrees[degree]
            cells.append(f"{entry['ours'] * 1e3:.2f} / {entry['sollya'] * 1e3:.2f}")
        rows.append(f"{family} & " + " & ".join(cells) + r" \\")
    return (
        r"\begin{tabular}{lcccc}"
        "\n"
        r"\toprule"
        "\n"
        r"Family & D3 & D4 & D5 & D6 \\"
        "\n"
        r"\midrule"
        "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-input", type=Path, default=DEFAULT_BF16)
    parser.add_argument("--fp16-input", type=Path, default=DEFAULT_FP16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tex-output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--sollya", default="sollya")
    parser.add_argument("--samples", type=int, default=20_001)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.bf16_input, args.fp16_input):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = generate(
        args.bf16_input,
        args.fp16_input,
        sollya=args.sollya,
        samples=args.samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    outputs: list[tuple[Path, str]] = [(args.output, "Sollya comparison JSON")]
    if args.tex_output:
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        args.tex_output.write_text(
            "% Forward (ours / Sollya; maximum absolute error x1e-3)\n"
            + latex_table("forward", result)
            + "\n% Backward (ours / Sollya; maximum absolute error x1e-3)\n"
            + latex_table("backward", result)
        )
        outputs.append((args.tex_output, "Sollya comparison LaTeX"))
    receipt = args.receipt or args.output.with_suffix(
        args.output.suffix + ".receipt.json"
    )
    write_receipt(
        receipt,
        artifact_type="paper-sollya-comparison",
        generator=Path(__file__),
        inputs=[
            (args.bf16_input, "BF16 coefficient sweep"),
            (args.fp16_input, "FP16 coefficient sweep"),
        ],
        outputs=outputs,
        parameters={
            "sollya_executable": Path(args.sollya).name,
            "samples": args.samples,
            "degrees": [3, 4, 5, 6],
        },
        packages=("numpy",),
        notes=(
            "Auxiliary sensitivity analysis only; this is not the manuscript Table 2 producer.",
            "Sigmoid D3 uses the earlier sweep row and clamp 4.75, not the later deployed fit and clamp 6.0.",
            "The later tanh D4 device refit used by the table is absent from the sweep input.",
            "The auxiliary Sollya side fits the SiLU residual directly; the retained table composes both columns from their corresponding sigmoid coefficients.",
            "GELU and GELU' use the FP16 sweep and 11-bit Sollya coefficients here; the retained table uses a deployed BF16 header and 8-bit/BF16 Sollya control.",
            "Separately, the retained table producer has asymmetric coefficient rounding, uses host NumPy rather than device error, and lacks endpoint-constrained least-squares lineage for every selected row.",
            "Within this auxiliary comparison, monomials, clamps, precision budgets, and reconstruction are matched.",
        ),
    )
    print(f"Wrote {args.output}")
    if args.tex_output:
        print(f"Wrote {args.tex_output}")
    print(f"Wrote {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

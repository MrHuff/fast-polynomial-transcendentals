#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replot the paper's paired B1--B4 loss curves from local artifacts only.

The transformation reads the five scientific columns required for the figure.
It neither imports a service client nor copies unrelated CSV or provenance
fields into its output receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.rcParams["pdf.fonttype"] = 42

ROLES = ("baseline", "candidate")
REQUIRED_COLUMNS = ("case", "role", "curve_label", "tokens_seen", "loss")
EXPECTED_SMOOTHING_KIND = "causal-token-ewm"


@dataclass(frozen=True)
class CurveSpec:
    label: str


@dataclass(frozen=True)
class CaseSpec:
    title: str
    baseline: CurveSpec
    candidate: CurveSpec
    plot_max_tokens: float | None = None


CASES = {
    "B1": CaseSpec(
        title="B1: dense SiLU",
        baseline=CurveSpec(label="Native SiLU"),
        candidate=CurveSpec(label="D3 polynomial SiLU"),
    ),
    "B2": CaseSpec(
        title="B2: attention softcap",
        baseline=CurveSpec(label="SFU tanh"),
        candidate=CurveSpec(label="D4 polynomial tanh"),
        plot_max_tokens=100_270_080_000,
    ),
    "B3": CaseSpec(
        title="B3: sigmoid attention",
        baseline=CurveSpec(label="SFU sigmoid"),
        candidate=CurveSpec(label="D3/D4 polynomial sigmoid"),
    ),
    "B4": CaseSpec(
        title="B4: expert SiLU in routed SwiGLU",
        baseline=CurveSpec(label="Native expert SiLU"),
        candidate=CurveSpec(label="D3 polynomial routed-expert SiLU"),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plot_protocol(provenance_path: Path, data_path: Path) -> dict[str, float]:
    """Validate the source binding and return the retained plot parameters."""

    try:
        document = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("input provenance is not readable JSON") from error
    if not isinstance(document, dict):
        raise ValueError("input provenance must contain a JSON object")

    recorded_digest = document.get("artifact_sha256")
    if not isinstance(recorded_digest, str) or len(recorded_digest) != 64:
        raise ValueError("input provenance has no valid artifact SHA-256")
    if not all(character in "0123456789abcdef" for character in recorded_digest):
        raise ValueError("input provenance artifact SHA-256 must be lowercase hex")
    if sha256_file(data_path) != recorded_digest:
        raise ValueError("input CSV does not match its provenance SHA-256")

    smoothing = document.get("smoothing")
    if not isinstance(smoothing, dict):
        raise ValueError("input provenance has no smoothing protocol")
    if smoothing.get("kind") != EXPECTED_SMOOTHING_KIND:
        raise ValueError("input provenance declares an unsupported smoothing method")
    try:
        half_life = float(smoothing["half_life_tokens"])
        grid_step = float(smoothing["plot_grid_step_tokens"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("input provenance has invalid plot parameters") from error
    if not math.isfinite(half_life) or half_life <= 0:
        raise ValueError("smoothing half-life must be finite and positive")
    if not math.isfinite(grid_step) or grid_step <= 0:
        raise ValueError("plot grid step must be finite and positive")
    return {
        "smoothing_half_life_tokens": half_life,
        "plot_grid_step_tokens": grid_step,
    }


def add_token_ewm(history: pd.DataFrame, half_life_tokens: float) -> pd.DataFrame:
    """Add the paper's causal EWM, with decay measured in tokens."""

    history = history.sort_values("tokens_seen").reset_index(drop=True).copy()
    losses = history["loss"].to_numpy(dtype=float)
    tokens = history["tokens_seen"].to_numpy(dtype=float)
    smoothed = losses.copy()
    for index in range(1, len(losses)):
        delta_tokens = max(0.0, tokens[index] - tokens[index - 1])
        alpha = 1.0 - math.exp2(-delta_tokens / half_life_tokens)
        smoothed[index] = alpha * losses[index] + (1.0 - alpha) * smoothed[index - 1]
    history["smoothed_loss"] = smoothed
    return history


def load_histories(
    data_path: Path, half_life_tokens: float
) -> dict[str, dict[str, pd.DataFrame]]:
    """Read only the scientific columns used by the figure."""

    try:
        columns = tuple(pd.read_csv(data_path, nrows=0).columns)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError("input CSV is not readable") from error
    missing = set(REQUIRED_COLUMNS) - set(columns)
    if missing:
        raise ValueError(f"input CSV is missing columns: {sorted(missing)}")

    try:
        data = pd.read_csv(data_path, usecols=list(REQUIRED_COLUMNS))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError("input CSV is not readable") from error
    if data.empty or data[list(REQUIRED_COLUMNS)].isnull().any(axis=None):
        raise ValueError("input CSV contains empty or missing scientific values")

    for column in ("tokens_seen", "loss"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"input CSV contains non-finite {column} values")
    if (data["tokens_seen"] < 0).any():
        raise ValueError("input CSV contains negative token coordinates")

    observed_cases = set(data["case"].astype(str))
    if observed_cases != set(CASES):
        raise ValueError("input CSV must contain exactly cases B1--B4")
    observed_roles = set(data["role"].astype(str))
    if observed_roles != set(ROLES):
        raise ValueError("input CSV must contain exactly baseline and candidate roles")

    histories: dict[str, dict[str, pd.DataFrame]] = {}
    for case_name, case in CASES.items():
        histories[case_name] = {}
        for role in ROLES:
            history = data[(data["case"] == case_name) & (data["role"] == role)].copy()
            if history.empty:
                raise ValueError(f"input CSV has no rows for {case_name} {role}")
            if history["tokens_seen"].duplicated().any():
                raise ValueError(
                    f"input CSV has duplicate token coordinates for {case_name} {role}"
                )
            expected_label = getattr(case, role).label
            labels = set(history["curve_label"].astype(str))
            if labels != {expected_label}:
                raise ValueError(
                    f"input CSV has an unexpected {case_name} {role} label"
                )
            histories[case_name][role] = add_token_ewm(history, half_life_tokens)

        baseline_start = float(histories[case_name]["baseline"]["tokens_seen"].min())
        candidate_start = float(histories[case_name]["candidate"]["tokens_seen"].min())
        baseline_end = float(histories[case_name]["baseline"]["tokens_seen"].max())
        candidate_end = float(histories[case_name]["candidate"]["tokens_seen"].max())
        if max(baseline_start, candidate_start) > min(baseline_end, candidate_end):
            raise ValueError(f"input CSV has no common token horizon for {case_name}")
    return histories


def uniformly_thinned(history: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    """Limit raw plot density without changing the source artifact."""

    if len(history) <= max_rows:
        return history
    stride = math.ceil(len(history) / max_rows)
    thinned = history.iloc[::stride]
    if thinned.index[-1] != history.index[-1]:
        thinned = pd.concat([thinned, history.tail(1)])
    return thinned


def common_smoothed_grid(
    displayed_histories: dict[str, pd.DataFrame],
    common_tokens: float,
    grid_step_tokens: float,
) -> dict[str, pd.DataFrame]:
    """Interpolate already-smoothed arms onto identical token coordinates."""

    common_start = max(
        float(displayed_histories[role]["tokens_seen"].min()) for role in ROLES
    )
    grid = np.arange(common_start, common_tokens, grid_step_tokens, dtype=float)
    grid = np.unique(np.append(grid, common_tokens))
    result: dict[str, pd.DataFrame] = {}
    for role in ROLES:
        history = displayed_histories[role]
        result[role] = pd.DataFrame(
            {
                "tokens_seen": grid,
                "smoothed_loss": np.interp(
                    grid,
                    history["tokens_seen"].to_numpy(dtype=float),
                    history["smoothed_loss"].to_numpy(dtype=float),
                ),
            }
        )
    return result


def plot_cases(
    histories: dict[str, dict[str, pd.DataFrame]],
    output_path: Path,
    grid_step_tokens: float,
) -> dict[str, float]:
    """Render the four-panel paper figure and return each common horizon."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.6), constrained_layout=True)
    colors = {"baseline": "#54565a", "candidate": "#ff5f72"}
    common_horizons: dict[str, float] = {}

    for axis, (case_name, case) in zip(axes.flat, CASES.items(), strict=True):
        case_histories = histories[case_name]
        displayed_histories: dict[str, pd.DataFrame] = {}
        observed_tokens: dict[str, float] = {}
        for role in ROLES:
            history = case_histories[role]
            observed_tokens[role] = float(history["tokens_seen"].max())
            if case.plot_max_tokens is not None:
                observed_tokens[role] = min(
                    observed_tokens[role], float(case.plot_max_tokens)
                )
            displayed_histories[role] = history[
                history["tokens_seen"] <= observed_tokens[role]
            ]

        common_tokens = min(observed_tokens.values())
        common_horizons[case_name] = common_tokens
        paired_smoothed = common_smoothed_grid(
            case_histories, common_tokens, grid_step_tokens
        )
        for role in ROLES:
            curve = getattr(case, role)
            history = displayed_histories[role]
            paired = history[history["tokens_seen"] <= common_tokens]
            tail = history[history["tokens_seen"] > common_tokens]
            raw_for_plot = uniformly_thinned(history)
            axis.plot(
                raw_for_plot["tokens_seen"] / 1e9,
                raw_for_plot["loss"],
                color=colors[role],
                linewidth=0.45,
                alpha=0.15,
            )
            axis.plot(
                paired_smoothed[role]["tokens_seen"] / 1e9,
                paired_smoothed[role]["smoothed_loss"],
                color=colors[role],
                linewidth=1.45,
                label=curve.label,
            )
            if not tail.empty:
                tail_with_boundary = pd.concat(
                    [paired.tail(1), tail], ignore_index=True
                )
                axis.plot(
                    tail_with_boundary["tokens_seen"] / 1e9,
                    tail_with_boundary["smoothed_loss"],
                    color=colors[role],
                    linewidth=1.45,
                    linestyle=(0, (3, 2)),
                )

        zoom_start = max(0.0, common_tokens - 25e9)
        zoom_axis = axis.inset_axes([0.49, 0.12, 0.48, 0.36])
        zoom_losses: list[float] = []
        for role in ROLES:
            zoom = paired_smoothed[role][
                (paired_smoothed[role]["tokens_seen"] >= zoom_start)
                & (paired_smoothed[role]["tokens_seen"] <= common_tokens)
            ]
            zoom_axis.plot(
                zoom["tokens_seen"] / 1e9,
                zoom["smoothed_loss"],
                color=colors[role],
                linewidth=1.0,
            )
            zoom_losses.extend(zoom["smoothed_loss"].tolist())

        zoom_min = min(zoom_losses)
        zoom_max = max(zoom_losses)
        zoom_padding = max(0.02, 0.08 * (zoom_max - zoom_min))
        zoom_axis.set_xlim(zoom_start / 1e9, common_tokens / 1e9)
        zoom_axis.set_ylim(zoom_min - zoom_padding, zoom_max + zoom_padding)
        zoom_axis.set_title("Final 25 billion tokens", fontsize=7.2, pad=2.0)
        zoom_axis.grid(axis="y", color="#d8d8d8", linewidth=0.4, alpha=0.65)
        zoom_axis.tick_params(labelsize=6.8, length=2.5, pad=1.5)
        zoom_axis.locator_params(axis="x", nbins=2)
        zoom_axis.locator_params(axis="y", nbins=3)
        zoom_axis.set_facecolor("white")
        for spine in zoom_axis.spines.values():
            spine.set_color("#9a9a9a")
            spine.set_linewidth(0.5)

        axis.set_title(case.title, fontsize=10)
        axis.set_xlim(left=0, right=max(observed_tokens.values()) / 1e9)
        axis.grid(axis="y", color="#d8d8d8", linewidth=0.5, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
        axis.legend(frameon=False, fontsize=7.4, loc="upper right")
        axis.text(
            0.02,
            0.04,
            f"Common horizon:\n{common_tokens / 1e9:.3f} billion tokens",
            transform=axis.transAxes,
            fontsize=7.2,
            color="#54565a",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 1.2,
            },
        )

    for axis in axes[:, 0]:
        axis.set_ylabel("Training loss", fontsize=9)
    for axis in axes[-1, :]:
        axis.set_xlabel("Tokens seen (billions)", fontsize=9)
    fig.savefig(
        output_path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return common_horizons


def source_receipt(script_path: Path) -> dict[str, Any]:
    """Describe this transformation without retaining machine-local paths."""

    return {
        "script": "extras/paper/plot_paired_loss_curves.py",
        "script_sha256": sha256_file(script_path),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }


def write_output_provenance(
    receipt_path: Path,
    *,
    data_path: Path,
    input_provenance_path: Path,
    output_path: Path,
    protocol: dict[str, float],
    histories: dict[str, dict[str, pd.DataFrame]],
    common_horizons: dict[str, float],
) -> None:
    """Write a path-neutral receipt that never copies source metadata fields."""

    receipt = {
        "schema_version": 1,
        "artifact_type": "paired-loss-paper-figure",
        "scope": "offline deterministic transformation of retained local data",
        "inputs": {
            "csv_sha256": sha256_file(data_path),
            "csv_size_bytes": data_path.stat().st_size,
            "source_provenance_sha256": sha256_file(input_provenance_path),
            "source_hash_verified": True,
        },
        "output": {
            "figure_sha256": sha256_file(output_path),
            "figure_size_bytes": output_path.stat().st_size,
            "format": "pdf",
        },
        "protocol": {
            "smoothing_kind": EXPECTED_SMOOTHING_KIND,
            **protocol,
            "common_horizon_policy": "minimum displayed endpoint per paired case",
            "inset_window_tokens": 25_000_000_000.0,
            "b2_display_cap_tokens": CASES["B2"].plot_max_tokens,
            "raw_curve_max_rows": 2000,
        },
        "rows": {
            f"{case_name}.{role}": len(histories[case_name][role])
            for case_name in CASES
            for role in ROLES
        },
        "common_horizon_tokens": common_horizons,
        "transformation": source_receipt(Path(__file__).resolve()),
        "command": [
            "python",
            "extras/paper/plot_paired_loss_curves.py",
            "--data-path",
            "<input-csv>",
            "--input-provenance",
            "<input-provenance>",
            "--output-path",
            "<output-pdf>",
            "--output-provenance",
            "<output-provenance>",
        ],
        "network_access": False,
        "source_metadata_copied": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(
    data_path: Path,
    input_provenance_path: Path,
    output_path: Path,
    output_provenance_path: Path,
) -> None:
    resolved_paths = {
        path.resolve()
        for path in (
            data_path,
            input_provenance_path,
            output_path,
            output_provenance_path,
        )
    }
    if len(resolved_paths) != 4:
        raise ValueError("input and output paths must all be distinct")
    if output_path.suffix.lower() != ".pdf":
        raise ValueError("output path must use the .pdf extension")

    protocol = load_plot_protocol(input_provenance_path, data_path)
    histories = load_histories(data_path, protocol["smoothing_half_life_tokens"])
    common_horizons = plot_cases(
        histories, output_path, protocol["plot_grid_step_tokens"]
    )
    write_output_provenance(
        output_provenance_path,
        data_path=data_path,
        input_provenance_path=input_provenance_path,
        output_path=output_path,
        protocol=protocol,
        histories=histories,
        common_horizons=common_horizons,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--input-provenance", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--output-provenance", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(
        args.data_path,
        args.input_provenance,
        args.output_path,
        args.output_provenance,
    )
    print(f"Wrote {args.output_path}")
    print(f"Wrote {args.output_provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

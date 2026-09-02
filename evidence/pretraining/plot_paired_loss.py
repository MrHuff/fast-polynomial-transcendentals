#!/usr/bin/env python3
"""Replot the checked-in paired B1--B4 training-loss trajectories offline."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPOSITORY_ROOT / "evidence/report-data/b1_b4_paired_loss_curves.csv"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "evidence/figures/training/b1_b4_paired_loss_curves.pdf"
HALF_LIFE_TOKENS = 1_000_000_000.0
GRID_STEP_TOKENS = 50_000_000.0
CASES = {
    "B1": "Dense SiLU",
    "B2": "Attention softcap",
    "B3": "Sigmoid attention",
    "B4": "Routed-expert SwiGLU",
}
ROLE_COLORS = {"baseline": "#6A6D70", "candidate": "#C84553"}


def token_ewm(tokens: np.ndarray, losses: np.ndarray, half_life: float) -> np.ndarray:
    """Compute a causal exponentially weighted mean in token space."""
    if half_life <= 0:
        raise ValueError("half-life must be positive")
    output = losses.astype(float, copy=True)
    for index in range(1, len(output)):
        delta = max(0.0, float(tokens[index] - tokens[index - 1]))
        alpha = 1.0 - math.exp2(-delta / half_life)
        output[index] = alpha * losses[index] + (1.0 - alpha) * output[index - 1]
    return output


def normalized_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "n_tokens_seen": "tokens_seen",
        "loss_metrics/global_avg_loss": "loss",
    }
    frame = frame.rename(columns=aliases)
    required = {"case", "role", "tokens_seen", "loss"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("input is missing columns: " + ", ".join(sorted(missing)))
    frame = frame.loc[:, [column for column in frame.columns if column in required | {"curve_label"}]]
    frame["case"] = frame["case"].astype(str).str.upper()
    frame["role"] = frame["role"].astype(str).str.lower()
    frame["tokens_seen"] = pd.to_numeric(frame["tokens_seen"], errors="raise")
    frame["loss"] = pd.to_numeric(frame["loss"], errors="raise")
    return frame.dropna(subset=["tokens_seen", "loss"])


def plot(input_path: Path, output_path: Path, *, half_life: float) -> None:
    frame = normalized_columns(pd.read_csv(input_path))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.7), sharex=False)
    for axis, (case, title) in zip(axes.flat, CASES.items(), strict=True):
        case_frame = frame[frame["case"] == case]
        if case_frame.empty:
            raise ValueError(f"input has no rows for {case}")
        for role in ("baseline", "candidate"):
            curve = case_frame[case_frame["role"] == role].sort_values("tokens_seen")
            if curve.empty:
                raise ValueError(f"input has no {role} rows for {case}")
            tokens = curve["tokens_seen"].to_numpy(dtype=float)
            losses = curve["loss"].to_numpy(dtype=float)
            smooth = token_ewm(tokens, losses, half_life)
            label = (
                str(curve["curve_label"].iloc[0])
                if "curve_label" in curve and curve["curve_label"].notna().any()
                else role.capitalize()
            )
            color = ROLE_COLORS[role]
            axis.plot(tokens / 1e9, losses, color=color, alpha=0.13, linewidth=0.45)
            grid = np.arange(tokens.min(), tokens.max() + GRID_STEP_TOKENS, GRID_STEP_TOKENS)
            axis.plot(
                grid / 1e9,
                np.interp(grid, tokens, smooth),
                color=color,
                linewidth=1.6,
                label=label,
            )
        axis.set_title(f"{case}: {title}")
        axis.set_xlabel("Tokens (billions)")
        axis.set_ylabel("Training loss")
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--half-life-tokens", type=float, default=HALF_LIFE_TOKENS)
    args = parser.parse_args()
    plot(args.input, args.output, half_life=args.half_life_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

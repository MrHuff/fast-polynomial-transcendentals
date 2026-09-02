#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Verify the deployed B3 polynomial against its exact sigmoid target.

This is a source-bound arithmetic audit, not a replacement for the missing
coefficient fitter. It reads constants from the pinned FlashAttention-4
manifest and emulates the packed-BF16 PTX rounding sequence on a declared
score grid.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FA4_ROOT = REPOSITORY_ROOT / "flash-attention"
EXPECTED_FA4_REVISION = "38afdedda24b0bf26e6904d3bed7807c19a6906e"
FA4_GITLINK = Path("flash-attention")
MANIFEST_PATH = Path("flash_attn/cute/polynomial_manifest.py")
PTX_PATH = Path("flash_attn/cute/handwritten_spline_ptx.py")
MAX_GRID_POINTS = 1_000_001

_CONSTANT_NAMES = (
    "FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH",
    "FLASH_SIGMOID_DIRECT_CLAMP",
    "FLASH_SIGMOID_DIRECT_SPLIT",
    "FLASH_SIGMOID_DIRECT_MIDPOINTS",
    "FLASH_SIGMOID_DIRECT_D3_COEFFS",
    "FLASH_SIGMOID_DIRECT_GRAD_D4_FACTOR",
)


class AuditError(RuntimeError):
    """Raised when the audit cannot establish its source or arithmetic inputs."""


@dataclass(frozen=True)
class DirectB3Spec:
    sequence_length: int
    clamp: float
    split: float
    midpoints: tuple[float, float]
    forward_rows: tuple[tuple[float, ...], tuple[float, ...]]
    gradient_factors: tuple[tuple[float, float], tuple[float, float]]

    @property
    def deployed_forward_rows(self) -> tuple[tuple[float, ...], ...]:
        scale = 1.0 / self.sequence_length
        return tuple(
            tuple(round_bf16(coefficient * scale) for coefficient in row)
            for row in self.forward_rows
        )

    @property
    def deployed_gradient_factors(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(round_bf16(coefficient) for coefficient in row)
            for row in self.gradient_factors
        )


@dataclass(frozen=True)
class DeployedValues:
    input_bf16: float
    clamped_bf16: float
    row_index: int
    forward: float
    derivative: float
    derivative_factor: float


@dataclass
class ErrorAccumulator:
    count: int = 0
    absolute_sum: float = 0.0
    square_sum: float = 0.0
    max_absolute: float = -1.0
    max_record: dict[str, float | int] | None = None

    def add(
        self,
        *,
        grid_index: int,
        declared_score: float,
        input_bf16: float,
        deployed: float,
        exact: float,
    ) -> None:
        error = deployed - exact
        absolute = abs(error)
        self.count += 1
        self.absolute_sum += absolute
        self.square_sum += error * error
        if absolute > self.max_absolute:
            self.max_absolute = absolute
            self.max_record = {
                "grid_index": grid_index,
                "declared_score": declared_score,
                "input_bf16": input_bf16,
                "deployed": deployed,
                "exact": exact,
                "signed_error": error,
            }

    def result(self) -> dict[str, Any]:
        if not self.count or self.max_record is None:
            raise AuditError("cannot summarize an empty grid")
        return {
            "sample_count": self.count,
            "max_abs_error": self.max_absolute,
            "mean_abs_error": self.absolute_sum / self.count,
            "rmse": math.sqrt(self.square_sum / self.count),
            "maximum": self.max_record,
        }


def _git_text(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuditError(
            f"failed to inspect Git repository {repository}: {' '.join(arguments)}"
        ) from error
    return completed.stdout.strip()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuditError(
            f"failed to inspect Git repository {repository}: {' '.join(arguments)}"
        ) from error
    return completed.stdout


def _root_gitlink_revision(repository_root: Path) -> str:
    listing = _git_text(
        repository_root,
        "ls-tree",
        "HEAD",
        "--",
        FA4_GITLINK.as_posix(),
    )
    fields = listing.split()
    if len(fields) < 4 or fields[0] != "160000" or fields[1] != "commit":
        raise AuditError("top-level HEAD does not contain the FlashAttention Gitlink")
    return fields[2]


def _repository_state(repository: Path) -> tuple[str | None, bool | None, int | None]:
    try:
        revision = _git_text(repository, "rev-parse", "HEAD")
        status = _git_text(
            repository,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
    except AuditError:
        return None, None, None
    entries = tuple(entry for entry in status.split("\0") if entry)
    untracked = sum(entry.startswith("?? ") for entry in entries)
    return revision, bool(entries), untracked


def verify_fa4_source(
    repository_root: Path,
    fa4_root: Path,
    *,
    expected_revision: str = EXPECTED_FA4_REVISION,
) -> dict[str, Any]:
    """Fail closed unless the two audited FA4 files match the pinned commit."""

    gitlink_revision = _root_gitlink_revision(repository_root)
    observed_revision = _git_text(fa4_root, "rev-parse", "HEAD")
    if gitlink_revision != expected_revision:
        raise AuditError(
            "top-level FlashAttention pin mismatch: "
            f"expected {expected_revision}, found {gitlink_revision}"
        )
    if observed_revision != expected_revision:
        raise AuditError(
            "checked-out FlashAttention revision mismatch: "
            f"expected {expected_revision}, found {observed_revision}"
        )

    sha256: dict[str, str] = {}
    git_blobs: dict[str, str] = {}
    for relative_path in (MANIFEST_PATH, PTX_PATH):
        working_path = fa4_root / relative_path
        if not working_path.is_file():
            raise AuditError(f"missing pinned FA4 source file: {working_path}")
        working_content = working_path.read_bytes()
        committed_content = _git_bytes(
            fa4_root,
            "show",
            f"{expected_revision}:{relative_path.as_posix()}",
        )
        if working_content != committed_content:
            raise AuditError(
                f"FA4 source differs from pinned commit: {relative_path.as_posix()}"
            )
        public_path = (FA4_GITLINK / relative_path).as_posix()
        sha256[public_path] = hashlib.sha256(working_content).hexdigest()
        git_blobs[public_path] = _git_text(
            fa4_root,
            "rev-parse",
            f"{expected_revision}:{relative_path.as_posix()}",
        )

    return {
        "expected_revision": expected_revision,
        "gitlink_revision": gitlink_revision,
        "observed_revision": observed_revision,
        "audited_files_match_commit": True,
        "git_blobs": git_blobs,
        "sha256": sha256,
    }


def _literal_assignments(source: str) -> dict[str, Any]:
    tree = ast.parse(source, filename=MANIFEST_PATH.as_posix())
    assignments: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in _CONSTANT_NAMES:
            continue
        try:
            assignments[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError) as error:
            raise AuditError(f"FA4 constant {target.id} is not a literal") from error
    missing = sorted(set(_CONSTANT_NAMES) - assignments.keys())
    if missing:
        raise AuditError(f"missing direct B3 constants in FA4 manifest: {missing}")
    return assignments


def _float_row(value: Any, *, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise AuditError(f"{name} must contain {length} coefficients")
    row = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in row):
        raise AuditError(f"{name} contains a non-finite coefficient")
    return row


def read_direct_b3_spec(manifest: Path) -> DirectB3Spec:
    assignments = _literal_assignments(manifest.read_text(encoding="utf-8"))
    sequence_length = assignments["FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH"]
    if not isinstance(sequence_length, int) or sequence_length != 4096:
        raise AuditError("the deployed direct B3 audit requires sequence length 4096")
    clamp = float(assignments["FLASH_SIGMOID_DIRECT_CLAMP"])
    split = float(assignments["FLASH_SIGMOID_DIRECT_SPLIT"])
    midpoints = _float_row(
        assignments["FLASH_SIGMOID_DIRECT_MIDPOINTS"],
        name="FLASH_SIGMOID_DIRECT_MIDPOINTS",
        length=2,
    )
    if midpoints != (0.0, 0.0):
        raise AuditError("the pinned fused D3/D4 path assumes zero midpoints")

    forward_source = assignments["FLASH_SIGMOID_DIRECT_D3_COEFFS"]
    factor_source = assignments["FLASH_SIGMOID_DIRECT_GRAD_D4_FACTOR"]
    if not isinstance(forward_source, (tuple, list)) or len(forward_source) != 2:
        raise AuditError("FLASH_SIGMOID_DIRECT_D3_COEFFS must contain two rows")
    if not isinstance(factor_source, (tuple, list)) or len(factor_source) != 2:
        raise AuditError("FLASH_SIGMOID_DIRECT_GRAD_D4_FACTOR must contain two rows")

    return DirectB3Spec(
        sequence_length=sequence_length,
        clamp=clamp,
        split=split,
        midpoints=(midpoints[0], midpoints[1]),
        forward_rows=(
            _float_row(forward_source[0], name="D3 row 0", length=4),
            _float_row(forward_source[1], name="D3 row 1", length=4),
        ),
        gradient_factors=(
            _float_row(factor_source[0], name="D4 factor row 0", length=2),
            _float_row(factor_source[1], name="D4 factor row 1", length=2),
        ),
    )


def _float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _bits_float32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def round_bf16(value: float) -> float:
    """Round a scalar through float32 to BF16, ties to even."""

    bits = _float32_bits(value)
    if bits & 0x7F800000 == 0x7F800000:
        return _bits_float32(bits)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    return _bits_float32(bits & 0xFFFF0000)


def bf16_fma(left: float, right: float, addend: float) -> float:
    """Emulate one `fma.rn.bf16x2` lane with a single BF16 rounding."""

    return round_bf16(left * right + addend)


def bf16_multiply(left: float, right: float) -> float:
    """Emulate one `mul.rn.bf16x2` lane."""

    return round_bf16(left * right)


def evaluate_deployed(score: float, spec: DirectB3Spec) -> DeployedValues:
    """Emulate one lane of the fused direct-D3/factored-D4 inline PTX."""

    input_bf16 = round_bf16(score)
    split_bf16 = round_bf16(spec.split)
    clamp_bf16 = round_bf16(spec.clamp)
    row_index = int(input_bf16 >= split_bf16)
    clamped = min(max(input_bf16, -clamp_bf16), clamp_bf16)

    coefficients = spec.deployed_forward_rows[row_index]
    forward = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        forward = bf16_fma(clamped, forward, coefficient)
    forward = max(round_bf16(0.0), forward)

    intercept, slope = spec.deployed_gradient_factors[row_index]
    derivative_factor = bf16_fma(clamped, slope, intercept)
    derivative = bf16_multiply(forward, derivative_factor)
    return DeployedValues(
        input_bf16=input_bf16,
        clamped_bf16=clamped,
        row_index=row_index,
        forward=forward,
        derivative=derivative,
        derivative_factor=derivative_factor,
    )


def exact_targets(score: float, sequence_length: int) -> tuple[float, float]:
    shifted = score - math.log(sequence_length)
    if shifted >= 0.0:
        negative_exp = math.exp(-shifted)
        probability = 1.0 / (1.0 + negative_exp)
    else:
        positive_exp = math.exp(shifted)
        probability = positive_exp / (1.0 + positive_exp)
    return probability, probability * (1.0 - probability)


def _constants_document(spec: DirectB3Spec) -> dict[str, Any]:
    return {
        "sequence_length": spec.sequence_length,
        "clamp": spec.clamp,
        "split": spec.split,
        "midpoints": list(spec.midpoints),
        "manifest_d3_rows_before_1_over_n_scaling": [
            list(row) for row in spec.forward_rows
        ],
        "deployed_d3_rows_after_1_over_n_and_bf16_rounding": [
            list(row) for row in spec.deployed_forward_rows
        ],
        "manifest_factored_d4_rows_intercept_then_slope": [
            list(row) for row in spec.gradient_factors
        ],
        "deployed_factored_d4_rows_bf16": [
            list(row) for row in spec.deployed_gradient_factors
        ],
    }


def audit_grid(
    spec: DirectB3Spec,
    *,
    grid_min: float,
    grid_max: float,
    grid_points: int,
) -> dict[str, Any]:
    if not math.isfinite(grid_min) or not math.isfinite(grid_max):
        raise AuditError("grid bounds must be finite")
    if grid_min >= grid_max:
        raise AuditError("grid minimum must be smaller than grid maximum")
    if grid_points < 2 or grid_points > MAX_GRID_POINTS:
        raise AuditError(f"grid points must be between 2 and {MAX_GRID_POINTS:,}")

    declared_forward = ErrorAccumulator()
    declared_derivative = ErrorAccumulator()
    rounded_forward = ErrorAccumulator()
    rounded_derivative = ErrorAccumulator()
    unique_inputs: set[int] = set()
    zero_forward_count = 0
    step = (grid_max - grid_min) / (grid_points - 1)

    for index in range(grid_points):
        score = grid_max if index == grid_points - 1 else grid_min + index * step
        deployed = evaluate_deployed(score, spec)
        unique_inputs.add(_float32_bits(deployed.input_bf16))
        zero_forward_count += deployed.forward == 0.0
        exact_forward, exact_derivative = exact_targets(score, spec.sequence_length)
        rounded_exact_forward, rounded_exact_derivative = exact_targets(
            deployed.input_bf16,
            spec.sequence_length,
        )
        shared = {
            "grid_index": index,
            "declared_score": score,
            "input_bf16": deployed.input_bf16,
        }
        declared_forward.add(
            **shared,
            deployed=deployed.forward,
            exact=exact_forward,
        )
        declared_derivative.add(
            **shared,
            deployed=deployed.derivative,
            exact=exact_derivative,
        )
        rounded_forward.add(
            **shared,
            deployed=deployed.forward,
            exact=rounded_exact_forward,
        )
        rounded_derivative.add(
            **shared,
            deployed=deployed.derivative,
            exact=rounded_exact_derivative,
        )

    boundary_scores = sorted(
        {grid_min, -spec.clamp, 0.0, spec.split, spec.clamp, grid_max}
    )
    boundary_samples = []
    for score in boundary_scores:
        deployed = evaluate_deployed(score, spec)
        exact_forward, exact_derivative = exact_targets(score, spec.sequence_length)
        boundary_samples.append(
            {
                "declared_score": score,
                "input_bf16": deployed.input_bf16,
                "clamped_bf16": deployed.clamped_bf16,
                "row_index": deployed.row_index,
                "deployed_forward": deployed.forward,
                "exact_forward": exact_forward,
                "deployed_derivative": deployed.derivative,
                "exact_derivative": exact_derivative,
                "deployed_derivative_factor": deployed.derivative_factor,
            }
        )

    return {
        "grid": {
            "coordinate": "raw attention score before deployed BF16 conversion",
            "minimum": grid_min,
            "maximum": grid_max,
            "points": grid_points,
            "inclusive_endpoints": True,
            "uniform_step": step,
            "unique_bf16_inputs": len(unique_inputs),
            "zero_forward_outputs": zero_forward_count,
        },
        "metrics": {
            "exact_target_at_declared_grid_score": {
                "forward": declared_forward.result(),
                "derivative": declared_derivative.result(),
            },
            "exact_target_at_deployed_bf16_input": {
                "forward": rounded_forward.result(),
                "derivative": rounded_derivative.result(),
            },
        },
        "boundary_samples": boundary_samples,
    }


def build_document(
    *,
    repository_root: Path,
    fa4_root: Path,
    grid_min: float,
    grid_max: float,
    grid_points: int,
) -> dict[str, Any]:
    source_verification = verify_fa4_source(repository_root, fa4_root)
    spec = read_direct_b3_spec(fa4_root / MANIFEST_PATH)
    grid_result = audit_grid(
        spec,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_points=grid_points,
    )
    revision, dirty, untracked_files = _repository_state(repository_root)
    input_sha256 = dict(source_verification["sha256"])
    input_sha256["scripts/audit_b3_deployed_fit.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return {
        "schema_version": 1,
        "experiment": {
            "id": "b3-deployed-fit-audit",
            "description": (
                "CPU verification of the pinned B3 direct-D3 forward and "
                "factored-D4 derivative arithmetic"
            ),
            "provenance_class": "new-measurement",
            "command": [
                "python",
                "scripts/audit_b3_deployed_fit.py",
                "--grid-min",
                str(grid_min),
                "--grid-max",
                str(grid_max),
                "--grid-points",
                str(grid_points),
            ],
            "evidence_role": "source-bound-verification",
            "verification_only": True,
            "original_fitter_recovered": False,
            "claim_boundary": (
                "This audit verifies deployed constants and arithmetic. It is "
                "not the missing original fitter and makes no assumption about "
                "the original weighting, samples, or optimizer."
            ),
        },
        "source": {
            "repository": "MrHuff/fast-polynomial-transcendentals",
            "revision": revision,
            "dirty": dirty,
            "untracked_files": untracked_files,
            "external_components": {
                "flash-attention": source_verification["observed_revision"],
            },
            "input_sha256": input_sha256,
            "flash_attention": source_verification,
        },
        "environment": {
            "device_name": "CPU arithmetic emulator",
            "gpu": None,
            "python": platform.python_version(),
        },
        "measurement": {
            "dtype": "deployed packed-BF16 lane emulation",
            "iterations": grid_points,
            "rounding_sequence": [
                "float32 input to BF16, round-to-nearest-even",
                "manifest D3 coefficients scaled by 1/4096 then rounded to BF16",
                "three fused multiply-add stages, each rounded once to BF16",
                "nonnegative clamp in BF16",
                "factored-D4 affine fused multiply-add rounded to BF16",
                "forward times derivative factor rounded to BF16",
            ],
            "reference": {
                "forward": "sigmoid(score - log(4096))",
                "derivative": "p * (1 - p)",
            },
        },
        "results": {
            "constants": _constants_document(spec),
            **grid_result,
        },
    }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fa4-root",
        type=Path,
        default=DEFAULT_FA4_ROOT,
        help="Pinned FlashAttention-4 checkout (default: ./flash-attention).",
    )
    parser.add_argument("--grid-min", type=float, default=-6.0)
    parser.add_argument("--grid-max", type=float, default=6.0)
    parser.add_argument(
        "--grid-points",
        type=int,
        default=24_577,
        help=f"Inclusive uniform grid size, at most {MAX_GRID_POINTS:,}.",
    )
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    try:
        document = build_document(
            repository_root=args.repository_root.resolve(),
            fa4_root=args.fa4_root.resolve(),
            grid_min=args.grid_min,
            grid_max=args.grid_max,
            grid_points=args.grid_points,
        )
    except AuditError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

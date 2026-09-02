# Copyright 2026 Robert Hu
# SPDX-License-Identifier: Apache-2.0
"""Check manuscript table values against released, credential-free evidence.

The checker never reads or emits run identifiers. It derives only the fields
used by the quantitative paper tables, applies the manuscript's stated
rounding, and compares them with ``paper_table_claims.json``. For Table 2 this
is a numeric source-to-typeset check only; it does not validate the table's
coefficient-rounding, device-arithmetic, or fitter-lineage semantics. See
``FUNCTION_TABLE_REVIEW.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .artifact_utils import artifact_record, sha256_file
except ImportError:  # Direct script execution.
    from artifact_utils import artifact_record, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLAIMS = Path(__file__).with_name("paper_table_claims.json")
DEFAULT_FUNCTION_LINEAGE = Path(__file__).with_name("function_table_lineage.json")
DEFAULT_EVIDENCE = REPOSITORY_ROOT / "evidence/report-data"
DEFAULT_FUNCTION_COMPARISON = (
    REPOSITORY_ROOT
    / "autonumerics_zero/cuda_benchmarks/analysis_results/sollya_device_bf16.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs/paper/table_audit.json"

RETAINED_FUNCTION_ARTIFACT_SHA256 = (
    "6c51affbb0ba593e21ad456c967bc9afe4d486d0a3e9eb9370ece9710c30b918"
)
AUDITED_GENERATOR_SNAPSHOT_SHA256 = (
    "7c3c4c053326b5d29076965616ec15edfd802476241e5bf1404f8f747fc4fbe2"
)
CURRENT_HEADER_SNAPSHOT_SHA256 = (
    "3207d0691ecaedbbfb64e83efb79929013a015609a2485a7ec6cb20dbe275950"
)
EXPECTED_FUNCTION_MEASUREMENT = {
    "metric": "maximum absolute error",
    "evaluation": "host NumPy real-arithmetic Horner reconstruction",
    "grid": "20,001 uniformly spaced points on each row's closed interval [-Lc, Lc]",
    "current_coefficients": (
        "decimal literals parsed from the deployed CUDA header and evaluated "
        "directly without BF16 pre-rounding"
    ),
    "sollya_coefficients": (
        "Sollya fpminimax coefficients constrained to 8-bit precision and cast "
        "to BF16 before host evaluation"
    ),
    "intermediate_rounding": (
        "none; target-precision intermediate rounding is not replayed"
    ),
    "device_measurement": (
        "not a device measurement; this is a host-side coefficient control"
    ),
}
REQUIRED_FUNCTION_REVIEW_IDS = {
    "asymmetric-coefficient-rounding",
    "current-fit-method-label-not-established",
}
REQUIRED_PROVENANCE_LIMITATION_IDS = {"original-producing-environment-not-captured"}


def rounded_equal(observed: float, expected: float, decimals: int) -> bool:
    return f"{observed:.{decimals}f}" == f"{expected:.{decimals}f}"


def bound_manuscript_source(claims_document: dict[str, Any]) -> Path:
    """Return the repository-local manuscript source bound by the claim map."""

    source = claims_document.get("manuscript_source")
    if not isinstance(source, dict):
        raise ValueError("claim map has no manuscript_source binding")
    raw_path = source.get("path")
    expected_sha256 = source.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("claim map manuscript path must be a non-empty string")
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("claim map manuscript path must stay inside the repository")
    manuscript_path = (REPOSITORY_ROOT / relative_path).resolve()
    if not manuscript_path.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("claim map manuscript path resolves outside the repository")
    if not manuscript_path.is_file():
        raise ValueError("claim map manuscript source is missing")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("claim map manuscript SHA-256 is invalid")
    if sha256_file(manuscript_path) != expected_sha256:
        raise ValueError("claim map does not match the packaged manuscript source")
    return manuscript_path


def compare_rows(
    expected_rows: list[dict[str, Any]],
    observed_rows: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
    fields: dict[str, int],
) -> dict[str, Any]:
    expected = {tuple(row[key] for key in keys): row for row in expected_rows}
    observed = {tuple(row[key] for key in keys): row for row in observed_rows}
    mismatches: list[dict[str, Any]] = []
    if set(expected) != set(observed):
        mismatches.append(
            {
                "kind": "row-set",
                "missing": [list(key) for key in sorted(set(expected) - set(observed))],
                "unexpected": [
                    list(key) for key in sorted(set(observed) - set(expected))
                ],
            }
        )
    for key in sorted(set(expected) & set(observed)):
        for field, decimals in fields.items():
            expected_value = float(expected[key][field])
            observed_value = float(observed[key][field])
            if not rounded_equal(observed_value, expected_value, decimals):
                mismatches.append(
                    {
                        "kind": "value",
                        "row": list(key),
                        "field": field,
                        "decimals": decimals,
                        "expected": expected_value,
                        "observed": observed_value,
                    }
                )
    return {
        "status": "pass" if not mismatches else "mismatch",
        "mismatches": mismatches,
    }


def integration_rows(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text())
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise ValueError("component evidence has no rows array")
    result = []
    for row in rows:
        result.append(
            {
                "case": str(row["case"]),
                "forward": float(row["forward"]["speedup"]),
                "backward": float(row["backward"]["speedup"]),
            }
        )
    return result


def model_timing_rows(full_model_path: Path, b4_path: Path) -> list[dict[str, Any]]:
    full_model = json.loads(full_model_path.read_text())
    comparisons = full_model.get("comparisons", {})
    result = []
    for case in ("B1", "B2", "B3"):
        row = comparisons[case]
        result.append(
            {
                "case": case,
                "forward": float(row["forward_speedup"]),
                "backward": float(row["backward_speedup"]),
                "gpu_step": float(row["gpu_step_speedup"]),
                "throughput": float(row["token_throughput_speedup"]),
            }
        )
    b4 = json.loads(b4_path.read_text())["results_steps_20_100"][
        "speedup_ratio_of_medians"
    ]
    result.append(
        {
            "case": "B4",
            "forward": float(b4["forward"]),
            "backward": float(b4["backward"]),
            "gpu_step": float(b4["gpu_step"]),
            "throughput": float(b4["tokens_per_s"]),
        }
    )
    return result


def downstream_rows(path: Path) -> list[dict[str, Any]]:
    allowed_fields = {
        "model",
        "fit",
        "quality_mean_delta_pp",
        "wikitext_word_perplexity",
        "prefill_speedup",
        "decode_speedup",
    }
    with path.open(newline="") as handle:
        raw_rows = [
            {field: row[field] for field in allowed_fields}
            for row in csv.DictReader(handle)
        ]
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in raw_rows:
        model = row["model"]
        fit = row["fit"]
        if fit in grouped[model]:
            raise ValueError(f"duplicate downstream row for {model!r}, {fit!r}")
        grouped[model][fit] = row
    output = []
    for model, fits in grouped.items():
        if set(fits) != {"native", "current", "sollya"}:
            raise ValueError(
                f"downstream evidence has incomplete fit set for {model!r}"
            )
        native_perplexity = float(fits["native"]["wikitext_word_perplexity"])
        for fit in ("current", "sollya"):
            row = fits[fit]
            output.append(
                {
                    "model": model,
                    "fit": fit,
                    "mean_accuracy_change_pp": float(row["quality_mean_delta_pp"]),
                    "native_perplexity": native_perplexity,
                    "variant_perplexity": float(row["wikitext_word_perplexity"]),
                    "prefill": float(row["prefill_speedup"]),
                    "decode": float(row["decode_speedup"]),
                }
            )
    return output


def token_ewm(
    tokens: list[float], losses: list[float], half_life: float
) -> list[float]:
    if half_life <= 0:
        raise ValueError("smoothing half-life must be positive")
    result = list(losses)
    for index in range(1, len(result)):
        delta = max(0.0, tokens[index] - tokens[index - 1])
        alpha = 1.0 - math.exp2(-delta / half_life)
        result[index] = alpha * losses[index] + (1.0 - alpha) * result[index - 1]
    return result


def interpolate(tokens: list[float], values: list[float], coordinate: float) -> float:
    if coordinate < tokens[0] or coordinate > tokens[-1]:
        raise ValueError("interpolation coordinate lies outside source range")
    for index, token in enumerate(tokens):
        if token == coordinate:
            return values[index]
        if token > coordinate:
            lower_token = tokens[index - 1]
            weight = (coordinate - lower_token) / (token - lower_token)
            return values[index - 1] + weight * (values[index] - values[index - 1])
    return values[-1]


def pretraining_rows(path: Path, half_life: float) -> list[dict[str, Any]]:
    curves: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case", "role", "tokens_seen", "loss"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("paired-loss CSV lacks required columns")
        for row in reader:
            case = row["case"].upper()
            role = row["role"].lower()
            if case in {"B1", "B2", "B3", "B4"} and role in {"baseline", "candidate"}:
                curves[(case, role)].append(
                    (float(row["tokens_seen"]), float(row["loss"]))
                )
    result = []
    for case in ("B1", "B2", "B3", "B4"):
        processed = {}
        for role in ("baseline", "candidate"):
            points = sorted(curves[(case, role)])
            if not points or len({token for token, _ in points}) != len(points):
                raise ValueError(
                    f"missing or duplicate paired-loss coordinates for {case} {role}"
                )
            tokens = [point[0] for point in points]
            losses = [point[1] for point in points]
            processed[role] = (tokens, token_ewm(tokens, losses, half_life))
        common = min(processed[role][0][-1] for role in ("baseline", "candidate"))
        native = interpolate(*processed["baseline"], common)
        polynomial = interpolate(*processed["candidate"], common)
        result.append(
            {
                "case": case,
                "horizon_billions": common / 1e9,
                "native": native,
                "polynomial": polynomial,
                "change": polynomial - native,
            }
        )
    return result


def function_rows(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text())
    if "families" in document:
        mappings = {
            "sigmoid_fwd": ("forward", "sigmoid"),
            "tanh_fwd": ("forward", "tanh"),
            "swish_fwd": ("forward", "swish"),
            "gelu_fwd": ("forward", "gelu"),
            "sigmoid_bwd": ("backward", "sigmoid'"),
            "tanh_bwd": ("backward", "tanh'"),
            "swish_bwd": ("backward", "swish'"),
            "gelu_bwd": ("backward", "gelu'"),
        }
        output = []
        for source_family, (split, family) in mappings.items():
            for degree, row in document["families"][source_family].items():
                output.append(
                    {
                        "split": split,
                        "family": family,
                        "degree": degree,
                        "ours": float(row["current_max_error"]) * 1000.0,
                        "sollya": float(row["sollya_max_error"]) * 1000.0,
                    }
                )
        return output

    results = document.get("results", document)
    output = []
    for split, families in results.items():
        if split not in {"forward", "backward"}:
            continue
        for family, degrees in families.items():
            for degree, row in degrees.items():
                output.append(
                    {
                        "split": split,
                        "family": family,
                        "degree": degree,
                        "ours": float(row["ours"]) * 1000.0,
                        "sollya": float(row["sollya"]) * 1000.0,
                    }
                )
    return output


def _object_with_ids(
    value: object,
    *,
    label: str,
    required_ids: set[str],
    required_fields: tuple[str, ...],
    errors: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        errors.append(f"{label} must be an array of objects")
        return []
    rows = list(value)
    ids = [row.get("id") for row in rows]
    if not all(isinstance(item, str) and item for item in ids):
        errors.append(f"{label} entries must have non-empty string ids")
        return rows
    if len(ids) != len(set(ids)):
        errors.append(f"{label} ids must be unique")
    missing = required_ids - set(ids)
    if missing:
        errors.append(f"{label} is missing required ids: {sorted(missing)}")
    for index, row in enumerate(rows):
        for field in required_fields:
            if not isinstance(row.get(field), str) or not row[field].strip():
                errors.append(f"{label}[{index}].{field} must be a non-empty string")
    return rows


def validate_function_lineage(
    lineage_path: Path,
    comparison_path: Path,
) -> dict[str, Any]:
    """Validate retained or freshly generated function-table semantics.

    The legacy retained artifact is accepted only at its pinned SHA and takes
    its measurement semantics from the sidecar. A different artifact must
    carry an equivalent top-level ``measurement`` object itself.
    """
    errors: list[str] = []
    if not lineage_path.is_file():
        return {
            "status": "invalid",
            "mode": None,
            "errors": ["function lineage sidecar is missing"],
            "review_required": [],
            "provenance_limitations": [],
        }
    try:
        lineage = json.loads(lineage_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "invalid",
            "mode": None,
            "errors": [f"function lineage sidecar is not valid JSON: {error}"],
            "review_required": [],
            "provenance_limitations": [],
        }
    if not isinstance(lineage, dict):
        errors.append("function lineage sidecar must be a JSON object")
        lineage = {}
    if lineage.get("schema_version") != 1:
        errors.append("function lineage schema_version must be 1")
    if lineage.get("artifact_type") != "manuscript-function-table-lineage":
        errors.append("function lineage artifact_type is invalid")
    if lineage.get("measurement") != EXPECTED_FUNCTION_MEASUREMENT:
        errors.append("function lineage measurement semantics are missing or incorrect")

    retained = lineage.get("retained_artifact")
    if not isinstance(retained, dict):
        errors.append("function lineage retained_artifact must be an object")
        retained = {}
    if retained.get("sha256") != RETAINED_FUNCTION_ARTIFACT_SHA256:
        errors.append("function lineage retained artifact SHA-256 is incorrect")
    if retained.get("path") != (
        "autonumerics_zero/cuda_benchmarks/analysis_results/sollya_device_bf16.json"
    ):
        errors.append("function lineage retained artifact path is incorrect")

    generator = lineage.get("audited_generator_snapshot")
    if not isinstance(generator, dict):
        errors.append("function lineage audited_generator_snapshot must be an object")
        generator = {}
    if generator.get("sha256") != AUDITED_GENERATOR_SNAPSHOT_SHA256:
        errors.append("function lineage audited generator SHA-256 is incorrect")
    if generator.get("path") != (
        "autonumerics_zero/spline_ops/generate_sollya_structs_bf16.py"
    ):
        errors.append("function lineage audited generator path is incorrect")

    header = lineage.get("current_header_snapshot")
    if not isinstance(header, dict):
        errors.append("function lineage current_header_snapshot must be an object")
        header = {}
    if header.get("sha256") != CURRENT_HEADER_SNAPSHOT_SHA256:
        errors.append("function lineage current-header SHA-256 is incorrect")
    if header.get("path") != (
        "autonumerics_zero/spline_ops/spline_structs_odd_bf16.cuh"
    ):
        errors.append("function lineage current-header path is incorrect")

    review_required = _object_with_ids(
        lineage.get("review_required"),
        label="function lineage review_required",
        required_ids=REQUIRED_FUNCTION_REVIEW_IDS,
        required_fields=("category", "observed", "manuscript_risk", "required_review"),
        errors=errors,
    )
    limitations = _object_with_ids(
        lineage.get("provenance_limitations"),
        label="function lineage provenance_limitations",
        required_ids=REQUIRED_PROVENANCE_LIMITATION_IDS,
        required_fields=("detail",),
        errors=errors,
    )

    comparison_sha = sha256_file(comparison_path)
    if comparison_sha == RETAINED_FUNCTION_ARTIFACT_SHA256:
        mode = "retained-artifact-sidecar"
    else:
        mode = "generated-artifact-embedded-measurement"
        try:
            comparison = json.loads(comparison_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"function comparison is not valid JSON: {error}")
            comparison = {}
        if not isinstance(comparison, dict) or comparison.get("measurement") != (
            EXPECTED_FUNCTION_MEASUREMENT
        ):
            errors.append(
                "non-retained function comparison lacks equivalent embedded measurement semantics"
            )

    return {
        "status": "invalid" if errors else "pass",
        "mode": mode,
        "comparison_sha256": comparison_sha,
        "errors": errors,
        "review_required": review_required,
        "provenance_limitations": limitations,
    }


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    claims_document = json.loads(args.claims.read_text())
    manuscript_source = bound_manuscript_source(claims_document)
    claims = claims_document["tables"]
    function_lineage = getattr(args, "function_lineage", DEFAULT_FUNCTION_LINEAGE)
    allow_review_required = bool(getattr(args, "allow_review_required", False))
    sources = {
        "function": args.function_comparison,
        "integration": args.evidence_dir
        / "b1_b4_100_iteration_phase_probes_gb200.json",
        "full_model": args.evidence_dir / "b1_b4_full_model_100step_gb200.json",
        "b4_model": args.evidence_dir / "b4_27a4b_local_100step_gb200.json",
        "downstream": args.evidence_dir / "sollya_eval_results.csv",
        "pretraining": args.evidence_dir / "b1_b4_paired_loss_curves.csv",
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    tables: dict[str, Any] = {}
    tables["integration-summary"] = compare_rows(
        claims["integration-summary"]["rows"],
        integration_rows(sources["integration"]),
        keys=("case",),
        fields={"forward": 3, "backward": 3},
    )
    tables["model-timing-summary"] = compare_rows(
        claims["model-timing-summary"]["rows"],
        model_timing_rows(sources["full_model"], sources["b4_model"]),
        keys=("case",),
        fields={"forward": 3, "backward": 3, "gpu_step": 3, "throughput": 3},
    )
    tables["same-checkpoint-eval"] = compare_rows(
        claims["same-checkpoint-eval"]["rows"],
        downstream_rows(sources["downstream"]),
        keys=("model", "fit"),
        fields={
            "mean_accuracy_change_pp": 2,
            "native_perplexity": 3,
            "variant_perplexity": 3,
            "prefill": 3,
            "decode": 3,
        },
    )
    half_life = float(claims["pretraining-summary"]["smoothing_half_life_tokens"])
    tables["pretraining-summary"] = compare_rows(
        claims["pretraining-summary"]["rows"],
        pretraining_rows(sources["pretraining"], half_life),
        keys=("case",),
        fields={
            "horizon_billions": 4,
            "native": 4,
            "polynomial": 4,
            "change": 4,
        },
    )
    expected_function_rows = claims["function-summary"]["rows"]
    selected = {
        (row["split"], row["family"], row["degree"]) for row in expected_function_rows
    }
    observed = [
        row
        for row in function_rows(sources["function"])
        if (row["split"], row["family"], row["degree"]) in selected
    ]
    tables["function-summary"] = compare_rows(
        expected_function_rows,
        observed,
        keys=("split", "family", "degree"),
        fields={"ours": 2, "sollya": 2},
    )
    numeric_status = tables["function-summary"]["status"]
    lineage_result = validate_function_lineage(
        function_lineage,
        sources["function"],
    )
    tables["function-summary"]["numeric_status"] = numeric_status
    tables["function-summary"]["lineage"] = lineage_result
    tables["function-summary"]["review_required_acknowledged"] = allow_review_required
    if numeric_status == "mismatch":
        tables["function-summary"]["status"] = "mismatch"
    elif lineage_result["status"] != "pass":
        tables["function-summary"]["status"] = "invalid-lineage"
    elif lineage_result["review_required"] and not allow_review_required:
        tables["function-summary"]["status"] = "review-required"
    else:
        tables["function-summary"]["status"] = "pass"

    statuses = [table["status"] for table in tables.values()]
    if "mismatch" in statuses:
        overall = "mismatch"
    elif "invalid-lineage" in statuses:
        overall = "invalid-lineage"
    elif "review-required" in statuses:
        overall = "review-required"
    else:
        overall = "pass"
    inputs = [
        (args.claims, "manuscript table claims"),
        (manuscript_source, "SHA-bound manuscript source"),
        *[(path, name) for name, path in sources.items()],
    ]
    if function_lineage.is_file():
        inputs.append((function_lineage, "function table lineage"))
    report = {
        "schema_version": 1,
        "artifact_type": "paper-table-evidence-audit",
        "overall_status": overall,
        "tables": tables,
        "inputs": [artifact_record(path, role=role) for path, role in inputs],
        "privacy": "identifier columns are neither loaded into derived rows nor emitted",
        "function_table_scope": (
            "numeric source-to-typeset consistency only; method and arithmetic "
            "semantics remain under review in extras/paper/FUNCTION_TABLE_REVIEW.md"
        ),
    }
    return (
        report,
        {
            "pass": 0,
            "mismatch": 1,
            "invalid-lineage": 1,
            "review-required": 2,
        }[overall],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument(
        "--function-comparison", type=Path, default=DEFAULT_FUNCTION_COMPARISON
    )
    parser.add_argument(
        "--function-lineage", type=Path, default=DEFAULT_FUNCTION_LINEAGE
    )
    parser.add_argument(
        "--allow-review-required",
        action="store_true",
        help=(
            "Acknowledge the scientific-semantic review items recorded in the "
            "validated function-table lineage. This never permits numeric or "
            "lineage mismatches."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report, status = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{report['overall_status'].upper()}: wrote {args.output}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

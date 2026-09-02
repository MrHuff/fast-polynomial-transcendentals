# Copyright 2026 Robert Hu
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from extras.paper import check_paper_tables as table_checker
from extras.paper import generate_accuracy_figures as accuracy
from extras.paper import generate_method_figures as method
from extras.paper import generate_sollya_comparison as sollya
from autonumerics_zero.spline_ops import (
    generate_sollya_structs_bf16 as deployed_sollya,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_method_figures_use_selected_outputs_and_portable_receipt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "figures"
    receipt = tmp_path / "receipt.json"
    status = method.main(
        [
            "--output-dir",
            str(output),
            "--format",
            "png",
            "--receipt",
            str(receipt),
        ]
    )
    assert status == 0
    assert (output / "symmetry_reduction.png").is_file()
    assert (output / "constrained_fit_replay.png").is_file()
    document = json.loads(receipt.read_text())
    assert document["artifact_type"] == "paper-method-figures"
    assert {row["name"] for row in document["outputs"]} == {
        "symmetry_reduction.png",
        "constrained_fit_replay.png",
    }
    assert str(REPOSITORY_ROOT) not in receipt.read_text()


def test_bf16_round_is_round_to_nearest_even() -> None:
    values = np.asarray([1.0, 1.00390625, 1.01171875], dtype=np.float32)
    rounded = method.bf16_round(values)
    np.testing.assert_array_equal(
        rounded, np.asarray([1.0, 1.0, 1.015625], dtype=np.float32)
    )


def test_sollya_monomials_match_each_program_shape() -> None:
    monomials = {spec.display_name: spec.monomials(3) for spec in sollya.build_specs()}
    assert monomials["sigmoid"] == [1, 2, 3]
    assert monomials["swish"] == [2, 3, 4]
    assert monomials["sigmoid'"] == [0, 1, 2, 3]


def test_sollya_parser_ignores_non_numeric_output(monkeypatch) -> None:
    completed = SimpleNamespace(stdout="banner\n1.0\n2.0\n3.0\n", stderr="")
    monkeypatch.setattr(sollya.subprocess, "run", lambda *args, **kwargs: completed)
    assert sollya.run_sollya("sollya", "x", [1, 2, 3], 8, 4.0) == [1.0, 2.0, 3.0]


def table_audit_args(**overrides):
    values = {
        "claims": REPOSITORY_ROOT / "extras/paper/paper_table_claims.json",
        "evidence_dir": REPOSITORY_ROOT / "evidence/report-data",
        "function_comparison": (
            REPOSITORY_ROOT
            / "autonumerics_zero/cuda_benchmarks/analysis_results/sollya_device_bf16.json"
        ),
        "function_lineage": (
            REPOSITORY_ROOT / "extras/paper/function_table_lineage.json"
        ),
        "allow_review_required": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_table_checker_requires_review_for_released_function_table() -> None:
    args = table_audit_args()
    report, status = table_checker.audit(args)
    assert status == 2
    assert report["overall_status"] == "review-required"
    function_table = report["tables"]["function-summary"]
    assert function_table["status"] == "review-required"
    assert function_table["numeric_status"] == "pass"
    assert function_table["lineage"]["status"] == "pass"
    assert function_table["lineage"]["mode"] == "retained-artifact-sidecar"
    assert {
        item["id"] for item in function_table["lineage"]["review_required"]
    } == table_checker.REQUIRED_FUNCTION_REVIEW_IDS
    for table in (
        "function-speed",
        "integration-summary",
        "model-timing-summary",
        "same-checkpoint-eval",
        "pretraining-summary",
    ):
        assert report["tables"][table] == {"status": "pass", "mismatches": []}
    serialized = json.dumps(report)
    assert "run_id" not in serialized
    assert "run_name" not in serialized


def test_table_checker_allows_acknowledged_review_items() -> None:
    args = table_audit_args(allow_review_required=True)
    report, status = table_checker.audit(args)
    assert status == 0
    assert report["overall_status"] == "pass"
    function_table = report["tables"]["function-summary"]
    assert function_table["status"] == "pass"
    assert function_table["review_required_acknowledged"] is True


def test_table_checker_requires_claim_map_to_bind_packaged_manuscript(
    tmp_path: Path,
) -> None:
    claims = json.loads(table_audit_args().claims.read_text())
    claims["manuscript_source"]["sha256"] = "0" * 64
    claims_path = tmp_path / "claims.json"
    claims_path.write_text(json.dumps(claims))
    with np.testing.assert_raises_regex(
        ValueError, "does not match the packaged manuscript"
    ):
        table_checker.audit(table_audit_args(claims=claims_path))


def test_table_checker_accepts_equivalent_embedded_measurement(
    tmp_path: Path,
) -> None:
    source = table_audit_args().function_comparison
    document = json.loads(source.read_text())
    document["measurement"] = dict(table_checker.EXPECTED_FUNCTION_MEASUREMENT)
    comparison = tmp_path / "fresh-comparison.json"
    comparison.write_text(json.dumps(document))
    args = table_audit_args(
        function_comparison=comparison,
        allow_review_required=True,
    )
    report, status = table_checker.audit(args)
    assert status == 0
    assert report["tables"]["function-summary"]["lineage"]["mode"] == (
        "generated-artifact-embedded-measurement"
    )


def test_table_checker_semantics_match_core_generator() -> None:
    assert deployed_sollya.ERROR_MEASUREMENT == (
        table_checker.EXPECTED_FUNCTION_MEASUREMENT
    )


def test_function_speed_rows_bind_both_memory_endpoints() -> None:
    rows = table_checker.function_speed_rows(
        REPOSITORY_ROOT
        / "evidence/report-data/isolated_function_speedups_fp16.csv"
    )
    assert len(rows) == 8
    sigmoid_forward = next(
        row
        for row in rows
        if (row["direction"], row["function"], row["degree"])
        == ("forward", "sigmoid", "D3")
    )
    assert sigmoid_forward["l2_speedup"] == 1.701769587017282
    assert sigmoid_forward["hbm_speedup"] == 1.2486912411869147


def test_table_checker_rejects_changed_function_cell(tmp_path: Path) -> None:
    document = json.loads(table_audit_args().function_comparison.read_text())
    document["families"]["sigmoid_fwd"]["D3"]["current_max_error"] += 0.001
    document["measurement"] = dict(table_checker.EXPECTED_FUNCTION_MEASUREMENT)
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(document))
    args = table_audit_args(
        function_comparison=comparison,
        allow_review_required=True,
    )
    report, status = table_checker.audit(args)
    assert status == 1
    assert report["overall_status"] == "mismatch"
    assert report["tables"]["function-summary"]["mismatches"]


def test_table_checker_rejects_missing_or_wrong_lineage_semantics(
    tmp_path: Path,
) -> None:
    missing = table_audit_args(
        function_lineage=tmp_path / "missing.json",
        allow_review_required=True,
    )
    report, status = table_checker.audit(missing)
    assert status == 1
    assert report["overall_status"] == "invalid-lineage"

    lineage = json.loads(table_audit_args().function_lineage.read_text())
    lineage["measurement"]["evaluation"] = "device"
    wrong_path = tmp_path / "wrong.json"
    wrong_path.write_text(json.dumps(lineage))
    wrong = table_audit_args(
        function_lineage=wrong_path,
        allow_review_required=True,
    )
    report, status = table_checker.audit(wrong)
    assert status == 1
    assert report["tables"]["function-summary"]["status"] == "invalid-lineage"

    lineage = json.loads(table_audit_args().function_lineage.read_text())
    lineage["retained_artifact"]["sha256"] = "0" * 64
    lineage["review_required"][0].pop("observed")
    incomplete_path = tmp_path / "incomplete.json"
    incomplete_path.write_text(json.dumps(lineage))
    incomplete = table_audit_args(
        function_lineage=incomplete_path,
        allow_review_required=True,
    )
    report, status = table_checker.audit(incomplete)
    assert status == 1
    errors = report["tables"]["function-summary"]["lineage"]["errors"]
    assert any("artifact SHA-256" in error for error in errors)
    assert any("observed" in error for error in errors)


def test_new_function_artifact_requires_embedded_measurement(tmp_path: Path) -> None:
    document = json.loads(table_audit_args().function_comparison.read_text())
    document["extra"] = "changes the retained SHA"
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(document))
    report, status = table_checker.audit(
        table_audit_args(
            function_comparison=comparison,
            allow_review_required=True,
        )
    )
    assert status == 1
    assert report["tables"]["function-summary"]["status"] == "invalid-lineage"


def test_accuracy_cli_defaults_outside_evidence() -> None:
    args = accuracy.parse_args([])
    assert args.output_dir == REPOSITORY_ROOT / "outputs/paper/accuracy"
    assert "evidence" not in args.output_dir.parts


def test_accuracy_loader_rejects_module_outside_selected_directory(
    tmp_path: Path, monkeypatch
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    module = SimpleNamespace(__file__=str(tmp_path / "other/spline_ops.so"))
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)
    with np.testing.assert_raises_regex(RuntimeError, "does not come from"):
        accuracy.load_extension(selected)

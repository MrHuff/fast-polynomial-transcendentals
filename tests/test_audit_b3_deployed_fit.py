# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

from pathlib import Path

import pytest

from scripts import audit_b3_deployed_fit as audit
from sfu_repro.artifact import validate_result


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FA4_ROOT = REPOSITORY_ROOT / "flash-attention"


def test_bf16_rounds_halfway_values_to_even() -> None:
    assert audit.round_bf16(1.00390625) == 1.0
    assert audit.round_bf16(1.01171875) == 1.015625
    assert audit.round_bf16(-1.00390625) == -1.0


def test_pinned_source_and_constants_are_read_from_fa4() -> None:
    source = audit.verify_fa4_source(REPOSITORY_ROOT, FA4_ROOT)
    spec = audit.read_direct_b3_spec(FA4_ROOT / audit.MANIFEST_PATH)

    assert source["expected_revision"] == audit.EXPECTED_FA4_REVISION
    assert source["observed_revision"] == audit.EXPECTED_FA4_REVISION
    assert source["audited_files_match_commit"] is True
    assert spec.sequence_length == 4096
    assert spec.forward_rows == (
        (1.0, 1.0546875, 0.53515625, 0.11328125),
        (-8.5625, 17.5, -9.0625, 2.140625),
    )
    assert spec.gradient_factors == (
        (1.0, -0.0087890625),
        (1.0078125, -0.0028076171875),
    )


def test_revision_mismatch_fails_closed() -> None:
    with pytest.raises(audit.AuditError, match="pin mismatch"):
        audit.verify_fa4_source(
            REPOSITORY_ROOT,
            FA4_ROOT,
            expected_revision="0" * 40,
        )


def test_deployed_emulation_uses_scaled_horner_and_factored_gradient() -> None:
    spec = audit.read_direct_b3_spec(FA4_ROOT / audit.MANIFEST_PATH)
    at_zero = audit.evaluate_deployed(0.0, spec)
    assert at_zero.row_index == 0
    assert at_zero.forward == audit.round_bf16(1.0 / 4096.0)
    assert at_zero.derivative_factor == 1.0
    assert at_zero.derivative == at_zero.forward

    high = audit.evaluate_deployed(spec.split, spec)
    assert high.row_index == 1
    intercept, slope = spec.deployed_gradient_factors[1]
    expected_factor = audit.bf16_fma(high.clamped_bf16, slope, intercept)
    assert high.derivative_factor == expected_factor
    assert high.derivative == audit.bf16_multiply(high.forward, expected_factor)

    below = audit.evaluate_deployed(-20.0, spec)
    lower_edge = audit.evaluate_deployed(-6.0, spec)
    assert below.clamped_bf16 == lower_edge.clamped_bf16
    assert below.forward == lower_edge.forward
    assert below.derivative == lower_edge.derivative

    above = audit.evaluate_deployed(20.0, spec)
    upper_edge = audit.evaluate_deployed(6.0, spec)
    assert above.clamped_bf16 == upper_edge.clamped_bf16
    assert above.forward == upper_edge.forward
    assert above.derivative == upper_edge.derivative


def test_grid_result_is_a_verification_envelope() -> None:
    document = audit.build_document(
        repository_root=REPOSITORY_ROOT,
        fa4_root=FA4_ROOT,
        grid_min=-6.0,
        grid_max=6.0,
        grid_points=257,
    )

    assert validate_result(document) == []
    assert document["experiment"]["verification_only"] is True
    assert document["experiment"]["original_fitter_recovered"] is False
    assert "not the missing original fitter" in document["experiment"]["claim_boundary"]
    assert document["results"]["grid"]["points"] == 257
    assert 0 < document["results"]["grid"]["unique_bf16_inputs"] <= 257
    assert document["results"]["metrics"]["exact_target_at_declared_grid_score"]


@pytest.mark.parametrize(
    "grid_min,grid_max,grid_points,match",
    (
        (0.0, 0.0, 2, "minimum"),
        (-6.0, 6.0, 1, "grid points"),
        (-6.0, 6.0, audit.MAX_GRID_POINTS + 1, "grid points"),
    ),
)
def test_grid_bounds_are_enforced(
    grid_min: float,
    grid_max: float,
    grid_points: int,
    match: str,
) -> None:
    spec = audit.read_direct_b3_spec(FA4_ROOT / audit.MANIFEST_PATH)
    with pytest.raises(audit.AuditError, match=match):
        audit.audit_grid(
            spec,
            grid_min=grid_min,
            grid_max=grid_max,
            grid_points=grid_points,
        )

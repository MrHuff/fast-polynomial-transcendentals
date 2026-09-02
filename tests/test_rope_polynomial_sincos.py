# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified 2026-09-02 for the standalone SFU reproduction package.

import math
import shutil

import numpy as np
import pytest
import torch

from sfu_repro.rope.fit_polynomial_sincos import (
    error_metrics,
    fit_full_pair,
    fit_parity_pair,
    fit_sollya_pair,
    reduce_to_half_turn,
)
from sfu_repro.artifact import validate_result
from sfu_repro.rope.benchmark_polynomial_sincos import rope_result_metadata
from sfu_repro.rope.polynomial_sincos import (
    SOLLYA_D3_D4,
    UNIFORM_D5_D4,
    UNIFORM_D7_D6,
    polynomial_sincos,
    reduce_to_quarter_turn,
)


def _apply_complex_rope(values: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    pairs = values.float().reshape(*values.shape[:-1], values.shape[-1] // 2, 2)
    complex_values = torch.view_as_complex(pairs)
    frequencies = torch.complex(
        torch.cos(angles).float(),
        torch.sin(angles).float(),
    )[None, :, None, :]
    return torch.view_as_real(complex_values * frequencies).flatten(-2)


def test_rope_benchmark_metadata_forms_a_valid_result_envelope() -> None:
    document = {
        **rope_result_metadata(
            "rope-test",
            repository_state={
                "revision": "a" * 40,
                "dirty": False,
                "untracked_files": 0,
            },
            attestations={},
            source_bound=True,
        ),
        "measurement": {"summary_statistic": "test"},
        "results": [],
    }

    assert validate_result(document) == []
    assert document["experiment"]["command"]


def _apply_real_rope(
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    pairs = values.float().reshape(*values.shape[:-1], values.shape[-1] // 2, 2)
    even, odd = pairs.unbind(-1)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        dim=-1,
    )
    return rotated.flatten(-2)


def test_polynomial_degrees_and_coefficient_budget() -> None:
    assert SOLLYA_D3_D4.sin_degree == 3
    assert SOLLYA_D3_D4.cos_degree == 4
    assert len(SOLLYA_D3_D4.sin_coefficients) == 2
    assert len(SOLLYA_D3_D4.cos_coefficients) == 3
    assert UNIFORM_D5_D4.sin_degree == 5
    assert UNIFORM_D5_D4.cos_degree == 4
    assert len(UNIFORM_D5_D4.sin_coefficients) == 3
    assert len(UNIFORM_D5_D4.cos_coefficients) == 3
    assert UNIFORM_D7_D6.sin_degree == 7
    assert UNIFORM_D7_D6.cos_degree == 6
    assert len(UNIFORM_D7_D6.sin_coefficients) == 4
    assert len(UNIFORM_D7_D6.cos_coefficients) == 4


def test_d3_d4_candidate_is_within_bfloat16_rotation_budget() -> None:
    angles = torch.linspace(-131_072.0, 131_072.0, 262_145)
    cos, sin = polynomial_sincos(angles, polynomial=SOLLYA_D3_D4)
    reference_cos = torch.cos(angles)
    reference_sin = torch.sin(angles)
    phase_error = torch.atan2(
        sin * reference_cos - cos * reference_sin,
        cos * reference_cos + sin * reference_sin,
    )

    assert phase_error.abs().max().item() <= 1.6e-4
    assert (cos.square() + sin.square() - 1.0).abs().max().item() <= 2.1e-4


def test_accurate_polynomial_matches_float32_sincos_over_wide_range() -> None:
    angles = torch.linspace(-131_072.0, 131_072.0, 262_145)
    cos, sin = polynomial_sincos(angles)

    torch.testing.assert_close(cos, torch.cos(angles), rtol=0.0, atol=1.2e-7)
    torch.testing.assert_close(sin, torch.sin(angles), rtol=0.0, atol=1.2e-7)
    assert (cos.square() + sin.square() - 1.0).abs().max().item() <= 2.4e-7


def test_d5_d4_candidate_is_within_bfloat16_rotation_budget() -> None:
    angles = torch.linspace(-131_072.0, 131_072.0, 262_145)
    cos, sin = polynomial_sincos(angles, polynomial=UNIFORM_D5_D4)
    reference_cos = torch.cos(angles)
    reference_sin = torch.sin(angles)
    phase_error = torch.atan2(
        sin * reference_cos - cos * reference_sin,
        cos * reference_cos + sin * reference_sin,
    )

    assert phase_error.abs().max().item() <= 1.5e-5
    assert (cos.square() + sin.square() - 1.0).abs().max().item() <= 4.5e-5


def test_reducer_covers_quadrants_and_primary_interval() -> None:
    angles = torch.tensor(
        [-2.0 * math.pi, -math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi]
    )
    reduced, quadrant = reduce_to_quarter_turn(angles)

    assert reduced.abs().max().item() <= math.pi / 4.0
    assert quadrant.tolist() == [0, 2, 3, 0, 1, 2]


def test_fast_reducer_has_explicit_long_context_error_bound() -> None:
    angles = torch.linspace(0.0, 8192.0, 262_145)
    accurate_cos, accurate_sin = polynomial_sincos(angles)
    fast_cos, fast_sin = polynomial_sincos(angles, accurate_reduction=False)
    error = torch.stack((fast_cos - accurate_cos, fast_sin - accurate_sin)).abs().max()

    assert 1.0e-5 < error.item() < 3.0e-4


def test_rope_weighting_can_fit_parity_and_unrestricted_bases() -> None:
    angles = np.linspace(-math.pi / 4.0, math.pi / 4.0, 8193)
    parity_sin, parity_cos = fit_parity_pair(
        angles,
        sin_terms=4,
        cos_terms=4,
    )
    full_sin, full_cos = fit_full_pair(
        angles,
        sin_degree=7,
        cos_degree=6,
    )

    parity_metrics = error_metrics(angles, parity_sin, parity_cos)
    full_metrics = error_metrics(
        angles,
        full_sin,
        full_cos,
        basis="full",
    )
    assert parity_metrics["phase_max_abs_rad"] < 5.0e-8
    assert full_metrics["phase_max_abs_rad"] < 5.0e-8
    assert len(parity_sin) + len(parity_cos) == 8
    assert len(full_sin) + len(full_cos) == 15


def test_half_turn_least_squares_fit_meets_bfloat16_error_budget() -> None:
    angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, 65_537)
    sin_coefficients, cos_coefficients = fit_parity_pair(
        angles,
        sin_terms=3,
        cos_terms=3,
    )
    metrics = error_metrics(
        angles,
        sin_coefficients.astype(np.float32),
        cos_coefficients.astype(np.float32),
        reduction="half-turn",
    )

    assert metrics["phase_max_abs_rad"] < 1.4e-3
    assert metrics["unit_norm_max_abs"] < 9.0e-4


@pytest.mark.skipif(shutil.which("sollya") is None, reason="Sollya not installed")
def test_half_turn_sollya_fit_improves_max_phase_error() -> None:
    angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, 65_537)
    least_squares = fit_parity_pair(angles, sin_terms=3, cos_terms=3)
    sollya = fit_sollya_pair(
        sin_terms=3,
        cos_terms=3,
        interval_max=math.pi / 2.0,
        coefficient_dtype="float32",
    )
    least_squares_metrics = error_metrics(
        angles,
        *least_squares,
        reduction="half-turn",
    )
    sollya_metrics = error_metrics(
        angles,
        *sollya,
        reduction="half-turn",
    )

    assert sollya_metrics["phase_max_abs_rad"] < 6.1e-4
    assert (
        sollya_metrics["phase_max_abs_rad"] < least_squares_metrics["phase_max_abs_rad"]
    )


def test_half_turn_reducer_uses_one_shared_sign_parity() -> None:
    angles = np.asarray(
        [-2.0 * math.pi, -math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi]
    )
    reduced, parity = reduce_to_half_turn(angles)

    assert np.max(np.abs(reduced)) <= math.pi / 2.0
    assert parity.tolist() == [0, 1, 0, 0, 0, 1]


def test_polynomial_table_and_rotation_match_complex_reference() -> None:
    torch.manual_seed(1234)
    head_dim = 128
    sequence_length = 512
    dimensions = torch.arange(0, head_dim, 2, dtype=torch.float64)
    frequencies = torch.pow(500_000.0, -dimensions / head_dim)
    positions = torch.arange(8192, dtype=torch.float64)
    angles = torch.outer(positions, frequencies)
    cos, sin = polynomial_sincos(angles)

    reference_cos = torch.cos(angles).float()
    reference_sin = torch.sin(angles).float()
    table_error = torch.stack((cos - reference_cos, sin - reference_sin))
    assert table_error.abs().max().item() <= 1.2e-7

    values = torch.randn(1, sequence_length, 4, head_dim)
    reference = _apply_complex_rope(values, angles[:sequence_length])
    candidate = _apply_real_rope(
        values,
        cos[:sequence_length],
        sin[:sequence_length],
    )
    torch.testing.assert_close(candidate, reference, rtol=1.0e-6, atol=6.0e-7)

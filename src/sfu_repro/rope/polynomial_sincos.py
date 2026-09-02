#
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified 2026-09-02 for the standalone SFU reproduction package.
#

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SincosPolynomial:
    """Parity-preserving polynomial pair on [-pi/4, pi/4]."""

    sin_coefficients: tuple[float, ...]
    cos_coefficients: tuple[float, ...]
    fit_name: str

    @property
    def sin_degree(self) -> int:
        return 2 * len(self.sin_coefficients) - 1

    @property
    def cos_degree(self) -> int:
        return 2 * (len(self.cos_coefficients) - 1)


# The runtime forms are sin(r) = r * P(r^2) and cos(r) = Q(r^2). The
# D3/D4 pair is a Sollya fpminimax fit with float32 coefficients; D5/D4 and
# D7/D6 are uniform least-squares controls. D3/D4 targets BF16 rotation,
# while D7/D6 provides a float32-accurate control.
SOLLYA_D3_D4 = SincosPolynomial(
    sin_coefficients=(
        0.9990314245223999,
        -0.16034401953220367,
    ),
    cos_coefficients=(
        0.9999900460243225,
        -0.4997082054615021,
        0.04039861634373665,
    ),
    fit_name="sollya_fpminimax_d3_d4_f32",
)

UNIFORM_D5_D4 = SincosPolynomial(
    sin_coefficients=(
        0.9999962449073792,
        -0.16661198437213898,
        0.008137642405927181,
    ),
    cos_coefficients=(
        0.999993085861206,
        -0.499763548374176,
        0.04051213711500168,
    ),
    fit_name="uniform_d5_d4_f32",
)

UNIFORM_D7_D6 = SincosPolynomial(
    sin_coefficients=(
        1.0,
        -0.1666664183139801,
        0.00833179522305727,
        -0.00019484198128338903,
    ),
    cos_coefficients=(
        1.0,
        -0.49999886751174927,
        0.041656624525785446,
        -0.0013605882413685322,
    ),
    fit_name="uniform_d7_d6_f32",
)

_INV_HALF_PI = 2.0 / math.pi
_HALF_PI = math.pi / 2.0

# Cody-Waite split used by the candidate FP32 reducer.
_HALF_PI_F32_HI = 1.570796251296997
_HALF_PI_F32_LO = 7.549789415861596e-08


def _horner(x: torch.Tensor, coefficients: tuple[float, ...]) -> torch.Tensor:
    value = torch.full_like(x, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        value = value * x + coefficient
    return value


def reduce_to_quarter_turn(
    angles: torch.Tensor,
    *,
    accurate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce angles to approximately [-pi/4, pi/4] and return quadrant indices.

    The accurate path performs table-generation range reduction in float64 and
    returns a float32 residual. It is intended for cached RoPE tables, where
    setup accuracy matters more than FP64 throughput. The fast path is a
    candidate for a fused GPU kernel. It stays in float32, but its residual
    error grows with the angle magnitude and must not silently replace the
    accurate path for long-context tables.
    """

    if not torch.is_floating_point(angles):
        raise TypeError("angles must be a floating-point tensor")

    if accurate:
        work = angles.to(torch.float64)
        quadrant_float = torch.round(work * _INV_HALF_PI)
        reduced = work - quadrant_float * _HALF_PI
    else:
        work = angles.to(torch.float32)
        quadrant_float = torch.round(work * _INV_HALF_PI)
        reduced = (
            work - quadrant_float * _HALF_PI_F32_HI
        ) - quadrant_float * _HALF_PI_F32_LO

    quadrant = quadrant_float.to(torch.int64).bitwise_and(3)
    return reduced.to(torch.float32), quadrant


def evaluate_reduced_sincos(
    reduced: torch.Tensor,
    polynomial: SincosPolynomial = UNIFORM_D7_D6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the paired parity polynomials on a reduced float32 argument."""

    reduced = reduced.to(torch.float32)
    squared = reduced * reduced
    sin_reduced = reduced * _horner(squared, polynomial.sin_coefficients)
    cos_reduced = _horner(squared, polynomial.cos_coefficients)
    return sin_reduced, cos_reduced


def polynomial_sincos(
    angles: torch.Tensor,
    *,
    polynomial: SincosPolynomial = UNIFORM_D7_D6,
    accurate_reduction: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate sine and cosine together using one range reduction.

    Returns cos then sin to match the packed Real-RoPE table convention.
    Polynomial evaluation is float32 even when the source angle tensor uses a
    lower precision.
    """

    reduced, quadrant = reduce_to_quarter_turn(
        angles,
        accurate=accurate_reduction,
    )
    sin_reduced, cos_reduced = evaluate_reduced_sincos(reduced, polynomial)

    swap = quadrant.bitwise_and(1).bool()
    sin_magnitude = torch.where(swap, cos_reduced, sin_reduced)
    cos_magnitude = torch.where(swap, sin_reduced, cos_reduced)

    sin_negative = quadrant.bitwise_and(2).bool()
    cos_negative = (quadrant + 1).bitwise_and(2).bool()
    sin = torch.where(sin_negative, -sin_magnitude, sin_magnitude)
    cos = torch.where(cos_negative, -cos_magnitude, cos_magnitude)
    return cos, sin

from __future__ import annotations

import numpy as np

from autonumerics_zero.evolution import fit_gelu_fp16
from autonumerics_zero.evolution.constrained_ls_fitter import dual_constrained_ls_fit


def test_gelu_symmetries() -> None:
    fit_gelu_fp16.verify_symmetries()


def test_historical_d5_forward_fit_regenerates_deployed_source_coefficients() -> None:
    coefficients = dual_constrained_ls_fit(
        fit_gelu_fp16.gaussian_cdf_odd,
        (0.0, 3.0),
        5,
    )
    rounded = [float(np.float16(value)) for value in coefficients]

    assert rounded == [
        0.0,
        0.396240234375,
        0.017822265625,
        -0.10528564453125,
        0.0362548828125,
        -0.0038852691650390625,
    ]

from __future__ import annotations

import numpy as np

from autonumerics_zero.evolution import fit_flash_sigmoid_exp2_d2 as fitter
from autonumerics_zero.spline_ops import audit_sincos_sass


def test_result_document_records_replay_inputs_and_float32_literals() -> None:
    args = fitter.parse_args(
        [
            "--sequence-length",
            "4096",
            "--score-sigma",
            "1.25",
            "--seed",
            "7",
            "--maxiter",
            "9",
            "--json-out",
            "result.json",
        ]
    )
    coefficients = np.asarray([1.0, 0.5, 0.25], dtype=np.float64)
    metrics = fitter.FitMetrics(
        forward_relative_l1=0.1,
        gradient_relative_l1=0.2,
        central_max_relative=0.3,
        endpoint_jump=0.0,
        minimum_mantissa=1.0,
    )

    document = fitter.result_document(args, coefficients, metrics)

    assert document["sequence_length"] == 4096
    assert document["score_sigma"] == 1.25
    assert document["seed"] == 7
    assert document["maxiter"] == 9
    assert document["coefficients_float32"] == [1.0, 0.5, 0.25]
    assert document["ptx_hex_high_to_low"] == [
        "0x3E800000",
        "0x3F000000",
        "0x3F800000",
    ]
    assert document["metrics"]["minimum_mantissa"] == 1.0


def test_sass_audit_accepts_a_caller_selected_json_path() -> None:
    args = audit_sincos_sass.parse_args(
        ["build/spline_ops.so", "--json-out", "results/sass.json"]
    )

    assert args.extension.as_posix() == "build/spline_ops.so"
    assert args.json_out.as_posix() == "results/sass.json"

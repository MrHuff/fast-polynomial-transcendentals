# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPOSITORY_ROOT / "autonumerics_zero/evolution/generate_bf16_structs.py"
FIT_JSON = (
    REPOSITORY_ROOT / "autonumerics_zero/cuda_benchmarks/analysis_results/"
    "all_degree_coefficients_bf16.json"
)
FALLBACK = REPOSITORY_ROOT / "autonumerics_zero/spline_ops/spline_structs_odd_bf16.cuh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generator_writes_caller_selected_header_and_bound_receipt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "generated.cuh"
    receipt = tmp_path / "generated.receipt.json"
    fallback_before = sha256(FALLBACK)

    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--bf16-input",
            str(FIT_JSON),
            "--fallback-header",
            str(FALLBACK),
            "--output",
            str(output),
            "--receipt-out",
            str(receipt),
            "--allow-unbound-source",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert fallback_before == sha256(FALLBACK)
    generated = output.read_text(encoding="utf-8")
    assert "struct SIGMOID_FWD_D3_ODD_BF16" in generated
    assert "struct GELU_BWD_D6_ODD_BF16" in generated
    document = json.loads(receipt.read_text(encoding="utf-8"))
    assert document["artifact_type"] == "bf16_activation_header_generation_receipt"
    assert document["artifact"]["sha256"] == sha256(output)
    assert "generated_header" not in document["source"]["input_sha256"]
    assert document["transformation"]["copied_fallback_families"] == [
        "erf_forward",
        "gelu_forward",
        "gelu_backward",
    ]
    assert all(not token.startswith(str(tmp_path)) for token in document["command"])

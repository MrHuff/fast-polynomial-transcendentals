from __future__ import annotations

import json
import os
from pathlib import Path
import re

import numpy as np

from autonumerics_zero.evolution import fit_all_degrees
from autonumerics_zero.evolution import fit_all_degrees_bf16
from autonumerics_zero.evolution import fit_exp2_softmax
from autonumerics_zero.evolution import fit_fa4_tanh_backends
from autonumerics_zero.evolution import fit_flash_sigmoid_exp2_d2
from autonumerics_zero.evolution import fit_gelu_fp16
from autonumerics_zero.evolution import fit_provenance


EXPECTED_INPUTS = {
    "fit_all_degrees.py": ("fit_all_degrees.py", "constrained_ls_fitter.py"),
    "fit_all_degrees_bf16.py": (
        "fit_all_degrees_bf16.py",
        "constrained_ls_fitter.py",
    ),
    "fit_fa4_tanh_backends.py": (
        "fit_fa4_tanh_backends.py",
        "constrained_ls_fitter.py",
    ),
    "fit_gelu_fp16.py": (
        "fit_gelu_fp16.py",
        "fit_all_degrees.py",
        "constrained_ls_fitter.py",
    ),
    "fit_exp2_softmax.py": ("fit_exp2_softmax.py",),
    "fit_flash_sigmoid_exp2_d2.py": ("fit_flash_sigmoid_exp2_d2.py",),
}
EXPECTED_PACKAGES = {
    "fit_all_degrees.py": ("numpy", "scipy"),
    "fit_all_degrees_bf16.py": ("numpy", "scipy", "torch"),
    "fit_fa4_tanh_backends.py": ("numpy", "scipy", "torch"),
    "fit_gelu_fp16.py": ("numpy", "scipy"),
    "fit_exp2_softmax.py": ("numpy",),
    "fit_flash_sigmoid_exp2_d2.py": ("numpy", "scipy"),
}


def assert_fit_provenance(
    document: dict[str, object],
    script_name: str,
    output_path: Path,
) -> None:
    provenance = document["_provenance"]
    assert isinstance(provenance, dict)
    assert provenance["schema_version"] == 1
    assert provenance["artifact_type"] == "generated-coefficient-fit"
    assert provenance["provenance_class"] == "generated-fit"
    assert "does not promote" in provenance["claim_scope"]
    assert provenance["numerical_payload_sha256"] == (
        fit_provenance.numerical_payload_sha256(document)
    )

    command = provenance["command"]
    assert isinstance(command, list)
    assert command[1].endswith(script_name)
    assert not any(str(output_path.parent) in token for token in command)
    assert f"<external>/{output_path.name}" in command

    source = provenance["source"]
    assert isinstance(source, dict)
    if source["dirty"] is None:
        assert source["clean"] is None
    else:
        assert source["clean"] is (not source["dirty"])
    assert source["revision"] is None or re.fullmatch(
        r"[0-9a-f]{40}", str(source["revision"])
    )
    hashes = source["input_sha256"]
    assert isinstance(hashes, dict)
    for source_name in (*EXPECTED_INPUTS[script_name], "fit_provenance.py"):
        assert any(path.endswith(source_name) for path in hashes)
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values())

    environment = provenance["environment"]
    assert isinstance(environment, dict)
    assert environment["python"]
    assert environment["platform"]
    assert environment["machine"]
    packages = environment["packages"]
    assert isinstance(packages, dict)
    for package in EXPECTED_PACKAGES[script_name]:
        assert isinstance(packages.get(package), str) and packages[package]


def test_fp16_and_bf16_fit_outputs_record_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def cheap_sweep(*args, **kwargs):
        return {"d3": {"err": 0.0}}

    monkeypatch.setattr(fit_all_degrees, "sweep_2d_and_fit", cheap_sweep)
    monkeypatch.setattr(fit_all_degrees_bf16, "sweep_2d_and_fit", cheap_sweep)

    fp16_output = tmp_path / "fp16.json"
    bf16_output = tmp_path / "bf16.json"
    fit_all_degrees.main(["--json-out", str(fp16_output)])
    fit_all_degrees_bf16.main(["--json-out", str(bf16_output)])

    fp16_document = json.loads(fp16_output.read_text(encoding="utf-8"))
    bf16_document = json.loads(bf16_output.read_text(encoding="utf-8"))
    assert_fit_provenance(fp16_document, "fit_all_degrees.py", fp16_output)
    assert_fit_provenance(
        bf16_document,
        "fit_all_degrees_bf16.py",
        bf16_output,
    )
    assert fp16_document["sigmoid_fwd_odd"] == {"d3": {"err": 0.0}}
    assert bf16_document["sigmoid_fwd_odd"] == {"d3": {"err": 0.0}}


def test_backend_and_gelu_fit_outputs_record_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        fit_fa4_tanh_backends,
        "sweep_backend",
        lambda **kwargs: {
            "d3": {
                "Li": 1.0,
                "Lc": 1.0,
                "max_error": 0.0,
                "mean_error": 0.0,
                "coeffs_runtime": [0.0],
            }
        },
    )
    monkeypatch.setattr(
        fit_gelu_fp16,
        "fit_all",
        lambda: {"gelu_fwd_odd": {"d3": {"err": 0.0}}},
    )

    backend_output = tmp_path / "backend.json"
    gelu_output = tmp_path / "gelu.json"
    fit_fa4_tanh_backends.main(["--degrees", "3", "--output", str(backend_output)])
    fit_gelu_fp16.main(["--json-out", str(gelu_output)])

    backend_document = json.loads(backend_output.read_text(encoding="utf-8"))
    gelu_document = json.loads(gelu_output.read_text(encoding="utf-8"))
    assert_fit_provenance(
        backend_document,
        "fit_fa4_tanh_backends.py",
        backend_output,
    )
    assert_fit_provenance(gelu_document, "fit_gelu_fp16.py", gelu_output)
    assert backend_document["function"] == "tanh_fwd_odd"
    assert "gelu_fwd_odd" in gelu_document


def test_exp2_fit_outputs_record_provenance(tmp_path: Path, monkeypatch) -> None:
    sampled_output = tmp_path / "sampled-exp2.json"
    fit_exp2_softmax.main(["--samples", "3", "--json-out", str(sampled_output)])

    coefficients = np.asarray([1.0, 0.5, 0.25], dtype=np.float64)
    metrics = fit_flash_sigmoid_exp2_d2.FitMetrics(
        forward_relative_l1=0.1,
        gradient_relative_l1=0.2,
        central_max_relative=0.3,
        endpoint_jump=0.0,
        minimum_mantissa=1.0,
    )
    monkeypatch.setattr(
        fit_flash_sigmoid_exp2_d2,
        "fit",
        lambda args: (coefficients, metrics),
    )
    sequence_output = tmp_path / "sequence-exp2.json"
    fit_flash_sigmoid_exp2_d2.main(
        [
            "--sequence-length",
            "4096",
            "--score-sigma",
            "1.0",
            "--seed",
            "1234",
            "--maxiter",
            "1",
            "--json-out",
            str(sequence_output),
        ]
    )

    sampled_document = json.loads(sampled_output.read_text(encoding="utf-8"))
    sequence_document = json.loads(sequence_output.read_text(encoding="utf-8"))
    assert_fit_provenance(
        sampled_document,
        "fit_exp2_softmax.py",
        sampled_output,
    )
    assert_fit_provenance(
        sequence_document,
        "fit_flash_sigmoid_exp2_d2.py",
        sequence_output,
    )
    assert sampled_document["samples"] == 3
    assert sequence_document["coefficients_float32"] == [1.0, 0.5, 0.25]


def test_safe_command_normalizes_relative_paths_outside_repository(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fit.json"
    current_header = tmp_path / "current.cuh"
    generated_header = tmp_path / "generated.cuh"
    relative_output = os.path.relpath(output, Path.cwd())
    relative_current = os.path.relpath(current_header, Path.cwd())
    relative_generated = os.path.relpath(generated_header, Path.cwd())

    command = fit_provenance.safe_command(
        Path(fit_exp2_softmax.__file__),
        [
            "--current-header",
            relative_current,
            f"--header-out={relative_generated}",
            "--json-out",
            relative_output,
            f"--output={relative_output}",
        ],
    )

    assert relative_output not in command
    assert relative_current not in command
    assert f"--header-out={relative_generated}" not in command
    assert "<external>/current.cuh" in command
    assert "--header-out=<external>/generated.cuh" in command
    assert command.count("<external>/fit.json") == 1
    assert command.count("--output=<external>/fit.json") == 1


def test_git_state_tolerates_a_missing_git_executable(monkeypatch) -> None:
    monkeypatch.setattr(
        fit_provenance.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    state = fit_provenance.git_state()

    assert state["revision"] is None
    assert state["dirty"] is None
    assert state["clean"] is None
    assert state["untracked_files"] is None


def test_fit_payload_digest_detects_coefficient_edits() -> None:
    document: dict[str, object] = {"coefficients": [1.0, 0.5]}
    provenance: dict[str, object] = {}

    fit_provenance.bind_fit_payload(document, provenance)
    bound_digest = provenance["numerical_payload_sha256"]
    document["coefficients"] = [1.0, 0.25]

    assert fit_provenance.numerical_payload_sha256(document) != bound_digest


def test_provenance_is_snapshotted_before_a_fit_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        fit_all_degrees,
        "build_fit_provenance",
        lambda **kwargs: events.append("provenance") or {"sentinel": True},
    )

    def cheap_sweep(*args, **kwargs):
        events.append("fit")
        return {"d3": {"err": 0.0}}

    monkeypatch.setattr(fit_all_degrees, "sweep_2d_and_fit", cheap_sweep)
    output = tmp_path / "ordered.json"

    fit_all_degrees.main(["--json-out", str(output)])

    assert events[0] == "provenance"
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["_provenance"]["sentinel"] is True
    assert written["_provenance"]["numerical_payload_sha256"] == (
        fit_provenance.numerical_payload_sha256(written)
    )


def test_fit_source_binding_detects_numerical_payload_edits() -> None:
    document: dict[str, object] = {"fit": {"coefficient": 1.0}}
    provenance = fit_provenance.build_fit_provenance(
        script=Path(fit_all_degrees_bf16.__file__),
        arguments=["--json-out", "outputs/fit.json"],
        source_files=[
            Path(fit_all_degrees_bf16.__file__).with_name("constrained_ls_fitter.py")
        ],
        distributions=(),
    )
    source = provenance["source"]
    assert isinstance(source, dict)
    revision = "a" * 40
    source["revision"] = revision
    source["dirty"] = False
    source["clean"] = True
    fit_provenance.bind_fit_payload(document, provenance)
    repository_state = {
        "revision": revision,
        "dirty": False,
        "untracked_files": 0,
    }

    assert fit_provenance.fit_output_is_source_bound(
        document,
        expected_script="fit_all_degrees_bf16.py",
        repository_state=repository_state,
    )

    document["fit"] = {"coefficient": 2.0}
    assert not fit_provenance.fit_output_is_source_bound(
        document,
        expected_script="fit_all_degrees_bf16.py",
        repository_state=repository_state,
    )

from pathlib import Path
import hashlib
import json

from autonumerics_zero.evolution import fit_all_degrees
from autonumerics_zero.evolution import fit_all_degrees_bf16
from autonumerics_zero.spline_ops import generate_sollya_structs_bf16


def test_fp16_fit_accepts_caller_selected_output(tmp_path: Path) -> None:
    output = tmp_path / "fp16.json"

    args = fit_all_degrees.get_parser().parse_args(["--json-out", str(output)])

    assert args.json_out == output


def test_bf16_fit_accepts_caller_selected_output(tmp_path: Path) -> None:
    output = tmp_path / "bf16.json"

    args = fit_all_degrees_bf16.get_parser().parse_args(["--json-out", str(output)])

    assert args.json_out == output


def test_sollya_generator_accepts_caller_selected_outputs(tmp_path: Path) -> None:
    header = tmp_path / "input.cuh"
    generated = tmp_path / "generated.cuh"
    result = tmp_path / "fit.json"

    args = generate_sollya_structs_bf16.get_parser().parse_args(
        [
            "--current-header",
            str(header),
            "--header-out",
            str(generated),
            "--json-out",
            str(result),
        ]
    )

    assert args.current_header == header
    assert args.header_out == generated
    assert args.json_out == result


def test_sollya_generator_main_attaches_fit_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    header = tmp_path / "input.cuh"
    header.write_text("// input\n", encoding="utf-8")
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return {
            "families": {
                name: {}
                for name in ("sigmoid_fwd", "tanh_fwd", "swish_fwd", "gelu_fwd")
            }
        }

    monkeypatch.setattr(generate_sollya_structs_bf16, "generate", fake_generate)
    monkeypatch.setattr(
        generate_sollya_structs_bf16, "sollya_version", lambda: "Sollya 8.0"
    )

    generate_sollya_structs_bf16.main(
        [
            "--current-header",
            str(header),
            "--header-out",
            str(tmp_path / "generated.cuh"),
            "--json-out",
            str(tmp_path / "fit.json"),
        ]
    )

    provenance = captured["provenance"]
    assert provenance["artifact_type"] == "generated-coefficient-fit"
    assert provenance["environment"]["sollya"] == "Sollya 8.0"
    assert provenance["source"]["input_sha256"]["<external>/input.cuh"]


def test_sollya_generator_hashes_the_generated_header(
    tmp_path: Path, monkeypatch
) -> None:
    current = tmp_path / "current.cuh"
    generated = tmp_path / "generated.cuh"
    result = tmp_path / "result.json"
    current.write_text("// empty fixture\n", encoding="utf-8")
    monkeypatch.setattr(generate_sollya_structs_bf16, "BASE_SPECS", ())
    monkeypatch.setattr(generate_sollya_structs_bf16, "SWISH_FWD_DEGREES", ())
    provenance = {"artifact_type": "generated-coefficient-fit"}

    document = generate_sollya_structs_bf16.generate(
        current_header=current,
        sollya_header=generated,
        out_json=result,
        provenance=provenance,
    )

    expected = hashlib.sha256(generated.read_bytes()).hexdigest()
    measurement = document["measurement"]
    assert set(measurement) == {
        "metric",
        "evaluation",
        "grid",
        "current_coefficients",
        "sollya_coefficients",
        "intermediate_rounding",
        "device_measurement",
    }
    assert all(isinstance(value, str) for value in measurement.values())
    assert measurement["metric"] == "maximum absolute error"
    assert "host NumPy" in measurement["evaluation"]
    assert "real-arithmetic Horner" in measurement["evaluation"]
    assert "20,001 uniformly spaced points" in measurement["grid"]
    assert "[-Lc, Lc]" in measurement["grid"]
    assert "deployed CUDA header" in measurement["current_coefficients"]
    assert "without BF16 pre-rounding" in measurement["current_coefficients"]
    assert "8-bit precision" in measurement["sollya_coefficients"]
    assert "cast to BF16" in measurement["sollya_coefficients"]
    assert "not replayed" in measurement["intermediate_rounding"]
    assert "not a device measurement" in measurement["device_measurement"]
    assert document["_provenance"]["generated_header_sha256"] == expected
    assert (
        json.loads(result.read_text(encoding="utf-8"))["_provenance"][
            "generated_header_sha256"
        ]
        == expected
    )

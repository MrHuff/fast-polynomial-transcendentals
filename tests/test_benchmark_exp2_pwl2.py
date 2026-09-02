from pathlib import Path

from autonumerics_zero.experiments import benchmark_exp2_pwl2


def test_parser_accepts_caller_selected_output(tmp_path: Path) -> None:
    output = tmp_path / "isolated.json"

    args = benchmark_exp2_pwl2.get_parser().parse_args(
        [
            "--device",
            "2",
            "--json-out",
            str(output),
            "--allow-unbound-source",
        ]
    )

    assert args.device == "2"
    assert args.json_out == output
    assert args.allow_unbound_source is True


def test_parse_record_preserves_numeric_fields() -> None:
    assert benchmark_exp2_pwl2.parse_record(
        "RESULT name=pwl2 milliseconds=0.125 iterations=100"
    ) == {"name": "pwl2", "milliseconds": 0.125, "iterations": 100}
    assert benchmark_exp2_pwl2.parse_record("SAMPLES ms=0.2,0.1,0.3") == {
        "ms": "0.2,0.1,0.3"
    }


def test_sha256_file(tmp_path: Path) -> None:
    value = tmp_path / "value.bin"
    value.write_bytes(b"abc")

    assert benchmark_exp2_pwl2.sha256_file(value) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )

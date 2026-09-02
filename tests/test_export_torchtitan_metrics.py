from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import pytest

from scripts import export_torchtitan_metrics as metrics
from scripts import run_torchtitan as runner


def _samples(*items: tuple[int, float]) -> tuple[metrics.ScalarSample, ...]:
    return tuple(metrics.ScalarSample(step, value) for step, value in items)


def _probe_scalars(
    *,
    times: tuple[tuple[int, float], ...],
    throughputs: tuple[tuple[int, float], ...],
) -> dict[str, tuple[metrics.ScalarSample, ...]]:
    return {
        metrics.TIME_METRIC: _samples(*times),
        metrics.THROUGHPUT_METRIC: _samples(*throughputs),
    }


def _write_bound_receipt(
    run_directory: Path,
    *,
    case: str,
    variant: str,
    phase: str,
    validation: bool = False,
) -> Path:
    values = [
        "--case",
        case,
        "--variant",
        variant,
        "--phase",
        phase,
        f"--config-arg=--job.dump-folder={run_directory}",
    ]
    checkpoint_input = None
    if phase == "pretraining":
        checkpoint = run_directory.parent / "seed-checkpoint"
        checkpoint.mkdir(exist_ok=True)
        shard = checkpoint / "model-shard.bin"
        if not shard.exists():
            shard.write_bytes(b"shared-seed-weights")
        values.extend(["--seed-checkpoint", str(checkpoint)])
        checkpoint_input = runner._checkpoint_tree_identity(checkpoint)
    if validation:
        values.append("--validation")
    args = runner.parse_args(values)
    manifest = runner.load_manifest(runner.DEFAULT_MANIFEST)
    command, config = runner.build_command(args, manifest)
    config_payload, config_document = runner._read_config(config)
    receipt = runner.build_launch_receipt(
        args=args,
        manifest=manifest,
        manifest_path=runner.DEFAULT_MANIFEST,
        command=command,
        config=config,
        config_payload=config_payload,
        config_document=config_document,
        root_state=("d" * 40, False),
        torchtitan_state=(manifest["torchtitan_commit"], False),
        flash_attention_state=(manifest["flash_attention_commit"], False),
        selected_assets=runner.REPOSITORY_ROOT / "assets/hf/Llama-3.1-8B",
        tokenizer_manifest_sha256="e" * 64,
        tokenizer_verified=True,
        checkpoint_input=checkpoint_input,
    )
    path = run_directory / runner.RECEIPT_FILENAME
    runner.write_receipt_once(path, receipt)
    return path


def _write_event(run_directory: Path, payload: bytes) -> Path:
    run_directory.mkdir(parents=True, exist_ok=True)
    path = run_directory / "events.out.tfevents.synthetic"
    path.write_bytes(payload)
    return path


def test_probe_summary_uses_inclusive_steady_window_and_ratio_of_medians() -> None:
    native = _probe_scalars(
        times=((19, 99.0), (20, 10.0), (40, 8.0), (100, 6.0), (101, 1.0)),
        throughputs=(
            (19, 1.0),
            (20, 100.0),
            (40, 120.0),
            (100, 140.0),
            (101, 999.0),
        ),
    )
    polynomial = _probe_scalars(
        times=((19, 99.0), (20, 5.0), (40, 4.0), (100, 3.0), (101, 1.0)),
        throughputs=(
            (19, 1.0),
            (20, 150.0),
            (40, 180.0),
            (100, 210.0),
            (101, 999.0),
        ),
    )

    result = metrics.summarize_pair(native, polynomial, phase="model-probe")

    native_time = result["arms"]["native"][metrics.TIME_METRIC]
    assert [point["step"] for point in native_time["series"]] == [20, 40, 100]
    assert native_time["summary"]["median"] == 8.0
    assert result["arms"]["polynomial"][metrics.TIME_METRIC]["summary"]["median"] == 4.0
    assert (
        result["comparisons"]["end_to_end_time_speedup_native_over_polynomial"]["value"]
        == 2.0
    )
    assert (
        result["comparisons"]["throughput_ratio_polynomial_over_native"]["value"] == 1.5
    )


def test_b5_probe_summary_preserves_candidate_name_and_20_to_80_window() -> None:
    native = _probe_scalars(
        times=((19, 99.0), (20, 8.0), (80, 6.0), (81, 1.0)),
        throughputs=((19, 1.0), (20, 100.0), (80, 140.0), (81, 999.0)),
    )
    candidate = _probe_scalars(
        times=((19, 99.0), (20, 4.0), (80, 2.0), (81, 1.0)),
        throughputs=((19, 1.0), (20, 150.0), (80, 210.0), (81, 999.0)),
    )

    result = metrics.summarize_pair(
        native,
        candidate,
        phase="model-probe",
        candidate_name="pwl2_safe_f16",
        probe_first_step=20,
        probe_last_step=80,
    )

    assert set(result["arms"]) == {"native", "pwl2_safe_f16"}
    assert [
        point["step"]
        for point in result["arms"]["pwl2_safe_f16"][metrics.TIME_METRIC]["series"]
    ] == [20, 80]
    assert result["comparisons"]["end_to_end_time_speedup_native_over_pwl2_safe_f16"][
        "value"
    ] == pytest.approx(7 / 3)
    assert (
        result["comparisons"]["throughput_ratio_pwl2_safe_f16_over_native"]["value"]
        == 1.5
    )


def test_validation_series_and_summaries_are_exported_when_present() -> None:
    scalars = {
        metrics.TRAINING_LOSS_METRIC: _samples((100, 4.0)),
        metrics.VALIDATION_LOSS_METRIC: _samples((300, 2.5), (100, 3.5), (200, 3.0)),
        metrics.VALIDATION_THROUGHPUT_METRIC: _samples(
            (100, 80.0), (200, 100.0), (300, 90.0)
        ),
    }

    result = metrics.summarize_arm(scalars, phase="pretraining")

    loss = result[metrics.VALIDATION_LOSS_METRIC]
    assert loss["series"] == [
        {"step": 100, "value": 3.5},
        {"step": 200, "value": 3.0},
        {"step": 300, "value": 2.5},
    ]
    assert loss["summary"] == {
        "sample_count": 3,
        "first_step": 100,
        "last_step": 300,
        "minimum": 2.5,
        "maximum": 3.5,
        "median": 3.0,
        "last": 2.5,
    }
    assert result[metrics.VALIDATION_THROUGHPUT_METRIC]["summary"]["median"] == 90.0


def test_pretraining_exports_loss_and_optional_progress_and_timing_series() -> None:
    scalars = {
        metrics.TRAINING_LOSS_METRIC: _samples((50, 4.0), (25, 5.0)),
        metrics.TOKENS_SEEN_METRIC: _samples((25, 1024.0), (50, 2048.0)),
        metrics.TIME_METRIC: _samples((25, 2.0), (50, 4.0)),
        metrics.THROUGHPUT_METRIC: _samples((25, 512.0), (50, 256.0)),
    }

    result = metrics.summarize_arm(scalars, phase="pretraining")

    assert result[metrics.TRAINING_LOSS_METRIC]["series"] == [
        {"step": 25, "value": 5.0},
        {"step": 50, "value": 4.0},
    ]
    assert result[metrics.TRAINING_LOSS_METRIC]["summary"]["median"] == 4.5
    assert result[metrics.TOKENS_SEEN_METRIC]["summary"]["last"] == 2048.0
    assert result[metrics.TIME_METRIC]["summary"]["median"] == 3.0
    assert result[metrics.THROUGHPUT_METRIC]["summary"]["median"] == 384.0

    with pytest.raises(metrics.MetricsExportError, match="missing required"):
        metrics.summarize_arm({}, phase="pretraining")


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        (_samples((20, 1.0), (20, 2.0)), "duplicate step"),
        (_samples((20, float("nan"))), "finite numbers"),
        (_samples((20, 0.0)), "must be positive"),
    ],
)
def test_probe_metric_rejects_ambiguous_or_invalid_values(
    samples: tuple[metrics.ScalarSample, ...],
    message: str,
) -> None:
    with pytest.raises(metrics.MetricsExportError, match=message):
        metrics.summarize_series(
            samples,
            first_step=metrics.PROBE_FIRST_STEP,
            last_step=metrics.PROBE_LAST_STEP,
            require_positive=True,
        )


def test_probe_requires_both_timing_metrics_and_an_in_window_sample() -> None:
    with pytest.raises(metrics.MetricsExportError, match="missing required"):
        metrics.summarize_arm(
            {metrics.TIME_METRIC: _samples((20, 1.0))},
            phase="model-probe",
        )

    out_of_window = _probe_scalars(
        times=((19, 1.0), (101, 1.0)),
        throughputs=((20, 1.0),),
    )
    with pytest.raises(metrics.MetricsExportError, match="requested step window"):
        metrics.summarize_arm(out_of_window, phase="model-probe")


def test_arm_directory_must_contain_exactly_one_direct_event_file(
    tmp_path: Path,
) -> None:
    arm = tmp_path / "arm"
    arm.mkdir()
    with pytest.raises(metrics.MetricsExportError, match="exactly one"):
        metrics.resolve_event_file(arm)

    event = arm / "events.out.tfevents.first"
    event.write_bytes(b"first")
    nested = arm / "nested"
    nested.mkdir()
    (nested / "events.out.tfevents.ignored").write_bytes(b"nested")
    assert metrics.resolve_event_file(arm) == event

    (arm / "events.out.tfevents.second").write_bytes(b"second")
    with pytest.raises(metrics.MetricsExportError, match="found 2"):
        metrics.resolve_event_file(arm)


def test_event_reading_can_be_tested_without_tensorboard_and_preserves_sha256(
    tmp_path: Path,
) -> None:
    arm = tmp_path / "arm"
    arm.mkdir()
    payload = b"synthetic event bytes"
    event = arm / "events.out.tfevents.synthetic"
    event.write_bytes(payload)
    calls: list[Path] = []

    def fake_loader(path: Path):
        calls.append(path)
        return {metrics.VALIDATION_LOSS_METRIC: _samples((10, 2.0))}

    result = metrics.read_event_data(arm, scalar_loader=fake_loader)

    assert calls == [event]
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert set(result.scalars) == {metrics.VALIDATION_LOSS_METRIC}


def test_tensorboard_is_an_optional_lazy_import(monkeypatch, tmp_path: Path) -> None:
    real_import = builtins.__import__

    def import_without_tensorboard(name, *args, **kwargs):
        if name == "tensorboard" or name.startswith("tensorboard."):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_tensorboard)
    with pytest.raises(metrics.MetricsExportError, match="optional tensorboard"):
        metrics._load_tensorboard_scalars(tmp_path / "unused-event-file")


def test_document_declares_metric_semantics_and_omits_source_paths() -> None:
    native_scalars = _probe_scalars(
        times=((20, 2.0), (100, 2.0)),
        throughputs=((20, 10.0), (100, 10.0)),
    )
    polynomial_scalars = _probe_scalars(
        times=((20, 1.0), (100, 1.0)),
        throughputs=((20, 20.0), (100, 20.0)),
    )
    document = metrics.build_document(
        case="b1",
        phase="model-probe",
        torchtitan_commit="a" * 40,
        native=metrics.EventData("b" * 64, native_scalars),
        polynomial=metrics.EventData("c" * 64, polynomial_scalars),
    )

    encoded = json.dumps(document)
    assert document["arms"]["native"]["source_event"] == {"sha256": "b" * 64}
    assert "wall-clock end-to-end" in document["measurement"]["timing_semantics"]
    assert (
        "not the historical synchronized CUDA-event"
        in document["measurement"]["historical_timing_distinction"]
    )
    assert "source_path" not in encoded
    assert "event_file" not in encoded


def test_b5_document_records_named_candidate_and_manifest_window() -> None:
    native_scalars = _probe_scalars(
        times=((20, 2.0), (80, 2.0)),
        throughputs=((20, 10.0), (80, 10.0)),
    )
    candidate_scalars = _probe_scalars(
        times=((20, 1.0), (80, 1.0)),
        throughputs=((20, 20.0), (80, 20.0)),
    )
    document = metrics.build_document(
        case="b5",
        phase="model-probe",
        torchtitan_commit="a" * 40,
        native=metrics.EventData("b" * 64, native_scalars),
        polynomial=metrics.EventData("c" * 64, candidate_scalars),
        candidate_name="d2_safe",
        probe_first_step=20,
        probe_last_step=80,
    )

    assert document["candidate"] == "d2_safe"
    assert set(document["arms"]) == {"native", "d2_safe"}
    assert document["measurement"]["model_probe_step_window"] == {
        "first_step": 20,
        "last_step": 80,
        "inclusive": True,
        "summary_statistic": "median",
    }


def test_probe_windows_come_from_public_run_manifest() -> None:
    assert metrics.load_model_probe_window(metrics.DEFAULT_MANIFEST, "b1") == (
        20,
        100,
    )
    assert metrics.load_model_probe_window(metrics.DEFAULT_MANIFEST, "b5") == (
        20,
        80,
    )


def test_export_selection_distinguishes_b5_named_candidates() -> None:
    b5 = metrics.parse_args(
        [
            "--case",
            "b5",
            "--phase",
            "model-probe",
            "--native-dir",
            "native",
            "--candidate-dir",
            "candidate",
            "--candidate-name",
            "pwl2_safe_f16",
            "--output",
            "result.json",
        ]
    )
    assert metrics.resolve_comparison_selection(b5) == (
        Path("candidate"),
        "pwl2_safe_f16",
    )

    with pytest.raises(metrics.MetricsExportError, match="candidate-dir"):
        metrics.resolve_comparison_selection(
            metrics.parse_args(
                [
                    "--case",
                    "b5",
                    "--native-dir",
                    "native",
                    "--polynomial-dir",
                    "polynomial",
                    "--output",
                    "result.json",
                ]
            )
        )

    with pytest.raises(metrics.MetricsExportError, match="B1--B4"):
        metrics.resolve_comparison_selection(
            metrics.parse_args(
                [
                    "--case",
                    "b1",
                    "--native-dir",
                    "native",
                    "--candidate-dir",
                    "candidate",
                    "--candidate-name",
                    "d2_safe",
                    "--output",
                    "result.json",
                ]
            )
        )


def test_atomic_json_write_replaces_output_and_leaves_no_temporary_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "metrics.json"
    metrics.atomic_write_json(output, {"value": 1})
    metrics.atomic_write_json(output, {"value": 2})

    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 2}
    assert list(output.parent.glob(".metrics.json.*.tmp")) == []


def test_invalid_arguments_do_not_echo_potential_credentials(capsys) -> None:
    sentinel_value = "do-not-print-this-secret"
    with pytest.raises(SystemExit) as error:
        metrics.parse_args(["--api-key", sentinel_value])

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert sentinel_value not in captured.out
    assert sentinel_value not in captured.err


def test_main_binds_event_files_to_matching_launch_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native_dir = tmp_path / "native"
    polynomial_dir = tmp_path / "polynomial"
    native_event = _write_event(native_dir, b"native-event")
    polynomial_event = _write_event(polynomial_dir, b"polynomial-event")
    native_receipt = _write_bound_receipt(
        native_dir,
        case="b1",
        variant="native",
        phase="model-probe",
    )
    polynomial_receipt = _write_bound_receipt(
        polynomial_dir,
        case="b1",
        variant="polynomial",
        phase="model-probe",
    )
    scalars = _probe_scalars(
        times=((20, 2.0), (100, 1.0)),
        throughputs=((20, 10.0), (100, 20.0)),
    )

    def fake_read(directory: Path, **kwargs) -> metrics.EventData:
        del kwargs
        event = native_event if directory == native_dir else polynomial_event
        return metrics.EventData(
            hashlib.sha256(event.read_bytes()).hexdigest(), scalars
        )

    monkeypatch.setattr(metrics, "read_event_data", fake_read)
    output = tmp_path / "result.json"
    assert (
        metrics.main(
            [
                "--case",
                "b1",
                "--phase",
                "model-probe",
                "--native-dir",
                str(native_dir),
                "--polynomial-dir",
                str(polynomial_dir),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    binding = document["provenance_binding"]
    assert binding["status"] == "bound"
    assert binding["source"]["repository"] == {
        "revision": "d" * 40,
        "dirty": False,
    }
    assert binding["arms"]["native"]["launch_receipt_sha256"] == (
        hashlib.sha256(native_receipt.read_bytes()).hexdigest()
    )
    assert binding["arms"]["polynomial"]["launch_receipt_sha256"] == (
        hashlib.sha256(polynomial_receipt.read_bytes()).hexdigest()
    )
    assert document["arms"]["native"]["source_event"] == {
        "sha256": hashlib.sha256(native_event.read_bytes()).hexdigest()
    }
    assert document["arms"]["native"]["source_launch_receipt"] == {
        "sha256": hashlib.sha256(native_receipt.read_bytes()).hexdigest()
    }


def test_export_fails_closed_without_receipts_and_diagnostic_is_marked_unbound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native_dir = tmp_path / "native"
    polynomial_dir = tmp_path / "polynomial"
    _write_event(native_dir, b"native-event")
    _write_event(polynomial_dir, b"polynomial-event")
    scalars = _probe_scalars(
        times=((20, 2.0), (100, 1.0)),
        throughputs=((20, 10.0), (100, 20.0)),
    )
    monkeypatch.setattr(
        metrics,
        "read_event_data",
        lambda directory, **kwargs: metrics.EventData("f" * 64, scalars),
    )
    output = tmp_path / "result.json"
    common = [
        "--case",
        "b1",
        "--native-dir",
        str(native_dir),
        "--polynomial-dir",
        str(polynomial_dir),
        "--output",
        str(output),
    ]

    assert metrics.main(common) == 1
    assert not output.exists()
    assert metrics.main([*common, "--allow-unbound-receipts"]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["provenance_binding"]["status"] == "unbound"
    assert document["provenance_binding"]["arms"] == {}
    assert document["torchtitan"]["revision_basis"].endswith(
        "launch provenance is unbound"
    )


def test_receipt_label_mismatch_is_rejected(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidate"
    receipt_path = _write_bound_receipt(
        candidate_dir,
        case="b1",
        variant="native",
        phase="model-probe",
    )
    manifest, manifest_sha256 = metrics.load_protocol_manifest(
        metrics.DEFAULT_MANIFEST,
        "b1",
    )

    with pytest.raises(metrics.MetricsExportError, match="labelled arm"):
        metrics.validate_launch_receipt(
            receipt_path,
            case="b1",
            variant="polynomial",
            phase="model-probe",
            manifest=manifest,
            manifest_sha256=manifest_sha256,
        )


def test_external_receipt_output_binds_only_its_declared_tree(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    event_directory = output_root / "tensorboard" / "run"
    event_directory.mkdir(parents=True)
    receipt = metrics.ValidatedReceipt(
        sha256="a" * 64,
        selection={},
        protocol={},
        config_sha256="b" * 64,
        command_sha256="c" * 64,
        source={},
        runtime={},
        topology={},
        output_folder=runner.normalize_repository_path(
            str(output_root), runner.REPOSITORY_ROOT
        ),
        pair_config={},
        pair_arguments=(),
    )

    metrics.validate_event_location(event_directory, receipt)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(metrics.MetricsExportError, match="outside"):
        metrics.validate_event_location(outside, receipt)


def test_validation_export_requires_matching_receipts_and_both_validation_series(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native_dir = tmp_path / "native"
    polynomial_dir = tmp_path / "polynomial"
    _write_event(native_dir, b"native-validation")
    _write_event(polynomial_dir, b"polynomial-validation")
    _write_bound_receipt(
        native_dir,
        case="b1",
        variant="native",
        phase="pretraining",
        validation=True,
    )
    _write_bound_receipt(
        polynomial_dir,
        case="b1",
        variant="polynomial",
        phase="pretraining",
        validation=True,
    )
    native_scalars = {
        metrics.TRAINING_LOSS_METRIC: _samples((1, 3.0)),
        metrics.VALIDATION_LOSS_METRIC: _samples((1, 2.5)),
    }
    polynomial_scalars = {
        metrics.TRAINING_LOSS_METRIC: _samples((1, 3.1)),
    }

    def fake_read(directory: Path, **kwargs) -> metrics.EventData:
        del kwargs
        selected = native_scalars if directory == native_dir else polynomial_scalars
        return metrics.EventData(
            "a" * 64 if directory == native_dir else "b" * 64, selected
        )

    monkeypatch.setattr(metrics, "read_event_data", fake_read)
    output = tmp_path / "validation.json"
    arguments = [
        "--case",
        "b1",
        "--phase",
        "pretraining",
        "--require-validation",
        "--native-dir",
        str(native_dir),
        "--polynomial-dir",
        str(polynomial_dir),
        "--output",
        str(output),
    ]
    assert metrics.main(arguments) == 1
    assert not output.exists()

    polynomial_scalars[metrics.VALIDATION_LOSS_METRIC] = _samples((1, 2.6))
    assert metrics.main(arguments) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["provenance_binding"]["status"] == "bound"
    assert document["provenance_binding"]["validation"] is True


def test_require_validation_rejects_nonvalidation_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native_dir = tmp_path / "native"
    polynomial_dir = tmp_path / "polynomial"
    _write_event(native_dir, b"native")
    _write_event(polynomial_dir, b"polynomial")
    _write_bound_receipt(
        native_dir,
        case="b1",
        variant="native",
        phase="pretraining",
    )
    _write_bound_receipt(
        polynomial_dir,
        case="b1",
        variant="polynomial",
        phase="pretraining",
    )
    scalars = {
        metrics.TRAINING_LOSS_METRIC: _samples((1, 3.0)),
        metrics.VALIDATION_LOSS_METRIC: _samples((1, 2.5)),
    }
    monkeypatch.setattr(
        metrics,
        "read_event_data",
        lambda directory, **kwargs: metrics.EventData("a" * 64, scalars),
    )

    output = tmp_path / "validation.json"
    assert (
        metrics.main(
            [
                "--case",
                "b1",
                "--phase",
                "pretraining",
                "--require-validation",
                "--native-dir",
                str(native_dir),
                "--polynomial-dir",
                str(polynomial_dir),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()

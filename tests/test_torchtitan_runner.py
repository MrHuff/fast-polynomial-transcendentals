from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_torchtitan as runner
from sfu_repro.torchtitan.config import JobConfig, SFUConfig


ROOT = Path(__file__).resolve().parents[1]


def _arguments(*values: str):
    return runner.parse_args(list(values))


def _command(*values: str) -> list[str]:
    manifest = runner.load_manifest(runner.DEFAULT_MANIFEST)
    command, _ = runner.build_command(_arguments(*values), manifest)
    return command


def _load_toml(relative: str) -> dict:
    with (ROOT / relative).open("rb") as stream:
        return tomllib.load(stream)


def test_custom_config_annotation_is_resolvable_by_torchtitan() -> None:
    # TorchTitan recursively merges actual dataclass types. A postponed string
    # annotation here fails inside tyro before a run starts.
    assert JobConfig.__annotations__["sfu"] is SFUConfig


def test_model_probe_emits_valid_boolean_overrides() -> None:
    command = _command(
        "--case",
        "b1",
        "--variant",
        "native",
        "--phase",
        "model-probe",
    )
    assert "--checkpoint.no-enable" in command
    assert "--validation.no-enable" in command
    assert "--compile.enable" in command
    assert "--lr-scheduler.total-steps=100" in command
    dataset_path = ROOT / "torchtitan/tests/assets/c4_test"
    assert dataset_path.is_dir()
    assert f"--training.dataset-path={dataset_path}" in command
    assert not any(".enable.enable" in item for item in command)
    assert not any(".enable.no-enable" in item for item in command)


def test_seed_checkpoints_are_case_specific() -> None:
    b1 = _command(
        "--case",
        "b1",
        "--variant",
        "native",
        "--phase",
        "seed-checkpoint",
    )
    b3 = _command(
        "--case",
        "b3",
        "--variant",
        "native",
        "--phase",
        "seed-checkpoint",
    )
    assert "--job.dump-folder=./outputs/torchtitan/seeds/b1_llama3_8b" in b1
    assert "--job.dump-folder=./outputs/torchtitan/seeds/b3_llama3_8b" in b3


def test_pretraining_requires_a_shared_seed_checkpoint() -> None:
    with pytest.raises(ValueError, match="requires --seed-checkpoint"):
        _command(
            "--case",
            "b1",
            "--variant",
            "native",
            "--phase",
            "pretraining",
        )


def test_seed_checkpoint_identity_hashes_contents_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    shard = checkpoint / "model-shard.bin"
    shard.write_bytes(b"first")

    first = runner._checkpoint_tree_identity(checkpoint)
    shard.write_bytes(b"second")
    second = runner._checkpoint_tree_identity(checkpoint)
    assert first["sha256"] != second["sha256"]
    assert first["file_count"] == second["file_count"] == 1
    assert first["path"].startswith("${EXTERNAL_PATH:")

    (checkpoint / "unsafe-link").symlink_to(shard)
    with pytest.raises(RuntimeError, match="symbolic links"):
        runner._checkpoint_tree_identity(checkpoint)


def test_manifest_token_horizons_match_configs() -> None:
    manifest = runner.load_manifest(runner.DEFAULT_MANIFEST)
    for case, record in manifest["cases"].items():
        if "pretraining" not in record:
            continue
        phase = record["pretraining"]
        assert phase["tokens"] == (
            phase["steps"] * phase["global_batch_size"] * phase["sequence_length"]
        )
        for variant, relative in record["configs"].items():
            config = _load_toml(relative)
            assert config["sfu"] == {
                "case": case,
                "variant": variant,
                "strict": True,
            }
            assert config["training"]["steps"] == phase["steps"]
            assert config["training"]["global_batch_size"] == phase["global_batch_size"]
            assert config["training"]["local_batch_size"] == phase["local_batch_size"]
            assert config["training"]["seq_len"] == phase["sequence_length"]
            world_size = phase["world_size"]
            assert (
                phase["global_batch_size"] % (phase["local_batch_size"] * world_size)
                == 0
            )


def test_public_dataset_snapshots_are_immutable_and_match_plugin() -> None:
    manifest = runner.load_manifest(runner.DEFAULT_MANIFEST)
    datasets = manifest["dataset_snapshots"]

    from sfu_repro.torchtitan import pins

    slim = datasets["b1_b3_training_and_all_validation"]
    assert slim["repository"] == pins.SLIMPAJAMA_REPOSITORY
    assert slim["revision"] == pins.SLIMPAJAMA_REVISION
    assert len(slim["revision"]) == 40

    olmo = datasets["b4_training"]
    assert olmo["repository"] == pins.OLMO_MIX_REPOSITORY
    assert olmo["revision"] == pins.OLMO_MIX_REVISION
    assert len(olmo["revision"]) == 40

    tokenizers = manifest["tokenizer_snapshots"]
    assert tokenizers["b1_b3"]["repository"] == pins.LLAMA_TOKENIZER_REPOSITORY
    assert tokenizers["b1_b3"]["revision"] == pins.LLAMA_TOKENIZER_REVISION
    assert tokenizers["b4"]["repository"] == pins.DEEPSEEK_TOKENIZER_REPOSITORY
    assert tokenizers["b4"]["revision"] == pins.DEEPSEEK_TOKENIZER_REVISION


def test_native_and_polynomial_configs_change_only_variant_metadata() -> None:
    manifest = runner.load_manifest(runner.DEFAULT_MANIFEST)
    for case, record in manifest["cases"].items():
        if "polynomial" not in record["configs"]:
            continue
        native = _load_toml(record["configs"]["native"])
        polynomial = _load_toml(record["configs"]["polynomial"])
        for document in (native, polynomial):
            document["job"]["dump_folder"] = "<variant>"
            document["job"]["description"] = "<variant>"
            document["sfu"]["variant"] = "<variant>"
        assert native == polynomial, case


def test_b5_manifest_and_configs_match_retained_public_probe() -> None:
    manifest = runner.load_manifest(runner.DEFAULT_MANIFEST)
    record = manifest["cases"]["b5"]
    assert record["scope"] == "new-public-rerun"
    assert record["configs"] == {
        "native": "configs/torchtitan/b5_native.toml",
        "pwl2_safe_f16": "configs/torchtitan/b5_pwl2_safe_f16.toml",
        "d2_safe": "configs/torchtitan/b5_d2_safe.toml",
    }
    assert record["exp2_routes"] == {
        "native": {
            "backend": "d3",
            "forward_frequency": 0,
            "backward_frequency": 0,
        },
        "pwl2_safe_f16": {
            "backend": "pwl2_safe_f16",
            "forward_frequency": 12,
            "backward_frequency": 32,
        },
        "d2_safe": {
            "backend": "d2_safe",
            "forward_frequency": 12,
            "backward_frequency": 32,
        },
    }
    assert record["model_probe"] == {
        "world_size": 1,
        "steps": 80,
        "local_batch_size": 1,
        "global_batch_size": 1,
        "sequence_length": 4096,
        "dataset": "c4_test",
        "dataset_path": "torchtitan/tests/assets/c4_test",
        "compile": False,
        "steady_state_first_step": 20,
    }
    assert record["protocol_provenance"] == {
        "model_shape_batch_sequence_steps_and_route_fractions": ("retained-artifact"),
        "integer_route_frequencies": ("retained-component-artifact-and-source-derived"),
        "compile_disabled": "historical-source-derived-not-runtime-attested",
        "steady_state_first_step": ("retained-for-d2-and-new-public-choice-for-pwl2"),
        "seed_dataset_tokenizer_and_torchtitan_runtime": "new-public-protocol",
    }
    assert record["retained_evidence"] == [
        "evidence/report-data/b5_exp2_pwl2_safe_8b_probe_fwd_bwd_gb200.json",
        "evidence/report-data/b5_exp2_d2_safe_8b_probe_fwd_bwd_gb200.json",
        "evidence/report-data/b5_exp2_apples_to_apples_d3_pwl2_d2_gb200.json",
    ]
    assert all((ROOT / relative).is_file() for relative in record["retained_evidence"])

    normalized: list[dict] = []
    for variant, relative in record["configs"].items():
        config = _load_toml(relative)
        assert config["model"]["name"] == "sfu_llama3"
        assert config["model"]["flavor"] == "8B"
        assert config["training"]["dataset"] == "c4_test"
        assert config["training"]["local_batch_size"] == 1
        assert config["training"]["global_batch_size"] == 1
        assert config["training"]["seq_len"] == 4096
        assert config["training"]["steps"] == 80
        assert config["compile"]["enable"] is False
        assert config["sfu"] == {
            "case": "b5",
            "variant": variant,
            "strict": True,
        }
        config["job"]["dump_folder"] = "<variant>"
        config["job"]["description"] = "<variant>"
        config["sfu"]["variant"] = "<variant>"
        normalized.append(config)

    assert normalized[1:] == [normalized[0], normalized[0]]


@pytest.mark.parametrize("variant", ("native", "pwl2_safe_f16", "d2_safe"))
def test_b5_model_probe_command_is_exact_and_compile_disabled(variant: str) -> None:
    command = _command(
        "--case",
        "b5",
        "--variant",
        variant,
        "--phase",
        "model-probe",
    )
    config = ROOT / f"configs/torchtitan/b5_{variant}.toml"
    dataset_path = ROOT / "torchtitan/tests/assets/c4_test"
    assert f"--job.config-file={config}" in command
    assert "--training.steps=80" in command
    assert "--lr-scheduler.total-steps=80" in command
    assert "--training.local-batch-size=1" in command
    assert "--training.global-batch-size=1" in command
    assert "--training.seq-len=4096" in command
    assert "--training.dataset=c4_test" in command
    assert f"--training.dataset-path={dataset_path}" in command
    assert "--compile.no-enable" in command
    assert "--compile.enable" not in command
    assert "--checkpoint.no-enable" in command
    assert "--validation.no-enable" in command
    assert f"--job.dump-folder=./outputs/torchtitan/probes/b5/{variant}" in command


@pytest.mark.parametrize("phase", ("pretraining", "seed-checkpoint"))
def test_b5_rejects_undefined_training_phases(phase: str) -> None:
    with pytest.raises(ValueError, match="only the new public model-probe"):
        _command(
            "--case",
            "b5",
            "--variant",
            "native",
            "--phase",
            phase,
        )


def test_b4_probe_requires_paper_world_size_without_override() -> None:
    with pytest.raises(ValueError, match="expects world size 4"):
        _command(
            "--case",
            "b4",
            "--variant",
            "native",
            "--phase",
            "model-probe",
            "--nproc-per-node",
            "1",
            "--nnodes",
            "1",
        )


def test_launcher_rejects_nonpositive_topology_and_validation_counts() -> None:
    with pytest.raises(SystemExit):
        _arguments("--case", "b1", "--variant", "native", "--nnodes", "0")
    with pytest.raises(SystemExit):
        _arguments(
            "--case",
            "b1",
            "--variant",
            "native",
            "--validation-steps",
            "-1",
        )


def test_changed_world_size_requires_valid_batch_and_expert_divisibility() -> None:
    with pytest.raises(ValueError, match="global-batch-size must be divisible"):
        _command(
            "--case",
            "b1",
            "--variant",
            "native",
            "--phase",
            "pretraining",
            "--seed-checkpoint",
            str(ROOT / "outputs/test-seed/checkpoint/step-0"),
            "--nproc-per-node",
            "3",
            "--nnodes",
            "1",
            "--allow-world-size-change",
        )


def test_tokenizer_manifest_verification_checks_revision_and_hashes(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    config = tmp_path / "tokenizer_config.json"
    tokenizer.write_text("tokenizer", encoding="utf-8")
    config.write_text("config", encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "schema_version": 1,
        "asset_type": "tokenizer",
        "repository": "owner/model",
        "revision": "a" * 40,
        "files": [
            {"path": tokenizer.name, "sha256": digest(tokenizer)},
            {"path": config.name, "sha256": digest(config)},
        ],
    }
    (tmp_path / "tokenizer-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    runner.verify_tokenizer_assets(
        tmp_path,
        expected_repository="owner/model",
        expected_revision="a" * 40,
    )

    tokenizer.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        runner.verify_tokenizer_assets(
            tmp_path,
            expected_repository="owner/model",
            expected_revision="a" * 40,
        )

    with pytest.raises(ValueError, match="expert-parallel degree"):
        _command(
            "--case",
            "b4",
            "--variant",
            "native",
            "--phase",
            "model-probe",
            "--nproc-per-node",
            "6",
            "--nnodes",
            "1",
            "--allow-world-size-change",
            "--config-arg=--training.global-batch-size=6",
        )


def test_torchtitan_submodule_matches_manifest_pin() -> None:
    manifest = runner.load_manifest(runner.DEFAULT_MANIFEST)
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT / "torchtitan",
        text=True,
    ).strip()
    assert observed == manifest["torchtitan_commit"]
    assert manifest["flash_attention_commit"] == (
        "38afdedda24b0bf26e6904d3bed7807c19a6906e"
    )


def test_launch_receipt_binds_config_sources_and_redacts_transport_hosts() -> None:
    args = _arguments(
        "--case",
        "b1",
        "--variant",
        "native",
        "--phase",
        "model-probe",
        "--torchrun-arg=--rdzv-endpoint=private.scheduler.example:29400",
        "--torchrun-arg=--rdzv-id=private-run-name",
    )
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
        selected_assets=ROOT / "assets/hf/Llama-3.1-8B",
        tokenizer_manifest_sha256="e" * 64,
        tokenizer_verified=True,
    )

    encoded = json.dumps(receipt)
    assert "private.scheduler.example" not in encoded
    assert "private-run-name" not in encoded
    assert "<redacted-host>" in encoded
    assert "<redacted-transport-id>" in encoded
    assert receipt["binding"] == {"status": "bound", "reasons": []}
    assert receipt["selection"] == {
        "case": "b1",
        "variant": "native",
        "phase": "model-probe",
        "validation": False,
    }
    assert base64.b64decode(receipt["config"]["bytes_base64"]) == config_payload
    assert receipt["config"]["sha256"] == hashlib.sha256(config_payload).hexdigest()
    assert receipt["source"]["repository"] == {
        "revision": "d" * 40,
        "dirty": False,
    }
    assert receipt["topology"]["world_size"] == 1


def test_launch_receipt_is_create_only_and_multi_node_idempotent(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / runner.RECEIPT_FILENAME
    document = {"schema_version": 1, "value": "same-on-every-node"}
    first = runner.write_receipt_once(receipt_path, document)
    second = runner.write_receipt_once(receipt_path, document)

    assert first == second == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    with pytest.raises(runner.ReceiptError, match="different launch receipt"):
        runner.write_receipt_once(
            receipt_path, {"schema_version": 1, "value": "changed"}
        )


def test_receipt_command_sanitizer_never_retains_secret_options() -> None:
    sentinel = "do-not-store-this-credential"
    sanitized, redactions, secret_redacted = runner.sanitize_command(
        ["torchrun", f"--model.api-key={sentinel}", "-m", "torchtitan.train"],
        repository_root=ROOT,
    )

    assert sentinel not in json.dumps(sanitized)
    assert sanitized[1] == "--model.api-key=<redacted-secret>"
    assert "secret" in redactions
    assert secret_redacted is True


def test_external_receipt_paths_are_anonymous_but_stable(tmp_path: Path) -> None:
    first = tmp_path / "first" / "results"
    second = tmp_path / "second" / "results"

    first_label = runner.normalize_repository_path(str(first), ROOT)
    assert first_label.startswith("${EXTERNAL_PATH:")
    assert first_label.endswith("}/results")
    assert str(tmp_path) not in first_label
    assert first_label == runner.normalize_repository_path(str(first), ROOT)
    assert first_label != runner.normalize_repository_path(str(second), ROOT)


def test_execute_writes_receipt_and_only_prints_sanitized_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = runner.load_manifest(runner.DEFAULT_MANIFEST)

    def clean_state(path: Path) -> tuple[str, bool]:
        if path.name == "torchtitan":
            return manifest["torchtitan_commit"], False
        if path.name == "flash-attention":
            return manifest["flash_attention_commit"], False
        return "d" * 40, False

    launched: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        launched.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner, "git_checkout_state", clean_state)
    monkeypatch.setattr(runner, "verify_tokenizer_assets", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_tokenizer_manifest_digest", lambda path: "e" * 64)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    receipt_path = tmp_path / "launch.json"
    seed_checkpoint = tmp_path / "seed-checkpoint"
    seed_checkpoint.mkdir()
    (seed_checkpoint / "model-shard.bin").write_bytes(b"shared-seed-weights")
    exit_code = runner.main(
        [
            "--case",
            "b1",
            "--variant",
            "native",
            "--phase",
            "pretraining",
            "--seed-checkpoint",
            str(seed_checkpoint),
            "--torchrun-arg=--rdzv-endpoint=private.scheduler.example:29400",
            "--torchrun-arg=--rdzv-id=private-run-name",
            "--receipt-path",
            str(receipt_path),
            "--execute",
        ]
    )

    assert exit_code == 0
    assert len(launched) == 1
    assert "private.scheduler.example" in " ".join(launched[0])
    output = capsys.readouterr().out
    assert "private.scheduler.example" not in output
    assert "private-run-name" not in output
    assert "<redacted-host>" in output
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["binding"]["status"] == "bound"
    assert receipt["protocol"]["seed_checkpoint"]["file_count"] == 1
    assert receipt["protocol"]["seed_checkpoint"]["sha256"]
    assert receipt["topology"] == {
        "nproc_per_node": 8,
        "nnodes": 4,
        "world_size": 32,
    }

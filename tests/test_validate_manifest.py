from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from scripts import validate_manifest as manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_MANIFEST = REPOSITORY_ROOT / "repro/experiments.json"
REQUIRED_WORKFLOWS = {
    "activation-bf16-header-generation",
    "activation-fp16-bf16-fit-sweeps",
    "b2-tanh-backend-fit",
    "gelu-fp16-fit",
    "b3-deployed-fit-audit",
    "open-weight-quality-evaluation",
    "open-weight-throughput-evaluation",
    "open-weight-result-summary",
    "b5-routed-exp2-torchtitan-model-probe-public-rerun",
    "flash-sigmoid-sequence-specific-d2-fit",
    "rope-polynomial-fitting",
    "rope-portable-numerical-check",
    "rope-cache-hbm-and-repeated-evaluation",
    "rope-table-and-repeated-evaluator",
    "rope-fused-integration",
    "rope-sass-audit",
}
REAL_COMMAND_FIELDS = (
    "command",
    "setup_commands",
    "additional_commands",
    "postprocess_commands",
)


def _load_release_manifest() -> dict[str, object]:
    return json.loads(EXPERIMENT_MANIFEST.read_text(encoding="utf-8"))


def _command_semantics() -> dict[str, str]:
    return {
        "command": "real primary command",
        "preview_command": "safe non-executing preview",
        "setup_commands": "real setup commands",
        "additional_commands": "real peer commands",
        "postprocess_commands": "real post-processing commands",
    }


def _minimal_document(repository_root: Path) -> dict[str, object]:
    scripts = repository_root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "tool.py").write_text("pass\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "command_semantics": _command_semantics(),
        "experiments": [
            {
                "id": "test-workflow",
                "status": "runnable",
                "command": [
                    "python",
                    "scripts/tool.py",
                    "--output",
                    "outputs/result.json",
                ],
                "prerequisites": {
                    "repository_paths": ["scripts/tool.py"],
                    "generated_inputs": [],
                    "software": ["Python"],
                    "hardware": [],
                    "external_assets": [],
                },
                "workflow": {
                    "kind": "test",
                    "output_class": "derived-artifact",
                    "outputs": ["outputs/result.json"],
                    "substitutions": {},
                    "notes": "Test fixture.",
                },
            }
        ],
    }


def _real_commands(experiment: dict[str, object]) -> list[list[str]]:
    commands: list[list[str]] = []
    primary = experiment.get("command")
    if isinstance(primary, list) and primary:
        commands.append(primary)
    for field in REAL_COMMAND_FIELDS[1:]:
        nested = experiment.get(field, [])
        assert isinstance(nested, list)
        commands.extend(nested)
    return commands


def test_release_manifest_is_valid_and_covers_public_workflows() -> None:
    document = _load_release_manifest()

    assert manifest.validate(document, repository_root=REPOSITORY_ROOT) == []
    experiment_ids = {experiment["id"] for experiment in document["experiments"]}
    assert REQUIRED_WORKFLOWS <= experiment_ids


def test_release_manifest_keeps_preview_and_execution_semantics_distinct() -> None:
    document = _load_release_manifest()

    for experiment in document["experiments"]:
        for command in _real_commands(experiment):
            assert "--dry-run" not in command
            if Path(command[1]).name == "run_torchtitan.py":
                assert "--execute" in command

        preview = experiment.get("preview_command")
        if preview is None:
            continue
        assert "--execute" not in preview
        if Path(preview[1]).name == "run_open_weight_suite.py":
            assert "--dry-run" in preview


def test_downstream_matrix_uses_one_pinned_interpreter_per_profile() -> None:
    document = _load_release_manifest()
    experiments = {item["id"]: item for item in document["experiments"]}

    for experiment_id in (
        "open-weight-quality-evaluation",
        "open-weight-throughput-evaluation",
    ):
        experiment = experiments[experiment_id]
        commands = _real_commands(experiment)
        assert len(commands) == 3
        assert all("paper" not in command for command in commands)
        interpreters = {command[command.index("--python") + 1] for command in commands}
        assert interpreters == {
            "HF_4_48_PYTHON",
            "HF_5_9_PYTHON",
            "HF_4_57_PYTHON",
        }


def test_validation_workflow_covers_every_b1_b4_pair() -> None:
    document = _load_release_manifest()
    experiment = next(
        item
        for item in document["experiments"]
        if item["id"] == "torchtitan-held-out-validation"
    )
    commands = _real_commands(experiment)
    run_commands = [
        command
        for command in commands
        if Path(command[1]).name == "run_torchtitan.py" and "pretraining" in command
    ]
    selections = {
        (
            command[command.index("--case") + 1],
            command[command.index("--variant") + 1],
        )
        for command in run_commands
    }

    assert selections == {
        (case, variant)
        for case in ("b1", "b2", "b3", "b4")
        for variant in ("native", "polynomial")
    }
    assert all("--validation" in command for command in run_commands)
    assert len(experiment["postprocess_commands"]) == 4


def test_validator_rejects_missing_entrypoint_and_escaping_output(
    tmp_path: Path,
) -> None:
    document = _minimal_document(tmp_path)
    missing = deepcopy(document)
    missing["experiments"][0]["command"][1] = "scripts/missing.py"

    failures = manifest.validate(missing, repository_root=tmp_path)
    assert any(
        "entry point is missing: scripts/missing.py" in item for item in failures
    )

    escaping = deepcopy(document)
    escaping["experiments"][0]["command"][-1] = "../escape.json"

    failures = manifest.validate(escaping, repository_root=tmp_path)
    assert any("path escapes repository: ../escape.json" in item for item in failures)


def test_validator_rejects_real_preview_semantic_confusion(tmp_path: Path) -> None:
    document = _minimal_document(tmp_path)
    scripts = tmp_path / "scripts"
    (scripts / "run_torchtitan.py").write_text("pass\n", encoding="utf-8")
    (scripts / "run_open_weight_suite.py").write_text("pass\n", encoding="utf-8")

    torchtitan = deepcopy(document)
    torchtitan["experiments"][0]["command"] = [
        "python",
        "scripts/run_torchtitan.py",
    ]
    failures = manifest.validate(torchtitan, repository_root=tmp_path)
    assert any(
        "TorchTitan execution must contain --execute" in item for item in failures
    )

    downstream = deepcopy(document)
    downstream["experiments"][0]["command"] = [
        "python",
        "scripts/run_open_weight_suite.py",
        "--dry-run",
    ]
    failures = manifest.validate(downstream, repository_root=tmp_path)
    assert any("real command but contains --dry-run" in item for item in failures)

    preview = deepcopy(document)
    preview["experiments"][0]["command"] = [
        "python",
        "scripts/run_open_weight_suite.py",
    ]
    preview["experiments"][0]["preview_command"] = [
        "python",
        "scripts/run_open_weight_suite.py",
    ]
    failures = manifest.validate(preview, repository_root=tmp_path)
    assert any(
        "open-weight preview must contain --dry-run" in item for item in failures
    )


def test_validator_checks_paths_in_nested_command_arrays(tmp_path: Path) -> None:
    document = _minimal_document(tmp_path)
    experiment = document["experiments"][0]
    experiment["additional_commands"] = [
        [
            "python",
            "scripts/tool.py",
            "--config",
            "configs/missing.json",
            "--output",
            "outputs/peer.json",
        ]
    ]
    experiment["workflow"]["outputs"].append("outputs/peer.json")

    failures = manifest.validate(document, repository_root=tmp_path)
    assert any(
        "additional_commands[0] --config references a missing input path: "
        "configs/missing.json" in item
        for item in failures
    )

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import check_open_weight_environment as checker
from scripts import run_open_weight_suite as suite


def loaded_inputs() -> tuple[suite.SuiteConfig, dict[str, object]]:
    return suite.load_config(), checker.load_environment_profiles(
        checker.DEFAULT_ENVIRONMENTS
    )


def test_quality_protocol_reports_harness_seeds_pins_and_all_profiles() -> None:
    config, environments = loaded_inputs()

    report = checker.inspect_quality_task_protocol(
        config,
        environments,
        check_installed=False,
    )

    assert report["path"] == "configs/lm_eval_paper_tasks.json"
    assert report["all_quality_tasks_covered"] is True
    assert report["harness"] == {
        "distribution": "lm-eval",
        "version": "0.4.12",
        "repository": "https://github.com/EleutherAI/lm-evaluation-harness",
        "revision": "6d642546f4688648fced259eb3302efd36ece5af",
        "submodule_path": "lm-evaluation-harness",
    }
    assert report["evaluator_seeds"] == {
        "random_seed": 0,
        "numpy_random_seed": 1234,
        "torch_random_seed": 1234,
        "fewshot_random_seed": 1234,
    }
    assert tuple(report["dataset_pins"]) == config.protocol.quality_tasks
    assert report["dataset_pins"]["mmlu"] == {
        "dataset_path": "cais/mmlu",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "revision_provenance": "public-protocol-selection",
        "num_fewshot": 5,
    }
    assert len(report["environment_profile_versions"]) == 3
    assert all(
        item["declared_version"] == "0.4.12" and item["matches_protocol"] is True
        for item in report["environment_profile_versions"]
    )
    assert report["harness_checkout"]["checked"] is False


def test_quality_protocol_rejects_an_uncovered_quality_task(tmp_path: Path) -> None:
    document = json.loads(suite.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    document["protocol"]["quality_tasks"].append("uncovered_task")
    document["protocol"]["quality_fewshot"]["uncovered_task"] = 0
    config_path = tmp_path / "open_weight.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    config = suite.load_config(config_path)
    environments = checker.load_environment_profiles(checker.DEFAULT_ENVIRONMENTS)

    with pytest.raises(ValueError, match="does not cover: uncovered_task"):
        checker.inspect_quality_task_protocol(
            config,
            environments,
            check_installed=False,
        )


def test_profile_lm_eval_mismatch_is_a_readiness_failure(tmp_path: Path) -> None:
    config, environments = loaded_inputs()
    environments = copy.deepcopy(environments)
    requirements = tmp_path / "mismatch.requirements.txt"
    requirements.write_text("lm-eval==0.4.11\n", encoding="utf-8")
    environments["profiles"]["hf_4_48_qwen2p5"]["requirements"] = str(requirements)

    task_report = checker.inspect_quality_task_protocol(
        config,
        environments,
        check_installed=False,
    )
    report = {
        "quality_task_protocol": task_report,
        "check_installed": False,
        "model_revisions": [],
    }

    assert task_report["environment_profile_versions"][0]["matches_protocol"] is False
    assert checker.failures(report, allow_unrecorded_revisions=False) == [
        "hf_4_48_qwen2p5: requirements declare lm-eval 0.4.11, expected "
        "0.4.12 from the task protocol"
    ]


def test_installed_check_reports_verified_harness_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, environments = loaded_inputs()
    calls: list[tuple[str, str, Path]] = []

    def checkout_state(protocol: object) -> SimpleNamespace:
        return SimpleNamespace(
            revision=protocol.harness_revision,
            clean=True,
            module_file="lm_eval/__init__.py",
        )

    def inspect(protocol: object, repository_root: Path) -> SimpleNamespace:
        calls.append(("inspect", protocol.harness_revision, repository_root))
        return checkout_state(protocol)

    def activate(protocol: object, repository_root: Path) -> SimpleNamespace:
        calls.append(("activate", protocol.harness_revision, repository_root))
        return checkout_state(protocol)

    monkeypatch.setattr(checker, "inspect_harness_checkout", inspect)
    monkeypatch.setattr(checker, "activate_harness_checkout", activate)
    report = checker.inspect_quality_task_protocol(
        config,
        environments,
        check_installed=True,
    )

    assert calls == [
        (
            "inspect",
            "6d642546f4688648fced259eb3302efd36ece5af",
            checker.REPOSITORY_ROOT,
        ),
        (
            "activate",
            "6d642546f4688648fced259eb3302efd36ece5af",
            checker.REPOSITORY_ROOT,
        ),
    ]
    assert report["harness_checkout"] == {
        "checked": True,
        "path": "lm-evaluation-harness",
        "expected_revision": "6d642546f4688648fced259eb3302efd36ece5af",
        "revision": "6d642546f4688648fced259eb3302efd36ece5af",
        "clean": True,
        "module_file": "lm_eval/__init__.py",
        "matches": True,
        "error": None,
    }


def test_installed_check_retains_git_state_when_source_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, environments = loaded_inputs()
    revision = "6d642546f4688648fced259eb3302efd36ece5af"
    monkeypatch.setattr(
        checker,
        "inspect_harness_checkout",
        lambda protocol, repository_root: SimpleNamespace(
            revision=revision,
            clean=True,
            module_file=None,
        ),
    )

    def fail_import(protocol: object, repository_root: Path) -> None:
        raise RuntimeError("lm_eval did not import from the pinned checkout")

    monkeypatch.setattr(checker, "activate_harness_checkout", fail_import)
    report = checker.inspect_quality_task_protocol(
        config,
        environments,
        check_installed=True,
    )

    assert report["harness_checkout"]["revision"] == revision
    assert report["harness_checkout"]["clean"] is True
    assert report["harness_checkout"]["matches"] is False
    assert "did not import" in report["harness_checkout"]["error"]


def test_main_json_includes_quality_protocol(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert checker.main(["--models", "qwen2p5_7b", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["declared_inputs_ready"] is True
    assert report["quality_task_protocol"]["all_quality_tasks_covered"] is True
    assert len(report["quality_task_protocol"]["dataset_pins"]) == 8

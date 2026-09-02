from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import lm_eval_task_pins as pins


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG = REPOSITORY_ROOT / "configs" / "lm_eval_paper_tasks.json"


def test_public_task_protocol_pins_all_paper_tasks_and_default_seeds() -> None:
    protocol = pins.load_task_protocol(TASK_CONFIG)
    assert tuple(protocol.tasks) == (
        "mmlu",
        "hellaswag",
        "arc_challenge",
        "winogrande",
        "piqa",
        "gsm8k",
        "truthfulqa_mc2",
        "wikitext",
    )
    assert protocol.harness_version == "0.4.12"
    assert protocol.harness_revision == ("6d642546f4688648fced259eb3302efd36ece5af")
    assert protocol.harness_submodule_path == "lm-evaluation-harness"
    assert protocol.evaluator_seeds == {
        "random_seed": 0,
        "numpy_random_seed": 1234,
        "torch_random_seed": 1234,
        "fewshot_random_seed": 1234,
    }
    assert all(
        pin.revision_provenance == "public-protocol-selection"
        for pin in protocol.tasks.values()
    )


def test_task_protocol_rejects_an_unpinned_requested_task() -> None:
    protocol = pins.load_task_protocol(TASK_CONFIG)
    with pytest.raises(ValueError, match="does not cover"):
        protocol.select(["not_a_paper_task"])


def test_dataset_loader_enforces_revision_and_restores_original() -> None:
    protocol = pins.load_task_protocol(TASK_CONFIG)
    selected = protocol.select(["mmlu", "hellaswag"])
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def original(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        return "dataset"

    dataset_module = SimpleNamespace(load_dataset=original)
    with pins.enforce_dataset_revisions(dataset_module, selected) as observed:
        assert (
            dataset_module.load_dataset(path="cais/mmlu", name="abstract_algebra")
            == "dataset"
        )
        assert dataset_module.load_dataset("Rowan/hellaswag", None) == "dataset"
        with pytest.raises(RuntimeError, match="unpinned dataset"):
            dataset_module.load_dataset(path="unknown/data")

    assert dataset_module.load_dataset is original
    assert calls[0][1]["revision"] == ("c30699e8356da336a370243923dbaf21066bb9fe")
    assert calls[1][1]["revision"] == ("218ec52e09a7e7462a5400043bb9a69a41d06b76")
    pins.require_all_datasets_observed(selected, observed)


def test_dataset_loader_rejects_conflicting_revision() -> None:
    protocol = pins.load_task_protocol(TASK_CONFIG)
    selected = protocol.select(["piqa"])
    dataset_module = SimpleNamespace(load_dataset=lambda *args, **kwargs: None)
    with pins.enforce_dataset_revisions(dataset_module, selected):
        with pytest.raises(RuntimeError, match="public protocol pins"):
            dataset_module.load_dataset(
                path="baber/piqa",
                revision="0" * 40,
            )


def test_observation_check_rejects_a_dataset_that_was_not_loaded() -> None:
    protocol = pins.load_task_protocol(TASK_CONFIG)
    selected = protocol.select(["wikitext"])
    with pytest.raises(RuntimeError, match="did not load"):
        pins.require_all_datasets_observed(selected, {})


def _initialize_harness_checkout(path: Path) -> str:
    path.mkdir()
    (path / "lm_eval").mkdir()
    (path / "lm_eval" / "__init__.py").write_text("", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def test_harness_checkout_requires_observed_revision_and_clean_tree(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "harness"
    revision = _initialize_harness_checkout(checkout)
    protocol = replace(
        pins.load_task_protocol(TASK_CONFIG),
        harness_submodule_path="harness",
        harness_revision=revision,
    )

    state = pins.inspect_harness_checkout(protocol, tmp_path)
    assert state.revision == revision
    assert state.clean is True
    assert state.relative_path == "harness"

    (checkout / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="local modifications or untracked"):
        pins.inspect_harness_checkout(protocol, tmp_path)


def test_harness_activation_records_observed_module_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "harness"
    revision = _initialize_harness_checkout(checkout)
    protocol = replace(
        pins.load_task_protocol(TASK_CONFIG),
        harness_submodule_path="harness",
        harness_revision=revision,
    )
    fake_module = SimpleNamespace(__file__=str(checkout / "lm_eval" / "__init__.py"))
    monkeypatch.setattr(pins.sys, "path", list(pins.sys.path))
    monkeypatch.setattr(pins.importlib, "import_module", lambda name: fake_module)

    state = pins.activate_harness_checkout(protocol, tmp_path)

    assert state.revision == revision
    assert state.clean is True
    assert state.module_file == "lm_eval/__init__.py"


def test_sample_coverage_uses_expanded_leaf_tasks_not_group_rows() -> None:
    pins.require_sample_coverage(
        {
            "results": {
                "mmlu": {"acc,none": 0.5},
                "mmlu_anatomy": {"acc,none": 0.5},
                "mmlu_astronomy": {"acc,none": 0.5},
            },
            "configs": {
                "mmlu_anatomy": {"task": "mmlu_anatomy"},
                "mmlu_astronomy": {"task": "mmlu_astronomy"},
            },
            "samples": {
                "mmlu_anatomy": [{"doc_id": 0}],
                "mmlu_astronomy": [{"doc_id": 0}],
            },
            "n-samples": {
                "mmlu_anatomy": {"original": 1, "effective": 1},
                "mmlu_astronomy": {"original": 1, "effective": 1},
            },
        }
    )


def test_sample_coverage_rejects_missing_or_empty_leaf_samples() -> None:
    with pytest.raises(RuntimeError, match="did not return samples"):
        pins.require_sample_coverage(
            {
                "configs": {"piqa": {}, "hellaswag": {}},
                "samples": {"piqa": [{"doc_id": 0}]},
            }
        )
    with pytest.raises(RuntimeError, match="empty or invalid"):
        pins.require_sample_coverage({"configs": {"piqa": {}}, "samples": {"piqa": []}})


@pytest.mark.parametrize(
    "counts,samples,match",
    (
        ({"original": 2, "effective": 1}, [{"doc_id": 0}], "evaluated only"),
        ({"original": 2, "effective": 2}, [{"doc_id": 0}], "retained 1 samples"),
        ({"original": 1, "effective": 1.0}, [{"doc_id": 0}], "non-integral"),
        ({"original": 1, "effective": 0}, [{"doc_id": 0}], "non-positive"),
    ),
)
def test_sample_coverage_binds_full_effective_counts(
    counts: dict[str, object],
    samples: list[dict[str, int]],
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        pins.require_sample_coverage(
            {
                "configs": {"piqa": {}},
                "samples": {"piqa": samples},
                "n-samples": {"piqa": counts},
            }
        )


def test_sample_coverage_requires_n_sample_counts() -> None:
    with pytest.raises(RuntimeError, match="did not contain n-samples"):
        pins.require_sample_coverage(
            {"configs": {"piqa": {}}, "samples": {"piqa": [{"doc_id": 0}]}}
        )

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts import benchmark_components
from scripts import benchmark_fa4_exp2_mix
from scripts import benchmark_open_weights
from sfu_repro.artifact import validate_result


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def initialize_repository(path: Path) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
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
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def test_all_drivers_count_untracked_files(tmp_path: Path) -> None:
    revision = initialize_repository(tmp_path)
    (tmp_path / "untracked.py").write_text("VALUE = 1\n", encoding="utf-8")

    assert benchmark_components.git_worktree_state(tmp_path) == (True, 1)
    assert benchmark_fa4_exp2_mix.git_worktree_state(tmp_path) == (True, 1)
    assert benchmark_open_weights.git_worktree_state(tmp_path) == (True, 1)

    state = benchmark_open_weights.repository_state(tmp_path)
    assert state["revision"] == revision
    assert state["dirty"] is True
    assert state["untracked_files"] == 1


def test_flash_attention_revision_reads_the_checked_out_repository(
    tmp_path: Path,
) -> None:
    flash_attention = tmp_path / "flash-attention"
    flash_attention.mkdir()
    revision = initialize_repository(flash_attention)

    assert benchmark_components.flash_attention_revision(tmp_path) == revision
    assert benchmark_fa4_exp2_mix.flash_attention_revision(tmp_path) == revision


def test_open_weight_result_records_revisions_and_package_versions() -> None:
    args = benchmark_open_weights.get_parser().parse_args(
        [
            "--model",
            "example/model",
            "--variant",
            "native",
            "--revision",
            "0123456789abcdef0123456789abcdef01234567",
            "--device",
            "cpu",
            "--suite-config",
            str(REPOSITORY_ROOT / "configs/open_weight_paper.json"),
        ]
    )
    args._recorded_command = [
        "python",
        "scripts/benchmark_open_weights.py",
        "--model",
        "example/model",
        "--variant",
        "native",
    ]

    document = benchmark_open_weights.result_document([], args)

    experiment = document["experiment"]
    assert experiment["requested_model_revisions"] == {
        "example/model": "0123456789abcdef0123456789abcdef01234567"
    }
    assert experiment["requested_tokenizer_revisions"] == {
        "example/model": "0123456789abcdef0123456789abcdef01234567"
    }
    assert experiment["revision_provenance"] == "user-specified"
    assert experiment["command"] == args._recorded_command
    assert document["measurement"]["eval_limit"] is None
    assert document["measurement"]["eval_batch_size"] == "auto"
    assert document["measurement"]["eval_task_config"] is None
    assert document["measurement"]["eval_task_config_sha256"] is None
    assert document["measurement"]["suite_environment_preflight"] == "not-run"
    assert document["measurement"]["suite_config"] == "configs/open_weight_paper.json"
    assert len(document["measurement"]["suite_config_sha256"]) == 64
    assert (
        document["measurement"]["environment_profiles"]
        == "configs/eval_environments/profiles.json"
    )
    assert len(document["measurement"]["environment_profiles_sha256"]) == 64
    assert document["environment"]["packages"]["transformers"]
    assert "lm-eval" in document["environment"]["packages"]
    assert validate_result(document) == []


def test_open_weight_benchmark_accepts_fractional_lm_eval_limit() -> None:
    args = benchmark_open_weights.get_parser().parse_args(
        [
            "--mode",
            "eval",
            "--eval-tasks",
            "piqa",
            "--eval-limit",
            "0.25",
            "--device",
            "cpu",
        ]
    )

    assert args.eval_limit == 0.25


def test_lm_eval_merge_preserves_samples_from_each_fewshot_group() -> None:
    merged = benchmark_open_weights.merge_lm_eval_results(
        [
            {
                "results": {"mmlu": {"acc,none": 0.5}},
                "samples": {"mmlu_abstract_algebra": [{"doc_id": 0}]},
            },
            {
                "results": {"hellaswag": {"acc_norm,none": 0.6}},
                "samples": {"hellaswag": [{"doc_id": 1}]},
            },
        ]
    )

    assert set(merged["results"]) == {"mmlu", "hellaswag"}
    assert set(merged["samples"]) == {"mmlu_abstract_algebra", "hellaswag"}


def test_exception_sanitizer_removes_tokens_and_provider_urls(
    monkeypatch,
) -> None:
    sensitive_value = "unit-test-sensitive-value"
    monkeypatch.setenv("HF_TOKEN", sensitive_value)
    credential_label = "access_" + "token"
    error = RuntimeError(
        "download failed at https://provider.invalid/object?token=signed-value "
        f"with {credential_label}={sensitive_value}"
    )

    sanitized = benchmark_open_weights.sanitize_exception(error)

    assert sanitized.startswith("RuntimeError: download failed")
    assert sensitive_value not in sanitized
    assert "signed-value" not in sanitized
    assert "https://" not in sanitized
    assert "[redacted" in sanitized

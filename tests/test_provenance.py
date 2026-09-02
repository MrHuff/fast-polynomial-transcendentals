from __future__ import annotations

import subprocess
from pathlib import Path

from scripts import benchmark_components
from scripts import benchmark_fa4_exp2_mix
from scripts import benchmark_open_weights
from sfu_repro.artifact import validate_result


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
        ]
    )

    document = benchmark_open_weights.result_document([], args)

    experiment = document["experiment"]
    assert experiment["requested_model_revisions"] == {
        "example/model": "0123456789abcdef0123456789abcdef01234567"
    }
    assert experiment["requested_tokenizer_revisions"] == {
        "example/model": "0123456789abcdef0123456789abcdef01234567"
    }
    assert document["environment"]["packages"]["transformers"]
    assert "lm-eval" in document["environment"]["packages"]
    assert validate_result(document) == []


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

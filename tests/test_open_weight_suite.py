from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts import run_open_weight_suite as suite


def test_variant_parser_accepts_paper_glm_intervention() -> None:
    variant = "fused_swiglu_d4_current+spline_router_sigmoid_d4_current"
    assert suite.parse_variant_parts(variant) == (
        "fused_swiglu_d4_current",
        "spline_router_sigmoid_d4_current",
    )


def test_patch_scope_accepts_router_only_intervention() -> None:
    row = {
        "patched_silu_modules": 0,
        "patched_router_sigmoid_modules": 16,
        "patched_gemma_softcap": False,
    }

    assert suite.result_patch_scope_matches(row, "spline_router_sigmoid_d4_current")


@pytest.mark.parametrize(
    "variant",
    ("", "native+fused_swiglu_d4_current", "fused_swiglu_d7_current", "secret"),
)
def test_variant_parser_rejects_invalid_variants(variant: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        suite.parse_variant_parts(variant)


def test_paper_config_preserves_revisions_and_glm_router_variant() -> None:
    config = suite.load_config()
    cases = {case.key: case for case in config.cases}
    assert set(cases) == {
        "qwen2p5_7b",
        "qwen3_30b_a3b_base",
        "glm4p7_flash",
        "gpt_oss_120b",
        "kimi_linear_48b_a3b_base",
    }
    assert cases["qwen2p5_7b"].revision == ("d149729398750b98c0af14eb82c78cfe92750796")
    assert cases["qwen2p5_7b"].revision_provenance == "public-protocol-selection"
    assert cases["qwen3_30b_a3b_base"].revision == (
        "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
    )
    assert cases["qwen3_30b_a3b_base"].revision_provenance == (
        "public-protocol-selection"
    )
    assert cases["glm4p7_flash"].revision == (
        "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
    )
    assert (
        "fused_swiglu_d4_current+spline_router_sigmoid_d4_current"
        in cases["glm4p7_flash"].variants
    )
    assert cases["gpt_oss_120b"].revision == (
        "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
    )
    assert cases["kimi_linear_48b_a3b_base"].revision == (
        "3b171c17bfc4ee348599b6781a2ca8715c21c8dc"
    )


def test_dry_run_command_is_standalone_and_never_contains_a_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HF_TOKEN", "must-not-appear")
    result = suite.main(
        [
            "--models",
            "glm4p7_flash",
            "--mode",
            "eval",
            "--quality-eval",
            "--dry-run",
            "--python",
            "python",
            "--output-dir",
            "outputs/test",
        ]
    )
    assert result == 0
    command = capsys.readouterr().out.strip()
    assert command.startswith("python scripts/benchmark_open_weights.py")
    assert "--revision 7dd20894a642a0aa287e9827cb1a1f7f91386b67" in command
    assert "--revision-provenance historical-source-derived" in command
    assert "--experts-implementation grouped_mm" in command
    assert "--seq-len 2048" in command
    assert "--steps 20" in command
    assert "--warmup 5" in command
    assert "--decode-steps 64" in command
    assert "--decode-repeats 10" in command
    assert "--decode-warmup 2" in command
    assert "--eval-task-config configs/lm_eval_paper_tasks.json" in command
    assert "--eval-log-samples" in command
    assert "--suite-environment-preflight planned" in command
    assert "--suite-config" in command
    assert "configs/open_weight_paper.json" in command
    assert (
        "--eval-tasks "
        "mmlu,hellaswag,arc_challenge,winogrande,piqa,gsm8k,"
        "truthfulqa_mc2,wikitext"
    ) in command
    assert (
        "--variant fused_swiglu_d4_current+spline_router_sigmoid_d4_current" in command
    )
    assert "--token" not in command
    assert "must-not-appear" not in command
    assert "low-bits-training" not in command
    assert "torchtitan" not in command.lower()


def test_execution_runs_environment_preflight_before_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = suite.load_config()
    case = next(case for case in config.cases if case.key == "qwen2p5_7b")
    args = suite.get_parser().parse_args(
        [
            "--models",
            case.key,
            "--mode",
            "eval",
            "--quality-eval",
            "--python",
            "python",
            "--output-dir",
            str(tmp_path),
        ]
    )
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> object:
        del kwargs
        calls.append(command)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(suite.subprocess, "run", run)

    assert suite.run_case(case, args, config.protocol, tmp_path) == 0
    assert calls[0][:2] == ["python", "scripts/check_open_weight_environment.py"]
    assert calls[0][-3:] == ["--models", case.key, "--check-installed"]
    assert calls[1][1] == "scripts/benchmark_open_weights.py"
    preflight_index = calls[1].index("--suite-environment-preflight")
    assert calls[1][preflight_index + 1] == "passed"
    config_index = calls[1].index("--suite-config")
    assert Path(calls[1][config_index + 1]) == suite.DEFAULT_CONFIG.resolve()


def test_public_quality_protocol_covers_all_tasks_and_shot_counts() -> None:
    config = suite.load_config()
    assert config.protocol.quality_tasks == (
        "mmlu",
        "hellaswag",
        "arc_challenge",
        "winogrande",
        "piqa",
        "gsm8k",
        "truthfulqa_mc2",
        "wikitext",
    )
    assert config.protocol.quality_fewshot == {
        "mmlu": 5,
        "hellaswag": 10,
        "arc_challenge": 25,
        "winogrande": 5,
        "piqa": 0,
        "gsm8k": 5,
        "truthfulqa_mc2": 0,
        "wikitext": 0,
    }
    assert config.protocol.quality_log_samples is True
    assert config.protocol.device == "cuda"
    assert config.protocol.seed == 1234
    assert config.protocol.throughput_mode == "both"
    assert config.protocol.quality_mode == "eval"
    assert config.protocol.quality_eval_limit is None
    assert config.protocol.quality_default_num_fewshot == 0


def test_skipped_preflight_is_propagated_to_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = suite.load_config()
    case = next(case for case in config.cases if case.key == "qwen2p5_7b")
    args = suite.get_parser().parse_args(
        [
            "--models",
            case.key,
            "--mode",
            "eval",
            "--quality-eval",
            "--skip-environment-check",
            "--python",
            "python",
            "--output-dir",
            str(tmp_path),
        ]
    )
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> object:
        del kwargs
        calls.append(command)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(suite.subprocess, "run", run)

    assert suite.run_case(case, args, config.protocol, tmp_path) == 0
    assert len(calls) == 1
    preflight_index = calls[0].index("--suite-environment-preflight")
    assert calls[0][preflight_index + 1] == "skipped"


def test_eval_only_uses_historical_non_consumed_metadata_length() -> None:
    config = suite.load_config()
    kimi = next(case for case in config.cases if case.key == "kimi_linear_48b_a3b_base")
    args = suite.get_parser().parse_args(
        ["--mode", "eval", "--quality-eval", "--python", "python"]
    )
    command = suite.build_command(
        kimi,
        args,
        config.protocol,
        Path("/tmp/open-weight-test-output"),
    )
    seq_index = command.index("--seq-len")
    assert command[seq_index + 1] == "8192"
    assert "--trust-remote-code" in command
    assert "--token" not in command


def test_eval_limit_is_forwarded_as_a_supported_float() -> None:
    config = suite.load_config()
    case = next(case for case in config.cases if case.key == "qwen2p5_7b")
    args = suite.get_parser().parse_args(
        [
            "--mode",
            "eval",
            "--quality-eval",
            "--eval-limit",
            "0.25",
            "--python",
            "python",
        ]
    )
    command = suite.build_command(
        case,
        args,
        config.protocol,
        Path("/tmp/open-weight-test-output"),
    )
    limit_index = command.index("--eval-limit")
    assert command[limit_index + 1] == "0.25"


def test_completed_output_accepts_versioned_result_envelope(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "results": [
                    {
                        "variant": "native",
                        "error": None,
                        "patched_silu_modules": 0,
                        "patched_router_sigmoid_modules": 0,
                        "patched_gemma_softcap": False,
                    },
                    {
                        "variant": "fused_swiglu_d4_current",
                        "error": None,
                        "patched_silu_modules": 32,
                        "patched_router_sigmoid_modules": 0,
                        "patched_gemma_softcap": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    assert suite.completed_output(path, ("native", "fused_swiglu_d4_current"))


def test_completed_output_rejects_zero_patch_or_reordered_variant(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    native = {
        "variant": "native",
        "error": None,
        "patched_silu_modules": 0,
        "patched_router_sigmoid_modules": 0,
        "patched_gemma_softcap": False,
    }
    polynomial = {
        "variant": "fused_swiglu_d4_current",
        "error": None,
        "patched_silu_modules": 0,
        "patched_router_sigmoid_modules": 0,
        "patched_gemma_softcap": False,
    }
    path.write_text(
        json.dumps({"schema_version": 1, "results": [native, polynomial]}),
        encoding="utf-8",
    )
    expected = ("native", "fused_swiglu_d4_current")

    assert not suite.completed_output(path, expected)

    polynomial["patched_silu_modules"] = 32
    path.write_text(
        json.dumps({"schema_version": 1, "results": [polynomial, native]}),
        encoding="utf-8",
    )
    assert not suite.completed_output(path, expected)


def test_completed_quality_output_requires_attested_sample_logs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quality.json"
    row = {
        "variant": "native",
        "error": None,
        "patched_silu_modules": 0,
        "patched_router_sigmoid_modules": 0,
        "patched_gemma_softcap": False,
        "eval": {
            "results": {"piqa": {"acc,none": 0.5}},
            "samples": {"piqa": [{"doc_id": 0}]},
            "public_task_protocol": {
                "log_samples": True,
                "lm_eval_source_clean": True,
            },
        },
    }
    path.write_text(
        json.dumps({"schema_version": 1, "results": [row]}),
        encoding="utf-8",
    )

    assert suite.completed_output(
        path,
        ("native",),
        require_evaluation=True,
        require_samples=True,
    )

    row["eval"]["samples"] = {}
    path.write_text(
        json.dumps({"schema_version": 1, "results": [row]}),
        encoding="utf-8",
    )
    assert not suite.completed_output(
        path,
        ("native",),
        require_evaluation=True,
        require_samples=True,
    )

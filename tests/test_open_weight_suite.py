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
    assert "--experts-implementation grouped_mm" in command
    assert "--seq-len 512" in command
    assert "--steps 20" in command
    assert "--warmup 5" in command
    assert "--decode-steps 64" in command
    assert "--decode-repeats 10" in command
    assert "--decode-warmup 2" in command
    assert (
        "--variant fused_swiglu_d4_current+spline_router_sigmoid_d4_current"
        in command
    )
    assert "--token" not in command
    assert "must-not-appear" not in command
    assert "low-bits-training" not in command
    assert "torchtitan" not in command.lower()


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


def test_completed_output_accepts_versioned_result_envelope(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "results": [
                    {"variant": "native", "error": None},
                    {"variant": "fused_swiglu_d4_current", "error": None},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert suite.completed_output(
        path, ("native", "fused_swiglu_d4_current")
    )

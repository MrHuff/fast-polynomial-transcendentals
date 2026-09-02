"""Check dependencies without importing optional GPU runtimes eagerly."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sfu_repro.torchtitan.assets import verify_tokenizer_assets
from sfu_repro.torchtitan.pins import (
    DEEPSEEK_TOKENIZER_REPOSITORY,
    DEEPSEEK_TOKENIZER_REVISION,
    LLAMA_TOKENIZER_REPOSITORY,
    LLAMA_TOKENIZER_REVISION,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def module_check(name: str, *, required: bool) -> Check:
    try:
        present = importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        present = False
    return Check(
        f"python:{name}",
        "pass" if present else ("fail" if required else "warn"),
        "available" if present else "not importable",
    )


def command_check(name: str, *, required: bool) -> Check:
    path = shutil.which(name)
    return Check(
        f"command:{name}",
        "pass" if path else ("fail" if required else "warn"),
        path or "not on PATH",
    )


def distribution_check(name: str, expected: str, *, required: bool) -> Check:
    try:
        installed = version(name)
    except PackageNotFoundError:
        installed = None
    matches = installed == expected
    return Check(
        f"distribution:{name}",
        "pass" if matches else ("fail" if required else "warn"),
        f"{installed or 'not installed'}; expected {expected}",
    )


def submodule_check(name: str, expected: str) -> Check:
    path = REPOSITORY_ROOT / name
    if not path.is_dir():
        return Check(
            f"source:{name}",
            "fail",
            "not initialized; run git submodule update --init --recursive",
        )
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    revision = completed.stdout.strip()
    if completed.returncode or revision != expected:
        return Check(
            f"source:{name}",
            "fail",
            f"expected {expected}; found {revision or 'unresolved'}",
        )
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if status.returncode or status.stdout.strip():
        return Check(
            f"source:{name}",
            "fail",
            "revision matches but the submodule has uncommitted or untracked files",
        )
    return Check(f"source:{name}", "pass", revision)


def tokenizer_check(
    name: str,
    relative_path: str,
    repository: str,
    revision: str,
) -> Check:
    path = REPOSITORY_ROOT / relative_path
    try:
        verify_tokenizer_assets(
            path,
            expected_repository=repository,
            expected_revision=revision,
        )
    except RuntimeError as error:
        return Check(f"tokenizer:{name}", "fail", str(error))
    return Check(
        f"tokenizer:{name}",
        "pass",
        f"{repository}@{revision}; file digests verified",
    )


def checks_for(profile: str) -> list[Check]:
    checks = [
        command_check("git", required=True),
        module_check("numpy", required=False),
    ]
    if profile in {"component", "eval", "all"}:
        checks.extend(
            [
                module_check("torch", required=True),
                module_check("spline_ops", required=True),
                module_check("flash_attn.cute.interface", required=True),
                module_check("flash_attn.cute.polynomial_manifest", required=True),
                submodule_check(
                    "flash-attention",
                    "38afdedda24b0bf26e6904d3bed7807c19a6906e",
                ),
            ]
        )
    if profile in {"eval", "all"}:
        checks.extend(
            [
                module_check("transformers", required=True),
                module_check("lm_eval", required=True),
                distribution_check("lm-eval", "0.4.12", required=True),
                submodule_check(
                    "lm-evaluation-harness",
                    "6d642546f4688648fced259eb3302efd36ece5af",
                ),
            ]
        )
    if profile in {"train", "all"}:
        checks.extend(
            [
                distribution_check("torch", "2.12.0.dev20260220+cu126", required=True),
                distribution_check(
                    "torchao", "0.17.0.dev20260220+cu126", required=True
                ),
                module_check("datasets", required=True),
                module_check("torchdata", required=True),
                module_check("tokenizers", required=True),
                module_check("tyro", required=True),
                module_check("tensorboard", required=True),
                module_check("spline_ops", required=True),
                module_check("flash_attn.cute.interface", required=True),
                submodule_check(
                    "torchtitan",
                    "73a0e6979dd10b6b1904098eb3c8f62c18ab87ce",
                ),
                submodule_check(
                    "flash-attention",
                    "38afdedda24b0bf26e6904d3bed7807c19a6906e",
                ),
                tokenizer_check(
                    "llama3",
                    "assets/hf/Llama-3.1-8B",
                    LLAMA_TOKENIZER_REPOSITORY,
                    LLAMA_TOKENIZER_REVISION,
                ),
                tokenizer_check(
                    "deepseek-v3",
                    "assets/hf/DeepSeek-V3.1-Base",
                    DEEPSEEK_TOKENIZER_REPOSITORY,
                    DEEPSEEK_TOKENIZER_REVISION,
                ),
            ]
        )
    return list(dict.fromkeys(checks))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("analysis", "component", "train", "eval", "all"),
        default="component",
    )
    args = parser.parse_args()
    checks = checks_for(args.profile)
    for check in checks:
        print(f"{check.status.upper():4} {check.name}: {check.detail}")
    return int(any(check.status == "fail" for check in checks))


if __name__ == "__main__":
    raise SystemExit(main())

"""Check dependencies without importing optional GPU runtimes eagerly."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


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


def submodule_check() -> Check:
    path = REPOSITORY_ROOT / "flash-attention"
    if not path.is_dir():
        return Check(
            "source:flash-attention",
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
    expected = "38afdedda24b0bf26e6904d3bed7807c19a6906e"
    if completed.returncode or revision != expected:
        return Check(
            "source:flash-attention",
            "fail",
            f"expected {expected}; found {revision or 'unresolved'}",
        )
    return Check("source:flash-attention", "pass", revision)


def checks_for(profile: str) -> list[Check]:
    checks = [command_check("git", required=True), module_check("numpy", required=False)]
    if profile in {"component", "eval", "all"}:
        checks.extend(
            [
                module_check("torch", required=True),
                module_check("spline_ops", required=True),
                module_check("flash_attn.cute.interface", required=True),
                module_check("flash_attn.cute.polynomial_manifest", required=True),
                submodule_check(),
            ]
        )
    if profile in {"eval", "all"}:
        checks.extend(
            [
                module_check("transformers", required=True),
                module_check("lm_eval", required=True),
            ]
        )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("analysis", "component", "eval", "all"), default="component"
    )
    args = parser.parse_args()
    checks = checks_for(args.profile)
    for check in checks:
        print(f"{check.status.upper():4} {check.name}: {check.detail}")
    return int(any(check.status == "fail" for check in checks))


if __name__ == "__main__":
    raise SystemExit(main())

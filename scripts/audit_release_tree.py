#!/usr/bin/env python3
"""Audit the standalone tree for accidental credentials and internal plumbing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", "flash-attention", "__pycache__", ".pytest_cache"}
BINARY_SUFFIXES = {".csv", ".pdf", ".png", ".jpg", ".jpeg", ".so"}
MAX_GIT_FILE_BYTES = 100 * 1024 * 1024
RULES = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "slack-token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "credential-assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
    "machine-path": re.compile(r"/(?:workspace|home)/[A-Za-z0-9_.-]+/"),
    "private-object-store": re.compile(r"\bs3://"),
}


def candidates() -> list[Path]:
    paths: list[Path] = []
    for path in REPOSITORY_ROOT.rglob("*"):
        relative = path.relative_to(REPOSITORY_ROOT)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.is_file():
            paths.append(path)
    return paths


def audit() -> list[tuple[str, Path]]:
    findings: list[tuple[str, Path]] = []
    for path in candidates():
        relative = path.relative_to(REPOSITORY_ROOT)
        lower_name = path.name.lower()
        if lower_name.startswith(".env") or any(
            marker in lower_name for marker in ("credentials", "private_key")
        ):
            findings.append(("sensitive-filename", relative))
        if path.stat().st_size > MAX_GIT_FILE_BYTES:
            findings.append(("github-size-limit", relative))
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for rule, pattern in RULES.items():
            if pattern.search(text):
                findings.append((rule, relative))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-machine-path",
        action="store_true",
        help="Ignore machine-path findings in reviewed historical metadata.",
    )
    args = parser.parse_args()
    findings = audit()
    if args.allow_machine_path:
        findings = [item for item in findings if item[0] != "machine-path"]
    for rule, path in findings:
        # Deliberately report no matched value: audit output is safe to share.
        print(f"FAIL {rule}: {path}")
    if not findings:
        print("PASS release-tree credential and internal-path audit")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())

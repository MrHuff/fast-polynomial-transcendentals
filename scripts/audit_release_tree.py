#!/usr/bin/env python3
"""Audit the current standalone tree for credentials and internal plumbing.

The filesystem walk includes staged, unstaged, untracked, and ignored files.
Git history, submodules, LFS objects, and release archives require separate
audits, as documented in ``docs/PUBLIC_RELEASE_CHECKLIST.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    "flash-attention",
    "lm-evaluation-harness",
    "torchtitan",
    "__pycache__",
    ".pytest_cache",
}
MAX_GIT_FILE_BYTES = 100 * 1024 * 1024
HISTORICAL_IDENTIFIER_FIELDS = {
    "contributing_run_ids",
    "contributing_runs",
    "run_created_at",
    "run_id",
    "run_name",
    "run_url",
    "source_project",
}
HISTORICAL_SERVICE_FIELDS = {"wandb"}
SENSITIVE_FILENAMES = {
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
SENSITIVE_SUFFIXES = {".key", ".p12", ".pfx"}
RULES = {
    "private-key": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"
    ),
    "github-token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
    ),
    "gitlab-token": re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    "huggingface-token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "openai-token": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "anthropic-token": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "google-api-key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "npm-token": re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"),
    "pypi-token": re.compile(r"\bpypi-AgEIcH[A-Za-z0-9_-]{30,}\b"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "slack-token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "slack-webhook": re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]+"),
    "bearer-token": re.compile(
        r"(?i)\bauthorization\s*[:=]\s*['\"]?bearer\s+[A-Za-z0-9._~+/-]{20,}"
    ),
    "credential-in-url": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
    "signed-url": re.compile(
        r"(?i)[?&](?:X-Amz-(?:Credential|Signature)|GoogleAccessId|Signature)="
    ),
    "credential-assignment": re.compile(
        r"(?x)(?:"
        r"(?i:\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
        r"session[_-]?token|token|password|passwd|"
        r"secret(?:[_-]?(?:access[_-]?key|key))?))|"
        r"\b(?:[A-Z][A-Z0-9]*_)+(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|"
        r"SESSION_TOKEN|TOKEN|PASSWORD|PASSWD|"
        r"SECRET(?:_(?:ACCESS_KEY|KEY))?)"
        r")\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{16,}"
    ),
    "machine-path": re.compile(
        r"/(?:workspace|volt/restore/workspace)(?:/|$)"
        r"|/(?:home|scratch|lustre|fsx|nfs|mnt)/[A-Za-z0-9_.-]+/"
    ),
    "private-object-store": re.compile(r"(?i)\b(?:s3|gs|az)://"),
}


def _json_contains_historical_identifier(value: object) -> bool:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            if key in HISTORICAL_IDENTIFIER_FIELDS and child not in (None, "", [], {}):
                return True
            if (
                key in HISTORICAL_SERVICE_FIELDS
                and isinstance(child, dict)
                and bool(child)
            ):
                return True
            if _json_contains_historical_identifier(child):
                return True
    elif isinstance(value, list):
        for child in value:
            if _json_contains_historical_identifier(child):
                return True
    return False


def contains_historical_identifiers(path: Path, relative: Path) -> bool:
    """Detect review-gated service/run metadata without exposing its values."""

    if not relative.parts or relative.parts[0] != "evidence":
        return False
    try:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                header = next(csv.reader(stream), [])
            fields = {field.strip().lower() for field in header}
            return bool(HISTORICAL_IDENTIFIER_FIELDS & fields)
        if path.suffix.lower() == ".json":
            document = json.loads(path.read_text(encoding="utf-8"))
            return _json_contains_historical_identifier(document)
    except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError):
        return False
    return False


def matching_content_rules(data: bytes) -> set[str]:
    """Return matching rule names while keeping any matched values private."""

    text = data.decode("utf-8", errors="ignore")
    return {rule for rule, pattern in RULES.items() if pattern.search(text)}


def candidates() -> list[Path]:
    paths: list[Path] = []
    for path in REPOSITORY_ROOT.rglob("*"):
        relative = path.relative_to(REPOSITORY_ROOT)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.is_file():
            paths.append(path)
    return sorted(paths)


def audit() -> list[tuple[str, Path]]:
    findings: list[tuple[str, Path]] = []
    for path in candidates():
        relative = path.relative_to(REPOSITORY_ROOT)
        lower_name = path.name.lower()
        if (
            lower_name.startswith(".env")
            or lower_name in SENSITIVE_FILENAMES
            or path.suffix.lower() in SENSITIVE_SUFFIXES
            or any(marker in lower_name for marker in ("credentials", "private_key"))
        ):
            findings.append(("sensitive-filename", relative))
        if path.stat().st_size > MAX_GIT_FILE_BYTES:
            findings.append(("github-size-limit", relative))
        if contains_historical_identifiers(path, relative):
            findings.append(("historical-experiment-identifiers", relative))
        try:
            data = path.read_bytes()
        except OSError:
            continue
        for rule in sorted(matching_content_rules(data)):
            findings.append((rule, relative))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-machine-path",
        action="store_true",
        help="Ignore machine-path findings in reviewed historical metadata.",
    )
    parser.add_argument(
        "--allow-historical-identifiers",
        action="store_true",
        help=(
            "Ignore reviewed historical run/service metadata. Use only after "
            "publication approval or documented sanitization."
        ),
    )
    args = parser.parse_args()
    findings = audit()
    if args.allow_machine_path:
        findings = [item for item in findings if item[0] != "machine-path"]
    if args.allow_historical_identifiers:
        findings = [
            item for item in findings if item[0] != "historical-experiment-identifiers"
        ]
    for rule, path in findings:
        # Deliberately report no matched value: audit output is safe to share.
        print(f"FAIL {rule}: {path}")
    if not findings:
        print("PASS release-tree credential and internal-path audit")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())

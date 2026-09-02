#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified in 2026 for the standalone fast-polynomial-transcendentals release.

"""Portable provenance records for standalone coefficient-fitting outputs."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_URL = "https://github.com/MrHuff/fast-polynomial-transcendentals"
FIT_SCOPE = (
    "Generated coefficient-search output only. This record does not promote "
    "coefficients into a deployed kernel or attest compiled-device accuracy."
)
PATH_OPTIONS = frozenset(("--current-header", "--header-out", "--json-out", "--output"))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one required regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_label(path: Path) -> str:
    """Render a path without retaining a machine-specific directory prefix."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def safe_command(script: Path, arguments: Sequence[str]) -> list[str]:
    """Build a path-normalized command receipt for a direct script invocation."""

    rendered = [Path(sys.executable).name, repository_label(script)]
    path_value_expected = False
    for raw in arguments:
        token = str(raw)
        option, separator, value = token.partition("=")
        option_path = bool(separator and option in PATH_OPTIONS)
        if path_value_expected or option_path:
            candidate = value if separator else token
            safe_value = repository_label(Path(candidate))
            rendered.append(f"{option}={safe_value}" if separator else safe_value)
            path_value_expected = False
        elif token in PATH_OPTIONS:
            rendered.append(token)
            path_value_expected = True
        elif Path(token).is_absolute():
            rendered.append(repository_label(Path(token)))
        else:
            rendered.append(token)
    return rendered


def git_state() -> dict[str, object]:
    """Record the enclosing repository revision and exact dirty state."""

    def git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=REPOSITORY_ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return None
        value = completed.stdout.strip()
        return value if completed.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    entries = tuple(item for item in (status or "").split("\0") if item)
    dirty = bool(entries) if status is not None else None
    return {
        "repository": REPOSITORY_URL,
        "revision": revision if revision is not None and len(revision) == 40 else None,
        "dirty": dirty,
        "clean": (not dirty) if dirty is not None else None,
        "untracked_files": (
            sum(item.startswith("?? ") for item in entries)
            if status is not None
            else None
        ),
    }


def runtime_versions(distributions: Iterable[str]) -> dict[str, object]:
    """Record the Python runtime and requested installed distribution versions."""

    packages: dict[str, str | None] = {}
    for distribution in sorted(set(distributions)):
        try:
            packages[distribution] = package_version(distribution)
        except PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def build_fit_provenance(
    *,
    script: Path,
    arguments: Sequence[str],
    source_files: Iterable[Path] = (),
    distributions: Iterable[str] = (),
) -> dict[str, object]:
    """Build metadata shared by the standalone fitting entry points."""

    required_sources = {script.resolve(), Path(__file__).resolve()}
    required_sources.update(Path(path).resolve() for path in source_files)
    source_hashes = {
        repository_label(path): sha256_file(path)
        for path in sorted(required_sources, key=lambda item: repository_label(item))
    }
    return {
        "schema_version": 1,
        "artifact_type": "generated-coefficient-fit",
        "provenance_class": "generated-fit",
        "claim_scope": FIT_SCOPE,
        "command": safe_command(script, arguments),
        "source": {
            **git_state(),
            "input_sha256": source_hashes,
        },
        "environment": runtime_versions(distributions),
    }


def numerical_payload_sha256(document: dict[str, object]) -> str:
    """Hash all numerical/result fields while excluding mutable provenance."""

    payload = {key: value for key, value in document.items() if key != "_provenance"}
    encoded = json.dumps(
        payload,
        allow_nan=True,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_fit_payload(
    document: dict[str, object], provenance: dict[str, object]
) -> None:
    """Attach provenance and bind it to the completed fit payload."""

    provenance["numerical_payload_sha256"] = numerical_payload_sha256(document)
    document["_provenance"] = provenance


def fit_output_is_source_bound(
    document: dict[str, object],
    *,
    expected_script: str,
    repository_state: dict[str, object],
) -> bool:
    """Verify an ignored fit artifact against its clean recorded source state."""

    provenance = document.get("_provenance")
    if not isinstance(provenance, dict):
        return False
    source = provenance.get("source")
    command = provenance.get("command")
    if not isinstance(source, dict) or not isinstance(command, list):
        return False
    source_hashes = source.get("input_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        return False
    if not (
        provenance.get("artifact_type") == "generated-coefficient-fit"
        and source.get("revision") == repository_state.get("revision")
        and source.get("dirty") is False
        and repository_state.get("revision") is not None
        and repository_state.get("dirty") is False
        and len(command) >= 2
        and str(command[1]).endswith(expected_script)
        and provenance.get("numerical_payload_sha256")
        == numerical_payload_sha256(document)
    ):
        return False
    for label, expected_digest in source_hashes.items():
        if not isinstance(label, str) or not isinstance(expected_digest, str):
            return False
        relative = Path(label)
        if (
            len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
            or label.startswith("<external>/")
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            return False
        try:
            observed_digest = sha256_file(REPOSITORY_ROOT / relative)
        except OSError:
            return False
        if observed_digest != expected_digest:
            return False
    return True


__all__ = [
    "FIT_SCOPE",
    "PATH_OPTIONS",
    "REPOSITORY_ROOT",
    "bind_fit_payload",
    "build_fit_provenance",
    "git_state",
    "fit_output_is_source_bound",
    "numerical_payload_sha256",
    "repository_label",
    "runtime_versions",
    "safe_command",
    "sha256_file",
]

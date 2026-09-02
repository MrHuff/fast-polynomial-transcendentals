# Copyright 2026 Robert Hu
# SPDX-License-Identifier: Apache-2.0
"""Small, dependency-free helpers for paper artifact generators."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    """Return an installed package version, or ``None`` when absent."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def artifact_record(path: Path, *, role: str) -> dict[str, object]:
    """Describe a file without embedding its machine-specific absolute path."""
    return {
        "role": role,
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_receipt(
    path: Path,
    *,
    artifact_type: str,
    generator: Path,
    inputs: Iterable[tuple[Path, str]],
    outputs: Iterable[tuple[Path, str]],
    parameters: dict[str, object],
    packages: Iterable[str] = (),
    notes: Iterable[str] = (),
) -> None:
    """Write a portable receipt for a deterministic or measurement artifact.

    Paths are intentionally reduced to basenames. The hashes bind the exact
    bytes without leaking workstation directory layouts.
    """
    payload = {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "generator": artifact_record(generator, role="generator"),
        "inputs": [artifact_record(item, role=role) for item, role in inputs],
        "outputs": [artifact_record(item, role=role) for item, role in outputs],
        "parameters": parameters,
        "runtime": {
            "python": platform.python_version(),
            "packages": {name: package_version(name) for name in packages},
        },
        "notes": list(notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

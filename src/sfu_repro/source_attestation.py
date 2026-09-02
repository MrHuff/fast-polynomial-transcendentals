# SPDX-License-Identifier: Apache-2.0
"""Source-origin receipts for locally built benchmark modules.

The benchmark drivers use these helpers to distinguish a clean build from the
checked-out source from an unrelated wheel that happens to expose the same
Python API.  The receipt contains hashes and repository-relative labels, never
credentials or hostnames.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from importlib.metadata import (
    PackageNotFoundError,
    distribution as distribution_metadata,
    version as distribution_version,
)
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence
from urllib.parse import unquote, urlparse


class SourceAttestationError(RuntimeError):
    """Raised when a measured runtime is not bound to the declared source."""


def sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def git_state(repository: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        value = completed.stdout.strip()
        return value if completed.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    entries = tuple(item for item in (status or "").split("\0") if item)
    return {
        "revision": revision if revision is not None and len(revision) == 40 else None,
        "dirty": bool(entries) if status is not None else None,
        "untracked_files": (
            sum(item.startswith("?? ") for item in entries)
            if status is not None
            else None
        ),
    }


def _safe_path_label(path: Path | None, repository_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except (OSError, ValueError):
        return f"<external>/{path.name}"


def _distribution_source(distribution: str) -> Path | None:
    try:
        metadata = distribution_metadata(distribution)
    except PackageNotFoundError:
        return None
    raw = metadata.read_text("direct_url.json")
    if not raw:
        return None
    try:
        document = json.loads(raw)
        parsed = urlparse(str(document.get("url", "")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return None
    return Path(unquote(parsed.path)).resolve()


def _distribution_version(distribution: str) -> str | None:
    try:
        return distribution_version(distribution)
    except PackageNotFoundError:
        return None


def attest_module_source(
    module_name: str,
    *,
    source_checkout: Path,
    repository_root: Path,
    expected_revision: str | None,
    distribution: str | None = None,
    expected_version: str | None = None,
    module: ModuleType | None = None,
) -> dict[str, Any]:
    """Describe and verify one imported module's local source binding."""

    loaded = importlib.import_module(module_name) if module is None else module
    raw_module_file = getattr(loaded, "__file__", None)
    module_file = (
        Path(raw_module_file).resolve()
        if isinstance(raw_module_file, str) and raw_module_file
        else None
    )
    source_checkout = source_checkout.resolve()
    module_within_source = bool(
        module_file is not None and module_file.is_relative_to(source_checkout)
    )
    direct_source = _distribution_source(distribution) if distribution else None
    direct_source_matches = direct_source == source_checkout
    state = git_state(source_checkout)
    observed_version = _distribution_version(distribution) if distribution else None
    version_matches = expected_version is None or observed_version == expected_version
    revision_matches = (
        expected_revision is not None
        and state["revision"] is not None
        and state["revision"] == expected_revision
    )
    binary_digest = sha256_file(module_file) if module_file is not None else None
    bound = bool(
        module_file is not None
        and binary_digest is not None
        and (module_within_source or direct_source_matches)
        and revision_matches
        and state["dirty"] is False
        and version_matches
    )
    return {
        "module": module_name,
        "module_file": _safe_path_label(module_file, repository_root),
        "module_sha256": binary_digest,
        "distribution": distribution,
        "package_version": observed_version,
        "expected_package_version": expected_version,
        "source_path": _safe_path_label(source_checkout, repository_root),
        "direct_url_source": _safe_path_label(direct_source, repository_root),
        "origin_matches_source": module_within_source or direct_source_matches,
        "binding_method": (
            "module-within-source"
            if module_within_source
            else "pep610-direct-url" if direct_source_matches else None
        ),
        "source_revision": state["revision"],
        "expected_source_revision": expected_revision,
        "source_revision_matches": revision_matches,
        "source_dirty": state["dirty"],
        "source_untracked_files": state["untracked_files"],
        "bound": bound,
    }


def require_bound_attestations(
    attestations: dict[str, dict[str, Any]], *, allow_unbound: bool
) -> None:
    unbound = sorted(
        name for name, attestation in attestations.items() if not attestation["bound"]
    )
    if unbound and not allow_unbound:
        raise SourceAttestationError(
            "runtime source attestation failed for "
            + ", ".join(unbound)
            + "; rebuild from the clean pinned checkout or use "
            "--allow-unbound-source for a diagnostic result"
        )


def safe_command(arguments: Sequence[str], repository_root: Path) -> list[str]:
    """Normalize paths in an argv receipt without retaining machine prefixes."""

    rendered: list[str] = []
    for raw in arguments:
        token = str(raw)
        option, separator, value = token.partition("=")
        candidate = value if separator else token
        path = Path(candidate)
        if path.is_absolute():
            safe_value = _safe_path_label(path, repository_root)
            rendered.append(f"{option}={safe_value}" if separator else str(safe_value))
        else:
            rendered.append(token)
    return rendered


__all__ = [
    "SourceAttestationError",
    "attest_module_source",
    "git_state",
    "require_bound_attestations",
    "safe_command",
    "sha256_file",
]

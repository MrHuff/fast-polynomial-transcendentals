# SPDX-License-Identifier: Apache-2.0
"""Verification helpers for revision-pinned TorchTitan tokenizer assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tokenizer_assets(
    path: Path,
    *,
    expected_repository: str,
    expected_revision: str,
) -> None:
    """Verify one tokenizer manifest and every file digest it declares."""

    manifest_path = path / "tokenizer-manifest.json"
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"missing or invalid pinned tokenizer manifest at {manifest_path}; "
            "run scripts/prepare_torchtitan_assets.py --execute"
        ) from error
    if (
        document.get("schema_version") != 1
        or document.get("asset_type") != "tokenizer"
        or document.get("repository") != expected_repository
        or document.get("revision") != expected_revision
    ):
        raise RuntimeError(f"tokenizer manifest provenance mismatch at {manifest_path}")

    records = document.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"tokenizer manifest has no files at {manifest_path}")
    root = path.resolve()
    observed: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError(f"invalid tokenizer file record at {manifest_path}")
        relative = record.get("path")
        expected_digest = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_digest, str):
            raise RuntimeError(f"invalid tokenizer file record at {manifest_path}")
        candidate = (path / relative).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise RuntimeError(f"missing or unsafe tokenizer asset {relative!r}")
        if sha256_file(candidate) != expected_digest:
            raise RuntimeError(f"tokenizer asset digest mismatch for {relative!r}")
        observed.add(relative)
    required = {"tokenizer.json", "tokenizer_config.json"}
    if not required.issubset(observed):
        raise RuntimeError(
            "tokenizer manifest must include tokenizer.json and "
            "tokenizer_config.json"
        )

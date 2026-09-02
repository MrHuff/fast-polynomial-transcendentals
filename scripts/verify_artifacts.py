#!/usr/bin/env python3
"""Verify materialized paper evidence against the reviewed SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "evidence/SHA256SUMS"
MATERIALIZED_ROOTS = (
    REPOSITORY_ROOT / "autonumerics_zero/cuda_benchmarks/analysis_results",
    REPOSITORY_ROOT / "evidence/figures",
    REPOSITORY_ROOT / "evidence/report-data",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def entries(path: Path) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        expected, separator, name = line.partition("  ")
        if not separator or len(expected) != 64:
            raise ValueError(f"{path}:{line_number}: expected SHA256, two spaces, path")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{path}:{line_number}: path escapes repository")
        if relative in seen:
            raise ValueError(f"{path}:{line_number}: duplicate {relative}")
        seen.add(relative)
        parsed.append((expected, relative))
    if not parsed:
        raise ValueError(f"{path}: empty manifest")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    failures: list[str] = []
    manifest_entries = entries(args.manifest)
    listed = {relative for _, relative in manifest_entries}
    materialized = {
        path.relative_to(REPOSITORY_ROOT)
        for root in MATERIALIZED_ROOTS
        for path in root.rglob("*")
        if path.is_file()
    }
    for relative in sorted(materialized - listed):
        failures.append(f"unlisted materialized evidence {relative}")
    for relative in sorted(listed - materialized):
        failures.append(f"manifest path outside materialized evidence roots {relative}")
    for expected, relative in manifest_entries:
        candidate = REPOSITORY_ROOT / relative
        if not candidate.is_file():
            failures.append(f"missing {relative}")
        elif (actual := digest(candidate)) != expected:
            failures.append(f"hash mismatch {relative}: {actual}")
        else:
            print(f"PASS {relative}")
    for failure in failures:
        print(f"FAIL {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

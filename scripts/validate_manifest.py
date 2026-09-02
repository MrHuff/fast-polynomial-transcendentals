#!/usr/bin/env python3
"""Validate the experiment map and its repository-relative paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "repro/experiments.json"
STATUSES = {"runnable", "hardware-gated", "historical-only"}


def safe_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path escapes repository: {value}")
    return path


def validate(document: object) -> list[str]:
    failures: list[str] = []
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        return ["manifest must be a schema-version-1 object"]
    experiments = document.get("experiments")
    if not isinstance(experiments, list):
        return ["experiments must be an array"]
    ids: set[str] = set()
    for index, experiment in enumerate(experiments):
        label = f"experiments[{index}]"
        if not isinstance(experiment, dict):
            failures.append(f"{label} must be an object")
            continue
        experiment_id = experiment.get("id")
        if not isinstance(experiment_id, str) or not experiment_id:
            failures.append(f"{label}.id must be non-empty")
        elif experiment_id in ids:
            failures.append(f"duplicate id: {experiment_id}")
        else:
            ids.add(experiment_id)
        if experiment.get("status") not in STATUSES:
            failures.append(f"{label}.status is invalid")
        command = experiment.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            failures.append(f"{label}.command must be a string array")
        for field in ("artifact", "protocol", "provenance"):
            value = experiment.get(field)
            if value is None:
                continue
            try:
                relative = safe_path(value)
            except ValueError as error:
                failures.append(f"{label}.{field}: {error}")
                continue
            if not (REPOSITORY_ROOT / relative).is_file():
                failures.append(f"{label}.{field} is missing: {relative}")
        for value in experiment.get("artifacts", []):
            try:
                relative = safe_path(value)
            except ValueError as error:
                failures.append(f"{label}.artifacts: {error}")
                continue
            if not (REPOSITORY_ROOT / relative).is_file():
                failures.append(f"{label}.artifacts is missing: {relative}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = validate(document)
    for failure in failures:
        print(f"FAIL {failure}")
    if not failures:
        print(f"PASS {args.manifest}: {len(document['experiments'])} experiments")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

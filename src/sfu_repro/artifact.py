"""Validate and write portable experiment-result envelopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {
    "schema_version",
    "experiment",
    "source",
    "environment",
    "measurement",
    "results",
}


def validate_result(document: Any) -> list[str]:
    """Return structural validation failures for a result document."""
    failures: list[str] = []
    if not isinstance(document, dict):
        return ["document must be a JSON object"]
    missing = sorted(REQUIRED_TOP_LEVEL - document.keys())
    if missing:
        failures.append("missing top-level fields: " + ", ".join(missing))
    if document.get("schema_version") != 1:
        failures.append("schema_version must equal 1")
    experiment = document.get("experiment")
    if not isinstance(experiment, dict) or not experiment.get("id"):
        failures.append("experiment.id must be a non-empty value")
    for field in ("source", "environment", "measurement"):
        if field in document and not isinstance(document[field], dict):
            failures.append(f"{field} must be an object")
    if "results" in document and not isinstance(document["results"], (list, dict)):
        failures.append("results must be an object or array")
    return failures


def load_and_validate(path: Path) -> list[str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [str(error)]
    return validate_result(document)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    failed = False
    for path in args.paths:
        failures = load_and_validate(path)
        if failures:
            failed = True
            for failure in failures:
                print(f"FAIL {path}: {failure}")
        else:
            print(f"PASS {path}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

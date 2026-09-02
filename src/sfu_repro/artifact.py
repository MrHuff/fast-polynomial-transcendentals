"""Validate and write portable experiment-result envelopes."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema


@lru_cache(maxsize=1)
def result_schema() -> dict[str, Any]:
    resource = resources.files("sfu_repro.schemas").joinpath("result-v1.json")
    document = json.loads(resource.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(document)
    return document


@lru_cache(maxsize=1)
def result_validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(
        result_schema(),
        format_checker=jsonschema.FormatChecker(),
    )


def error_location(error: jsonschema.ValidationError) -> str:
    location = "$"
    for part in error.absolute_path:
        if isinstance(part, int):
            location += f"[{part}]"
        else:
            location += f".{part}"
    return location


def safe_error_message(error: jsonschema.ValidationError) -> str:
    """Describe a schema failure without echoing artifact values."""

    location = error_location(error)
    if error.validator == "required":
        required = set(error.validator_value)
        present = set(error.instance) if isinstance(error.instance, dict) else set()
        fields = ", ".join(sorted(required - present))
        return f"{location}: missing required field(s): {fields}"
    if error.validator == "type":
        expected = error.validator_value
        if isinstance(expected, list):
            expected = " or ".join(str(item) for item in expected)
        return f"{location}: must have type {expected}"
    if error.validator == "pattern":
        return f"{location}: must match the required pattern"
    if error.validator in {"minimum", "exclusiveMinimum", "maximum", "maxLength"}:
        return f"{location}: violates the {error.validator} constraint"
    if error.validator in {"const", "enum", "oneOf", "minLength", "minItems"}:
        return f"{location}: violates the {error.validator} constraint"
    return f"{location}: schema validation failed ({error.validator})"


def validate_result(document: Any) -> list[str]:
    """Return value-blind Draft 2020-12 schema-validation failures."""

    errors = sorted(
        result_validator().iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    return [safe_error_message(error) for error in errors]


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

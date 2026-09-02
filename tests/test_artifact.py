import json
from pathlib import Path

from sfu_repro.artifact import load_and_validate, result_schema, validate_result


def test_minimal_result_envelope() -> None:
    document = {
        "schema_version": 1,
        "experiment": {"id": "smoke"},
        "source": {},
        "environment": {},
        "measurement": {},
        "results": [],
    }
    assert validate_result(document) == []


def test_missing_fields_are_reported() -> None:
    failures = validate_result({"schema_version": 1})
    assert failures
    assert any("missing required field" in failure for failure in failures)


def test_validator_enforces_ids_revisions_and_iteration_bounds(
    tmp_path: Path,
) -> None:
    invalid = {
        "schema_version": 1,
        "experiment": {"id": "Bad ID!"},
        "source": {"revision": "not-a-commit"},
        "environment": {},
        "measurement": {"iterations": 0},
        "results": [],
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")

    failures = load_and_validate(path)

    assert len(failures) == 3
    assert any("$.experiment.id" in failure for failure in failures)
    assert any("$.source.revision" in failure for failure in failures)
    assert any("$.measurement.iterations" in failure for failure in failures)
    assert all("Bad ID!" not in failure for failure in failures)
    assert all("not-a-commit" not in failure for failure in failures)


def test_packaged_schema_matches_repository_copy() -> None:
    repository_schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "result-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert result_schema() == repository_schema

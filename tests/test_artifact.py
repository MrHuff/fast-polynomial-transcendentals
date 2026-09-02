from sfu_repro.artifact import validate_result


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
    assert any("missing top-level fields" in failure for failure in failures)

from __future__ import annotations

from pathlib import Path

from scripts.audit_release_tree import (
    contains_historical_identifiers,
    matching_content_rules,
)


def test_historical_identifier_scan_covers_csv_and_nested_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    csv_path.write_text("case, Run_ID ,value\nb1,opaque,1\n", encoding="utf-8")
    assert contains_historical_identifiers(
        csv_path, Path("evidence/report-data/history.csv")
    )

    json_path = tmp_path / "history.json"
    json_path.write_text(
        '{"outer": {"contributing_runs": [{"id": "opaque"}]}}\n',
        encoding="utf-8",
    )
    assert contains_historical_identifiers(
        json_path, Path("evidence/report-data/history.json")
    )


def test_identifier_scan_is_scoped_to_retained_evidence(tmp_path: Path) -> None:
    source_path = tmp_path / "example.json"
    source_path.write_text('{"run_id": "fixture"}\n', encoding="utf-8")

    assert not contains_historical_identifiers(
        source_path, Path("tests/fixtures/example.json")
    )


def test_wandb_version_and_disabled_flag_are_not_service_identifiers(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "tooling.json"
    json_path.write_text(
        '{"tool_versions": {"wandb": "0.24.0"}, "protocol": {"wandb": false}}\n',
        encoding="utf-8",
    )

    assert not contains_historical_identifiers(
        json_path, Path("evidence/report-data/tooling.json")
    )


def test_wandb_service_container_is_a_historical_identifier(tmp_path: Path) -> None:
    json_path = tmp_path / "service.json"
    json_path.write_text(
        '{"wandb": {"project": "review-required"}}\n', encoding="utf-8"
    )

    assert contains_historical_identifiers(
        json_path, Path("evidence/report-data/service.json")
    )


def test_content_rules_scan_binary_bytes_without_exposing_values() -> None:
    fake_hf_token = b"hf_" + (b"a" * 32)
    fake_private_key = b"-----BEGIN " + b"PRIVATE KEY-----"

    assert matching_content_rules(b"\xff" + fake_hf_token) == {"huggingface-token"}
    assert matching_content_rules(fake_private_key) == {"private-key"}


def test_content_rules_detect_prefixed_environment_secret_assignments() -> None:
    variable = b"WANDB_" + b"API_KEY"
    fake_value = b"a" * 40

    assert matching_content_rules(variable + b"=" + fake_value) == {
        "credential-assignment"
    }


def test_content_rules_ignore_environment_variable_references() -> None:
    assert not matching_content_rules(b"token = os.environ['HF_TOKEN']")

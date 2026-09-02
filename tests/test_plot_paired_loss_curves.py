from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from extras.paper.plot_paired_loss_curves import CASES, add_token_ewm, run


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    rows: list[dict[str, object]] = []
    for case_index, (case_name, case) in enumerate(CASES.items()):
        for role_index, role in enumerate(("baseline", "candidate")):
            label = getattr(case, role).label
            for point in range(7):
                rows.append(
                    {
                        "case": case_name,
                        "role": role,
                        "curve_label": label,
                        "tokens_seen": float(point * 10_000_000_000),
                        "loss": 5.0
                        - point * 0.2
                        + case_index * 0.01
                        + role_index * 0.005,
                        # This unrelated field must not reach the output receipt.
                        "private_metadata": "do-not-copy",
                    }
                )
    data_path = tmp_path / "curves.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)
    provenance_path = tmp_path / "source.json"
    provenance_path.write_text(
        json.dumps(
            {
                "artifact_sha256": sha256_file(data_path),
                "smoothing": {
                    "kind": "causal-token-ewm",
                    "half_life_tokens": 1_000_000_000.0,
                    "plot_grid_step_tokens": 50_000_000.0,
                },
                "private_metadata": "do-not-copy",
            }
        ),
        encoding="utf-8",
    )
    return data_path, provenance_path


def test_token_ewm_uses_token_delta() -> None:
    history = pd.DataFrame({"tokens_seen": [0.0, 10.0, 20.0], "loss": [4.0, 2.0, 2.0]})

    observed = add_token_ewm(history, half_life_tokens=10.0)

    assert observed["smoothed_loss"].tolist() == pytest.approx([4.0, 3.0, 2.5])


def test_offline_replot_is_deterministic_and_path_neutral(tmp_path: Path) -> None:
    data_path, source_path = write_fixture(tmp_path)
    first_figure = tmp_path / "first.pdf"
    first_receipt = tmp_path / "first.json"
    second_figure = tmp_path / "second.pdf"
    second_receipt = tmp_path / "second.json"

    run(data_path, source_path, first_figure, first_receipt)
    run(data_path, source_path, second_figure, second_receipt)

    assert first_figure.read_bytes() == second_figure.read_bytes()
    first = json.loads(first_receipt.read_text(encoding="utf-8"))
    second = json.loads(second_receipt.read_text(encoding="utf-8"))
    assert first == second
    assert first["inputs"]["source_hash_verified"] is True
    assert first["output"]["figure_sha256"] == sha256_file(first_figure)
    assert set(first["common_horizon_tokens"]) == set(CASES)
    assert "do-not-copy" not in first_receipt.read_text(encoding="utf-8")
    assert str(tmp_path) not in first_receipt.read_text(encoding="utf-8")


def test_replot_rejects_csv_that_does_not_match_provenance(tmp_path: Path) -> None:
    data_path, source_path = write_fixture(tmp_path)
    with data_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="does not match"):
        run(
            data_path,
            source_path,
            tmp_path / "figure.pdf",
            tmp_path / "receipt.json",
        )


def test_replot_rejects_duplicate_coordinates(tmp_path: Path) -> None:
    data_path, source_path = write_fixture(tmp_path)
    data = pd.read_csv(data_path)
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    data.to_csv(data_path, index=False)
    provenance = json.loads(source_path.read_text(encoding="utf-8"))
    provenance["artifact_sha256"] = sha256_file(data_path)
    source_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate token coordinates"):
        run(
            data_path,
            source_path,
            tmp_path / "figure.pdf",
            tmp_path / "receipt.json",
        )


def test_plotter_source_has_no_remote_or_identifier_client_code() -> None:
    source = (
        Path(__file__).parents[1] / "extras" / "paper" / "plot_paired_loss_curves.py"
    ).read_text(encoding="utf-8")
    prohibited = ("wandb", "requests", "urllib", "run_id", "api_key")

    assert not any(token in source.lower() for token in prohibited)

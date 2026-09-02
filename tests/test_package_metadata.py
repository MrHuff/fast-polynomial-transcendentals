from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _dependency_name(requirement: str) -> str:
    for marker in (";", "[", "=", "<", ">"):
        requirement = requirement.split(marker, 1)[0]
    return requirement.strip().lower()


def test_full_test_dependencies_are_optional() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    base = {_dependency_name(item) for item in project["dependencies"]}
    test = {_dependency_name(item) for item in project["optional-dependencies"]["test"]}

    assert {"torch", "transformers"}.isdisjoint(base)
    assert {"pytest", "torch", "transformers"} <= test

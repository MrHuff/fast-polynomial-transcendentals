from __future__ import annotations

import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from sfu_repro.source_attestation import (
    SourceAttestationError,
    attest_module_source,
    require_bound_attestations,
    safe_command,
)


def initialize_repository(path: Path) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "module.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def test_module_inside_clean_checkout_is_bound(tmp_path: Path) -> None:
    revision = initialize_repository(tmp_path)
    module = ModuleType("fixture_module")
    module.__file__ = str(tmp_path / "module.py")

    attestation = attest_module_source(
        "fixture_module",
        source_checkout=tmp_path,
        repository_root=tmp_path,
        expected_revision=revision,
        module=module,
    )

    assert attestation["bound"] is True
    assert attestation["binding_method"] == "module-within-source"
    assert attestation["module_file"] == "module.py"


def test_dirty_checkout_is_unbound_without_diagnostic_override(tmp_path: Path) -> None:
    revision = initialize_repository(tmp_path)
    module = ModuleType("fixture_module")
    module.__file__ = str(tmp_path / "module.py")
    (tmp_path / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    attestation = attest_module_source(
        "fixture_module",
        source_checkout=tmp_path,
        repository_root=tmp_path,
        expected_revision=revision,
        module=module,
    )

    with pytest.raises(SourceAttestationError, match="fixture_module"):
        require_bound_attestations({"fixture_module": attestation}, allow_unbound=False)
    require_bound_attestations({"fixture_module": attestation}, allow_unbound=True)


def test_safe_command_replaces_external_absolute_prefix(tmp_path: Path) -> None:
    assert safe_command(
        ["python", f"--output={tmp_path / 'result.json'}", "/outside/model.bin"],
        tmp_path,
    ) == ["python", "--output=result.json", "<external>/model.bin"]

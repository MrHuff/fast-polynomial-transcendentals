# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from autonumerics_zero.spline_ops import audit_sincos_sass
from sfu_repro.artifact import validate_result


def test_sass_audit_document_binds_the_exact_binary(
    tmp_path: Path, monkeypatch
) -> None:
    extension = tmp_path / "spline_ops.so"
    cuobjdump = tmp_path / "cuobjdump"
    extension.write_bytes(b"compiled-extension")
    cuobjdump.write_bytes(b"tool")
    monkeypatch.setattr(
        audit_sincos_sass,
        "tool_version",
        lambda _path: "Cuda compilation tools, release 13.0",
    )

    document = audit_sincos_sass.result_document(
        opcode_results={"native_sfu": {"mufu_instructions": ["MUFU.SIN"]}},
        extension=extension,
        cuobjdump=cuobjdump,
        repository_state={
            "revision": "a" * 40,
            "dirty": False,
            "untracked_files": 0,
        },
        command=["python", "audit_sincos_sass.py", "<external>/spline_ops.so"],
    )

    assert document["experiment"]["provenance_class"] == "new-measurement"
    assert document["measurement"]["artifact"] == "<external>/spline_ops.so"
    assert len(document["measurement"]["artifact_sha256"]) == 64
    assert (
        document["measurement"]["artifact_sha256"]
        == document["source"]["input_sha256"]["extension_binary"]
    )
    assert document["results"]["native_sfu"]["mufu_instructions"] == ["MUFU.SIN"]
    assert validate_result(document) == []


def test_sass_audit_dirty_source_is_labeled_diagnostic(
    tmp_path: Path, monkeypatch
) -> None:
    extension = tmp_path / "spline_ops.so"
    cuobjdump = tmp_path / "cuobjdump"
    extension.write_bytes(b"extension")
    cuobjdump.write_bytes(b"tool")
    monkeypatch.setattr(audit_sincos_sass, "tool_version", lambda _path: "13.0")

    document = audit_sincos_sass.result_document(
        opcode_results={},
        extension=extension,
        cuobjdump=cuobjdump,
        repository_state={
            "revision": "a" * 40,
            "dirty": True,
            "untracked_files": 1,
        },
        command=["python", "audit_sincos_sass.py"],
    )

    assert document["experiment"]["provenance_class"] == ("diagnostic-unbound-source")

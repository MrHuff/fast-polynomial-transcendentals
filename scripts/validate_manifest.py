#!/usr/bin/env python3
"""Validate the experiment map, command semantics, and local path references."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "repro/experiments.json"
STATUSES = {"runnable", "hardware-gated", "historical-only"}
OUTPUT_CLASSES = {
    "new-measurement",
    "generated-fit",
    "source-bound-verification",
    "binary-bound-verification",
    "derived-artifact",
    "historical-evidence",
}
REAL_COMMAND_LIST_FIELDS = (
    "setup_commands",
    "additional_commands",
    "postprocess_commands",
)
COMMAND_SEMANTIC_FIELDS = (
    "command",
    "preview_command",
    *REAL_COMMAND_LIST_FIELDS,
)
DEPRECATED_COMMAND_FIELDS = (
    "export_command",
    "export_commands",
    "preview_commands",
)
REPOSITORY_FILE_FIELDS = (
    "artifact",
    "protocol",
    "task_protocol",
    "historical_protocol",
    "provenance",
)
REPOSITORY_FILE_LIST_FIELDS = ("artifacts", "historical_evidence")
INPUT_PATH_OPTIONS = {
    "--bf16-input",
    "--candidate-dir",
    "--claims",
    "--coefficients",
    "--config",
    "--current-header",
    "--data-path",
    "--evidence-dir",
    "--environments",
    "--experiment-manifest",
    "--extension-dir",
    "--fa4-root",
    "--fallback-header",
    "--fp16-input",
    "--function-comparison",
    "--function-lineage",
    "--hf-assets-path",
    "--input",
    "--input-provenance",
    "--manifest",
    "--native-dir",
    "--polynomial-dir",
    "--repository-root",
    "--seed-checkpoint",
    "--task-config",
}
OUTPUT_PATH_OPTIONS = {
    "--csv-out",
    "--header-out",
    "--json-out",
    "--markdown-out",
    "--output",
    "--output-dir",
    "--output-path",
    "--output-provenance",
    "--receipt",
    "--receipt-out",
    "--tex-output",
}
PATH_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".pdf",
    ".py",
    ".toml",
    ".txt",
    ".zip",
}
PLACEHOLDER_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")


def safe_path(value: str) -> Path:
    """Return a repository-relative path and reject ambiguous escapes."""

    if not isinstance(value, str) or not value:
        raise ValueError("path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path escapes repository: {value}")
    return path


def _string_array(value: object, label: str, failures: list[str]) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        failures.append(f"{label} must be an array of non-empty strings")
        return []
    return value


def _command_array(
    value: object,
    label: str,
    failures: list[str],
    *,
    allow_empty: bool = False,
) -> list[str]:
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or not all(isinstance(item, str) and item for item in value)
    ):
        failures.append(f"{label} must be a non-empty-string array")
        return []
    return value


def _nested_command_arrays(
    value: object,
    label: str,
    failures: list[str],
) -> list[tuple[str, list[str]]]:
    if not isinstance(value, list):
        failures.append(f"{label} must be an array of command arrays")
        return []
    commands: list[tuple[str, list[str]]] = []
    for index, raw_command in enumerate(value):
        command_label = f"{label}[{index}]"
        command = _command_array(raw_command, command_label, failures)
        if command:
            commands.append((command_label, command))
    return commands


def _entrypoint(command: list[str]) -> tuple[str, ...]:
    if len(command) >= 3 and command[1] == "-m":
        return command[0], "-m", command[2]
    if len(command) >= 2:
        return command[0], command[1]
    return tuple(command)


def _uses_script(command: list[str], name: str) -> bool:
    return len(command) >= 2 and Path(command[1]).name == name


def _validate_command_semantics(
    command: list[str],
    label: str,
    failures: list[str],
    *,
    preview: bool,
) -> None:
    if preview:
        if "--execute" in command:
            failures.append(f"{label} is a preview but contains --execute")
        if (
            _uses_script(command, "run_open_weight_suite.py")
            and "--dry-run" not in command
        ):
            failures.append(f"{label} open-weight preview must contain --dry-run")
        if not (
            "--dry-run" in command
            or "--help" in command
            or _uses_script(command, "run_torchtitan.py")
        ):
            failures.append(f"{label} is not recognizably non-executing")
        return

    if "--dry-run" in command:
        failures.append(f"{label} is a real command but contains --dry-run")
    if _uses_script(command, "run_torchtitan.py") and "--execute" not in command:
        failures.append(f"{label} TorchTitan execution must contain --execute")


def _module_exists(module: str, repository_root: Path) -> bool:
    relative = Path(*module.split("."))
    candidates = (
        repository_root / relative.with_suffix(".py"),
        repository_root / relative / "__main__.py",
        repository_root / "src" / relative.with_suffix(".py"),
        repository_root / "src" / relative / "__main__.py",
    )
    return any(candidate.is_file() for candidate in candidates)


def _declared_placeholder(value: str, substitutions: set[str]) -> bool:
    names = set(PLACEHOLDER_PATTERN.findall(value))
    return bool(names) and names <= substitutions


def _looks_like_path(value: str) -> bool:
    return (
        "/" in value
        or value.startswith(".")
        or Path(value).suffix.lower() in PATH_SUFFIXES
    )


def _validate_path_value(
    value: str,
    label: str,
    failures: list[str],
    *,
    repository_root: Path,
    role: str,
    substitutions: set[str],
    generated_inputs: set[str],
    declared_outputs: set[str],
) -> None:
    try:
        relative = safe_path(value)
    except ValueError as error:
        failures.append(f"{label}: {error}")
        return
    normalized = relative.as_posix()
    if _declared_placeholder(value, substitutions):
        return
    if role == "output":
        if normalized not in declared_outputs:
            failures.append(
                f"{label} output path is not declared in workflow.outputs: {normalized}"
            )
        return
    if (repository_root / relative).exists():
        return
    if normalized in generated_inputs:
        return
    failures.append(f"{label} references a missing input path: {normalized}")


def _validate_command_paths(
    command: list[str],
    label: str,
    failures: list[str],
    *,
    repository_root: Path,
    substitutions: set[str],
    generated_inputs: set[str],
    declared_outputs: set[str],
) -> None:
    if not command:
        return
    if command[0] not in {"python", "python3"}:
        failures.append(f"{label} must invoke python or python3 explicitly")
        return

    if len(command) < 2:
        failures.append(f"{label} has no Python entry point")
        return
    if command[1] == "-m":
        if len(command) < 3:
            failures.append(f"{label} has no module after -m")
            return
        module = command[2]
        if module.startswith(
            ("sfu_repro.", "autonumerics_zero.")
        ) and not _module_exists(module, repository_root):
            failures.append(f"{label} references a missing local module: {module}")
        argument_start = 3
    else:
        try:
            entrypoint = safe_path(command[1])
        except ValueError as error:
            failures.append(f"{label} entry point: {error}")
            return
        if not (repository_root / entrypoint).is_file():
            failures.append(f"{label} entry point is missing: {entrypoint}")
        argument_start = 2

    index = argument_start
    while index < len(command):
        token = command[index]
        option, separator, inline_value = token.partition("=")
        if option in INPUT_PATH_OPTIONS or option in OUTPUT_PATH_OPTIONS:
            if separator:
                value = inline_value
            elif index + 1 < len(command):
                index += 1
                value = command[index]
            else:
                failures.append(f"{label} {option} has no path value")
                index += 1
                continue
            _validate_path_value(
                value,
                f"{label} {option}",
                failures,
                repository_root=repository_root,
                role="output" if option in OUTPUT_PATH_OPTIONS else "input",
                substitutions=substitutions,
                generated_inputs=generated_inputs,
                declared_outputs=declared_outputs,
            )
        elif token.startswith("--") and separator and _looks_like_path(inline_value):
            _validate_path_value(
                inline_value,
                f"{label} {option}",
                failures,
                repository_root=repository_root,
                role="input",
                substitutions=substitutions,
                generated_inputs=generated_inputs,
                declared_outputs=declared_outputs,
            )
        elif not token.startswith("-") and _looks_like_path(token):
            _validate_path_value(
                token,
                f"{label} argument",
                failures,
                repository_root=repository_root,
                role="input",
                substitutions=substitutions,
                generated_inputs=generated_inputs,
                declared_outputs=declared_outputs,
            )
        index += 1


def _validate_repository_file(
    value: object,
    label: str,
    failures: list[str],
    repository_root: Path,
) -> None:
    try:
        relative = safe_path(value)  # type: ignore[arg-type]
    except ValueError as error:
        failures.append(f"{label}: {error}")
        return
    if not (repository_root / relative).is_file():
        failures.append(f"{label} is missing: {relative}")


def validate(
    document: object,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> list[str]:
    """Return every structural, semantic, and path failure in a manifest."""

    failures: list[str] = []
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        return ["manifest must be a schema-version-1 object"]
    command_semantics = document.get("command_semantics")
    if not isinstance(command_semantics, dict) or not all(
        isinstance(command_semantics.get(field), str) and command_semantics[field]
        for field in COMMAND_SEMANTIC_FIELDS
    ):
        failures.append("command_semantics must describe every command-array field")
    experiments = document.get("experiments")
    if not isinstance(experiments, list):
        return [*failures, "experiments must be an array"]

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

        status = experiment.get("status")
        if status not in STATUSES:
            failures.append(f"{label}.status is invalid")

        workflow = experiment.get("workflow")
        declared_outputs: set[str] = set()
        substitutions: set[str] = set()
        if not isinstance(workflow, dict):
            failures.append(f"{label}.workflow must be an object")
        else:
            kind = workflow.get("kind")
            if not isinstance(kind, str) or not kind:
                failures.append(f"{label}.workflow.kind must be non-empty")
            if workflow.get("output_class") not in OUTPUT_CLASSES:
                failures.append(f"{label}.workflow.output_class is invalid")
            elif (
                status == "historical-only"
                and workflow["output_class"] != "historical-evidence"
            ):
                failures.append(
                    f"{label} historical-only workflow must use historical-evidence"
                )
            elif (
                status != "historical-only"
                and workflow["output_class"] == "historical-evidence"
            ):
                failures.append(
                    f"{label} executable workflow cannot use historical-evidence"
                )
            raw_outputs = _string_array(
                workflow.get("outputs"), f"{label}.workflow.outputs", failures
            )
            if status != "historical-only" and not raw_outputs:
                failures.append(f"{label} executable workflow needs declared outputs")
            for output_index, output in enumerate(raw_outputs):
                try:
                    declared_outputs.add(safe_path(output).as_posix())
                except ValueError as error:
                    failures.append(
                        f"{label}.workflow.outputs[{output_index}]: {error}"
                    )
            raw_substitutions = workflow.get("substitutions", {})
            if not isinstance(raw_substitutions, dict) or not all(
                isinstance(name, str)
                and PLACEHOLDER_PATTERN.fullmatch(name)
                and isinstance(description, str)
                and description
                for name, description in raw_substitutions.items()
            ):
                failures.append(
                    f"{label}.workflow.substitutions must map placeholders to text"
                )
            else:
                substitutions = set(raw_substitutions)
            if not isinstance(workflow.get("notes"), str) or not workflow["notes"]:
                failures.append(f"{label}.workflow.notes must be non-empty")

        prerequisites = experiment.get("prerequisites")
        generated_inputs: set[str] = set()
        if status == "historical-only" and prerequisites is None:
            prerequisites = {}
        if not isinstance(prerequisites, dict):
            failures.append(f"{label}.prerequisites must be an object")
            prerequisites = {}
        repository_paths = _string_array(
            prerequisites.get("repository_paths", []),
            f"{label}.prerequisites.repository_paths",
            failures,
        )
        for path_index, value in enumerate(repository_paths):
            try:
                relative = safe_path(value)
            except ValueError as error:
                failures.append(
                    f"{label}.prerequisites.repository_paths[{path_index}]: {error}"
                )
                continue
            if not (repository_root / relative).exists():
                failures.append(
                    f"{label}.prerequisites.repository_paths[{path_index}] "
                    f"is missing: {relative}"
                )
        raw_generated = _string_array(
            prerequisites.get("generated_inputs", []),
            f"{label}.prerequisites.generated_inputs",
            failures,
        )
        for path_index, value in enumerate(raw_generated):
            try:
                generated_inputs.add(safe_path(value).as_posix())
            except ValueError as error:
                failures.append(
                    f"{label}.prerequisites.generated_inputs[{path_index}]: {error}"
                )
        software = _string_array(
            prerequisites.get("software", []),
            f"{label}.prerequisites.software",
            failures,
        )
        hardware = _string_array(
            prerequisites.get("hardware", []),
            f"{label}.prerequisites.hardware",
            failures,
        )
        _string_array(
            prerequisites.get("external_assets", []),
            f"{label}.prerequisites.external_assets",
            failures,
        )
        if status != "historical-only" and (not repository_paths or not software):
            failures.append(
                f"{label} executable workflow needs repository_paths and software"
            )
        if status == "hardware-gated" and not hardware:
            failures.append(f"{label} hardware-gated workflow needs hardware metadata")
        if status == "hardware-gated" and (
            not isinstance(experiment.get("hardware"), str)
            or not experiment["hardware"]
        ):
            failures.append(f"{label} hardware-gated workflow needs a hardware summary")

        for field in DEPRECATED_COMMAND_FIELDS:
            if field in experiment:
                failures.append(f"{label}.{field} is deprecated")

        raw_primary = experiment.get("command")
        primary = _command_array(
            raw_primary,
            f"{label}.command",
            failures,
            allow_empty=status == "historical-only",
        )
        if status == "historical-only":
            if raw_primary != []:
                failures.append(f"{label} historical-only command must be empty")
            primary = []
        elif not primary:
            failures.append(f"{label} executable workflow needs a real command")

        real_commands: list[tuple[str, list[str]]] = []
        if primary:
            real_commands.append((f"{label}.command", primary))
        for field in REAL_COMMAND_LIST_FIELDS:
            if field in experiment:
                real_commands.extend(
                    _nested_command_arrays(
                        experiment[field], f"{label}.{field}", failures
                    )
                )
        if status == "historical-only" and real_commands:
            failures.append(f"{label} historical-only workflow has executable commands")

        preview: list[str] = []
        if "preview_command" in experiment:
            preview = _command_array(
                experiment["preview_command"],
                f"{label}.preview_command",
                failures,
            )
            if preview and primary and _entrypoint(preview) != _entrypoint(primary):
                failures.append(f"{label}.preview_command uses a different entry point")

        all_commands = [command for _, command in real_commands]
        if preview:
            all_commands.append(preview)
        observed_placeholders = {
            name
            for command in all_commands
            for token in command
            for name in PLACEHOLDER_PATTERN.findall(token)
        }
        undeclared = observed_placeholders - substitutions
        unused = substitutions - observed_placeholders
        if undeclared:
            failures.append(
                f"{label} has undeclared substitutions: "
                + ", ".join(sorted(undeclared))
            )
        if unused:
            failures.append(
                f"{label} has unused substitutions: " + ", ".join(sorted(unused))
            )

        for command_label, command in real_commands:
            _validate_command_semantics(command, command_label, failures, preview=False)
            _validate_command_paths(
                command,
                command_label,
                failures,
                repository_root=repository_root,
                substitutions=substitutions,
                generated_inputs=generated_inputs,
                declared_outputs=declared_outputs,
            )
        if preview:
            _validate_command_semantics(
                preview, f"{label}.preview_command", failures, preview=True
            )
            _validate_command_paths(
                preview,
                f"{label}.preview_command",
                failures,
                repository_root=repository_root,
                substitutions=substitutions,
                generated_inputs=generated_inputs,
                declared_outputs=declared_outputs,
            )

        for field in REPOSITORY_FILE_FIELDS:
            value = experiment.get(field)
            if value is not None:
                _validate_repository_file(
                    value, f"{label}.{field}", failures, repository_root
                )
        for field in REPOSITORY_FILE_LIST_FIELDS:
            values = experiment.get(field, [])
            if not isinstance(values, list):
                failures.append(f"{label}.{field} must be an array")
                continue
            for value_index, value in enumerate(values):
                _validate_repository_file(
                    value,
                    f"{label}.{field}[{value_index}]",
                    failures,
                    repository_root,
                )
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

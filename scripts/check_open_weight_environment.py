#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect the isolated environments required by the paper's downstream evals.

This checker does not download models, install packages, or import CUDA
extensions. It compares installed distribution metadata with the package pins
derived from the historical launcher and reports missing checkpoint revisions
separately from software readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from importlib.metadata import (
    PackageNotFoundError,
    distribution as distribution_metadata,
    version,
)
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_open_weight_suite as suite  # noqa: E402
from scripts.lm_eval_task_pins import (  # noqa: E402
    activate_harness_checkout,
    inspect_harness_checkout,
    load_task_protocol,
)


DEFAULT_ENVIRONMENTS = (
    REPOSITORY_ROOT / "configs" / "eval_environments" / "profiles.json"
)
_EXACT_PIN = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;]+)$")
_NATIVE_BINARY_SUFFIXES = (".so", ".pyd", ".dll", ".dylib")


def canonical_distribution(name: str) -> str:
    """Apply the distribution-name normalization used by package metadata."""

    return re.sub(r"[-_.]+", "-", name).lower()


def load_environment_profiles(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("environment profile schema_version must equal 1")
    profiles = document.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("environment profiles must be a non-empty object")
    return document


def requirement_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = _EXACT_PIN.fullmatch(line)
        if match is None:
            raise ValueError(
                f"{path}:{line_number}: expected an exact NAME==VERSION pin"
            )
        name = canonical_distribution(match.group("name"))
        if name in pins:
            raise ValueError(f"{path}:{line_number}: duplicate pin for {name}")
        pins[name] = match.group("version")
    if not pins:
        raise ValueError(f"{path}: no package pins found")
    return pins


def repository_path(reference: str, repository_root: Path) -> Path:
    """Resolve a repository-relative protocol path without changing the CWD."""

    path = Path(reference)
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def inspect_quality_task_protocol(
    open_weight_config: suite.SuiteConfig,
    environment_document: dict[str, Any],
    *,
    check_installed: bool,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Validate and report the task, dataset, and harness protocol."""

    protocol_config = repository_path(
        open_weight_config.protocol.quality_task_config,
        repository_root,
    )
    protocol = load_task_protocol(protocol_config)
    selected_pins = protocol.select(open_weight_config.protocol.quality_tasks)
    harness_distribution = canonical_distribution(protocol.harness_distribution)

    profile_versions: list[dict[str, Any]] = []
    for profile_name, raw_profile in environment_document["profiles"].items():
        requirements = repository_path(raw_profile["requirements"], repository_root)
        pins = requirement_pins(requirements)
        declared = pins.get(harness_distribution)
        profile_versions.append(
            {
                "profile": profile_name,
                "requirements": str(raw_profile["requirements"]),
                "declared_version": declared,
                "matches_protocol": declared == protocol.harness_version,
            }
        )

    checkout_report: dict[str, Any] = {
        "checked": check_installed,
        "path": protocol.harness_submodule_path,
        "expected_revision": protocol.harness_revision,
        "revision": None,
        "clean": None,
        "module_file": None,
        "matches": None,
        "error": None,
    }
    if check_installed:
        try:
            checkout = inspect_harness_checkout(protocol, repository_root)
        except Exception as error:  # readiness output should retain import failures
            checkout_report["matches"] = False
            checkout_report["error"] = f"{type(error).__name__}: {error}"
        else:
            checkout_report.update(
                {
                    "revision": checkout.revision,
                    "clean": checkout.clean,
                }
            )
            try:
                activated = activate_harness_checkout(protocol, repository_root)
            except Exception as error:  # report an import failure after Git succeeds
                checkout_report["matches"] = False
                checkout_report["error"] = f"{type(error).__name__}: {error}"
            else:
                checkout_report["module_file"] = activated.module_file
                checkout_report["matches"] = True

    try:
        protocol_path = str(protocol_config.relative_to(repository_root.resolve()))
    except ValueError:
        protocol_path = str(protocol_config)
    return {
        "path": protocol_path,
        "protocol_class": protocol.protocol_class,
        "selection_date": protocol.selection_date,
        "quality_tasks": list(open_weight_config.protocol.quality_tasks),
        "quality_log_samples": open_weight_config.protocol.quality_log_samples,
        "all_quality_tasks_covered": len(selected_pins)
        == len(open_weight_config.protocol.quality_tasks),
        "harness": {
            "distribution": protocol.harness_distribution,
            "version": protocol.harness_version,
            "repository": protocol.harness_repository,
            "revision": protocol.harness_revision,
            "submodule_path": protocol.harness_submodule_path,
        },
        "evaluator_seeds": dict(protocol.evaluator_seeds),
        "dataset_pins": {
            pin.task: {
                "dataset_path": pin.dataset_path,
                "revision": pin.revision,
                "revision_provenance": pin.revision_provenance,
                "num_fewshot": open_weight_config.protocol.quality_fewshot[pin.task],
            }
            for pin in selected_pins
        },
        "environment_profile_versions": profile_versions,
        "harness_checkout": checkout_report,
        "historical_boundary": protocol.historical_boundary,
    }


def case_to_profile(document: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for profile_name, raw_profile in document["profiles"].items():
        cases = raw_profile.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"profile {profile_name!r} must list cases")
        for case in cases:
            if not isinstance(case, str) or not case:
                raise ValueError(f"profile {profile_name!r} has an invalid case")
            if case in mapping:
                raise ValueError(f"case {case!r} appears in multiple profiles")
            mapping[case] = profile_name
    return mapping


def module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def git_revision(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and len(revision) == 40 else None


def git_worktree_state(path: Path) -> tuple[bool | None, int | None]:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=path,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode:
        return None, None
    entries = tuple(item for item in completed.stdout.split("\0") if item)
    return bool(entries), sum(item.startswith("?? ") for item in entries)


def module_file(module: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, AttributeError, ValueError):
        return None
    if (
        spec is None
        or not isinstance(spec.origin, str)
        or spec.origin
        in {
            "built-in",
            "frozen",
        }
    ):
        return None
    return Path(spec.origin).resolve()


def distribution_direct_source(distribution: str) -> Path | None:
    try:
        metadata = distribution_metadata(distribution)
    except PackageNotFoundError:
        return None
    raw = metadata.read_text("direct_url.json")
    if not raw:
        return None
    try:
        document = json.loads(raw)
        parsed = urlparse(str(document.get("url", "")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return None
    return Path(unquote(parsed.path)).resolve()


def _path_label(path: Path | None, repository_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def _sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def distribution_native_binary_hashes(distribution: str) -> dict[str, str]:
    try:
        metadata = distribution_metadata(distribution)
    except PackageNotFoundError:
        return {}
    binaries: dict[str, str] = {}
    for declared in getattr(metadata, "files", ()) or ():
        declared_text = str(declared)
        if not declared_text.lower().endswith(_NATIVE_BINARY_SUFFIXES):
            continue
        try:
            resolved = Path(metadata.locate_file(declared)).resolve()
        except (AttributeError, OSError, TypeError):
            continue
        digest = _sha256_file(resolved)
        if digest is None:
            continue
        declared_path = Path(declared_text)
        label = (
            declared_path.as_posix()
            if not declared_path.is_absolute() and ".." not in declared_path.parts
            else f"<external>/{resolved.name}"
        )
        binaries[label] = digest
    return dict(sorted(binaries.items()))


def inspect_local_build(
    name: str,
    build: dict[str, Any],
    *,
    check_installed: bool,
    repository_root: Path,
    repository_revision: str | None,
) -> dict[str, Any]:
    required = ("module", "distribution", "version", "path", "install")
    if any(
        not isinstance(build.get(field), str) or not build[field] for field in required
    ):
        raise ValueError(f"local_builds.{name} lacks module provenance metadata")
    source_relative = Path(build["path"])
    if source_relative.is_absolute() or ".." in source_relative.parts:
        raise ValueError(f"local_builds.{name}.path must stay inside the repository")
    source = (repository_root / source_relative).resolve()
    revision_source = build.get("revision_source")
    configured_revision = build.get("revision")
    if revision_source not in {None, "repository"}:
        raise ValueError(f"local_builds.{name}.revision_source is invalid")
    if revision_source is None and (
        not isinstance(configured_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", configured_revision) is None
    ):
        raise ValueError(f"local_builds.{name}.revision must be a full commit")
    expected_revision = (
        repository_revision if revision_source == "repository" else configured_revision
    )
    require_native_binary = build.get("require_native_binary", False)
    if not isinstance(require_native_binary, bool):
        raise ValueError(f"local_builds.{name}.require_native_binary must be boolean")

    observed_module_file = module_file(build["module"]) if check_installed else None
    direct_source = (
        distribution_direct_source(build["distribution"]) if check_installed else None
    )
    module_within_source = bool(
        observed_module_file is not None and observed_module_file.is_relative_to(source)
    )
    direct_url_matches_source = direct_source == source if check_installed else None
    origin_matches_source = (
        module_within_source or direct_url_matches_source if check_installed else None
    )
    observed_revision = (
        git_revision(source) if check_installed and source.exists() else None
    )
    dirty, untracked = (
        git_worktree_state(source)
        if check_installed and source.exists()
        else (None, None)
    )
    observed_version = (
        installed_version(build["distribution"]) if check_installed else None
    )
    native_binaries = (
        distribution_native_binary_hashes(build["distribution"])
        if check_installed
        else {}
    )
    return {
        "name": name,
        "module": build["module"],
        "module_file": _path_label(observed_module_file, repository_root),
        "distribution": build["distribution"],
        "expected_version": build["version"],
        "installed_version": observed_version,
        "package_matches": (
            observed_version == build["version"] if check_installed else None
        ),
        "path": source_relative.as_posix(),
        "install": build["install"],
        "direct_url_source": _path_label(direct_source, repository_root),
        "origin_matches_source": origin_matches_source,
        "expected_revision": expected_revision,
        "checkout_revision": observed_revision,
        "checkout_revision_matches": (
            observed_revision == expected_revision if check_installed else None
        ),
        "checkout_dirty": dirty,
        "checkout_untracked_files": untracked,
        "native_binary_required": require_native_binary,
        "native_binary_sha256": native_binaries,
    }


def inspect_profiles(
    selected_cases: Sequence[suite.OpenWeightCase],
    configured_cases: Sequence[suite.OpenWeightCase],
    environment_document: dict[str, Any],
    *,
    check_installed: bool,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    mapping = case_to_profile(environment_document)
    configured_keys = {case.key for case in configured_cases}
    unknown_profile_cases = set(mapping) - configured_keys
    if unknown_profile_cases:
        raise ValueError(
            "environment profiles contain unknown cases: "
            + ", ".join(sorted(unknown_profile_cases))
        )
    missing_profiles = {case.key for case in selected_cases} - set(mapping)
    if missing_profiles:
        raise ValueError(
            "selected cases lack environment profiles: "
            + ", ".join(sorted(missing_profiles))
        )

    selected_by_profile: dict[str, list[suite.OpenWeightCase]] = {}
    for case in selected_cases:
        selected_by_profile.setdefault(mapping[case.key], []).append(case)

    profile_reports: list[dict[str, Any]] = []
    for profile_name, cases in selected_by_profile.items():
        raw_profile = environment_document["profiles"][profile_name]
        requirements = repository_root / raw_profile["requirements"]
        pins = requirement_pins(requirements)
        package_checks: list[dict[str, Any]] = []
        for distribution, expected in pins.items():
            installed = installed_version(distribution) if check_installed else None
            package_checks.append(
                {
                    "distribution": distribution,
                    "expected": expected,
                    "installed": installed,
                    "matches": installed == expected if check_installed else None,
                }
            )

        required_modules = set(raw_profile.get("required_modules", []))
        per_case_modules = raw_profile.get("case_required_modules", {})
        for case in cases:
            required_modules.update(per_case_modules.get(case.key, []))
        module_checks = [
            {
                "module": module,
                "available": module_available(module) if check_installed else None,
            }
            for module in sorted(required_modules)
        ]
        profile_reports.append(
            {
                "name": profile_name,
                "cases": [case.key for case in cases],
                "requirements": str(requirements.relative_to(repository_root)),
                "packages": package_checks,
                "modules": module_checks,
            }
        )

    revision_reports = [
        {
            "case": case.key,
            "model": case.model_id,
            "revision": case.revision,
            "revision_provenance": case.revision_provenance,
            "historically_recorded": (
                case.revision_provenance == "historical-source-derived"
            ),
        }
        for case in selected_cases
    ]
    local_builds: list[dict[str, Any]] = []
    selected_keys = {case.key for case in selected_cases}
    repository_revision = git_revision(repository_root) if check_installed else None
    for name, build in environment_document.get("local_builds", {}).items():
        used_by = set(build.get("used_by", selected_keys))
        if not selected_keys & used_by:
            continue
        local_builds.append(
            inspect_local_build(
                name,
                build,
                check_installed=check_installed,
                repository_root=repository_root,
                repository_revision=repository_revision,
            )
        )

    return {
        "schema_version": 1,
        "readiness_scope": "declared public rerun inputs, not historical identity",
        "check_installed": check_installed,
        "python": sys.version.split()[0],
        "torch": installed_version("torch") if check_installed else None,
        "profiles": profile_reports,
        "model_revisions": revision_reports,
        "local_builds": local_builds,
        "historical_boundaries": environment_document["historical_boundaries"],
    }


def failures(report: dict[str, Any], *, allow_unrecorded_revisions: bool) -> list[str]:
    problems: list[str] = []
    task_protocol = report["quality_task_protocol"]
    for profile in task_protocol["environment_profile_versions"]:
        if not profile["matches_protocol"]:
            problems.append(
                f"{profile['profile']}: requirements declare lm-eval "
                f"{profile['declared_version'] or 'missing'}, expected "
                f"{task_protocol['harness']['version']} from the task protocol"
            )
    if report["check_installed"]:
        checkout = task_protocol["harness_checkout"]
        if not checkout["matches"]:
            problems.append(
                "lm-eval pinned checkout/import verification failed: "
                f"{checkout['error'] or 'unknown error'}"
            )
        if report["torch"] is None:
            problems.append("PyTorch is not installed")
        for profile in report["profiles"]:
            for package in profile["packages"]:
                if not package["matches"]:
                    problems.append(
                        f"{profile['name']}: {package['distribution']} "
                        f"is {package['installed'] or 'missing'}, expected "
                        f"{package['expected']}"
                    )
            for module in profile["modules"]:
                if not module["available"]:
                    problems.append(
                        f"{profile['name']}: Python module {module['module']} is missing"
                    )
        for build in report["local_builds"]:
            if not build["package_matches"]:
                problems.append(
                    f"{build['name']}: {build['distribution']} is "
                    f"{build['installed_version'] or 'missing'}, expected "
                    f"{build['expected_version']}"
                )
            if not build["origin_matches_source"]:
                problems.append(
                    f"{build['name']}: loaded module is not bound to local source "
                    f"{build['path']}"
                )
            if not build["checkout_revision_matches"]:
                problems.append(
                    f"{build['name']}: checkout is {build['checkout_revision'] or 'missing'}, "
                    f"expected {build['expected_revision']}"
                )
            if build["checkout_dirty"] is not False:
                problems.append(
                    f"{build['name']}: source checkout is dirty or unavailable"
                )
            if build["native_binary_required"] and not build["native_binary_sha256"]:
                problems.append(
                    f"{build['name']}: installed native binary inventory is empty"
                )
    if not allow_unrecorded_revisions:
        for model in report["model_revisions"]:
            if model["revision"] is None:
                problems.append(
                    f"{model['case']}: historical model revision was not recorded"
                )
    return problems


def print_human(report: dict[str, Any]) -> None:
    print(f"Readiness scope: {report['readiness_scope']}")
    task_protocol = report["quality_task_protocol"]
    harness = task_protocol["harness"]
    print(f"Quality task protocol: {task_protocol['path']}")
    print(
        f"  harness: {harness['distribution']}=={harness['version']} "
        f"@ {harness['revision']}"
    )
    print(
        "  evaluator seeds: "
        + ", ".join(
            f"{name}={value}"
            for name, value in task_protocol["evaluator_seeds"].items()
        )
    )
    print("  dataset pins:")
    for task, pin in task_protocol["dataset_pins"].items():
        print(
            f"    {task}: {pin['dataset_path']} @ {pin['revision']} "
            f"({pin['num_fewshot']}-shot)"
        )
    for profile in task_protocol["environment_profile_versions"]:
        marker = "ok" if profile["matches_protocol"] else "mismatch"
        print(
            f"  [{marker}] {profile['profile']} declares lm-eval "
            f"{profile['declared_version'] or 'missing'}"
        )
    checkout = task_protocol["harness_checkout"]
    if checkout["checked"]:
        marker = "ok" if checkout["matches"] else "mismatch"
        print(
            f"  [{marker}] harness checkout: {checkout['path']} "
            f"@ {checkout['revision'] or 'unavailable'}"
        )
        if checkout["module_file"]:
            print(f"    imported: {checkout['module_file']}")
    for profile in report["profiles"]:
        print(f"[{profile['name']}] {', '.join(profile['cases'])}")
        print(f"  requirements: {profile['requirements']}")
        if report["check_installed"]:
            for package in profile["packages"]:
                marker = "ok" if package["matches"] else "mismatch"
                print(
                    f"  [{marker}] {package['distribution']}=={package['expected']} "
                    f"(installed: {package['installed'] or 'missing'})"
                )
            for module in profile["modules"]:
                marker = "ok" if module["available"] else "missing"
                print(f"  [{marker}] import {module['module']}")
    print("Model revisions:")
    for model in report["model_revisions"]:
        revision = model["revision"] or "UNRECORDED"
        print(
            f"  {model['case']}: {model['model']} @ {revision} "
            f"({model['revision_provenance']})"
        )
    print("Local builds:")
    for build in report["local_builds"]:
        print(f"  {build['install']}")
        print(
            f"    {build['module']} from {build['distribution']}=="
            f"{build['expected_version']} @ {build['expected_revision']}"
        )
        if report["check_installed"]:
            marker = "ok" if build["origin_matches_source"] else "mismatch"
            print(
                f"    [{marker}] module: {build['module_file'] or 'missing'}; "
                f"source: {build['direct_url_source'] or 'unbound'}"
            )
            print(
                "    native binaries: " f"{len(build['native_binary_sha256'])} hashed"
            )
    if report["check_installed"]:
        print(f"PyTorch: {report['torch'] or 'missing'}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=suite.DEFAULT_CONFIG)
    parser.add_argument("--environments", type=Path, default=DEFAULT_ENVIRONMENTS)
    parser.add_argument(
        "--models",
        default="paper",
        help="Comma-separated paper case keys or model IDs; default: paper.",
    )
    parser.add_argument(
        "--check-installed",
        action="store_true",
        help="Compare the active Python environment and local checkouts with the pins.",
    )
    parser.add_argument(
        "--allow-unrecorded-revisions",
        action="store_true",
        help=(
            "Do not fail solely because a historical model revision is unavailable. "
            "The resulting run remains a new-revision reproduction."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    try:
        open_weight_config = suite.load_config(args.config.resolve())
        selected = suite.select_cases(open_weight_config.cases, args.models)
        environments = load_environment_profiles(args.environments.resolve())
        task_protocol = inspect_quality_task_protocol(
            open_weight_config,
            environments,
            check_installed=args.check_installed,
        )
        report = inspect_profiles(
            selected,
            open_weight_config.cases,
            environments,
            check_installed=args.check_installed,
        )
        report["quality_task_protocol"] = task_protocol
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid downstream-evaluation environment config: {error}")
    problems = failures(
        report,
        allow_unrecorded_revisions=args.allow_unrecorded_revisions,
    )
    report["declared_inputs_ready"] = not problems
    report["historical_exact_replay_established"] = False
    report["problems"] = problems
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
        if problems:
            print("Readiness blockers:")
            for problem in problems:
                print(f"  - {problem}")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())

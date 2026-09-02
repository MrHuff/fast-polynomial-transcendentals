#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and enforce immutable dataset revisions for the paper's lm-eval tasks."""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


_COMMIT = re.compile(r"[0-9a-f]{40}")
_SEED_KEYS = (
    "random_seed",
    "numpy_random_seed",
    "torch_random_seed",
    "fewshot_random_seed",
)


@dataclass(frozen=True)
class DatasetPin:
    task: str
    dataset_path: str
    revision: str
    revision_provenance: str


@dataclass(frozen=True)
class LMEvalTaskProtocol:
    protocol_class: str
    selection_date: str
    harness_distribution: str
    harness_version: str
    harness_repository: str
    harness_revision: str
    harness_submodule_path: str
    evaluator_seeds: dict[str, int]
    tasks: dict[str, DatasetPin]
    historical_boundary: str

    def select(self, task_names: Sequence[str]) -> tuple[DatasetPin, ...]:
        unknown = set(task_names) - set(self.tasks)
        if unknown:
            raise ValueError(
                "task pin config does not cover: " + ", ".join(sorted(unknown))
            )
        return tuple(self.tasks[name] for name in task_names)


def _commit(value: Any, field: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 40-character commit")
    return value


def load_task_protocol(path: Path) -> LMEvalTaskProtocol:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("lm-eval task config schema_version must equal 1")
    harness = document.get("harness")
    if not isinstance(harness, dict):
        raise ValueError("harness must be an object")
    raw_seeds = document.get("evaluator_seeds")
    if not isinstance(raw_seeds, dict) or set(raw_seeds) != set(_SEED_KEYS):
        raise ValueError(
            "evaluator_seeds must contain exactly " + ", ".join(_SEED_KEYS)
        )
    seeds: dict[str, int] = {}
    for key in _SEED_KEYS:
        value = raw_seeds[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"evaluator_seeds.{key} must be a non-negative integer")
        seeds[key] = value

    raw_tasks = document.get("tasks")
    if not isinstance(raw_tasks, dict) or not raw_tasks:
        raise ValueError("tasks must be a non-empty object")
    tasks: dict[str, DatasetPin] = {}
    revisions_by_path: dict[str, str] = {}
    for task, raw_pin in raw_tasks.items():
        if not isinstance(task, str) or not task or not isinstance(raw_pin, dict):
            raise ValueError("every task pin must be a named object")
        dataset_path = raw_pin.get("dataset_path")
        if not isinstance(dataset_path, str) or not dataset_path:
            raise ValueError(f"tasks.{task}.dataset_path must be a non-empty string")
        revision = _commit(
            raw_pin.get("dataset_revision"), f"tasks.{task}.dataset_revision"
        )
        previous = revisions_by_path.setdefault(dataset_path, revision)
        if previous != revision:
            raise ValueError(
                f"dataset {dataset_path!r} has conflicting revisions in task config"
            )
        provenance = raw_pin.get("dataset_revision_provenance")
        if provenance != "public-protocol-selection":
            raise ValueError(
                f"tasks.{task}.dataset_revision_provenance must be "
                "'public-protocol-selection'"
            )
        tasks[task] = DatasetPin(
            task=task,
            dataset_path=dataset_path,
            revision=revision,
            revision_provenance=provenance,
        )

    for field in (
        "protocol_class",
        "selection_date",
        "historical_boundary",
    ):
        if not isinstance(document.get(field), str) or not document[field]:
            raise ValueError(f"{field} must be a non-empty string")
    for field in ("distribution", "version", "repository"):
        if not isinstance(harness.get(field), str) or not harness[field]:
            raise ValueError(f"harness.{field} must be a non-empty string")

    return LMEvalTaskProtocol(
        protocol_class=document["protocol_class"],
        selection_date=document["selection_date"],
        harness_distribution=harness["distribution"],
        harness_version=harness["version"],
        harness_repository=harness["repository"],
        harness_revision=_commit(harness.get("revision"), "harness.revision"),
        harness_submodule_path=_relative_path(
            harness.get("submodule_path"), "harness.submodule_path"
        ),
        evaluator_seeds=seeds,
        tasks=tasks,
        historical_boundary=document["historical_boundary"],
    )


@dataclass(frozen=True)
class HarnessCheckout:
    relative_path: str
    revision: str
    clean: bool
    module_file: str | None = None


def _relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must stay inside the repository")
    return path.as_posix()


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise RuntimeError("cannot inspect the pinned lm-eval checkout with Git")
    return completed.stdout.strip()


def inspect_harness_checkout(
    protocol: LMEvalTaskProtocol,
    repository_root: Path,
) -> HarnessCheckout:
    """Require the declared lm-eval submodule revision and a clean checkout."""

    root = repository_root.resolve()
    checkout = (root / protocol.harness_submodule_path).resolve()
    if not checkout.is_relative_to(root) or not checkout.is_dir():
        raise RuntimeError(
            "pinned lm-eval checkout is missing; initialize Git submodules"
        )
    revision = _git(checkout, "rev-parse", "HEAD")
    if revision != protocol.harness_revision:
        raise RuntimeError(
            f"lm-eval checkout is {revision}, expected {protocol.harness_revision}"
        )
    status = _git(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise RuntimeError(
            "lm-eval checkout has local modifications or untracked files"
        )
    return HarnessCheckout(
        relative_path=protocol.harness_submodule_path,
        revision=revision,
        clean=True,
    )


def activate_harness_checkout(
    protocol: LMEvalTaskProtocol,
    repository_root: Path,
) -> HarnessCheckout:
    """Import lm-eval from the verified checkout, never from another install."""

    state = inspect_harness_checkout(protocol, repository_root)
    checkout = (repository_root.resolve() / state.relative_path).resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "lm_eval" and not name.startswith("lm_eval."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None or not Path(module_file).resolve().is_relative_to(
            checkout
        ):
            raise RuntimeError(
                f"{name} was imported outside the pinned lm-eval checkout"
            )
    checkout_string = str(checkout)
    sys.path[:] = [entry for entry in sys.path if entry != checkout_string]
    sys.path.insert(0, checkout_string)
    importlib.invalidate_caches()
    module = importlib.import_module("lm_eval")
    module_file = getattr(module, "__file__", None)
    if module_file is None or not Path(module_file).resolve().is_relative_to(checkout):
        raise RuntimeError("lm_eval did not import from the pinned checkout")
    # Recheck after import so the recorded state is observed, not copied from config.
    observed = inspect_harness_checkout(protocol, repository_root)
    return HarnessCheckout(
        relative_path=observed.relative_path,
        revision=observed.revision,
        clean=observed.clean,
        module_file=str(Path(module_file).resolve().relative_to(checkout)),
    )


def require_harness_version(
    protocol: LMEvalTaskProtocol, installed_version: str | None
) -> None:
    if installed_version != protocol.harness_version:
        raise RuntimeError(
            f"{protocol.harness_distribution} {protocol.harness_version} is required; "
            f"installed version is {installed_version or 'missing'}"
        )


@contextmanager
def enforce_dataset_revisions(
    dataset_module: Any,
    pins: Sequence[DatasetPin],
) -> Iterator[dict[str, set[str | None]]]:
    """Temporarily constrain ``datasets.load_dataset`` to selected revisions."""

    by_path = {pin.dataset_path: pin for pin in pins}
    original = dataset_module.load_dataset
    observations: dict[str, set[str | None]] = {}

    def load_dataset(*args: Any, **kwargs: Any) -> Any:
        dataset_path = kwargs.get("path")
        if dataset_path is None and args:
            dataset_path = args[0]
        pin = by_path.get(dataset_path)
        if pin is None:
            raise RuntimeError(
                f"lm-eval attempted unpinned dataset {dataset_path!r}; "
                "add it to the public task protocol before running"
            )
        requested_revision = kwargs.get("revision")
        if requested_revision is not None and requested_revision != pin.revision:
            raise RuntimeError(
                f"dataset {dataset_path!r} requested revision {requested_revision!r}, "
                f"but the public protocol pins {pin.revision}"
            )
        kwargs["revision"] = pin.revision
        dataset_name = kwargs.get("name")
        if dataset_name is None and len(args) > 1:
            dataset_name = args[1]
        observations.setdefault(pin.dataset_path, set()).add(dataset_name)
        return original(*args, **kwargs)

    dataset_module.load_dataset = load_dataset
    try:
        yield observations
    finally:
        dataset_module.load_dataset = original


def require_all_datasets_observed(
    pins: Sequence[DatasetPin], observations: dict[str, set[str | None]]
) -> None:
    missing = {pin.dataset_path for pin in pins} - set(observations)
    if missing:
        raise RuntimeError(
            "lm-eval did not load the pinned dataset(s): " + ", ".join(sorted(missing))
        )


def require_sample_coverage(evaluation: dict[str, Any]) -> None:
    """Require full, count-bound samples for every expanded leaf task."""

    leaf_configs = evaluation.get("configs")
    if not isinstance(leaf_configs, dict) or not leaf_configs:
        raise RuntimeError(
            "lm-eval sample coverage cannot be checked without leaf task configs"
        )
    samples = evaluation.get("samples")
    if not isinstance(samples, dict):
        raise RuntimeError(
            "lm-eval sample logging was requested but no samples were returned"
        )
    missing = set(leaf_configs) - set(samples)
    if missing:
        raise RuntimeError(
            "lm-eval did not return samples for expanded leaf task(s): "
            + ", ".join(sorted(missing))
        )
    empty_or_invalid = {
        task
        for task in leaf_configs
        if not isinstance(samples.get(task), list) or not samples[task]
    }
    if empty_or_invalid:
        raise RuntimeError(
            "lm-eval returned empty or invalid samples for expanded leaf task(s): "
            + ", ".join(sorted(empty_or_invalid))
        )
    sample_counts = evaluation.get("n-samples")
    if not isinstance(sample_counts, dict):
        raise RuntimeError("lm-eval result did not contain n-samples counts")
    missing_counts = set(leaf_configs) - set(sample_counts)
    if missing_counts:
        raise RuntimeError(
            "lm-eval did not return n-samples counts for expanded leaf task(s): "
            + ", ".join(sorted(missing_counts))
        )
    for task in leaf_configs:
        counts = sample_counts.get(task)
        if not isinstance(counts, dict):
            raise RuntimeError(f"lm-eval returned invalid n-samples counts for {task}")
        original = counts.get("original")
        effective = counts.get("effective")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (original, effective)
        ):
            raise RuntimeError(
                f"lm-eval returned non-positive or non-integral n-samples counts "
                f"for {task}"
            )
        if effective != original:
            raise RuntimeError(
                f"lm-eval evaluated only {effective} of {original} samples for {task}"
            )
        if len(samples[task]) != effective:
            raise RuntimeError(
                f"lm-eval retained {len(samples[task])} samples for {task}, "
                f"but n-samples.effective is {effective}"
            )

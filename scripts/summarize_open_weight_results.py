#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from report/generate_wandb_eval_results.py. This version consumes
# local, source-bound result envelopes and has no W&B or service dependency.

"""Validate and summarize public open-weight rerun result JSON files.

The benchmark suite writes quality and throughput measurements independently.
This tool joins them by immutable model revision and variant, checks the
declared public lm-eval protocol, and emits a comprehensive CSV plus a compact
Markdown table.  It never treats retained historical W&B summaries as inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/open_weight_paper.json"
DEFAULT_TASK_CONFIG = REPOSITORY_ROOT / "configs/lm_eval_paper_tasks.json"
DEFAULT_ENVIRONMENTS = REPOSITORY_ROOT / "configs/eval_environments/profiles.json"
DEFAULT_EXPERIMENT_MANIFEST = REPOSITORY_ROOT / "repro/experiments.json"
_EXACT_PIN = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;]+)$")
_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
EXPECTED_TASK_PROTOCOL_CLASS = "new-public-rerun"
EXPECTED_LM_EVAL_DISTRIBUTION = "lm-eval"
EXPECTED_LM_EVAL_VERSION = "0.4.12"
DOWNSTREAM_EXTERNAL_SOURCES = frozenset(("flash-attention", "lm-evaluation-harness"))
EVALUATOR_SEED_KEYS = frozenset(
    ("random_seed", "numpy_random_seed", "torch_random_seed", "fewshot_random_seed")
)

# These are the exact lm-eval v0.4.12 result keys used by the paper protocol.
# The seven bounded metrics enter the mean/max percentage-point summaries;
# word perplexity is reported separately and is never relabelled as accuracy.
TASK_METRICS: dict[str, str] = {
    "mmlu": "acc,none",
    "hellaswag": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "winogrande": "acc,none",
    "piqa": "acc_norm,none",
    "gsm8k": "exact_match,flexible-extract",
    "truthfulqa_mc2": "acc,none",
    "wikitext": "word_perplexity,none",
}
QUALITY_TASKS = tuple(task for task in TASK_METRICS if task != "wikitext")


@dataclass(frozen=True)
class CaseSpec:
    key: str
    paper_name: str
    model_id: str
    revision: str
    revision_provenance: str
    replacement: str
    variants: tuple[str, ...]
    attn_implementation: str
    experts_implementation: str | None
    trust_remote_code: bool
    eval_batch_size: str
    throughput_sequence_length: int
    quality_sequence_length_argument: int


@dataclass(frozen=True)
class MeasurementProtocol:
    batch_size: int
    device: str
    seed: int
    throughput_mode: str
    quality_mode: str
    prefill_measurements: int
    prefill_warmups: int
    decode_steps: int
    decode_measurements: int
    decode_warmups: int
    dtype: str
    quality_task_config: str
    quality_log_samples: bool
    quality_eval_limit: None
    quality_default_num_fewshot: int
    quality_tasks: tuple[str, ...]
    mxfp4_dequantize: bool
    freeze_granite_moe_routing: bool


@dataclass(frozen=True)
class TaskExpectations:
    protocol_class: str
    selection_date: str
    harness_distribution: str
    harness_version: str
    harness_revision: str
    harness_submodule_path: str
    task_pins: dict[str, dict[str, str]]
    evaluator_seeds: dict[str, int]


@dataclass(frozen=True)
class LocalModuleExpectation:
    name: str
    module: str
    distribution: str
    version: str
    source_path: str
    revision: str | None
    revision_source: str | None
    require_native_binary: bool


@dataclass
class VariantRecord:
    case: CaseSpec
    variant: str
    source_revision: str | None = None
    source_dirty: bool | None = None
    environment: dict[str, Any] | None = None
    quality: dict[str, float] | None = None
    prefill_tokens_per_s: float | None = None
    prefill_repetition_ms: list[float] | None = None
    decode_tokens_per_s: float | None = None
    decode_repetition_ms: list[float] | None = None
    patched_silu_modules: int | None = None
    patched_router_sigmoid_modules: int | None = None
    patched_gemma_softcap: bool | None = None
    input_paths: list[Path] = field(default_factory=list)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_samples(value: Any, label: str, expected_count: int) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    samples = [
        _positive_number(sample, f"{label}[{index}]")
        for index, sample in enumerate(value)
    ]
    if len(samples) != expected_count:
        raise ValueError(
            f"{label} contains {len(samples)} samples, expected {expected_count}"
        )
    return samples


def _full_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or _FULL_COMMIT.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 40-character commit")
    return value


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def _sha256_file(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.resolve().open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    return digest.hexdigest()


def _file_receipt(path: Path, label: str) -> tuple[str, str]:
    return _path_label(path), _sha256_file(path, label)


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{label} must stay inside the repository")
    return path.as_posix()


def load_cases(path: Path) -> tuple[CaseSpec, ...]:
    document = _object(json.loads(path.read_text(encoding="utf-8")), "config")
    if document.get("schema_version") != 1:
        raise ValueError("open-weight config schema_version must equal 1")
    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("open-weight config cases must be a non-empty array")
    cases: list[CaseSpec] = []
    for index, raw in enumerate(raw_cases):
        item = _object(raw, f"cases[{index}]")
        revision = _full_commit(item.get("revision"), f"cases[{index}].revision")
        replacement = item.get("replacement")
        if not isinstance(replacement, str) or not replacement:
            raise ValueError(f"cases[{index}].replacement must be declared")
        variants = item.get("variants")
        if not isinstance(variants, list) or not all(
            isinstance(value, str) and value for value in variants
        ):
            raise ValueError(f"cases[{index}].variants must be a string array")
        attn_implementation = item.get("attn_implementation")
        if not isinstance(attn_implementation, str) or not attn_implementation:
            raise ValueError(f"cases[{index}].attn_implementation must be declared")
        experts_implementation = item.get("experts_implementation")
        if experts_implementation is not None and not isinstance(
            experts_implementation, str
        ):
            raise ValueError(f"cases[{index}].experts_implementation is invalid")
        trust_remote_code = item.get("trust_remote_code")
        if not isinstance(trust_remote_code, bool):
            raise ValueError(f"cases[{index}].trust_remote_code must be boolean")
        eval_batch_size = item.get("eval_batch_size")
        if not isinstance(eval_batch_size, str) or not eval_batch_size:
            raise ValueError(f"cases[{index}].eval_batch_size must be declared")
        cases.append(
            CaseSpec(
                key=str(item["key"]),
                paper_name=str(item["paper_name"]),
                model_id=str(item["model_id"]),
                revision=revision,
                revision_provenance=str(item["revision_provenance"]),
                replacement=replacement,
                variants=tuple(variants),
                attn_implementation=attn_implementation,
                experts_implementation=experts_implementation,
                trust_remote_code=trust_remote_code,
                eval_batch_size=eval_batch_size,
                throughput_sequence_length=_positive_int(
                    item.get("throughput_sequence_length"),
                    f"cases[{index}].throughput_sequence_length",
                ),
                quality_sequence_length_argument=_positive_int(
                    item.get("unused_quality_sequence_length_argument"),
                    f"cases[{index}].unused_quality_sequence_length_argument",
                ),
            )
        )
    return tuple(cases)


def load_measurement_protocol(path: Path) -> MeasurementProtocol:
    document = _object(json.loads(path.read_text(encoding="utf-8")), "config")
    if document.get("schema_version") != 1:
        raise ValueError("open-weight config schema_version must equal 1")
    raw = _object(document.get("protocol"), "config protocol")
    device = raw.get("device")
    if device != "cuda":
        raise ValueError("protocol.device must equal 'cuda'")
    dtype = raw.get("dtype")
    if dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError("protocol.dtype is invalid")
    throughput_mode = raw.get("throughput_mode")
    if throughput_mode != "both":
        raise ValueError("protocol.throughput_mode must equal 'both'")
    quality_mode = raw.get("quality_mode")
    if quality_mode != "eval":
        raise ValueError("protocol.quality_mode must equal 'eval'")
    quality_task_config = _relative_path(
        raw.get("quality_task_config"), "protocol.quality_task_config"
    )
    quality_log_samples = raw.get("quality_log_samples")
    if not isinstance(quality_log_samples, bool):
        raise ValueError("protocol.quality_log_samples must be boolean")
    if raw.get("quality_eval_limit", object()) is not None:
        raise ValueError("protocol.quality_eval_limit must be null")
    raw_tasks = raw.get("quality_tasks")
    if not isinstance(raw_tasks, list) or not all(
        isinstance(task, str) and task for task in raw_tasks
    ):
        raise ValueError("protocol.quality_tasks must be a string array")
    if tuple(raw_tasks) != tuple(TASK_METRICS):
        raise ValueError("protocol.quality_tasks must preserve the paper task order")
    mxfp4_dequantize = raw.get("mxfp4_dequantize")
    freeze_routing = raw.get("freeze_granite_moe_routing")
    if not isinstance(mxfp4_dequantize, bool):
        raise ValueError("protocol.mxfp4_dequantize must be boolean")
    if not isinstance(freeze_routing, bool):
        raise ValueError("protocol.freeze_granite_moe_routing must be boolean")
    return MeasurementProtocol(
        batch_size=_positive_int(raw.get("batch_size"), "protocol.batch_size"),
        device=device,
        seed=_non_negative_int(raw.get("seed"), "protocol.seed"),
        throughput_mode=throughput_mode,
        quality_mode=quality_mode,
        prefill_measurements=_positive_int(
            raw.get("prefill_measurements"), "protocol.prefill_measurements"
        ),
        prefill_warmups=_non_negative_int(
            raw.get("prefill_warmups"), "protocol.prefill_warmups"
        ),
        decode_steps=_positive_int(raw.get("decode_steps"), "protocol.decode_steps"),
        decode_measurements=_positive_int(
            raw.get("decode_measurements"), "protocol.decode_measurements"
        ),
        decode_warmups=_non_negative_int(
            raw.get("decode_warmups"), "protocol.decode_warmups"
        ),
        dtype=dtype,
        quality_task_config=quality_task_config,
        quality_log_samples=quality_log_samples,
        quality_eval_limit=None,
        quality_default_num_fewshot=_non_negative_int(
            raw.get("quality_default_num_fewshot"),
            "protocol.quality_default_num_fewshot",
        ),
        quality_tasks=tuple(raw_tasks),
        mxfp4_dequantize=mxfp4_dequantize,
        freeze_granite_moe_routing=freeze_routing,
    )


def load_task_expectations(path: Path) -> TaskExpectations:
    document = _object(json.loads(path.read_text(encoding="utf-8")), "task config")
    if document.get("schema_version") != 1:
        raise ValueError("lm-eval task config schema_version must equal 1")
    protocol_class = document.get("protocol_class")
    if protocol_class != EXPECTED_TASK_PROTOCOL_CLASS:
        raise ValueError(
            "task config protocol_class must equal " f"{EXPECTED_TASK_PROTOCOL_CLASS!r}"
        )
    selection_date = document.get("selection_date")
    if not isinstance(selection_date, str) or not selection_date:
        raise ValueError("task config selection_date must be a non-empty string")
    harness = _object(document.get("harness"), "task config harness")
    distribution = harness.get("distribution")
    if distribution != EXPECTED_LM_EVAL_DISTRIBUTION:
        raise ValueError(
            "task config harness distribution must equal "
            f"{EXPECTED_LM_EVAL_DISTRIBUTION!r}"
        )
    version = harness.get("version")
    if version != EXPECTED_LM_EVAL_VERSION:
        raise ValueError(
            "task config harness version must equal " f"{EXPECTED_LM_EVAL_VERSION!r}"
        )
    revision = _full_commit(harness.get("revision"), "task config harness revision")
    submodule_path = _relative_path(
        harness.get("submodule_path"), "task config harness submodule_path"
    )
    raw_tasks = _object(document.get("tasks"), "task config tasks")
    if set(raw_tasks) != set(TASK_METRICS):
        raise ValueError(
            "task config tasks must contain exactly " + ", ".join(sorted(TASK_METRICS))
        )
    task_pins: dict[str, dict[str, str]] = {}
    for task in TASK_METRICS:
        raw_pin = _object(raw_tasks.get(task), f"task config {task}")
        dataset_path = raw_pin.get("dataset_path")
        if not isinstance(dataset_path, str) or not dataset_path:
            raise ValueError(f"task config {task} dataset_path must be non-empty")
        dataset_revision = _full_commit(
            raw_pin.get("dataset_revision"),
            f"task config {task} dataset_revision",
        )
        revision_provenance = raw_pin.get("dataset_revision_provenance")
        if revision_provenance != "public-protocol-selection":
            raise ValueError(
                f"task config {task} dataset_revision_provenance must equal "
                "'public-protocol-selection'"
            )
        task_pins[task] = {
            "dataset_path": dataset_path,
            "revision": dataset_revision,
            "revision_provenance": revision_provenance,
        }
    seeds = _object(document.get("evaluator_seeds"), "task config evaluator_seeds")
    if set(seeds) != EVALUATOR_SEED_KEYS:
        raise ValueError(
            "task config evaluator_seeds must contain exactly "
            + ", ".join(sorted(EVALUATOR_SEED_KEYS))
        )
    evaluator_seeds: dict[str, int] = {}
    for key, value in seeds.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"task config evaluator_seeds.{key} must be a non-negative integer"
            )
        evaluator_seeds[key] = value
    return TaskExpectations(
        protocol_class=protocol_class,
        selection_date=selection_date,
        harness_distribution=distribution,
        harness_version=version,
        harness_revision=revision,
        harness_submodule_path=submodule_path,
        task_pins=task_pins,
        evaluator_seeds=evaluator_seeds,
    )


def load_fewshot_expectations(path: Path) -> dict[str, int]:
    document = _object(json.loads(path.read_text(encoding="utf-8")), "config")
    if document.get("schema_version") != 1:
        raise ValueError("open-weight config schema_version must equal 1")
    protocol = _object(document.get("protocol"), "config protocol")
    raw = _object(protocol.get("quality_fewshot"), "quality_fewshot")
    if set(raw) != set(TASK_METRICS):
        raise ValueError(
            "quality_fewshot must contain exactly " + ", ".join(sorted(TASK_METRICS))
        )
    fewshot: dict[str, int] = {}
    for task, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"quality_fewshot.{task} must be a non-negative integer")
        fewshot[task] = value
    return fewshot


def _canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = _EXACT_PIN.fullmatch(line)
        if match is None:
            raise ValueError(f"{path}:{line_number} is not an exact package pin")
        name = _canonical_distribution(match.group("name"))
        if name in pins:
            raise ValueError(f"{path}:{line_number} repeats package {name}")
        pins[name] = match.group("version")
    return pins


def load_environment_expectations(
    path: Path, cases: tuple[CaseSpec, ...]
) -> dict[str, dict[str, str]]:
    document = _object(json.loads(path.read_text(encoding="utf-8")), "environments")
    if document.get("schema_version") != 1:
        raise ValueError("environment profile schema_version must equal 1")
    profiles = _object(document.get("profiles"), "environment profiles")
    expected: dict[str, dict[str, str]] = {}
    case_keys = {case.key for case in cases}
    for profile_name, raw_profile in profiles.items():
        profile = _object(raw_profile, f"environment profile {profile_name}")
        requirements = Path(str(profile["requirements"]))
        if requirements.is_absolute() or ".." in requirements.parts:
            raise ValueError(
                f"environment profile {profile_name} escapes the repository"
            )
        pins = _requirement_pins(REPOSITORY_ROOT / requirements)
        raw_cases = profile.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError(f"environment profile {profile_name} has no case list")
        for case_key in raw_cases:
            if case_key not in case_keys:
                raise ValueError(
                    f"environment profile {profile_name} names unknown case {case_key!r}"
                )
            if case_key in expected:
                raise ValueError(f"case {case_key} appears in multiple environments")
            expected[str(case_key)] = pins
    missing = case_keys - set(expected)
    if missing:
        raise ValueError("cases lack environments: " + ", ".join(sorted(missing)))
    return expected


def load_local_module_expectations(
    path: Path,
    cases: tuple[CaseSpec, ...],
) -> dict[str, tuple[LocalModuleExpectation, ...]]:
    document = _object(json.loads(path.read_text(encoding="utf-8")), "environments")
    profiles = _object(document.get("profiles"), "environment profiles")
    raw_builds = _object(document.get("local_builds"), "local_builds")
    builds: dict[str, LocalModuleExpectation] = {}
    used_by: dict[str, set[str] | None] = {}
    for name, raw_build in raw_builds.items():
        build = _object(raw_build, f"local_builds.{name}")
        module = build.get("module")
        distribution = build.get("distribution")
        version = build.get("version")
        if not all(
            isinstance(value, str) and value
            for value in (module, distribution, version)
        ):
            raise ValueError(f"local_builds.{name} lacks module package metadata")
        source_path = _relative_path(build.get("path"), f"local_builds.{name}.path")
        revision_source = build.get("revision_source")
        revision: str | None = None
        if revision_source == "repository":
            pass
        elif revision_source is None:
            revision = _full_commit(
                build.get("revision"), f"local_builds.{name}.revision"
            )
        else:
            raise ValueError(f"local_builds.{name}.revision_source is invalid")
        raw_used_by = build.get("used_by")
        if raw_used_by is None:
            used_by[str(name)] = None
        elif not isinstance(raw_used_by, list) or not all(
            isinstance(case, str) and case for case in raw_used_by
        ):
            raise ValueError(f"local_builds.{name}.used_by must be a case array")
        else:
            used_by[str(name)] = set(raw_used_by)
        require_native_binary = build.get("require_native_binary", False)
        if not isinstance(require_native_binary, bool):
            raise ValueError(
                f"local_builds.{name}.require_native_binary must be boolean"
            )
        builds[str(name)] = LocalModuleExpectation(
            name=str(name),
            module=module,
            distribution=distribution,
            version=version,
            source_path=source_path,
            revision=revision,
            revision_source=revision_source,
            require_native_binary=require_native_binary,
        )

    profile_by_case: dict[str, dict[str, Any]] = {}
    for profile_name, raw_profile in profiles.items():
        profile = _object(raw_profile, f"environment profile {profile_name}")
        raw_cases = profile.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError(f"environment profile {profile_name} has no case list")
        for case_key in raw_cases:
            profile_by_case[str(case_key)] = profile

    expected: dict[str, tuple[LocalModuleExpectation, ...]] = {}
    for case in cases:
        profile = profile_by_case.get(case.key)
        if profile is None:
            raise ValueError(f"case {case.key} lacks an environment profile")
        names = set(profile.get("required_modules", []))
        per_case = profile.get("case_required_modules", {})
        if not isinstance(per_case, dict):
            raise ValueError("case_required_modules must be an object")
        names.update(per_case.get(case.key, []))
        local_names = sorted(names & builds.keys())
        for name in local_names:
            selected = used_by[name]
            if selected is not None and case.key not in selected:
                raise ValueError(
                    f"local_builds.{name} is required by {case.key} but used_by excludes it"
                )
        expected[case.key] = tuple(builds[name] for name in local_names)
    return expected


def load_external_source_pins(path: Path) -> dict[str, Any]:
    document = _object(json.loads(path.read_text(encoding="utf-8")), "manifest")
    if document.get("schema_version") != 1:
        raise ValueError("experiment manifest schema_version must equal 1")
    raw = _object(document.get("external_source_pins"), "external_source_pins")
    return {
        str(name): revision
        for name, revision in raw.items()
        if name in DOWNSTREAM_EXTERNAL_SOURCES
    }


def _required_external_sources(
    case: CaseSpec,
    module_expectations: tuple[LocalModuleExpectation, ...],
    *,
    quality: bool,
) -> frozenset[str]:
    required: set[str] = set()
    if quality:
        required.add("lm-evaluation-harness")
    if case.attn_implementation.startswith("flash_attention_") or any(
        expectation.source_path == "flash-attention"
        for expectation in module_expectations
    ):
        required.add("flash-attention")
    return frozenset(required)


def expand_inputs(paths: Sequence[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved.is_dir():
            expanded.extend(sorted(resolved.glob("*.json")))
        elif resolved.is_file():
            expanded.append(resolved)
        else:
            raise FileNotFoundError(path)
    unique = sorted(set(expanded))
    if not unique:
        raise ValueError("no result JSON files were found")
    return unique


def variant_fit(variant: str) -> str:
    if variant == "native":
        return "native"
    parts = variant.split("+")
    if all(part.endswith("_current") for part in parts):
        return "current"
    if all(part.endswith("_sollya") for part in parts):
        return "sollya"
    raise ValueError(f"variant {variant!r} does not identify one fit family")


def variant_patch_requirements(variant: str) -> frozenset[str]:
    if variant == "native":
        return frozenset()
    required: set[str] = set()
    for part in variant.split("+"):
        if part.startswith(("spline_silu_", "spline_gelu_", "fused_swiglu_")):
            required.add("activation")
        elif part.startswith("spline_router_sigmoid_"):
            required.add("router")
        elif part.startswith(("gemma_tanh_", "gemma_fa4_tanh_")):
            required.add("softcap")
        else:
            raise ValueError(f"variant {variant!r} has an unknown intervention")
    if not required:
        raise ValueError(f"variant {variant!r} does not request an intervention")
    return frozenset(required)


def _merge_equal(
    record: VariantRecord, field_name: str, value: Any, path: Path
) -> None:
    if value is None:
        return
    previous = getattr(record, field_name)
    if previous is not None and previous != value:
        raise ValueError(
            f"conflicting {field_name} for {record.case.key}/{record.variant} "
            f"across result files (including {path})"
        )
    setattr(record, field_name, value)


def _check_quality_protocol(
    evaluation: dict[str, Any],
    *,
    expectations: TaskExpectations,
    fewshot: dict[str, int],
    label: str,
) -> None:
    public = _object(evaluation.get("public_task_protocol"), f"{label} protocol")
    if public.get("class") != expectations.protocol_class:
        raise ValueError(f"{label} used a different task protocol class")
    if public.get("selection_date") != expectations.selection_date:
        raise ValueError(f"{label} used a different task protocol selection date")
    if public.get("lm_eval_version") != expectations.harness_version:
        raise ValueError(f"{label} used a different lm-eval version")
    if public.get("lm_eval_source_revision") != expectations.harness_revision:
        raise ValueError(f"{label} used a different lm-eval source revision")
    if public.get("lm_eval_source_path") != expectations.harness_submodule_path:
        raise ValueError(f"{label} used a different lm-eval source path")
    if public.get("lm_eval_source_clean") is not True:
        raise ValueError(f"{label} did not use a clean lm-eval checkout")
    module_file = public.get("lm_eval_module_file")
    try:
        module_path = _relative_path(module_file, f"{label} lm_eval_module_file")
    except ValueError as error:
        raise ValueError(
            f"{label} did not record a safe lm-eval module path"
        ) from error
    if Path(module_path).parts[0] != "lm_eval":
        raise ValueError(f"{label} imported a module outside the lm_eval package")
    if public.get("log_samples") is not True:
        raise ValueError(f"{label} did not retain per-example samples")
    if public.get("dataset_pins") != expectations.task_pins:
        raise ValueError(f"{label} dataset pins differ from the task protocol")
    if public.get("evaluator_seeds") != expectations.evaluator_seeds:
        raise ValueError(f"{label} evaluator seeds differ from the task protocol")
    if evaluation.get("task_num_fewshot") != fewshot:
        raise ValueError(f"{label} few-shot counts differ from the paper protocol")
    configs = _object(evaluation.get("configs"), f"{label} leaf configs")
    samples = _object(evaluation.get("samples"), f"{label} samples")
    missing_samples = {
        task
        for task in configs
        if not isinstance(samples.get(task), list) or not samples[task]
    }
    if missing_samples:
        raise ValueError(
            f"{label} lacks nonempty samples for {', '.join(sorted(missing_samples))}"
        )
    sample_counts = _object(evaluation.get("n-samples"), f"{label} n-samples")
    missing_counts = set(configs) - set(sample_counts)
    if missing_counts:
        raise ValueError(
            f"{label} lacks n-samples counts for " + ", ".join(sorted(missing_counts))
        )
    for task in configs:
        counts = _object(sample_counts.get(task), f"{label} n-samples {task}")
        original = _positive_int(
            counts.get("original"), f"{label} {task} n-samples.original"
        )
        effective = _positive_int(
            counts.get("effective"), f"{label} {task} n-samples.effective"
        )
        if effective != original:
            raise ValueError(
                f"{label} {task} n-samples.effective differs from n-samples.original"
            )
        if len(samples[task]) != effective:
            raise ValueError(
                f"{label} retained {len(samples[task])} samples for {task}, "
                f"expected {effective}"
            )


def _check_recorded_command(
    experiment: dict[str, Any], label: str
) -> tuple[list[str], dict[str, list[str | None]]]:
    command = experiment.get("command")
    if (
        not isinstance(command, list)
        or len(command) < 2
        or not all(isinstance(token, str) and token for token in command)
    ):
        raise ValueError(f"{label} lacks the complete benchmark command")
    if Path(command[1]).as_posix() != "scripts/benchmark_open_weights.py":
        raise ValueError(f"{label} records a different benchmark entry point")
    options: dict[str, list[str | None]] = {}
    index = 2
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            raise ValueError(f"{label} benchmark command contains a stray argument")
        option, separator, inline_value = token.partition("=")
        if separator:
            value: str | None = inline_value
        elif index + 1 < len(command) and not command[index + 1].startswith("--"):
            index += 1
            value = command[index]
        else:
            value = None
        options.setdefault(option, []).append(value)
        index += 1
    return command, options


def _expect_command_values(
    options: dict[str, list[str | None]],
    option: str,
    expected: Sequence[str | None],
    label: str,
) -> None:
    if options.get(option, []) != list(expected):
        raise ValueError(f"{label} benchmark command has inconsistent {option}")


def _check_command_protocol(
    options: dict[str, list[str | None]],
    *,
    case: CaseSpec,
    protocol: MeasurementProtocol,
    fewshot: dict[str, int],
    config_receipts: dict[str, tuple[str | None, str | None]],
    quality: bool,
    label: str,
) -> None:
    sequence_length = (
        case.quality_sequence_length_argument
        if quality
        else case.throughput_sequence_length
    )
    expected_values: dict[str, str] = {
        "--model": case.model_id,
        "--attn-implementation": case.attn_implementation,
        "--mode": protocol.quality_mode if quality else protocol.throughput_mode,
        "--batch-size": str(protocol.batch_size),
        "--seq-len": str(sequence_length),
        "--steps": str(protocol.prefill_measurements),
        "--warmup": str(protocol.prefill_warmups),
        "--decode-steps": str(protocol.decode_steps),
        "--decode-repeats": str(protocol.decode_measurements),
        "--decode-warmup": str(protocol.decode_warmups),
        "--dtype": protocol.dtype,
        "--device": protocol.device,
        "--environment-profiles": str(config_receipts["environment_profiles"][0]),
        "--suite-config": str(config_receipts["suite_config"][0]),
        "--seed": str(protocol.seed),
        "--suite-environment-preflight": "passed",
        "--revision": case.revision,
        "--revision-provenance": case.revision_provenance,
    }
    for option, expected in expected_values.items():
        _expect_command_values(options, option, (expected,), label)
    if len(options.get("--json-out", [])) != 1 or options["--json-out"][0] is None:
        raise ValueError(f"{label} benchmark command lacks --json-out")
    _expect_command_values(options, "--variant", case.variants, label)

    optional_values = {
        "--experts-implementation": case.experts_implementation,
    }
    for option, expected in optional_values.items():
        _expect_command_values(
            options,
            option,
            () if expected is None else (expected,),
            label,
        )
    optional_flags = {
        "--trust-remote-code": case.trust_remote_code,
        "--mxfp4-dequantize": protocol.mxfp4_dequantize,
        "--freeze-granite-moe-routing": protocol.freeze_granite_moe_routing,
    }
    for option, enabled in optional_flags.items():
        _expect_command_values(options, option, (None,) if enabled else (), label)

    if "--eval-limit" in options:
        raise ValueError(f"{label} benchmark command contains --eval-limit")
    if quality:
        quality_values = {
            "--eval-tasks": ",".join(protocol.quality_tasks),
            "--eval-num-fewshot": str(protocol.quality_default_num_fewshot),
            "--eval-batch-size": case.eval_batch_size,
            "--eval-task-config": str(config_receipts["eval_task_config"][0]),
        }
        for option, expected in quality_values.items():
            _expect_command_values(options, option, (expected,), label)
        _expect_command_values(
            options,
            "--eval-task-fewshot",
            tuple(f"{task}={fewshot[task]}" for task in protocol.quality_tasks),
            label,
        )
        _expect_command_values(options, "--eval-log-samples", (None,), label)
    else:
        for option in (
            "--eval-tasks",
            "--eval-num-fewshot",
            "--eval-task-fewshot",
            "--eval-batch-size",
            "--eval-task-config",
            "--eval-log-samples",
        ):
            _expect_command_values(options, option, (), label)


def _check_measurement_protocol(
    measurement: dict[str, Any],
    *,
    case: CaseSpec,
    protocol: MeasurementProtocol,
    fewshot: dict[str, int],
    config_receipts: dict[str, tuple[str | None, str | None]],
    quality: bool,
    label: str,
) -> None:
    expected_common: dict[str, Any] = {
        "dtype": protocol.dtype,
        "device": protocol.device,
        "batch_size": protocol.batch_size,
        "prefill_measurements": protocol.prefill_measurements,
        "prefill_warmups": protocol.prefill_warmups,
        "decode_steps": protocol.decode_steps,
        "decode_measurements": protocol.decode_measurements,
        "decode_warmups": protocol.decode_warmups,
        "seed": protocol.seed,
        "attention_implementation": case.attn_implementation,
        "experts_implementation": case.experts_implementation,
        "trust_remote_code": case.trust_remote_code,
        "mxfp4_dequantize": protocol.mxfp4_dequantize,
        "freeze_granite_moe_routing": protocol.freeze_granite_moe_routing,
        "suite_environment_preflight": "passed",
        "suite_config": config_receipts["suite_config"][0],
        "suite_config_sha256": config_receipts["suite_config"][1],
        "environment_profiles": config_receipts["environment_profiles"][0],
        "environment_profiles_sha256": config_receipts["environment_profiles"][1],
    }
    for field_name, expected in expected_common.items():
        if measurement.get(field_name) != expected:
            raise ValueError(
                f"{label} measurement {field_name} differs from the paper protocol"
            )

    expected_mode = protocol.quality_mode if quality else protocol.throughput_mode
    if measurement.get("mode") != expected_mode:
        raise ValueError(f"{label} measurement mode differs from the paper protocol")
    expected_sequence_length = (
        case.quality_sequence_length_argument
        if quality
        else case.throughput_sequence_length
    )
    if measurement.get("sequence_length_argument") != expected_sequence_length:
        raise ValueError(
            f"{label} measurement sequence length differs from the paper protocol"
        )
    if measurement.get("eval_limit") is not protocol.quality_eval_limit:
        raise ValueError(f"{label} uses a limited evaluation")

    if quality:
        quality_expected: dict[str, Any] = {
            "eval_tasks": ",".join(protocol.quality_tasks),
            "eval_num_fewshot": protocol.quality_default_num_fewshot,
            "eval_task_fewshot": [
                f"{task}={fewshot[task]}" for task in protocol.quality_tasks
            ],
            "eval_batch_size": case.eval_batch_size,
            "eval_task_config": protocol.quality_task_config,
            "eval_task_config_sha256": config_receipts["eval_task_config"][1],
            "eval_log_samples": protocol.quality_log_samples,
        }
        for field_name, expected in quality_expected.items():
            if measurement.get(field_name) != expected:
                raise ValueError(
                    f"{label} measurement {field_name} differs from the paper protocol"
                )
    else:
        if measurement.get("eval_tasks") not in (None, ""):
            raise ValueError(f"{label} throughput artifact unexpectedly ran evaluation")
        if measurement.get("eval_task_config") is not None:
            raise ValueError(
                f"{label} throughput artifact records an evaluation task config"
            )
        if measurement.get("eval_task_config_sha256") is not None:
            raise ValueError(
                f"{label} throughput artifact records an evaluation task-config hash"
            )
        if measurement.get("eval_log_samples") is not False:
            raise ValueError(f"{label} throughput artifact enabled sample logging")


def _check_result_shape(
    row: dict[str, Any],
    *,
    case: CaseSpec,
    protocol: MeasurementProtocol,
    quality: bool,
    label: str,
) -> None:
    expected = {
        "dtype": protocol.dtype,
        "device": protocol.device,
        "batch_size": protocol.batch_size,
        "seq_len": (
            case.quality_sequence_length_argument
            if quality
            else case.throughput_sequence_length
        ),
    }
    for field_name, value in expected.items():
        if row.get(field_name) != value:
            raise ValueError(
                f"{label} result {field_name} differs from the paper protocol"
            )


def _quality_scores(evaluation: dict[str, Any], label: str) -> dict[str, float]:
    results = _object(evaluation.get("results"), f"{label} results")
    scores: dict[str, float] = {}
    for task, metric in TASK_METRICS.items():
        task_result = _object(results.get(task), f"{label} result {task}")
        scores[task] = _finite_number(
            task_result.get(metric), f"{label} {task}/{metric}"
        )
    return scores


def _check_module_origins(
    environment: dict[str, Any],
    *,
    expectations: tuple[LocalModuleExpectation, ...],
    source_revision: str,
    label: str,
) -> None:
    origins = _object(environment.get("module_origins"), f"{label} module_origins")
    packages = _object(environment.get("packages"), f"{label} packages")
    for expectation in expectations:
        raw = _object(
            origins.get(expectation.name),
            f"{label} module origin {expectation.name}",
        )
        expected_revision = (
            source_revision
            if expectation.revision_source == "repository"
            else expectation.revision
        )
        if raw.get("module") != expectation.module:
            raise ValueError(f"{label} records the wrong {expectation.name} module")
        if raw.get("distribution") != expectation.distribution:
            raise ValueError(
                f"{label} records the wrong {expectation.name} distribution"
            )
        if raw.get("package_version") != expectation.version:
            raise ValueError(
                f"{label} {expectation.name} package version is not "
                f"{expectation.version}"
            )
        if packages.get(expectation.distribution) != expectation.version:
            raise ValueError(
                f"{label} package inventory does not bind {expectation.distribution}"
            )
        if raw.get("source_path") != expectation.source_path:
            raise ValueError(f"{label} {expectation.name} source path is not declared")
        if raw.get("source_revision") != expected_revision:
            raise ValueError(
                f"{label} {expectation.name} source revision does not match"
            )
        if raw.get("expected_source_revision") != expected_revision:
            raise ValueError(
                f"{label} {expectation.name} expected revision was not recorded"
            )
        if raw.get("source_revision_matches") is not True:
            raise ValueError(f"{label} {expectation.name} source revision is unbound")
        if raw.get("source_dirty") is not False:
            raise ValueError(f"{label} {expectation.name} source checkout is dirty")
        if raw.get("module_loaded") is not True:
            raise ValueError(f"{label} {expectation.name} module was not loaded")
        if not isinstance(raw.get("module_file"), str) or not raw["module_file"]:
            raise ValueError(f"{label} {expectation.name} module file is missing")
        if (
            not isinstance(raw.get("module_sha256"), str)
            or _SHA256.fullmatch(raw["module_sha256"]) is None
        ):
            raise ValueError(f"{label} {expectation.name} module hash is missing")
        if raw.get("native_binary_required") is not expectation.require_native_binary:
            raise ValueError(
                f"{label} {expectation.name} native-binary policy is inconsistent"
            )
        if expectation.require_native_binary:
            native_binaries = _object(
                raw.get("native_binary_sha256"),
                f"{label} {expectation.name} native binaries",
            )
            if not native_binaries:
                raise ValueError(
                    f"{label} {expectation.name} native binary hashes are missing"
                )
            for binary_name, digest in native_binaries.items():
                if (
                    not isinstance(binary_name, str)
                    or not binary_name
                    or not isinstance(digest, str)
                    or _SHA256.fullmatch(digest) is None
                ):
                    raise ValueError(
                        f"{label} {expectation.name} native binary inventory is invalid"
                    )
        if raw.get("origin_matches_source") is not True:
            raise ValueError(f"{label} {expectation.name} module origin is unbound")
        if raw.get("binding_method") not in {
            "module-within-source",
            "pep610-direct-url",
        }:
            raise ValueError(f"{label} {expectation.name} binding method is invalid")
        if raw.get("bound") is not True:
            raise ValueError(f"{label} {expectation.name} attestation is unbound")


def collect_records(
    paths: Sequence[Path],
    cases: tuple[CaseSpec, ...],
    *,
    task_config: Path,
    open_weight_config: Path,
    external_source_pins: dict[str, Any],
    environment_config: Path = DEFAULT_ENVIRONMENTS,
    module_expectations: dict[str, tuple[LocalModuleExpectation, ...]] | None = None,
    allow_unbound_source: bool = False,
) -> dict[tuple[str, str], VariantRecord]:
    by_model = {case.model_id: case for case in cases}
    records = {
        (case.key, variant): VariantRecord(case=case, variant=variant)
        for case in cases
        for variant in case.variants
    }
    fewshot = load_fewshot_expectations(open_weight_config)
    measurement_protocol = load_measurement_protocol(open_weight_config)
    suite_config_receipt = _file_receipt(open_weight_config, "open-weight config")
    environment_receipt = _file_receipt(
        environment_config, "environment-profile config"
    )
    declared_task_config = (
        REPOSITORY_ROOT / measurement_protocol.quality_task_config
    ).resolve()
    task_expectations: TaskExpectations | None = None
    quality_task_config_receipt: tuple[str, str] | None = None
    if module_expectations is None:
        module_expectations = load_local_module_expectations(
            environment_config,
            cases,
        )

    for path in paths:
        document = _object(json.loads(path.read_text(encoding="utf-8")), str(path))
        if document.get("schema_version") != 1:
            raise ValueError(f"{path} is not a schema-version-1 result")
        experiment = _object(document.get("experiment"), f"{path} experiment")
        if experiment.get("id") != "open-weight-evaluation":
            raise ValueError(f"{path} is not an open-weight-evaluation result")
        _, command_options = _check_recorded_command(experiment, str(path))
        models = experiment.get("models")
        if (
            not isinstance(models, list)
            or len(models) != 1
            or models[0] not in by_model
        ):
            raise ValueError(f"{path} must contain exactly one configured model")
        model_id = models[0]
        case = by_model[model_id]
        requested = _object(
            experiment.get("requested_model_revisions"),
            f"{path} requested_model_revisions",
        )
        if requested.get(model_id) != case.revision:
            raise ValueError(f"{path} does not use the configured model revision")
        requested_tokenizer = _object(
            experiment.get("requested_tokenizer_revisions"),
            f"{path} requested_tokenizer_revisions",
        )
        if requested_tokenizer.get(model_id) != case.revision:
            raise ValueError(f"{path} does not use the configured tokenizer revision")
        if experiment.get("revision_provenance") != case.revision_provenance:
            raise ValueError(f"{path} has incompatible revision provenance")
        if experiment.get("variants") != list(case.variants):
            raise ValueError(f"{path} does not preserve the configured variant order")
        measurement = _object(document.get("measurement"), f"{path} measurement")
        raw_rows = document.get("results")
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError(f"{path} results must be a non-empty array")
        if [
            row.get("variant") if isinstance(row, dict) else None for row in raw_rows
        ] != list(case.variants):
            raise ValueError(f"{path} result rows do not preserve variant order")
        if not all(isinstance(row, dict) for row in raw_rows):
            raise ValueError(f"{path} result rows must be objects")
        quality_flags = [row.get("eval") is not None for row in raw_rows]
        if len(set(quality_flags)) != 1:
            raise ValueError(f"{path} mixes quality and throughput result rows")
        has_quality = quality_flags[0]

        if case.key not in module_expectations:
            raise ValueError(f"{path} lacks module expectations for {case.key}")
        case_module_expectations = module_expectations[case.key]
        required_external_sources = _required_external_sources(
            case,
            case_module_expectations,
            quality=has_quality,
        )
        missing_external_pins = required_external_sources - set(external_source_pins)
        if missing_external_pins:
            raise ValueError(
                "experiment manifest lacks required external source pins: "
                + ", ".join(sorted(missing_external_pins))
            )
        required_external_pins = {
            name: _full_commit(
                external_source_pins[name], f"external source pin {name}"
            )
            for name in required_external_sources
        }

        if has_quality:
            if task_config.resolve() != declared_task_config:
                raise ValueError(
                    "quality task config differs from the path declared by the "
                    "open-weight protocol"
                )
            if task_expectations is None:
                task_expectations = load_task_expectations(task_config)
                if (
                    required_external_pins["lm-evaluation-harness"]
                    != task_expectations.harness_revision
                ):
                    raise ValueError(
                        "task protocol lm-eval revision differs from the experiment "
                        "manifest"
                    )
                quality_task_config_receipt = _file_receipt(
                    task_config, "quality task config"
                )
            assert quality_task_config_receipt is not None
            eval_task_config_receipt: tuple[str | None, str | None] = (
                quality_task_config_receipt
            )
        else:
            eval_task_config_receipt = (None, None)

        config_receipts: dict[str, tuple[str | None, str | None]] = {
            "suite_config": suite_config_receipt,
            "environment_profiles": environment_receipt,
            "eval_task_config": eval_task_config_receipt,
        }
        source = _object(document.get("source"), f"{path} source")
        source_revision = source.get("revision")
        source_dirty = source.get("dirty")
        observed_components = source.get("external_components")
        observed_component_dirty = source.get("external_component_dirty")
        if not allow_unbound_source:
            if (
                not isinstance(source_revision, str)
                or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
            ):
                raise ValueError(f"{path} lacks an immutable repository revision")
            if source_dirty is not False:
                raise ValueError(f"{path} was produced from a dirty source tree")
            if required_external_sources and (
                not isinstance(observed_components, dict)
                or any(
                    observed_components.get(name) != required_external_pins[name]
                    for name in required_external_sources
                )
            ):
                raise ValueError(
                    f"{path} external source pins differ from the manifest"
                )
            if required_external_sources and (
                not isinstance(observed_component_dirty, dict)
                or any(
                    observed_component_dirty.get(name) is not False
                    for name in required_external_sources
                )
            ):
                raise ValueError(f"{path} has dirty or unavailable external sources")
        environment = _object(document.get("environment"), f"{path} environment")
        if not allow_unbound_source:
            _check_module_origins(
                environment,
                expectations=case_module_expectations,
                source_revision=source_revision,
                label=str(path),
            )
        _check_measurement_protocol(
            measurement,
            case=case,
            protocol=measurement_protocol,
            fewshot=fewshot,
            config_receipts=config_receipts,
            quality=has_quality,
            label=str(path),
        )
        _check_command_protocol(
            command_options,
            case=case,
            protocol=measurement_protocol,
            fewshot=fewshot,
            config_receipts=config_receipts,
            quality=has_quality,
            label=str(path),
        )
        for raw_row in raw_rows:
            row = _object(raw_row, f"{path} result row")
            if row.get("error"):
                raise ValueError(f"{path} contains a failed result row")
            if row.get("model") != model_id:
                raise ValueError(f"{path} result row model does not match its envelope")
            _check_result_shape(
                row,
                case=case,
                protocol=measurement_protocol,
                quality=has_quality,
                label=str(path),
            )
            variant = row.get("variant")
            if variant not in case.variants:
                raise ValueError(f"{path} contains undeclared variant {variant!r}")
            record = records[(case.key, str(variant))]
            _merge_equal(record, "source_revision", source_revision, path)
            _merge_equal(record, "source_dirty", source_dirty, path)
            _merge_equal(record, "environment", environment, path)
            if row.get("eval") is not None:
                assert task_expectations is not None
                evaluation = _object(row["eval"], f"{path} {variant} evaluation")
                _check_quality_protocol(
                    evaluation,
                    expectations=task_expectations,
                    fewshot=fewshot,
                    label=f"{path} {variant}",
                )
                _merge_equal(
                    record,
                    "quality",
                    _quality_scores(evaluation, f"{path} {variant}"),
                    path,
                )
            for field_name in ("prefill_tokens_per_s", "decode_tokens_per_s"):
                value = row.get(field_name)
                if value is not None:
                    _merge_equal(
                        record,
                        field_name,
                        _positive_number(value, f"{path} {variant} {field_name}"),
                        path,
                    )
            prefill_ms = row.get("prefill_ms")
            prefill_samples = row.get("prefill_repetition_ms")
            if prefill_ms is not None:
                expected_count = _positive_int(
                    measurement.get("prefill_measurements"),
                    f"{path} prefill_measurements",
                )
                samples = _positive_samples(
                    prefill_samples,
                    f"{path} {variant} prefill_repetition_ms",
                    expected_count,
                )
                observed_mean = _positive_number(
                    prefill_ms, f"{path} {variant} prefill_ms"
                )
                if not math.isclose(
                    observed_mean,
                    sum(samples) / len(samples),
                    rel_tol=1e-6,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        f"{path} {variant} prefill_ms does not match repetitions"
                    )
                _merge_equal(record, "prefill_repetition_ms", samples, path)
            elif prefill_samples not in (None, []):
                raise ValueError(
                    f"{path} {variant} has prefill repetitions without an aggregate"
                )

            decode_ms_per_token = row.get("decode_ms_per_token")
            decode_samples = row.get("decode_repetition_ms")
            if decode_ms_per_token is not None:
                expected_count = _positive_int(
                    measurement.get("decode_measurements"),
                    f"{path} decode_measurements",
                )
                decode_steps = _positive_int(
                    measurement.get("decode_steps"),
                    f"{path} decode_steps",
                )
                samples = _positive_samples(
                    decode_samples,
                    f"{path} {variant} decode_repetition_ms",
                    expected_count,
                )
                observed_per_token = _positive_number(
                    decode_ms_per_token,
                    f"{path} {variant} decode_ms_per_token",
                )
                expected_per_token = sum(samples) / (len(samples) * decode_steps)
                if not math.isclose(
                    observed_per_token,
                    expected_per_token,
                    rel_tol=1e-6,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        f"{path} {variant} decode_ms_per_token does not match repetitions"
                    )
                _merge_equal(record, "decode_repetition_ms", samples, path)
            elif decode_samples not in (None, []):
                raise ValueError(
                    f"{path} {variant} has decode repetitions without an aggregate"
                )
            for field_name in (
                "patched_silu_modules",
                "patched_router_sigmoid_modules",
            ):
                value = row.get(field_name)
                if value is not None:
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise ValueError(f"{path} {variant} {field_name} is invalid")
                    _merge_equal(record, field_name, value, path)
            patched_softcap = row.get("patched_gemma_softcap")
            if not isinstance(patched_softcap, bool):
                raise ValueError(f"{path} {variant} patched_gemma_softcap is invalid")
            _merge_equal(record, "patched_gemma_softcap", patched_softcap, path)
            record.input_paths.append(path)
    return records


def require_complete(
    records: dict[tuple[str, str], VariantRecord],
    cases: tuple[CaseSpec, ...],
    mode: str,
    environment_expectations: dict[str, dict[str, str]],
) -> None:
    failures: list[str] = []
    for case in cases:
        expected_packages = environment_expectations[case.key]
        if (
            mode in {"quality", "combined"}
            and expected_packages.get(EXPECTED_LM_EVAL_DISTRIBUTION)
            != EXPECTED_LM_EVAL_VERSION
        ):
            failures.append(
                f"{case.key}: quality environment does not pin "
                f"{EXPECTED_LM_EVAL_DISTRIBUTION}=={EXPECTED_LM_EVAL_VERSION}"
            )
        for variant in case.variants:
            record = records[(case.key, variant)]
            if not record.input_paths:
                failures.append(f"{case.key}/{variant}: no result")
                continue
            packages = (
                record.environment.get("packages")
                if isinstance(record.environment, dict)
                else None
            )
            if not isinstance(packages, dict):
                failures.append(f"{case.key}/{variant}: no package inventory")
            else:
                for distribution, expected in expected_packages.items():
                    if (
                        mode == "throughput"
                        and distribution == EXPECTED_LM_EVAL_DISTRIBUTION
                    ):
                        continue
                    if packages.get(distribution) != expected:
                        failures.append(
                            f"{case.key}/{variant}: {distribution} does not match "
                            f"the declared {expected} environment"
                        )
                if packages.get("sfu-spline-ops") is None:
                    failures.append(
                        f"{case.key}/{variant}: sfu-spline-ops build is not recorded"
                    )
            required_patches = variant_patch_requirements(variant)
            observed_patch_fields = (
                record.patched_silu_modules,
                record.patched_router_sigmoid_modules,
                record.patched_gemma_softcap,
            )
            if any(value is None for value in observed_patch_fields):
                failures.append(f"{case.key}/{variant}: patch scope is not recorded")
            else:
                activation_count = record.patched_silu_modules or 0
                router_count = record.patched_router_sigmoid_modules or 0
                softcap_patched = bool(record.patched_gemma_softcap)
                if (activation_count > 0) != ("activation" in required_patches):
                    failures.append(
                        f"{case.key}/{variant}: activation patch scope is unexpected"
                    )
                if (router_count > 0) != ("router" in required_patches):
                    failures.append(
                        f"{case.key}/{variant}: router patch scope is unexpected"
                    )
                if softcap_patched != ("softcap" in required_patches):
                    failures.append(
                        f"{case.key}/{variant}: softcap patch scope is unexpected"
                    )
            if mode in {"quality", "combined"} and record.quality is None:
                failures.append(f"{case.key}/{variant}: no quality result")
            if mode in {"throughput", "combined"}:
                if record.prefill_tokens_per_s is None:
                    failures.append(f"{case.key}/{variant}: no prefill result")
                if not record.prefill_repetition_ms:
                    failures.append(
                        f"{case.key}/{variant}: no prefill repetition timings"
                    )
                if record.decode_tokens_per_s is None:
                    failures.append(f"{case.key}/{variant}: no decode result")
                if not record.decode_repetition_ms:
                    failures.append(
                        f"{case.key}/{variant}: no decode repetition timings"
                    )
    if failures:
        raise ValueError("incomplete result set:\n  " + "\n  ".join(failures))


def build_rows(
    records: dict[tuple[str, str], VariantRecord],
    cases: tuple[CaseSpec, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        native = records[(case.key, "native")]
        for variant in case.variants:
            record = records[(case.key, variant)]
            quality_deltas: dict[str, float] = {}
            if record.quality is not None and native.quality is not None:
                quality_deltas = {
                    task: 100.0 * (record.quality[task] - native.quality[task])
                    for task in QUALITY_TASKS
                }
            row: dict[str, Any] = {
                "case": case.key,
                "model": case.paper_name,
                "model_id": case.model_id,
                "model_revision": case.revision,
                "revision_provenance": case.revision_provenance,
                "replacement": case.replacement,
                "fit": variant_fit(variant),
                "variant": variant,
                "variant_index": case.variants.index(variant),
                "source_revision": record.source_revision,
                "source_dirty": record.source_dirty,
                "quality_mean_delta_pp": (
                    sum(quality_deltas.values()) / len(quality_deltas)
                    if quality_deltas
                    else None
                ),
                "quality_max_abs_delta_pp": (
                    max(abs(value) for value in quality_deltas.values())
                    if quality_deltas
                    else None
                ),
                "wikitext/word_perplexity,none": (
                    record.quality["wikitext"] if record.quality else None
                ),
                "prefill_tokens_per_s": record.prefill_tokens_per_s,
                "prefill_speedup": (
                    record.prefill_tokens_per_s / native.prefill_tokens_per_s
                    if record.prefill_tokens_per_s is not None
                    and native.prefill_tokens_per_s is not None
                    else None
                ),
                "decode_tokens_per_s": record.decode_tokens_per_s,
                "decode_speedup": (
                    record.decode_tokens_per_s / native.decode_tokens_per_s
                    if record.decode_tokens_per_s is not None
                    and native.decode_tokens_per_s is not None
                    else None
                ),
                "patched_silu_modules": record.patched_silu_modules,
                "patched_router_sigmoid_modules": (
                    record.patched_router_sigmoid_modules
                ),
                "patched_gemma_softcap": record.patched_gemma_softcap,
            }
            for task in QUALITY_TASKS:
                metric = TASK_METRICS[task]
                row[f"{task}/{metric}"] = (
                    record.quality[task] if record.quality else None
                )
                row[f"{task}/{metric}_delta_pp"] = quality_deltas.get(task)
            rows.append(row)
    return rows


def _display(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(rows: Iterable[dict[str, Any]]) -> str:
    lines = [
        "| Model | Fit | Mean quality delta (pp) | Max | WikiText PPL | Prefill | Decode |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        prefill = row["prefill_speedup"]
        decode = row["decode_speedup"]
        lines.append(
            "| {model} | {fit} | {mean} | {maximum} | {ppl} | {prefill} | {decode} |".format(
                model=row["model"],
                fit=row["fit"],
                mean=_display(row["quality_mean_delta_pp"], 3),
                maximum=_display(row["quality_max_abs_delta_pp"], 3),
                ppl=_display(row["wikitext/word_perplexity,none"], 4),
                prefill="--" if prefill is None else f"{prefill:.4f}x",
                decode="--" if decode is None else f"{decode:.4f}x",
            )
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Result JSON files or directories containing result JSON files.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--environments", type=Path, default=DEFAULT_ENVIRONMENTS)
    parser.add_argument(
        "--experiment-manifest", type=Path, default=DEFAULT_EXPERIMENT_MANIFEST
    )
    parser.add_argument(
        "--mode", choices=("quality", "throughput", "combined"), default="combined"
    )
    parser.add_argument(
        "--csv-out", type=Path, default=Path("results/open_weight_summary.csv")
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("results/open_weight_summary.md"),
    )
    parser.add_argument(
        "--allow-unbound-source",
        action="store_true",
        help=(
            "Accept dirty, missing, or non-manifest source state for a diagnostic "
            "summary. Such a summary is outside the paper-quality protocol."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    try:
        config = args.config.resolve()
        task_config = args.task_config.resolve()
        cases = load_cases(config)
        environment_expectations = load_environment_expectations(
            args.environments.resolve(), cases
        )
        module_expectations = load_local_module_expectations(
            args.environments.resolve(), cases
        )
        external_source_pins = load_external_source_pins(
            args.experiment_manifest.resolve()
        )
        paths = expand_inputs(args.inputs)
        records = collect_records(
            paths,
            cases,
            task_config=task_config,
            open_weight_config=config,
            external_source_pins=external_source_pins,
            environment_config=args.environments.resolve(),
            module_expectations=module_expectations,
            allow_unbound_source=args.allow_unbound_source,
        )
        require_complete(records, cases, args.mode, environment_expectations)
        rows = build_rows(records, cases)
        csv_path = args.csv_out.resolve()
        markdown_path = args.markdown_out.resolve()
        write_csv(csv_path, rows)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown(rows), encoding="utf-8")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"wrote {len(rows)} rows to {csv_path}")
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export paired TorchTitan TensorBoard scalars to a provenance-safe JSON file.

The input directories must each contain exactly one completed TensorBoard event
file.  TensorBoard is imported only when an event file is actually read, so the
pure validation and summarization helpers remain usable in a CPU-only analysis
environment without that optional dependency. B1--B4 compare native and
polynomial arms. B5 compares native with one explicitly named routed-exp2
candidate at a time.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sfu_repro.torchtitan.provenance import (  # noqa: E402
    RECEIPT_ARTIFACT_TYPE,
    RECEIPT_SCHEMA_VERSION,
    ReceiptError,
    canonical_json_bytes,
    find_receipt,
    normalize_repository_path,
    read_receipt,
    sha256_bytes,
)


DEFAULT_MANIFEST = REPOSITORY_ROOT / "configs/torchtitan/paper_runs.json"

EVENT_PATTERN = "events.out.tfevents.*"
TRAINING_LOSS_METRIC = "loss_metrics/global_avg_loss"
TOKENS_SEEN_METRIC = "n_tokens_seen"
TIME_METRIC = "time_metrics/end_to_end(s)"
THROUGHPUT_METRIC = "throughput(tps)"
VALIDATION_LOSS_METRIC = "validation_metrics/loss"
VALIDATION_THROUGHPUT_METRIC = "validation_metrics/throughput(tps)"
SUPPORTED_METRICS = (
    TRAINING_LOSS_METRIC,
    TOKENS_SEEN_METRIC,
    TIME_METRIC,
    THROUGHPUT_METRIC,
    VALIDATION_LOSS_METRIC,
    VALIDATION_THROUGHPUT_METRIC,
)

PROBE_FIRST_STEP = 20
PROBE_LAST_STEP = 100
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_FILE_SHA256 = re.compile(r"[0-9a-f]{64}")


class MetricsExportError(RuntimeError):
    """A safe-to-display failure while validating or exporting metrics."""


class SafeArgumentParser(argparse.ArgumentParser):
    """Reject invalid arguments without reflecting their contents to stderr."""

    def error(self, message: str) -> None:
        del message
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid arguments\n")


@dataclass(frozen=True)
class ScalarSample:
    """One scalar value at one TorchTitan training step."""

    step: int
    value: float


@dataclass(frozen=True)
class EventData:
    """The allow-listed scalar data and content identity of one event file."""

    sha256: str
    scalars: Mapping[str, Sequence[ScalarSample]]


@dataclass(frozen=True)
class ValidatedReceipt:
    """One validated launch receipt and the safe fields exported from it."""

    sha256: str
    selection: Mapping[str, Any]
    protocol: Mapping[str, Any]
    config_sha256: str
    command_sha256: str
    source: Mapping[str, Any]
    runtime: Mapping[str, Any]
    topology: Mapping[str, int]
    output_folder: str
    pair_config: Mapping[str, Any]
    pair_arguments: tuple[str, ...]


ScalarLoader = Callable[[Path], Mapping[str, Sequence[ScalarSample]]]


def resolve_event_file(directory: Path) -> Path:
    """Return the sole direct-child TensorBoard event file in ``directory``."""

    if not directory.is_dir():
        raise MetricsExportError("each arm input must be an existing directory")
    try:
        matches = sorted(
            path for path in directory.glob(EVENT_PATTERN) if path.is_file()
        )
    except OSError:
        raise MetricsExportError("could not inspect an arm input directory") from None
    if len(matches) != 1:
        raise MetricsExportError(
            "each arm input directory must contain exactly one "
            f"{EVENT_PATTERN} file; found {len(matches)}"
        )
    return matches[0]


def sha256_file(path: Path) -> str:
    """Hash ``path`` without loading the event file into memory."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        raise MetricsExportError("could not hash a source event file") from None
    return digest.hexdigest()


def _load_tensorboard_scalars(
    event_file: Path,
) -> Mapping[str, Sequence[ScalarSample]]:
    """Read only the allow-listed scalar tags from one TensorBoard event file."""

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        raise MetricsExportError(
            "reading event files requires the optional tensorboard package"
        ) from None

    try:
        accumulator = EventAccumulator(
            str(event_file),
            size_guidance={"scalars": 0},
        )
        accumulator.Reload()
        scalar_tags = set(accumulator.Tags().get("scalars", ()))
        return {
            tag: tuple(
                ScalarSample(step=event.step, value=float(event.value))
                for event in accumulator.Scalars(tag)
            )
            for tag in SUPPORTED_METRICS
            if tag in scalar_tags
        }
    except MetricsExportError:
        raise
    except Exception:
        raise MetricsExportError(
            "could not read a source TensorBoard event file"
        ) from None


def read_event_data(
    directory: Path,
    *,
    scalar_loader: ScalarLoader | None = None,
) -> EventData:
    """Read and hash one stable event file from an arm input directory."""

    event_file = resolve_event_file(directory)
    loader = _load_tensorboard_scalars if scalar_loader is None else scalar_loader
    try:
        before = event_file.stat()
        scalars = loader(event_file)
        digest = sha256_file(event_file)
        after = event_file.stat()
    except MetricsExportError:
        raise
    except OSError:
        raise MetricsExportError("could not inspect a source event file") from None

    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise MetricsExportError(
            "a source event file changed while it was being exported; "
            "use a completed run"
        )
    return EventData(sha256=digest, scalars=scalars)


def _selected_samples(
    samples: Sequence[ScalarSample],
    *,
    first_step: int | None = None,
    last_step: int | None = None,
) -> list[ScalarSample]:
    if first_step is not None and last_step is not None and first_step > last_step:
        raise ValueError("first_step must not exceed last_step")

    selected: list[ScalarSample] = []
    seen_steps: set[int] = set()
    for sample in samples:
        if isinstance(sample.step, bool) or not isinstance(sample.step, int):
            raise MetricsExportError("scalar steps must be integers")
        if sample.step < 0:
            raise MetricsExportError("scalar steps must be non-negative")
        if first_step is not None and sample.step < first_step:
            continue
        if last_step is not None and sample.step > last_step:
            continue
        if sample.step in seen_steps:
            raise MetricsExportError("a scalar metric contains a duplicate step")
        if isinstance(sample.value, bool):
            raise MetricsExportError("scalar values must be finite numbers")
        try:
            value = float(sample.value)
        except (TypeError, ValueError, OverflowError):
            raise MetricsExportError("scalar values must be finite numbers") from None
        if not math.isfinite(value):
            raise MetricsExportError("scalar values must be finite numbers")
        seen_steps.add(sample.step)
        selected.append(ScalarSample(step=sample.step, value=value))

    selected.sort(key=lambda sample: sample.step)
    if not selected:
        if first_step is None and last_step is None:
            raise MetricsExportError("a present scalar metric contains no samples")
        raise MetricsExportError(
            "a required scalar metric has no samples in the requested step window"
        )
    return selected


def summarize_series(
    samples: Sequence[ScalarSample],
    *,
    first_step: int | None = None,
    last_step: int | None = None,
    require_positive: bool = False,
) -> dict[str, Any]:
    """Return a deterministic series and descriptive summary for scalar samples."""

    selected = _selected_samples(
        samples,
        first_step=first_step,
        last_step=last_step,
    )
    values = [sample.value for sample in selected]
    if require_positive and any(value <= 0.0 for value in values):
        raise MetricsExportError("timing and throughput values must be positive")
    return {
        "series": [{"step": sample.step, "value": sample.value} for sample in selected],
        "summary": {
            "sample_count": len(selected),
            "first_step": selected[0].step,
            "last_step": selected[-1].step,
            "minimum": min(values),
            "maximum": max(values),
            "median": statistics.median(values),
            "last": values[-1],
        },
    }


def summarize_arm(
    scalars: Mapping[str, Sequence[ScalarSample]],
    *,
    phase: str,
    probe_first_step: int = PROBE_FIRST_STEP,
    probe_last_step: int = PROBE_LAST_STEP,
) -> dict[str, dict[str, Any]]:
    """Summarize one arm without importing TensorBoard or accessing files."""

    if phase not in {"model-probe", "pretraining"}:
        raise ValueError(f"unsupported phase: {phase}")

    metrics: dict[str, dict[str, Any]] = {}
    if phase == "model-probe":
        for tag in (TIME_METRIC, THROUGHPUT_METRIC):
            samples = scalars.get(tag)
            if not samples:
                raise MetricsExportError(
                    f"model probe is missing required metric {tag}"
                )
            metrics[tag] = summarize_series(
                samples,
                first_step=probe_first_step,
                last_step=probe_last_step,
                require_positive=True,
            )
    else:
        training_loss = scalars.get(TRAINING_LOSS_METRIC)
        if not training_loss:
            raise MetricsExportError(
                f"pretraining run is missing required metric {TRAINING_LOSS_METRIC}"
            )
        metrics[TRAINING_LOSS_METRIC] = summarize_series(training_loss)
        for tag in (TOKENS_SEEN_METRIC, TIME_METRIC, THROUGHPUT_METRIC):
            samples = scalars.get(tag)
            if samples:
                metrics[tag] = summarize_series(
                    samples,
                    require_positive=tag in {TIME_METRIC, THROUGHPUT_METRIC},
                )

    for tag in (VALIDATION_LOSS_METRIC, VALIDATION_THROUGHPUT_METRIC):
        samples = scalars.get(tag)
        if samples:
            metrics[tag] = summarize_series(
                samples,
                require_positive=tag == VALIDATION_THROUGHPUT_METRIC,
            )
    return metrics


def summarize_pair(
    native_scalars: Mapping[str, Sequence[ScalarSample]],
    polynomial_scalars: Mapping[str, Sequence[ScalarSample]],
    *,
    phase: str,
    candidate_name: str = "polynomial",
    probe_first_step: int = PROBE_FIRST_STEP,
    probe_last_step: int = PROBE_LAST_STEP,
) -> dict[str, Any]:
    """Build per-arm summaries and one explicitly named comparison."""

    if candidate_name not in {"polynomial", "pwl2_safe_f16", "d2_safe"}:
        raise ValueError(f"unsupported candidate name: {candidate_name}")

    arms = {
        "native": summarize_arm(
            native_scalars,
            phase=phase,
            probe_first_step=probe_first_step,
            probe_last_step=probe_last_step,
        ),
        candidate_name: summarize_arm(
            polynomial_scalars,
            phase=phase,
            probe_first_step=probe_first_step,
            probe_last_step=probe_last_step,
        ),
    }
    comparisons: dict[str, Any] = {}
    if phase == "model-probe":
        native_time = arms["native"][TIME_METRIC]["summary"]["median"]
        candidate_time = arms[candidate_name][TIME_METRIC]["summary"]["median"]
        native_throughput = arms["native"][THROUGHPUT_METRIC]["summary"]["median"]
        candidate_throughput = arms[candidate_name][THROUGHPUT_METRIC]["summary"][
            "median"
        ]
        comparisons = {
            f"end_to_end_time_speedup_native_over_{candidate_name}": {
                "formula": (f"native median seconds / {candidate_name} median seconds"),
                "value": native_time / candidate_time,
            },
            f"throughput_ratio_{candidate_name}_over_native": {
                "formula": f"{candidate_name} median tps / native median tps",
                "value": candidate_throughput / native_throughput,
            },
        }
    return {"arms": arms, "comparisons": comparisons}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MetricsExportError(f"launch receipt has invalid {label}")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MetricsExportError(f"launch receipt has invalid {label}")
    return value


def _string_array(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise MetricsExportError(f"launch receipt has invalid {label}")
    return value


def load_protocol_manifest(path: Path, case: str) -> tuple[dict[str, Any], str]:
    """Load one public run manifest and hash its exact bytes."""

    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise MetricsExportError("could not read the TorchTitan run manifest") from None
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise MetricsExportError("unsupported TorchTitan run-manifest schema")
    cases = document.get("cases")
    if not isinstance(cases, dict) or case not in cases:
        raise MetricsExportError("the selected case is absent from the run manifest")
    commit = document.get("torchtitan_commit")
    flash_attention_commit = document.get("flash_attention_commit")
    if (
        not isinstance(commit, str)
        or _COMMIT_SHA.fullmatch(commit) is None
        or not isinstance(flash_attention_commit, str)
        or _COMMIT_SHA.fullmatch(flash_attention_commit) is None
    ):
        raise MetricsExportError("the run manifest has invalid source commits")
    return document, sha256_bytes(payload)


def _decode_receipt_config(
    config: Mapping[str, Any],
    *,
    expected_case: str,
    expected_variant: str,
    expected_path: str,
) -> tuple[str, dict[str, Any]]:
    digest = _string(config.get("sha256"), "config hash")
    encoded = _string(config.get("bytes_base64"), "config bytes")
    repository_path = _string(config.get("repository_path"), "config path")
    if _FILE_SHA256.fullmatch(digest) is None:
        raise MetricsExportError("launch receipt has invalid config hash")
    if repository_path != "${REPOSITORY_ROOT}/" + expected_path:
        raise MetricsExportError(
            "launch receipt config path does not match the selected arm"
        )
    try:
        payload = base64.b64decode(encoded, validate=True)
        if sha256_bytes(payload) != digest:
            raise MetricsExportError(
                "launch receipt config hash does not match its bytes"
            )
        document = tomllib.loads(payload.decode("utf-8"))
    except MetricsExportError:
        raise
    except (ValueError, UnicodeError, tomllib.TOMLDecodeError):
        raise MetricsExportError(
            "launch receipt contains invalid config bytes"
        ) from None
    if not isinstance(document, dict):
        raise MetricsExportError("launch receipt contains invalid config bytes")
    sfu = document.get("sfu")
    if not isinstance(sfu, dict) or (
        sfu.get("case") != expected_case
        or sfu.get("variant") != expected_variant
        or sfu.get("strict") is not True
    ):
        raise MetricsExportError(
            "launch receipt config does not select the labelled arm"
        )
    return digest, document


def _normalized_pair_config(document: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(document))
    job = normalized.get("job")
    if isinstance(job, dict):
        job["dump_folder"] = "<arm-output>"
        job["description"] = "<arm-description>"
    sfu = normalized.get("sfu")
    if isinstance(sfu, dict):
        sfu["variant"] = "<candidate-variant>"
    return normalized


def _normalized_pair_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for argument in arguments:
        if argument.startswith("--job.config-file="):
            normalized.append("--job.config-file=<arm-config>")
        elif argument.startswith("--job.dump-folder="):
            normalized.append("--job.dump-folder=<arm-output>")
        elif argument.startswith("--sfu.variant="):
            normalized.append("--sfu.variant=<candidate-variant>")
        else:
            normalized.append(argument)
    return tuple(normalized)


def _effective_receipt_override(arguments: Sequence[str], name: str) -> str | None:
    prefix = f"--{name}="
    selected: str | None = None
    for argument in arguments:
        if argument.startswith(prefix):
            selected = argument.removeprefix(prefix)
    return selected


def validate_launch_receipt(
    path: Path,
    *,
    case: str,
    variant: str,
    phase: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
) -> ValidatedReceipt:
    """Fail closed unless a receipt binds one exact public run arm."""

    try:
        document, receipt_sha256 = read_receipt(path)
    except ReceiptError as error:
        raise MetricsExportError(str(error)) from None
    if (
        document.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or document.get("artifact_type") != RECEIPT_ARTIFACT_TYPE
    ):
        raise MetricsExportError("unsupported launch-receipt schema")

    binding = _object(document.get("binding"), "binding status")
    reasons = binding.get("reasons")
    if binding.get("status") != "bound" or reasons != []:
        raise MetricsExportError("launch receipt is explicitly unbound")

    selection = _object(document.get("selection"), "selection")
    if (
        selection.get("case") != case
        or selection.get("variant") != variant
        or selection.get("phase") != phase
        or not isinstance(selection.get("validation"), bool)
    ):
        raise MetricsExportError("launch receipt does not match the labelled arm")

    try:
        case_record = manifest["cases"][case]
        expected_config = case_record["configs"][variant]
    except (KeyError, TypeError):
        raise MetricsExportError(
            "run manifest does not define the labelled arm"
        ) from None

    protocol = _object(document.get("protocol"), "protocol")
    if protocol.get("manifest_sha256") != manifest_sha256:
        raise MetricsExportError(
            "launch receipt protocol hash does not match the manifest"
        )
    tokenizer = _object(protocol.get("tokenizer"), "tokenizer provenance")
    tokenizer_key = "b4" if case == "b4" else "b1_b3"
    try:
        tokenizer_pin = manifest["tokenizer_snapshots"][tokenizer_key]
    except (KeyError, TypeError):
        raise MetricsExportError(
            "run manifest has no tokenizer pin for the case"
        ) from None
    if (
        tokenizer.get("repository") != tokenizer_pin.get("repository")
        or tokenizer.get("revision") != tokenizer_pin.get("revision")
        or tokenizer.get("verified") is not True
        or not isinstance(tokenizer.get("manifest_sha256"), str)
        or _FILE_SHA256.fullmatch(tokenizer["manifest_sha256"]) is None
    ):
        raise MetricsExportError("launch receipt has unverified tokenizer provenance")
    dataset = _object(protocol.get("dataset"), "dataset provenance")
    training_dataset = _object(dataset.get("training"), "training dataset provenance")
    validation_dataset = dataset.get("validation")
    try:
        slim_pin = manifest["dataset_snapshots"]["b1_b3_training_and_all_validation"]
        olmo_pin = manifest["dataset_snapshots"]["b4_training"]
    except (KeyError, TypeError):
        raise MetricsExportError("run manifest has incomplete dataset pins") from None
    if phase == "model-probe":
        expected_probe_path = "${REPOSITORY_ROOT}/torchtitan/tests/assets/c4_test"
        probe_source = training_dataset.get("source")
        if (
            training_dataset.get("name") != "c4_test"
            or training_dataset.get("repository_path") != expected_probe_path
            or not isinstance(probe_source, dict)
            or probe_source.get("submodule") != "torchtitan"
            or probe_source.get("revision") != manifest["torchtitan_commit"]
        ):
            raise MetricsExportError(
                "launch receipt has invalid model-probe dataset provenance"
            )
    else:
        expected_training_name = (
            "sfu_olmo_mix_1124" if case == "b4" else "sfu_slimpajama"
        )
        expected_training_pin = olmo_pin if case == "b4" else slim_pin
        if (
            training_dataset.get("name") != expected_training_name
            or training_dataset.get("pin") != expected_training_pin
        ):
            raise MetricsExportError(
                "launch receipt has invalid training-dataset provenance"
            )
    if selection["validation"]:
        if not isinstance(validation_dataset, dict) or (
            validation_dataset.get("name") != "sfu_slimpajama_validation"
            or validation_dataset.get("pin") != slim_pin
        ):
            raise MetricsExportError(
                "launch receipt has invalid validation-dataset provenance"
            )
    elif validation_dataset is not None:
        raise MetricsExportError("launch receipt records unexpected validation data")

    checkpoint = protocol.get("seed_checkpoint")
    if phase == "pretraining":
        checkpoint = _object(checkpoint, "seed checkpoint provenance")
        if (
            not isinstance(checkpoint.get("path"), str)
            or not checkpoint["path"]
            or checkpoint.get("tree_hash_algorithm") != "sha256-path-size-content-v1"
            or not isinstance(checkpoint.get("sha256"), str)
            or _FILE_SHA256.fullmatch(checkpoint["sha256"]) is None
            or isinstance(checkpoint.get("file_count"), bool)
            or not isinstance(checkpoint.get("file_count"), int)
            or checkpoint["file_count"] <= 0
            or isinstance(checkpoint.get("total_bytes"), bool)
            or not isinstance(checkpoint.get("total_bytes"), int)
            or checkpoint["total_bytes"] <= 0
        ):
            raise MetricsExportError(
                "launch receipt has invalid seed checkpoint provenance"
            )
    elif checkpoint is not None:
        raise MetricsExportError("launch receipt records an unexpected seed checkpoint")

    config = _object(document.get("config"), "config")
    config_sha256, config_document = _decode_receipt_config(
        config,
        expected_case=case,
        expected_variant=variant,
        expected_path=expected_config,
    )

    command = _object(document.get("command"), "command")
    sanitized_argv = _string_array(command.get("sanitized_argv"), "command")
    effective_arguments = _string_array(
        command.get("effective_torchtitan_arguments"),
        "effective TorchTitan arguments",
    )
    effective_overrides = _string_array(
        command.get("effective_overrides"),
        "effective overrides",
    )
    redactions = command.get("redactions")
    if not isinstance(redactions, list) or not all(
        isinstance(item, str) for item in redactions
    ):
        raise MetricsExportError("launch receipt has invalid command redactions")
    if any("<redacted-secret>" in item for item in sanitized_argv):
        raise MetricsExportError("launch receipt redacted a scientific command value")
    if len(effective_overrides) > len(effective_arguments) or (
        effective_overrides
        and effective_arguments[-len(effective_overrides) :] != effective_overrides
    ):
        raise MetricsExportError("launch receipt override list is inconsistent")
    config_override = next(
        (
            item.removeprefix("--job.config-file=")
            for item in effective_arguments
            if item.startswith("--job.config-file=")
        ),
        None,
    )
    if config_override != config.get("repository_path"):
        raise MetricsExportError(
            "launch receipt command does not use its embedded config"
        )
    for name, expected in (("sfu.case", case), ("sfu.variant", variant)):
        override = _effective_receipt_override(effective_arguments, name)
        if override is not None and override != expected:
            raise MetricsExportError("launch receipt overrides the labelled arm")

    source = _object(document.get("source"), "source provenance")
    repository = _object(source.get("repository"), "repository provenance")
    if (
        not isinstance(repository.get("revision"), str)
        or _COMMIT_SHA.fullmatch(repository["revision"]) is None
        or repository.get("dirty") is not False
    ):
        raise MetricsExportError("launch receipt has an unbound repository state")
    submodules = _object(source.get("submodules"), "submodule provenance")
    for name, expected_revision in (
        ("torchtitan", manifest["torchtitan_commit"]),
        ("flash-attention", manifest["flash_attention_commit"]),
    ):
        state = _object(submodules.get(name), f"{name} provenance")
        if (
            state.get("revision") != expected_revision
            or state.get("dirty") is not False
        ):
            raise MetricsExportError("launch receipt has an unbound submodule state")

    runtime = _object(document.get("runtime"), "runtime provenance")
    _object(runtime.get("python"), "Python runtime")
    _object(runtime.get("platform"), "platform runtime")
    _object(runtime.get("distributions"), "package runtime")
    _object(runtime.get("torch"), "Torch runtime")
    topology = _object(document.get("topology"), "topology")
    nproc_per_node = topology.get("nproc_per_node")
    nnodes = topology.get("nnodes")
    world_size = topology.get("world_size")
    if (
        isinstance(nproc_per_node, bool)
        or not isinstance(nproc_per_node, int)
        or nproc_per_node <= 0
        or isinstance(nnodes, bool)
        or not isinstance(nnodes, int)
        or nnodes <= 0
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size != nproc_per_node * nnodes
    ):
        raise MetricsExportError("launch receipt has invalid topology")
    try:
        phase_key = phase.replace("-", "_")
        expected_world_size = int(case_record[phase_key]["world_size"])
    except (KeyError, TypeError, ValueError):
        raise MetricsExportError(
            "run manifest has no topology for the selected phase"
        ) from None
    if world_size != expected_world_size:
        raise MetricsExportError(
            "launch receipt world size differs from the public protocol"
        )
    output = _object(document.get("output"), "output provenance")
    output_folder = _string(output.get("dump_folder"), "output folder")

    return ValidatedReceipt(
        sha256=receipt_sha256,
        selection=dict(selection),
        protocol={
            "manifest_sha256": manifest_sha256,
            "dataset": dataset,
            "seed_checkpoint": checkpoint,
            "tokenizer": {
                "repository": tokenizer["repository"],
                "revision": tokenizer["revision"],
                "manifest_sha256": tokenizer["manifest_sha256"],
                "verified": True,
            },
        },
        config_sha256=config_sha256,
        command_sha256=sha256_bytes(canonical_json_bytes(command)),
        source=source,
        runtime=runtime,
        topology={
            "nproc_per_node": nproc_per_node,
            "nnodes": nnodes,
            "world_size": world_size,
        },
        output_folder=output_folder,
        pair_config=_normalized_pair_config(config_document),
        pair_arguments=_normalized_pair_arguments(effective_arguments),
    )


def validate_event_location(run_directory: Path, receipt: ValidatedReceipt) -> None:
    """Require the event directory to live in the receipt's declared output tree."""

    prefix = "${REPOSITORY_ROOT}"
    if receipt.output_folder == prefix:
        output_folder = REPOSITORY_ROOT.resolve()
    elif receipt.output_folder.startswith(prefix + "/"):
        output_folder = (
            REPOSITORY_ROOT / receipt.output_folder.removeprefix(prefix + "/")
        ).resolve()
    try:
        resolved_run = run_directory.resolve(strict=True)
    except OSError:
        raise MetricsExportError("could not resolve an arm input directory") from None
    if receipt.output_folder.startswith("${EXTERNAL_PATH:"):
        declared_ancestor = any(
            normalize_repository_path(str(directory), REPOSITORY_ROOT)
            == receipt.output_folder
            for directory in (resolved_run, *resolved_run.parents)
        )
        if not declared_ancestor:
            raise MetricsExportError(
                "an arm event directory is outside its receipt-declared output tree"
            )
        return
    if not receipt.output_folder.startswith(prefix):
        raise MetricsExportError("launch receipt has an invalid output folder")
    if resolved_run != output_folder and not resolved_run.is_relative_to(output_folder):
        raise MetricsExportError(
            "an arm event directory is outside its receipt-declared output tree"
        )


def validate_receipt_pair(
    native: ValidatedReceipt,
    candidate: ValidatedReceipt,
) -> None:
    """Require matched controls apart from arm-specific code and output names."""

    if native.selection.get("variant") != "native":
        raise MetricsExportError("control launch receipt is not the native arm")
    for label, left, right in (
        ("case", native.selection.get("case"), candidate.selection.get("case")),
        ("phase", native.selection.get("phase"), candidate.selection.get("phase")),
        (
            "validation mode",
            native.selection.get("validation"),
            candidate.selection.get("validation"),
        ),
        ("protocol", native.protocol, candidate.protocol),
        ("source", native.source, candidate.source),
        ("runtime", native.runtime, candidate.runtime),
        ("topology", native.topology, candidate.topology),
        ("base configuration", native.pair_config, candidate.pair_config),
        ("effective arguments", native.pair_arguments, candidate.pair_arguments),
    ):
        if left != right:
            raise MetricsExportError(f"native and candidate receipts differ in {label}")


def _receipt_path(run_directory: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    try:
        return find_receipt(run_directory)
    except ReceiptError as error:
        raise MetricsExportError(str(error)) from None


def bound_pair_document(
    native: ValidatedReceipt,
    candidate: ValidatedReceipt,
    *,
    candidate_name: str,
) -> dict[str, Any]:
    return {
        "status": "bound",
        "reasons": [],
        "protocol": native.protocol,
        "source": native.source,
        "runtime": native.runtime,
        "topology": native.topology,
        "validation": native.selection["validation"],
        "arms": {
            "native": {
                "launch_receipt_sha256": native.sha256,
                "config_sha256": native.config_sha256,
                "command_sha256": native.command_sha256,
            },
            candidate_name: {
                "launch_receipt_sha256": candidate.sha256,
                "config_sha256": candidate.config_sha256,
                "command_sha256": candidate.command_sha256,
            },
        },
    }


def unbound_pair_document(manifest_sha256: str, reason: str) -> dict[str, Any]:
    return {
        "status": "unbound",
        "reasons": [reason],
        "protocol": {"manifest_sha256": manifest_sha256},
        "arms": {},
    }


def load_protocol_pin(manifest_path: Path, case: str) -> str:
    """Load and validate the TorchTitan revision declared for a public case."""

    document, _ = load_protocol_manifest(manifest_path, case)
    return document["torchtitan_commit"]


def load_model_probe_window(manifest_path: Path, case: str) -> tuple[int, int]:
    """Return the manifest-declared inclusive steady-state step window."""

    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        probe = document["cases"][case]["model_probe"]
        first_step = int(probe.get("steady_state_first_step", PROBE_FIRST_STEP))
        last_step = int(probe["steps"])
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        raise MetricsExportError(
            "could not read the model-probe window from the run manifest"
        ) from None
    if first_step < 0 or last_step < first_step:
        raise MetricsExportError("the run manifest has an invalid model-probe window")
    return first_step, last_step


def build_document(
    *,
    case: str,
    phase: str,
    torchtitan_commit: str,
    native: EventData,
    polynomial: EventData,
    candidate_name: str = "polynomial",
    probe_first_step: int = PROBE_FIRST_STEP,
    probe_last_step: int = PROBE_LAST_STEP,
    provenance_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete JSON-compatible export document."""

    if (
        not isinstance(torchtitan_commit, str)
        or _COMMIT_SHA.fullmatch(torchtitan_commit) is None
    ):
        raise MetricsExportError("invalid TorchTitan commit")
    for event in (native, polynomial):
        if (
            not isinstance(event.sha256, str)
            or _FILE_SHA256.fullmatch(event.sha256) is None
        ):
            raise MetricsExportError("invalid source event SHA256")

    summarized = summarize_pair(
        native.scalars,
        polynomial.scalars,
        phase=phase,
        candidate_name=candidate_name,
        probe_first_step=probe_first_step,
        probe_last_step=probe_last_step,
    )
    for arm_name, event in (("native", native), (candidate_name, polynomial)):
        receipt_sha256 = None
        if provenance_binding is not None:
            binding_arms = provenance_binding.get("arms")
            if isinstance(binding_arms, Mapping):
                binding_arm = binding_arms.get(arm_name)
                if isinstance(binding_arm, Mapping):
                    receipt_sha256 = binding_arm.get("launch_receipt_sha256")
        summarized["arms"][arm_name] = {
            "source_event": {"sha256": event.sha256},
            "metrics": summarized["arms"][arm_name],
        }
        if isinstance(receipt_sha256, str):
            summarized["arms"][arm_name]["source_launch_receipt"] = {
                "sha256": receipt_sha256
            }

    measurement: dict[str, Any] = {
        "source": "TorchTitan TensorBoard scalar metrics",
        "timing_semantics": (
            "TorchTitan wall-clock end-to-end iteration time from "
            "time.perf_counter(), divided by metrics.log_freq"
        ),
        "historical_timing_distinction": (
            "These are not the historical synchronized CUDA-event "
            "forward/backward/optimizer phase statistics."
        ),
    }
    if phase == "model-probe":
        measurement["model_probe_step_window"] = {
            "first_step": probe_first_step,
            "last_step": probe_last_step,
            "inclusive": True,
            "summary_statistic": "median",
        }

    if provenance_binding is None:
        provenance_binding = {
            "status": "unbound",
            "reasons": ["launch receipts were not supplied to the document builder"],
            "arms": {},
        }

    return {
        "schema_version": 1,
        "artifact_type": "paired_torchtitan_tensorboard_metrics",
        "case": case,
        "phase": phase,
        "candidate": candidate_name,
        "torchtitan": {
            "repository": "pytorch/torchtitan",
            "commit": torchtitan_commit,
            "revision_basis": (
                "validated launch receipts and repository run-manifest pin"
                if provenance_binding.get("status") == "bound"
                else "repository run-manifest pin only; launch provenance is unbound"
            ),
        },
        "provenance_binding": dict(provenance_binding),
        "measurement": measurement,
        **summarized,
    }


def atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically replace ``path`` with a strict, deterministic JSON document."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise MetricsExportError("could not create the output directory") from None

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError):
        raise MetricsExportError("could not write the metrics JSON output") from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = SafeArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", choices=("b1", "b2", "b3", "b4", "b5"), required=True)
    parser.add_argument(
        "--phase",
        choices=("model-probe", "pretraining"),
        default="model-probe",
    )
    parser.add_argument("--native-dir", type=Path, required=True)
    comparison = parser.add_mutually_exclusive_group(required=True)
    comparison.add_argument("--polynomial-dir", type=Path)
    comparison.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--native-receipt", type=Path)
    parser.add_argument("--polynomial-receipt", type=Path)
    parser.add_argument("--candidate-receipt", type=Path)
    parser.add_argument(
        "--candidate-name",
        choices=("pwl2_safe_f16", "d2_safe"),
    )
    parser.add_argument(
        "--allow-unbound-receipts",
        action="store_true",
        help=(
            "Diagnostic only: export event metrics when launch receipts are "
            "missing or inconsistent and mark the result unbound."
        ),
    )
    parser.add_argument(
        "--require-validation",
        action="store_true",
        help=(
            "Require a pretraining receipt pair with validation enabled and a "
            "held-out loss series from both arms."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def resolve_comparison_selection(args: argparse.Namespace) -> tuple[Path, str]:
    """Validate the case-specific comparison arm and return its path and name."""

    if args.require_validation and args.phase != "pretraining":
        raise MetricsExportError("--require-validation requires --phase pretraining")
    if args.case == "b5":
        if args.phase != "model-probe":
            raise MetricsExportError("B5 defines model-probe exports only")
        if args.candidate_dir is None or args.candidate_name is None:
            raise MetricsExportError("B5 requires --candidate-dir and --candidate-name")
        if args.polynomial_receipt is not None:
            raise MetricsExportError("B5 does not accept --polynomial-receipt")
        return args.candidate_dir, args.candidate_name
    if args.polynomial_dir is None or args.candidate_name is not None:
        raise MetricsExportError(
            "B1--B4 require --polynomial-dir without --candidate-name"
        )
    if args.candidate_receipt is not None:
        raise MetricsExportError("B1--B4 do not accept --candidate-receipt")
    return args.polynomial_dir, "polynomial"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        comparison_dir, candidate_name = resolve_comparison_selection(args)
        manifest, manifest_sha256 = load_protocol_manifest(args.manifest, args.case)
        commit = manifest["torchtitan_commit"]

        native_receipt_path: Path | None = None
        candidate_receipt_path: Path | None = None
        provenance_binding: dict[str, Any]
        try:
            native_receipt_path = _receipt_path(
                args.native_dir,
                args.native_receipt,
            )
            candidate_receipt_path = _receipt_path(
                comparison_dir,
                (
                    args.candidate_receipt
                    if args.case == "b5"
                    else args.polynomial_receipt
                ),
            )
            native_receipt = validate_launch_receipt(
                native_receipt_path,
                case=args.case,
                variant="native",
                phase=args.phase,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
            )
            candidate_receipt = validate_launch_receipt(
                candidate_receipt_path,
                case=args.case,
                variant=candidate_name,
                phase=args.phase,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
            )
            validate_event_location(args.native_dir, native_receipt)
            validate_event_location(comparison_dir, candidate_receipt)
            validate_receipt_pair(native_receipt, candidate_receipt)
            if args.require_validation and not native_receipt.selection["validation"]:
                raise MetricsExportError(
                    "--require-validation needs receipts from validation-enabled launches"
                )
            provenance_binding = bound_pair_document(
                native_receipt,
                candidate_receipt,
                candidate_name=candidate_name,
            )
        except MetricsExportError as receipt_error:
            if not args.allow_unbound_receipts:
                raise
            provenance_binding = unbound_pair_document(
                manifest_sha256,
                f"diagnostic override: {receipt_error}",
            )

        native_file = resolve_event_file(args.native_dir)
        comparison_file = resolve_event_file(comparison_dir)
        try:
            output = args.output.resolve()
            source_files = {native_file.resolve(), comparison_file.resolve()}
            for receipt_path in (native_receipt_path, candidate_receipt_path):
                if receipt_path is not None and receipt_path.exists():
                    source_files.add(receipt_path.resolve())
        except OSError:
            raise MetricsExportError(
                "could not resolve input and output paths"
            ) from None
        if native_file.resolve() == comparison_file.resolve():
            raise MetricsExportError(
                "native and comparison inputs must be different event files"
            )
        if output in source_files:
            raise MetricsExportError(
                "the JSON output must not replace a source event or launch receipt"
            )

        probe_first_step, probe_last_step = PROBE_FIRST_STEP, PROBE_LAST_STEP
        if args.phase == "model-probe":
            probe_first_step, probe_last_step = load_model_probe_window(
                args.manifest, args.case
            )
        native = read_event_data(args.native_dir)
        comparison = read_event_data(comparison_dir)

        if args.require_validation and not (
            native.scalars.get(VALIDATION_LOSS_METRIC)
            and comparison.scalars.get(VALIDATION_LOSS_METRIC)
        ):
            raise MetricsExportError(
                "--require-validation needs held-out validation metrics from both arms"
            )

        if provenance_binding.get("status") == "bound":
            validation_expected = provenance_binding.get("validation") is True
            native_has_validation = bool(native.scalars.get(VALIDATION_LOSS_METRIC))
            candidate_has_validation = bool(
                comparison.scalars.get(VALIDATION_LOSS_METRIC)
            )
            receipt_metric_error: str | None = None
            if validation_expected and not (
                native_has_validation and candidate_has_validation
            ):
                receipt_metric_error = (
                    "validated launch receipts require held-out validation metrics "
                    "from both arms"
                )
            elif not validation_expected and (
                native_has_validation or candidate_has_validation
            ):
                receipt_metric_error = (
                    "event metrics contain validation data but the launch receipts "
                    "do not enable validation"
                )
            if receipt_metric_error is not None:
                if not args.allow_unbound_receipts:
                    raise MetricsExportError(receipt_metric_error)
                provenance_binding = unbound_pair_document(
                    manifest_sha256,
                    f"diagnostic override: {receipt_metric_error}",
                )

        document = build_document(
            case=args.case,
            phase=args.phase,
            torchtitan_commit=commit,
            native=native,
            polynomial=comparison,
            candidate_name=candidate_name,
            probe_first_step=probe_first_step,
            probe_last_step=probe_last_step,
            provenance_binding=provenance_binding,
        )
        atomic_write_json(args.output, document)
    except MetricsExportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("Wrote paired TorchTitan metrics JSON.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plan or execute one public TorchTitan B1--B5 run.

The default is deliberately a dry run. Passing ``--execute`` is required to
start a job, and large multi-node runs still require the caller's ordinary
TorchElastic rendezvous or scheduler arguments.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sfu_repro.torchtitan.pins import (  # noqa: E402
    DEEPSEEK_TOKENIZER_REPOSITORY,
    DEEPSEEK_TOKENIZER_REVISION,
    LLAMA_TOKENIZER_REPOSITORY,
    LLAMA_TOKENIZER_REVISION,
)
from sfu_repro.torchtitan.assets import verify_tokenizer_assets  # noqa: E402
from sfu_repro.torchtitan.provenance import (  # noqa: E402
    RECEIPT_ARTIFACT_TYPE,
    RECEIPT_FILENAME,
    RECEIPT_SCHEMA_VERSION,
    ReceiptError,
    normalize_repository_path,
    sanitize_command,
    sha256_bytes,
    write_receipt_once,
)


DEFAULT_MANIFEST = REPOSITORY_ROOT / "configs/torchtitan/paper_runs.json"

_SENSITIVE_CONFIG_KEYS = {
    "access_key",
    "api_key",
    "auth_token",
    "credential",
    "credentials",
    "password",
    "private_key",
    "secret",
    "token",
}
_HOST_CONFIG_KEYS = {
    "endpoint",
    "host",
    "hostname",
    "local_addr",
    "master_addr",
    "rdzv_endpoint",
}


def load_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("unsupported TorchTitan run-manifest schema")
    return document


def git_checkout_state(path: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return revision, bool(status.strip())
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot inspect Git checkout at {path}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise RuntimeError("cannot hash a provenance input") from error
    return digest.hexdigest()


def _checkpoint_tree_identity(path: Path) -> dict[str, Any]:
    """Hash every regular file in one seed checkpoint deterministically."""

    try:
        root = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("seed checkpoint does not exist") from error
    if not root.is_dir():
        raise RuntimeError("seed checkpoint must be a directory")

    aggregate = hashlib.sha256()
    aggregate.update(b"sfu-checkpoint-tree-v1\0")
    file_count = 0
    total_bytes = 0
    try:
        entries = sorted(
            root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
        )
        for entry in entries:
            relative = entry.relative_to(root).as_posix()
            if entry.is_symlink():
                raise RuntimeError("seed checkpoint must not contain symbolic links")
            if not entry.is_file():
                continue
            size = entry.stat().st_size
            digest = _sha256_file(entry)
            aggregate.update(relative.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(str(size).encode("ascii"))
            aggregate.update(b"\0")
            aggregate.update(digest.encode("ascii"))
            aggregate.update(b"\0")
            file_count += 1
            total_bytes += size
    except OSError as error:
        raise RuntimeError("could not hash the seed checkpoint") from error
    if file_count == 0:
        raise RuntimeError("seed checkpoint contains no files")
    return {
        "path": normalize_repository_path(str(root), REPOSITORY_ROOT),
        "tree_hash_algorithm": "sha256-path-size-content-v1",
        "sha256": aggregate.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _read_config(config: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = config.read_bytes()
        document = tomllib.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(
            "cannot read the selected TorchTitan configuration"
        ) from error
    if not isinstance(document, dict):
        raise RuntimeError("the selected TorchTitan configuration is invalid")
    return payload, document


def _config_contains_sensitive_values(document: dict[str, Any]) -> bool:
    pending: list[dict[str, Any]] = [document]
    while pending:
        current = pending.pop()
        for key, value in current.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in (
                _SENSITIVE_CONFIG_KEYS | _HOST_CONFIG_KEYS
            ) and value not in (None, "", False):
                return True
            if isinstance(value, dict):
                pending.append(value)
            elif isinstance(value, str) and "://" in value:
                # Exact config bytes are embedded in the receipt. Refuse
                # endpoint-bearing configs rather than leak their hostnames.
                return True
    return False


def _runtime_versions() -> dict[str, Any]:
    distributions: dict[str, str | None] = {}
    for distribution in (
        "torch",
        "torchao",
        "torchtitan",
        "flash-attn",
        "sfu-repro",
        "sfu-spline-ops",
    ):
        try:
            distributions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            distributions[distribution] = None
    torch_runtime: dict[str, str | None] = {
        "version": distributions["torch"],
        "cuda_build": None,
        "git_revision": None,
    }
    try:
        import torch

        torch_runtime["version"] = str(torch.__version__)
        torch_runtime["cuda_build"] = torch.version.cuda
        torch_runtime["git_revision"] = getattr(torch.version, "git_version", None)
    except Exception:
        # The actual TorchTitan invocation will report an import failure. The
        # receipt still records every distribution version available before it.
        pass
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "distributions": distributions,
        "torch": torch_runtime,
    }


def _command_option(arguments: list[str], name: str, default: str | None) -> str | None:
    prefix = f"--{name}="
    selected = default
    for argument in arguments:
        if argument.startswith(prefix):
            selected = argument.removeprefix(prefix)
    return selected


def _topology_from_command(command: list[str]) -> dict[str, int]:
    try:
        nproc_per_node = int(
            next(
                item.split("=", 1)[1]
                for item in command
                if item.startswith("--nproc-per-node=")
            )
        )
        nnodes = int(
            next(
                item.split("=", 1)[1]
                for item in command
                if item.startswith("--nnodes=")
            )
        )
    except (StopIteration, ValueError, IndexError) as error:
        raise RuntimeError("could not resolve TorchTitan launch topology") from error
    return {
        "nproc_per_node": nproc_per_node,
        "nnodes": nnodes,
        "world_size": nproc_per_node * nnodes,
    }


def _dump_folder(
    command: list[str],
    config_document: dict[str, Any],
) -> Path:
    try:
        configured = str(config_document["job"]["dump_folder"])
    except (KeyError, TypeError) as error:
        raise RuntimeError("TorchTitan config has no job.dump_folder") from error
    selected = _command_option(command, "job.dump-folder", configured)
    if selected is None:
        raise RuntimeError("TorchTitan output folder could not be resolved")
    path = Path(selected)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _tokenizer_manifest_digest(path: Path) -> str:
    return _sha256_file(path / "tokenizer-manifest.json")


def _selected_dataset_protocol(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    command: list[str],
    config_document: dict[str, Any],
    torchtitan_revision: str,
) -> tuple[dict[str, Any], list[str]]:
    training = config_document.get("training", {})
    validation = config_document.get("validation", {})
    training_name = _command_option(
        command,
        "training.dataset",
        str(training.get("dataset", "")),
    )
    training_path = _command_option(
        command,
        "training.dataset-path",
        str(training.get("dataset_path", "")) or None,
    )
    result: dict[str, Any] = {
        "training": {"name": training_name},
        "validation": None,
    }
    reasons: list[str] = []
    pins = manifest.get("dataset_snapshots", {})
    if training_name == "sfu_slimpajama":
        result["training"]["pin"] = pins.get("b1_b3_training_and_all_validation")
    elif training_name == "sfu_olmo_mix_1124":
        result["training"]["pin"] = pins.get("b4_training")
    elif training_name == "c4_test":
        result["training"].update(
            {
                "repository_path": normalize_repository_path(
                    str(training_path or ""), REPOSITORY_ROOT
                ),
                "source": {
                    "submodule": "torchtitan",
                    "revision": torchtitan_revision,
                },
            }
        )
    elif args.phase != "seed-checkpoint":
        reasons.append("training dataset is outside the public protocol")

    if args.validation:
        validation_name = _command_option(
            command,
            "validation.dataset",
            str(validation.get("dataset", "")),
        )
        result["validation"] = {
            "name": validation_name,
            "pin": pins.get("b1_b3_training_and_all_validation"),
        }
        if validation_name != "sfu_slimpajama_validation":
            reasons.append("validation dataset is outside the public protocol")
    return result, reasons


def build_launch_receipt(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    command: list[str],
    config: Path,
    config_payload: bytes,
    config_document: dict[str, Any],
    root_state: tuple[str, bool],
    torchtitan_state: tuple[str, bool],
    flash_attention_state: tuple[str, bool],
    selected_assets: Path,
    tokenizer_manifest_sha256: str | None,
    tokenizer_verified: bool,
    checkpoint_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one deterministic receipt shared by all wrappers for a run arm."""

    receipt_command = [Path(command[0]).name, *command[1:]]
    sanitized_command, redactions, secret_redacted = sanitize_command(
        receipt_command,
        repository_root=REPOSITORY_ROOT,
    )
    try:
        module_index = sanitized_command.index("-m")
    except ValueError as error:
        raise RuntimeError("TorchTitan command has no module entry point") from error
    effective_arguments = sanitized_command[module_index + 2 :]
    config_argument_index = next(
        (
            index
            for index, item in enumerate(effective_arguments)
            if item.startswith("--job.config-file=")
        ),
        None,
    )
    if config_argument_index is None:
        raise RuntimeError("TorchTitan command has no config argument")
    effective_overrides = effective_arguments[config_argument_index + 1 :]

    root_revision, root_dirty = root_state
    torchtitan_revision, torchtitan_dirty = torchtitan_state
    flash_attention_revision, flash_attention_dirty = flash_attention_state
    dataset_protocol, dataset_reasons = _selected_dataset_protocol(
        args=args,
        manifest=manifest,
        command=command,
        config_document=config_document,
        torchtitan_revision=torchtitan_revision,
    )
    tokenizer_key = "b4" if args.case == "b4" else "b1_b3"
    tokenizer_pin = manifest["tokenizer_snapshots"][tokenizer_key]

    unbound_reasons = list(dataset_reasons)
    if root_dirty:
        unbound_reasons.append("top-level repository was dirty at launch")
    if torchtitan_dirty or flash_attention_dirty:
        unbound_reasons.append("a required source submodule was dirty at launch")
    if torchtitan_revision != manifest["torchtitan_commit"]:
        unbound_reasons.append("TorchTitan revision differs from the public pin")
    if flash_attention_revision != manifest["flash_attention_commit"]:
        unbound_reasons.append("FlashAttention revision differs from the public pin")
    if not tokenizer_verified:
        unbound_reasons.append("tokenizer assets were not verified")
    if args.allow_world_size_change:
        unbound_reasons.append("world size differs from the public protocol")
    if args.validation:
        validation_config = config_document.get("validation", {})
        if args.validation_frequency != int(
            validation_config.get("freq", 10000)
        ) or args.validation_steps != int(validation_config.get("steps", 64)):
            unbound_reasons.append(
                "validation schedule differs from the public protocol"
            )
    non_output_config_args = [
        item for item in args.config_arg if not item.startswith("--job.dump-folder=")
    ]
    if non_output_config_args:
        unbound_reasons.append("non-output config overrides alter the public protocol")
    if secret_redacted:
        unbound_reasons.append("a secret-bearing command option was redacted")

    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "artifact_type": RECEIPT_ARTIFACT_TYPE,
        "binding": {
            "status": "bound" if not unbound_reasons else "unbound",
            "reasons": sorted(set(unbound_reasons)),
        },
        "selection": {
            "case": args.case,
            "variant": args.variant,
            "phase": args.phase,
            "validation": bool(args.validation),
        },
        "protocol": {
            "manifest_sha256": _sha256_file(manifest_path),
            "dataset": dataset_protocol,
            "seed_checkpoint": checkpoint_input,
            "tokenizer": {
                "repository": tokenizer_pin["repository"],
                "revision": tokenizer_pin["revision"],
                "asset_path": normalize_repository_path(
                    str(selected_assets), REPOSITORY_ROOT
                ),
                "manifest_sha256": tokenizer_manifest_sha256,
                "verified": tokenizer_verified,
            },
        },
        "config": {
            "repository_path": normalize_repository_path(
                str(config.resolve()), REPOSITORY_ROOT
            ),
            "sha256": sha256_bytes(config_payload),
            "bytes_base64": base64.b64encode(config_payload).decode("ascii"),
        },
        "command": {
            "sanitized_argv": sanitized_command,
            "effective_torchtitan_arguments": effective_arguments,
            "effective_overrides": effective_overrides,
            "redactions": redactions,
        },
        "source": {
            "repository": {
                "revision": root_revision,
                "dirty": root_dirty,
            },
            "submodules": {
                "torchtitan": {
                    "revision": torchtitan_revision,
                    "dirty": torchtitan_dirty,
                },
                "flash-attention": {
                    "revision": flash_attention_revision,
                    "dirty": flash_attention_dirty,
                },
            },
        },
        "runtime": _runtime_versions(),
        "topology": _topology_from_command(command),
        "output": {
            "dump_folder": normalize_repository_path(
                str(_dump_folder(command, config_document)), REPOSITORY_ROOT
            )
        },
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _effective_integer_override(arguments: list[str], name: str, default: int) -> int:
    prefix = f"--{name}="
    selected = default
    for argument in arguments:
        if argument.startswith(prefix):
            try:
                selected = int(argument.removeprefix(prefix))
            except ValueError as error:
                raise ValueError(f"{name} override must be an integer") from error
    return selected


def _effective_text_override(arguments: list[str], name: str, default: str) -> str:
    prefix = f"--{name}="
    selected = default
    for argument in arguments:
        if argument.startswith(prefix):
            selected = argument.removeprefix(prefix)
    return selected


def tokenizer_provenance(case: str) -> tuple[Path, str, str]:
    if case == "b4":
        return (
            REPOSITORY_ROOT / "assets/hf/DeepSeek-V3.1-Base",
            DEEPSEEK_TOKENIZER_REPOSITORY,
            DEEPSEEK_TOKENIZER_REVISION,
        )
    return (
        REPOSITORY_ROOT / "assets/hf/Llama-3.1-8B",
        LLAMA_TOKENIZER_REPOSITORY,
        LLAMA_TOKENIZER_REVISION,
    )


def _override(name: str, value: object) -> str:
    if isinstance(value, bool):
        if value:
            return f"--{name}"
        section, separator, field = name.rpartition(".")
        prefix = f"{section}." if separator else ""
        return f"--{prefix}no-{field}"
    return f"--{name}={value}"


def build_command(
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> tuple[list[str], Path]:
    try:
        case = manifest["cases"][args.case]
        config = REPOSITORY_ROOT / case["configs"][args.variant]
    except KeyError as error:
        raise ValueError(f"unknown case/variant selection: {error}") from error
    if args.case == "b5" and args.phase != "model-probe":
        raise ValueError("B5 defines only the new public model-probe workflow")
    if args.phase == "pretraining" and args.seed_checkpoint is None:
        raise ValueError(
            "pretraining requires --seed-checkpoint so paired arms share exact weights"
        )

    if args.phase == "pretraining":
        phase = case["pretraining"]
        default_world_size = int(phase["world_size"])
    elif args.phase == "model-probe":
        phase = case["model_probe"]
        default_world_size = int(phase["world_size"])
    else:
        phase = {}
        default_world_size = 1

    nproc_per_node = args.nproc_per_node or min(8, default_world_size)
    nnodes = args.nnodes or max(1, default_world_size // nproc_per_node)
    world_size = nproc_per_node * nnodes
    if world_size != default_world_size and not args.allow_world_size_change:
        raise ValueError(
            f"{args.case} {args.phase} expects world size {default_world_size}, "
            f"got {nproc_per_node} x {nnodes}; pass --allow-world-size-change "
            "for a new configuration"
        )
    if args.phase in {"pretraining", "model-probe"}:
        local_batch_size = _effective_integer_override(
            args.config_arg,
            "training.local-batch-size",
            int(phase["local_batch_size"]),
        )
        global_batch_size = _effective_integer_override(
            args.config_arg,
            "training.global-batch-size",
            int(phase["global_batch_size"]),
        )
        if local_batch_size <= 0 or global_batch_size <= 0:
            raise ValueError("training batch sizes must be positive")
        batch_denominator = local_batch_size * world_size
        if global_batch_size % batch_denominator:
            raise ValueError(
                "training.global-batch-size must be divisible by "
                "training.local-batch-size times world size "
                f"({global_batch_size} % {batch_denominator} != 0)"
            )
        if args.case == "b4":
            expert_parallel_degree = _effective_integer_override(
                args.config_arg,
                "parallelism.expert-parallel-degree",
                int(phase["expert_parallel_degree"]),
            )
            if expert_parallel_degree <= 0 or world_size % expert_parallel_degree:
                raise ValueError(
                    "B4 world size must be divisible by expert-parallel degree "
                    f"({world_size} % {expert_parallel_degree} != 0)"
                )

    command = [
        args.torchrun,
        f"--nproc-per-node={nproc_per_node}",
        f"--nnodes={nnodes}",
    ]
    if nnodes == 1:
        command.append("--standalone")
    elif args.node_rank is not None:
        command.append(f"--node-rank={args.node_rank}")
    command.extend(args.torchrun_arg)
    command.extend(
        [
            "-m",
            "torchtitan.train",
            f"--job.config-file={config}",
        ]
    )

    if args.phase == "model-probe":
        dataset_path = REPOSITORY_ROOT / phase.get(
            "dataset_path", "torchtitan/tests/assets/c4_test"
        )
        command.extend(
            [
                _override("training.steps", phase["steps"]),
                _override("lr-scheduler.total-steps", phase["steps"]),
                _override("training.dataset", phase.get("dataset", "c4_test")),
                _override("training.dataset-path", dataset_path),
                _override("training.local-batch-size", phase["local_batch_size"]),
                _override("training.global-batch-size", phase["global_batch_size"]),
                _override("training.seq-len", phase["sequence_length"]),
                _override("checkpoint.enable", False),
                _override("validation.enable", False),
                _override("compile.enable", phase.get("compile", True)),
                _override("activation-checkpoint.mode", "none"),
                _override("metrics.log-freq", 1),
                _override(
                    "job.dump-folder",
                    f"./outputs/torchtitan/probes/{args.case}/{args.variant}",
                ),
            ]
        )
        if args.case == "b4":
            command.append(
                _override(
                    "parallelism.expert-parallel-degree",
                    phase["expert_parallel_degree"],
                )
            )
    elif args.phase == "seed-checkpoint":
        if args.variant != "native":
            raise ValueError("create one shared seed checkpoint from the native config")
        model_key = (
            f"{args.case}_deepseek_27a4b"
            if args.case == "b4"
            else f"{args.case}_llama3_8b"
        )
        command.extend(
            [
                _override("checkpoint.enable", True),
                _override("checkpoint.create-seed-checkpoint", True),
                _override("checkpoint.async-mode", "disabled"),
                _override("validation.enable", False),
                _override("compile.enable", False),
                _override("parallelism.data-parallel-shard-degree", 1),
                _override("parallelism.expert-parallel-degree", 1),
                _override("job.dump-folder", f"./outputs/torchtitan/seeds/{model_key}"),
            ]
        )

    if args.validation:
        if args.phase != "pretraining":
            raise ValueError("--validation is supported only for pretraining runs")
        command.extend(
            [
                _override("validation.enable", True),
                _override("validation.freq", args.validation_frequency),
                _override("validation.steps", args.validation_steps),
            ]
        )
    if args.seed_checkpoint:
        if args.phase == "seed-checkpoint":
            raise ValueError("--seed-checkpoint cannot be used while creating one")
        command.append(
            _override("checkpoint.initial-load-path", args.seed_checkpoint.resolve())
        )
    if args.hf_assets_path:
        command.append(_override("model.hf-assets-path", args.hf_assets_path.resolve()))
    command.extend(args.config_arg)
    return command, config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", choices=("b1", "b2", "b3", "b4", "b5"), required=True)
    parser.add_argument(
        "--variant",
        choices=("native", "polynomial", "pwl2_safe_f16", "d2_safe"),
        required=True,
    )
    parser.add_argument(
        "--phase",
        choices=("model-probe", "pretraining", "seed-checkpoint"),
        default="model-probe",
    )
    parser.add_argument("--nproc-per-node", type=positive_int)
    parser.add_argument("--nnodes", type=positive_int)
    parser.add_argument("--node-rank", type=nonnegative_int)
    parser.add_argument("--torchrun", default="torchrun")
    parser.add_argument(
        "--torchrun-arg",
        action="append",
        default=[],
        help="Repeat for scheduler/rendezvous arguments passed before -m.",
    )
    parser.add_argument(
        "--config-arg",
        action="append",
        default=[],
        help="Repeat for an explicit TorchTitan argument passed after the config.",
    )
    parser.add_argument("--hf-assets-path", type=Path)
    parser.add_argument("--seed-checkpoint", type=Path)
    parser.add_argument(
        "--receipt-path",
        type=Path,
        help=(
            "Write the launch receipt here instead of job.dump_folder/"
            f"{RECEIPT_FILENAME}. The file is create-only."
        ),
    )
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--validation-frequency", type=positive_int, default=10000)
    parser.add_argument("--validation-steps", type=positive_int, default=64)
    parser.add_argument("--allow-world-size-change", action="store_true")
    parser.add_argument("--allow-unpinned-torchtitan", action="store_true")
    parser.add_argument("--allow-unpinned-flash-attention", action="store_true")
    parser.add_argument("--allow-dirty-submodules", action="store_true")
    parser.add_argument("--allow-unverified-tokenizer-assets", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the command. Without this flag, print it and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    expected_revision = manifest["torchtitan_commit"]
    actual_revision, torchtitan_dirty = git_checkout_state(
        REPOSITORY_ROOT / "torchtitan"
    )
    if actual_revision != expected_revision and not args.allow_unpinned_torchtitan:
        raise RuntimeError(
            f"TorchTitan revision mismatch: expected {expected_revision}, "
            f"found {actual_revision}"
        )
    expected_fa_revision = manifest["flash_attention_commit"]
    actual_fa_revision, flash_attention_dirty = git_checkout_state(
        REPOSITORY_ROOT / "flash-attention"
    )
    if (
        actual_fa_revision != expected_fa_revision
        and not args.allow_unpinned_flash_attention
    ):
        raise RuntimeError(
            "FlashAttention revision mismatch: expected "
            f"{expected_fa_revision}, found {actual_fa_revision}"
        )
    if (torchtitan_dirty or flash_attention_dirty) and not args.allow_dirty_submodules:
        dirty_names = ", ".join(
            name
            for name, dirty in (
                ("torchtitan", torchtitan_dirty),
                ("flash-attention", flash_attention_dirty),
            )
            if dirty
        )
        raise RuntimeError(
            f"refusing a source-dirty run ({dirty_names}); commit or record the "
            "changes, or pass --allow-dirty-submodules for a new unbound run"
        )
    command, config = build_command(args, manifest)
    if not config.is_file():
        raise FileNotFoundError(config)

    printable_command, _, _ = sanitize_command(
        [Path(command[0]).name, *command[1:]],
        repository_root=REPOSITORY_ROOT,
    )

    selected_assets: Path | None = None
    tokenizer_manifest_sha256: str | None = None
    tokenizer_verified = False

    if args.execute:
        topology = _topology_from_command(command)
        if topology["world_size"] > 1:
            if (
                "--standalone" not in command
                and args.node_rank is None
                and not args.torchrun_arg
            ):
                raise RuntimeError(
                    "multi-node execution needs --node-rank or explicit "
                    "--torchrun-arg rendezvous settings"
                )

        config_payload, config_document = _read_config(config)
        if _config_contains_sensitive_values(config_document):
            raise RuntimeError(
                "refusing to embed a config containing credentials or host endpoints"
            )

        default_assets, expected_repository, expected_asset_revision = (
            tokenizer_provenance(args.case)
        )
        selected_assets = (
            args.hf_assets_path.resolve() if args.hf_assets_path else default_assets
        )
        selected_assets = Path(
            _effective_text_override(
                args.config_arg,
                "model.hf-assets-path",
                str(selected_assets),
            )
        ).resolve()
        if not args.allow_unverified_tokenizer_assets:
            tokenizer_digest_before = _tokenizer_manifest_digest(selected_assets)
            verify_tokenizer_assets(
                selected_assets,
                expected_repository=expected_repository,
                expected_revision=expected_asset_revision,
            )
            tokenizer_manifest_sha256 = _tokenizer_manifest_digest(selected_assets)
            if tokenizer_manifest_sha256 != tokenizer_digest_before:
                raise RuntimeError(
                    "tokenizer manifest changed while launch provenance was verified"
                )
            tokenizer_verified = True

        root_state = git_checkout_state(REPOSITORY_ROOT)
        checkpoint_input = (
            _checkpoint_tree_identity(args.seed_checkpoint)
            if args.phase == "pretraining" and args.seed_checkpoint is not None
            else None
        )
        receipt = build_launch_receipt(
            args=args,
            manifest=manifest,
            manifest_path=args.manifest,
            command=command,
            config=config,
            config_payload=config_payload,
            config_document=config_document,
            root_state=root_state,
            torchtitan_state=(actual_revision, torchtitan_dirty),
            flash_attention_state=(actual_fa_revision, flash_attention_dirty),
            selected_assets=selected_assets,
            tokenizer_manifest_sha256=tokenizer_manifest_sha256,
            tokenizer_verified=tokenizer_verified,
            checkpoint_input=checkpoint_input,
        )
        receipt_path = (
            args.receipt_path.resolve()
            if args.receipt_path is not None
            else _dump_folder(command, config_document) / RECEIPT_FILENAME
        )
        try:
            write_receipt_once(receipt_path, receipt)
        except ReceiptError as error:
            raise RuntimeError(str(error)) from error

    print(shlex.join(printable_command))
    if not args.execute:
        return 0
    environment = os.environ.copy()
    python_paths = [str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT / "torchtitan")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    return subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment).returncode


if __name__ == "__main__":
    sys.exit(main())

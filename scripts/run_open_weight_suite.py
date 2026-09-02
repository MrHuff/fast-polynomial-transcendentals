#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from Graphcore's low-bits-training evaluation launcher. Modified in
# 2026 to use a standalone, declarative protocol and environment-only secrets.

"""Run the paper's open-weight polynomial evaluation cases.

The suite is intentionally just orchestration. ``benchmark_open_weights.py``
contains the model patching and measurement logic. Hugging Face authentication,
when required, is read by that process from ``HF_TOKEN``; credentials are never
accepted as command-line arguments or included in dry-run output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "open_weight_paper.json"

_VARIANT_PATTERNS = (
    re.compile(r"native"),
    re.compile(r"spline_silu_d[3-6]_(?:current|sollya)"),
    re.compile(r"spline_gelu_d[3-6]_(?:current|sollya)"),
    re.compile(r"fused_swiglu_d[3-6]_(?:current|sollya)"),
    re.compile(r"spline_router_sigmoid_d[3-6]_(?:current|sollya)"),
    re.compile(r"gemma_tanh_current"),
    re.compile(r"gemma_fa4_tanh_native"),
    re.compile(r"gemma_fa4_tanh_d[3-6]_(?:current|sollya)"),
)


@dataclass(frozen=True)
class OpenWeightCase:
    key: str
    paper_name: str
    model_id: str
    revision: str | None
    variants: tuple[str, ...]
    attn_implementation: str
    experts_implementation: str | None
    trust_remote_code: bool
    eval_batch_size: str
    throughput_sequence_length: int
    unused_quality_sequence_length_argument: int


@dataclass(frozen=True)
class Protocol:
    batch_size: int
    prefill_measurements: int
    prefill_warmups: int
    decode_steps: int
    decode_measurements: int
    decode_warmups: int
    dtype: str
    quality_tasks: tuple[str, ...]
    quality_fewshot: dict[str, int]


@dataclass(frozen=True)
class SuiteConfig:
    protocol: Protocol
    cases: tuple[OpenWeightCase, ...]


def parse_variant_parts(raw: str) -> tuple[str, ...]:
    """Validate a ``+``-joined benchmark variant without importing GPU code."""

    parts = tuple(part.strip() for part in raw.split("+"))
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(f"invalid empty variant component in {raw!r}")
    for part in parts:
        if not any(pattern.fullmatch(part) for pattern in _VARIANT_PATTERNS):
            raise argparse.ArgumentTypeError(f"unknown variant component {part!r}")
    if "native" in parts and len(parts) != 1:
        raise argparse.ArgumentTypeError("native cannot be combined with another variant")
    return parts


def _positive_int(document: dict[str, Any], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def load_config(path: Path = DEFAULT_CONFIG) -> SuiteConfig:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("open-weight config schema_version must equal 1")

    raw_protocol = document.get("protocol")
    if not isinstance(raw_protocol, dict):
        raise ValueError("protocol must be an object")
    raw_tasks = raw_protocol.get("quality_tasks")
    raw_fewshot = raw_protocol.get("quality_fewshot")
    if not isinstance(raw_tasks, list) or not all(
        isinstance(task, str) and task for task in raw_tasks
    ):
        raise ValueError("protocol.quality_tasks must be a list of task names")
    if not isinstance(raw_fewshot, dict) or not all(
        isinstance(task, str)
        and isinstance(shots, int)
        and not isinstance(shots, bool)
        and shots >= 0
        for task, shots in raw_fewshot.items()
    ):
        raise ValueError("protocol.quality_fewshot must map task names to counts")
    missing_fewshot = set(raw_tasks) - set(raw_fewshot)
    if missing_fewshot:
        raise ValueError(
            "quality_fewshot is missing tasks: " + ", ".join(sorted(missing_fewshot))
        )
    dtype = raw_protocol.get("dtype")
    if dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError("protocol.dtype must be bf16, fp16, or fp32")
    protocol = Protocol(
        batch_size=_positive_int(raw_protocol, "batch_size"),
        prefill_measurements=_positive_int(raw_protocol, "prefill_measurements"),
        prefill_warmups=_positive_int(raw_protocol, "prefill_warmups"),
        decode_steps=_positive_int(raw_protocol, "decode_steps"),
        decode_measurements=_positive_int(raw_protocol, "decode_measurements"),
        decode_warmups=_positive_int(raw_protocol, "decode_warmups"),
        dtype=dtype,
        quality_tasks=tuple(raw_tasks),
        quality_fewshot={task: int(shots) for task, shots in raw_fewshot.items()},
    )

    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases must be a non-empty list")
    cases: list[OpenWeightCase] = []
    observed_keys: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"cases[{index}] must be an object")
        key = raw_case.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"cases[{index}].key must be a non-empty string")
        if key in observed_keys:
            raise ValueError(f"duplicate case key {key!r}")
        observed_keys.add(key)
        raw_variants = raw_case.get("variants")
        if not isinstance(raw_variants, list) or not raw_variants:
            raise ValueError(f"case {key!r} must define variants")
        variants = tuple(str(variant) for variant in raw_variants)
        for variant in variants:
            parse_variant_parts(variant)
        if variants[0] != "native":
            raise ValueError(f"case {key!r} must list native first")
        revision = raw_case.get("revision")
        if revision is not None and not isinstance(revision, str):
            raise ValueError(f"case {key!r} revision must be a string or null")
        cases.append(
            OpenWeightCase(
                key=key,
                paper_name=str(raw_case["paper_name"]),
                model_id=str(raw_case["model_id"]),
                revision=revision,
                variants=variants,
                attn_implementation=str(raw_case["attn_implementation"]),
                experts_implementation=raw_case.get("experts_implementation"),
                trust_remote_code=bool(raw_case.get("trust_remote_code", False)),
                eval_batch_size=str(raw_case["eval_batch_size"]),
                throughput_sequence_length=_positive_int(
                    raw_case, "throughput_sequence_length"
                ),
                unused_quality_sequence_length_argument=_positive_int(
                    raw_case, "unused_quality_sequence_length_argument"
                ),
            )
        )
    return SuiteConfig(protocol=protocol, cases=tuple(cases))


def select_cases(cases: tuple[OpenWeightCase, ...], selector: str) -> list[OpenWeightCase]:
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    if not requested or requested & {"all", "paper"}:
        return list(cases)
    selected = [
        case for case in cases if case.key in requested or case.model_id in requested
    ]
    matched = {case.key for case in selected} | {case.model_id for case in selected}
    unknown = requested - matched
    if unknown:
        valid = ", ".join(case.key for case in cases)
        raise ValueError(f"unknown model selector(s) {sorted(unknown)}; valid keys: {valid}")
    return selected


def parse_fewshot_overrides(values: Sequence[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        task, separator, raw_shots = value.partition("=")
        if not separator or not task:
            raise argparse.ArgumentTypeError(
                f"invalid --eval-task-fewshot {value!r}; expected TASK=SHOTS"
            )
        try:
            shots = int(raw_shots)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid few-shot count in {value!r}"
            ) from error
        if shots < 0:
            raise argparse.ArgumentTypeError("few-shot counts must be non-negative")
        parsed[task] = shots
    return parsed


def eval_tasks(args: argparse.Namespace, protocol: Protocol) -> tuple[str, ...]:
    if args.no_eval:
        return ()
    if args.quality_eval:
        return protocol.quality_tasks
    return tuple(task.strip() for task in args.eval_tasks.split(",") if task.strip())


def sequence_length(case: OpenWeightCase, args: argparse.Namespace) -> int:
    if args.seq_len is not None:
        return args.seq_len
    if args.mode == "eval":
        # lm-eval does not consume this synthetic-token argument. It is retained
        # in output metadata solely to reconstruct the historical invocation.
        return case.unused_quality_sequence_length_argument
    return case.throughput_sequence_length


def output_path(
    case: OpenWeightCase,
    args: argparse.Namespace,
    protocol: Protocol,
    output_dir: Path,
) -> Path:
    tasks = eval_tasks(args, protocol)
    suffix = "quality" if tasks else "throughput"
    return output_dir / f"{case.key}_{args.mode}_seq{sequence_length(case, args)}_{suffix}.json"


def build_command(
    case: OpenWeightCase,
    args: argparse.Namespace,
    protocol: Protocol,
    output_dir: Path,
) -> list[str]:
    batch_size = args.batch_size or protocol.batch_size
    steps = args.steps or protocol.prefill_measurements
    warmup = args.warmup or protocol.prefill_warmups
    decode_steps = args.decode_steps or protocol.decode_steps
    decode_repeats = args.decode_repeats or protocol.decode_measurements
    decode_warmup = args.decode_warmup or protocol.decode_warmups
    dtype = args.dtype or protocol.dtype
    command = [
        args.python,
        "scripts/benchmark_open_weights.py",
        "--model",
        case.model_id,
        "--attn-implementation",
        case.attn_implementation,
        "--mode",
        args.mode,
        "--batch-size",
        str(batch_size),
        "--seq-len",
        str(sequence_length(case, args)),
        "--steps",
        str(steps),
        "--warmup",
        str(warmup),
        "--decode-steps",
        str(decode_steps),
        "--decode-repeats",
        str(decode_repeats),
        "--decode-warmup",
        str(decode_warmup),
        "--dtype",
        dtype,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--json-out",
        str(output_path(case, args, protocol, output_dir)),
    ]
    if case.revision:
        command.extend(("--revision", case.revision))
    if case.experts_implementation:
        command.extend(("--experts-implementation", case.experts_implementation))
    if case.trust_remote_code:
        command.append("--trust-remote-code")
    if args.cache_dir:
        command.extend(("--cache-dir", str(args.cache_dir)))

    tasks = eval_tasks(args, protocol)
    if tasks:
        command.extend(
            (
                "--eval-tasks",
                ",".join(tasks),
                "--eval-num-fewshot",
                str(args.eval_num_fewshot),
                "--eval-batch-size",
                args.eval_batch_size or case.eval_batch_size,
            )
        )
        if args.eval_limit is not None:
            command.extend(("--eval-limit", str(args.eval_limit)))
        fewshot = dict(protocol.quality_fewshot) if args.quality_eval else {}
        fewshot.update(parse_fewshot_overrides(args.eval_task_fewshot))
        for task in tasks:
            if task in fewshot:
                command.extend(("--eval-task-fewshot", f"{task}={fewshot[task]}"))
        if args.eval_log_samples:
            command.append("--eval-log-samples")

    variants = tuple(args.variants) if args.variants else case.variants
    for variant in variants:
        parse_variant_parts(variant)
        command.extend(("--variant", variant))
    return command


def completed_output(path: Path, expected_variants: tuple[str, ...]) -> bool:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    rows = document.get("results") if isinstance(document, dict) else document
    if not isinstance(rows, list):
        return False
    successful = {
        row.get("variant")
        for row in rows
        if isinstance(row, dict) and row.get("variant") and not row.get("error")
    }
    return set(expected_variants).issubset(successful)


def run_case(
    case: OpenWeightCase,
    args: argparse.Namespace,
    protocol: Protocol,
    output_dir: Path,
) -> int:
    variants = tuple(args.variants) if args.variants else case.variants
    destination = output_path(case, args, protocol, output_dir)
    if args.skip_existing and completed_output(destination, variants):
        print(f"[skip] {case.key}: {destination}", flush=True)
        return 0
    command = build_command(case, args, protocol, output_dir)
    if args.dry_run:
        print(shlex.join(command), flush=True)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.cache_dir:
        cache_dir = args.cache_dir.resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        environment["HF_HOME"] = str(cache_dir)
        environment["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    print(f"[run] {case.key} ({case.model_id})", flush=True)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    )
    if completed.returncode and not args.keep_going:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--models",
        default="paper",
        help="Comma-separated case keys or model IDs; 'paper' and 'all' select all cases.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Override configured variants for every selected case; repeat as needed.",
    )
    parser.add_argument(
        "--mode", choices=("prefill", "decode", "both", "eval"), default="both"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--decode-steps", type=int)
    parser.add_argument("--decode-repeats", type=int)
    parser.add_argument("--decode-warmup", type=int)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Hugging Face cache directory on a filesystem with sufficient capacity.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--quality-eval",
        action="store_true",
        help="Run the complete paper quality task list and per-task few-shot settings.",
    )
    parser.add_argument("--eval-tasks", default="")
    parser.add_argument("--eval-num-fewshot", type=int, default=0)
    parser.add_argument(
        "--eval-task-fewshot", action="append", default=[], metavar="TASK=SHOTS"
    )
    parser.add_argument("--eval-limit", type=float)
    parser.add_argument("--eval-batch-size")
    parser.add_argument("--eval-log-samples", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/open_weight"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    if args.no_eval and (args.quality_eval or args.eval_tasks):
        raise SystemExit("--no-eval cannot be combined with an evaluation task selection")
    if args.mode == "eval" and not (args.quality_eval or args.eval_tasks):
        raise SystemExit("--mode eval requires --quality-eval or --eval-tasks")
    for name in (
        "batch_size",
        "seq_len",
        "steps",
        "warmup",
        "decode_steps",
        "decode_repeats",
        "decode_warmup",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    try:
        suite = load_config(args.config.resolve())
        cases = select_cases(suite.cases, args.models)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid open-weight config: {error}") from error

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    output_dir = output_dir.resolve()
    failed = False
    for case in cases:
        try:
            failed |= bool(run_case(case, args, suite.protocol, output_dir))
        except subprocess.CalledProcessError as error:
            failed = True
            if not args.keep_going:
                raise SystemExit(error.returncode) from error
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

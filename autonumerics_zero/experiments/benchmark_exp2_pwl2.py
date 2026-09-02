#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified in 2026 for the standalone fast-polynomial-transcendentals release.

"""Build and run the pure-CUDA two-piece exp2 benchmark.

This intentionally invokes NVCC directly. No Triton-generated kernel is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "autonumerics_zero/cuda_benchmarks/benchmark_exp2_pwl2.cu"
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sfu_repro.source_attestation import safe_command  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repository: Path = REPO_ROOT) -> dict[str, object]:
    def git(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        value = completed.stdout.strip()
        return value if completed.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    entries = tuple(item for item in (status or "").split("\0") if item)
    return {
        "repository": "https://github.com/MrHuff/fast-polynomial-transcendentals",
        "revision": revision if revision is not None and len(revision) == 40 else None,
        "dirty": bool(entries) if status is not None else None,
        "untracked_files": (
            sum(item.startswith("?? ") for item in entries)
            if status is not None
            else None
        ),
    }


def parse_value(value: str):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_record(line: str) -> dict[str, object]:
    fields = {}
    for token in shlex.split(line)[1:]:
        key, value = token.split("=", 1)
        fields[key] = parse_value(value)
    return fields


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nvcc", type=Path, default=Path("/usr/local/cuda-13.0/bin/nvcc")
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--allow-unbound-source",
        action="store_true",
        help="Permit a diagnostic measurement from a dirty or unversioned tree.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    args = get_parser().parse_args(argv)
    repository_state = git_state()
    source_bound = bool(
        repository_state.get("revision") is not None
        and repository_state.get("dirty") is False
    )
    if not source_bound and not args.allow_unbound_source:
        raise RuntimeError(
            "the repository is dirty or unversioned; use a clean checkout or "
            "--allow-unbound-source for a diagnostic measurement"
        )
    nvcc = args.nvcc.resolve()
    if not nvcc.is_file():
        raise RuntimeError("--nvcc must identify an existing compiler binary")

    compile_command = [
        str(nvcc),
        "-O3",
        "-std=c++17",
        "-arch=sm_100",
        "-lineinfo",
        str(SOURCE),
    ]
    with tempfile.TemporaryDirectory(prefix="exp2-pwl2-") as temporary_dir:
        binary = Path(temporary_dir) / "benchmark_exp2_pwl2"
        compile_command.extend(["-o", str(binary)])
        subprocess.run(compile_command, cwd=REPO_ROOT, check=True)
        binary_sha256 = sha256_file(binary)
        completed = subprocess.run(
            [str(binary)],
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": args.device,
            },
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )

    print(completed.stdout, end="")
    nvcc_version = subprocess.run(
        [str(nvcc), "--version"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "experiment": {
            "id": "fractional-exp2-isolated",
            "provenance_class": (
                "new-measurement" if source_bound else "diagnostic-unbound-source"
            ),
            "command": safe_command(
                [
                    Path(sys.executable).name,
                    Path(__file__).resolve().relative_to(REPO_ROOT),
                    *raw_arguments,
                ],
                REPO_ROOT,
            ),
        },
        "source": {
            **repository_state,
            "input_sha256": {
                str(SOURCE.relative_to(REPO_ROOT)): sha256_file(SOURCE),
                "autonumerics_zero/experiments/benchmark_exp2_pwl2.py": sha256_file(
                    Path(__file__).resolve()
                ),
            },
            "compiled_binary_sha256": binary_sha256,
        },
        "environment": {
            "python": platform.python_version(),
            "nvcc": nvcc_version,
            "nvcc_sha256": sha256_file(nvcc),
            "gpu": None,
        },
        "measurement": {
            "architecture": "sm_100",
            "summary_statistic": "median",
            "order_policy": "sequential source order",
            "measurement_order": [],
        },
        "protocol": {
            "device": args.device,
            "compile_command": safe_command(
                compile_command[:-2] + ["-o", "<temporary-binary>"], REPO_ROOT
            ),
            "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        },
        "results": {
            "errors": [],
            "timings": [],
        },
    }
    pending_samples: list[float] | None = None
    for line in completed.stdout.splitlines():
        if line.startswith("GPU "):
            payload["environment"]["gpu"] = line.removeprefix("GPU ")
        elif line.startswith("PROTOCOL "):
            payload["measurement"].update(parse_record(line))
        elif line.startswith("SAMPLES "):
            if pending_samples is not None:
                raise RuntimeError("benchmark emitted consecutive timing sample rows")
            raw_samples = parse_record(line).get("ms")
            if not isinstance(raw_samples, str):
                raise RuntimeError("benchmark emitted an invalid timing sample row")
            pending_samples = [float(value) for value in raw_samples.split(",")]
        elif line.startswith("ERROR "):
            payload["results"]["errors"].append(parse_record(line))
        elif line.startswith("RESULT "):
            if pending_samples is None:
                raise RuntimeError("benchmark result is missing raw timing samples")
            record = parse_record(line)
            expected_samples = payload["measurement"].get("timing_samples")
            if expected_samples != len(pending_samples):
                raise RuntimeError("benchmark timing sample count is inconsistent")
            record["samples_ms"] = pending_samples
            record["measurement_index"] = len(payload["results"]["timings"])
            payload["results"]["timings"].append(record)
            payload["measurement"]["measurement_order"].append(
                "/".join(str(record[key]) for key in ("regime", "datatype", "variant"))
            )
            pending_samples = None

    if pending_samples is not None:
        raise RuntimeError("benchmark emitted raw samples without a result row")
    if not payload["results"]["timings"]:
        raise RuntimeError("benchmark emitted no timing results")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"JSON path={args.json_out}")


if __name__ == "__main__":
    main()

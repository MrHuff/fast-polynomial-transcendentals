#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare the revision-pinned tokenizer assets used by TorchTitan configs.

The default invocation is an offline dry run.  ``--execute`` is required before
this script imports :mod:`huggingface_hub` or contacts the Hub.  Authentication
is intentionally left to huggingface_hub's normal ``HF_TOKEN`` and configured
provider mechanisms; credentials are never accepted as command-line options.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence


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
from sfu_repro.torchtitan.assets import sha256_file  # noqa: E402


MANIFEST_NAME = "tokenizer-manifest.json"
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_TOKENIZER_BASENAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
)


class AssetPreparationError(RuntimeError):
    """A safe-to-display tokenizer preparation failure."""


@dataclass(frozen=True)
class TokenizerAsset:
    """One immutable Hub snapshot and its TorchTitan destination."""

    key: str
    repository: str
    revision: str
    destination: Path
    files: tuple[str, ...]


TOKENIZER_ASSETS = (
    TokenizerAsset(
        key="llama3",
        repository=LLAMA_TOKENIZER_REPOSITORY,
        revision=LLAMA_TOKENIZER_REVISION,
        destination=Path("assets/hf/Llama-3.1-8B"),
        files=(
            "original/tokenizer.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    ),
    TokenizerAsset(
        key="deepseek-v3",
        repository=DEEPSEEK_TOKENIZER_REPOSITORY,
        revision=DEEPSEEK_TOKENIZER_REVISION,
        destination=Path("assets/hf/DeepSeek-V3.1-Base"),
        files=(
            "assets/chat_template.jinja",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    ),
)


ListRepoFiles = Callable[..., Iterable[str]]
HubDownload = Callable[..., str]


class SafeArgumentParser(argparse.ArgumentParser):
    """Do not reflect arbitrary command-line text back to stderr."""

    def error(self, message: str) -> None:
        del message
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid arguments\n")


def _validate_asset(asset: TokenizerAsset) -> None:
    if _COMMIT_SHA.fullmatch(asset.revision) is None:
        raise ValueError(f"{asset.key}: revision must be a full lowercase commit SHA")
    if not asset.repository or "/" not in asset.repository:
        raise ValueError(f"{asset.key}: invalid Hugging Face repository")
    if asset.destination.is_absolute() or ".." in asset.destination.parts:
        raise ValueError(f"{asset.key}: destination must stay within the repository")
    if not asset.files:
        raise ValueError(f"{asset.key}: no tokenizer files declared")
    for filename in asset.files:
        remote_path = PurePosixPath(filename)
        if (
            remote_path.is_absolute()
            or ".." in remote_path.parts
            or remote_path.name not in _TOKENIZER_BASENAMES
        ):
            raise ValueError(f"{asset.key}: unsafe tokenizer path {filename!r}")


for _asset in TOKENIZER_ASSETS:
    _validate_asset(_asset)


def _write_manifest(asset: TokenizerAsset, destination: Path) -> Path:
    files = []
    for filename in asset.files:
        local_path = destination.joinpath(*PurePosixPath(filename).parts)
        if not local_path.is_file():
            raise AssetPreparationError(
                f"{asset.key}: Hub download did not create {filename}"
            )
        files.append(
            {
                "path": filename,
                "sha256": sha256_file(local_path),
            }
        )

    document = {
        "schema_version": 1,
        "asset_type": "tokenizer",
        "repository": asset.repository,
        "revision": asset.revision,
        "files": files,
    }
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / MANIFEST_NAME
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination,
            prefix=f".{MANIFEST_NAME}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return manifest_path


def prepare_asset(
    asset: TokenizerAsset,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    list_repo_files: ListRepoFiles,
    hf_hub_download: HubDownload,
) -> Path:
    """Download one exact tokenizer snapshot and write its digest manifest."""

    try:
        available_files = tuple(
            list_repo_files(repo_id=asset.repository, revision=asset.revision)
        )
    except Exception as error:
        raise AssetPreparationError(
            f"{asset.key}: could not inspect the pinned Hub snapshot "
            f"({type(error).__name__})"
        ) from None

    missing = [filename for filename in asset.files if filename not in available_files]
    if missing:
        raise AssetPreparationError(
            f"{asset.key}: pinned snapshot is missing declared tokenizer files: "
            + ", ".join(missing)
        )

    destination = repository_root / asset.destination
    destination.mkdir(parents=True, exist_ok=True)
    # Never leave an older attestation in place while files are being updated.
    # If a later download fails, the partial directory is visibly unmanifested.
    (destination / MANIFEST_NAME).unlink(missing_ok=True)
    for filename in asset.files:
        try:
            hf_hub_download(
                repo_id=asset.repository,
                filename=filename,
                revision=asset.revision,
                local_dir=str(destination),
            )
        except Exception as error:
            raise AssetPreparationError(
                f"{asset.key}: Hub download failed for {filename} "
                f"({type(error).__name__})"
            ) from None

    return _write_manifest(asset, destination)


def _hub_functions() -> tuple[ListRepoFiles, HubDownload]:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise AssetPreparationError(
            "execution requires huggingface_hub in the active environment"
        ) from None
    return list_repo_files, hf_hub_download


def _selected_assets(keys: Sequence[str]) -> tuple[TokenizerAsset, ...]:
    selected = set(keys)
    return tuple(
        asset for asset in TOKENIZER_ASSETS if not selected or asset.key in selected
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = SafeArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(asset.key for asset in TOKENIZER_ASSETS),
        default=[],
        help="Prepare one model; repeat as needed. The default selects both.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Contact Hugging Face and prepare files. The default is a dry run.",
    )
    parser.epilog = (
        "Authentication is resolved by huggingface_hub from HF_TOKEN or a "
        "configured provider; credential command-line options are not supported."
    )
    return parser.parse_args(argv)


def _print_plan(assets: Sequence[TokenizerAsset]) -> None:
    print("TorchTitan tokenizer asset plan (dry run; no network access)")
    for asset in assets:
        print(f"{asset.key}: {asset.repository}@{asset.revision}")
        print(f"  destination: {REPOSITORY_ROOT / asset.destination}")
        for filename in asset.files:
            print(f"  file: {filename}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    assets = _selected_assets(args.model)
    if not args.execute:
        _print_plan(assets)
        return 0

    try:
        list_repo_files, hf_hub_download = _hub_functions()
        for asset in assets:
            manifest = prepare_asset(
                asset,
                list_repo_files=list_repo_files,
                hf_hub_download=hf_hub_download,
            )
            print(
                f"Prepared {asset.key}: {len(asset.files)} tokenizer files; "
                f"manifest: {manifest}"
            )
    except Exception as error:
        if isinstance(error, AssetPreparationError):
            message = str(error)
        else:
            message = f"asset preparation failed ({type(error).__name__})"
        print(f"error: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

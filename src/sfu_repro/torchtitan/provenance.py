# SPDX-License-Identifier: Apache-2.0
"""Sanitized, deterministic provenance helpers for public TorchTitan runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


RECEIPT_FILENAME = "sfu-launch-receipt.json"
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_ARTIFACT_TYPE = "sfu_torchtitan_launch_receipt"

_HOST_OPTIONS = {
    "local-addr",
    "master-addr",
    "rdzv-endpoint",
}
_TRANSPORT_IDENTIFIER_OPTIONS = {"node-rank", "rdzv-id"}
_SECRET_WORDS = {
    "accesskey",
    "apikey",
    "credential",
    "credentials",
    "password",
    "privatekey",
    "secret",
    "token",
}
_OPTION_WORDS = re.compile(r"[^a-z0-9]+")


class ReceiptError(RuntimeError):
    """A launch-receipt failure safe to display without user-provided values."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(document: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            document,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ReceiptError("launch receipt is not strict JSON") from error
    return encoded + b"\n"


def normalize_repository_path(value: str, repository_root: Path) -> str:
    """Replace checkout or external prefixes without losing path identity."""

    candidate = Path(value)
    try:
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (repository_root / candidate).resolve()
        )
        root = repository_root.resolve()
        if resolved == root:
            return "${REPOSITORY_ROOT}"
        if resolved.is_relative_to(root):
            return "${REPOSITORY_ROOT}/" + resolved.relative_to(root).as_posix()
        if candidate.is_absolute():
            path_digest = sha256_bytes(os.fsencode(str(resolved)))
            return f"${{EXTERNAL_PATH:{path_digest}}}/{resolved.name}"
    except OSError:
        pass
    return value


def _normalize_if_path(value: str, repository_root: Path) -> str:
    candidate = Path(value)
    root_text = str(repository_root.resolve())
    looks_like_path = (
        candidate.is_absolute()
        or value.startswith(("./", "../"))
        or value == root_text
        or value.startswith(root_text + os.sep)
        or ("/" in value and (repository_root / candidate).exists())
    )
    if looks_like_path:
        return normalize_repository_path(value, repository_root)
    return value


def _option_kind(option: str) -> str | None:
    normalized = option.lstrip("-").lower().replace("_", "-")
    final_name = normalized.rsplit(".", 1)[-1]
    if final_name in _HOST_OPTIONS:
        return "host"
    if final_name in _TRANSPORT_IDENTIFIER_OPTIONS:
        return "transport-identifier"
    compact_words = {word for word in _OPTION_WORDS.split(normalized) if word}
    compact = "".join(character for character in normalized if character.isalnum())
    final_compact = "".join(
        character for character in final_name if character.isalnum()
    )
    if (
        compact_words & _SECRET_WORDS
        or compact in _SECRET_WORDS
        or final_compact in _SECRET_WORDS
    ):
        return "secret"
    return None


def _redact_url_hostname(value: str) -> tuple[str, bool]:
    if "://" not in value:
        return value, False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<redacted-url>", True
    if not parsed.hostname:
        return value, False
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        return "<redacted-url>", True
    netloc = f"<redacted-host>{port}"
    return (
        urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)),
        True,
    )


def sanitize_command(
    command: Sequence[str],
    *,
    repository_root: Path,
) -> tuple[list[str], list[str], bool]:
    """Return a receipt-safe command and its redaction classes.

    Rendezvous hosts and identifiers are transport details, so their redaction
    does not weaken scientific binding. A secret-bearing experimental option
    does: callers use the final boolean to mark the receipt unbound.
    """

    sanitized: list[str] = []
    redactions: set[str] = set()
    secret_redacted = False
    pending_kind: str | None = None
    for raw_token in command:
        token = str(raw_token)
        if pending_kind is not None:
            replacement = (
                "<redacted-host>"
                if pending_kind == "host"
                else (
                    "<redacted-transport-id>"
                    if pending_kind == "transport-identifier"
                    else "<redacted-secret>"
                )
            )
            sanitized.append(replacement)
            redactions.add(pending_kind)
            secret_redacted = secret_redacted or pending_kind == "secret"
            pending_kind = None
            continue

        if token.startswith("--"):
            option, separator, value = token.partition("=")
            kind = _option_kind(option)
            if kind is not None:
                if separator:
                    replacement = (
                        "<redacted-host>"
                        if kind == "host"
                        else (
                            "<redacted-transport-id>"
                            if kind == "transport-identifier"
                            else "<redacted-secret>"
                        )
                    )
                    sanitized.append(f"{option}={replacement}")
                    redactions.add(kind)
                    secret_redacted = secret_redacted or kind == "secret"
                else:
                    sanitized.append(option)
                    pending_kind = kind
                continue
            if separator:
                safe_value, hostname_redacted = _redact_url_hostname(value)
                if hostname_redacted:
                    redactions.add("host")
                safe_value = _normalize_if_path(safe_value, repository_root)
                sanitized.append(f"{option}={safe_value}")
                continue

        safe_token, hostname_redacted = _redact_url_hostname(token)
        if hostname_redacted:
            redactions.add("host")
        sanitized.append(_normalize_if_path(safe_token, repository_root))

    if pending_kind is not None:
        # Preserve the malformed option without manufacturing a value. The
        # launcher/Torchrun will reject it, while the receipt remains safe.
        redactions.add(pending_kind)
        secret_redacted = secret_redacted or pending_kind == "secret"
    return sanitized, sorted(redactions), secret_redacted


def write_receipt_once(path: Path, document: Mapping[str, Any]) -> str:
    """Create an immutable receipt, tolerating identical multi-node writers."""

    payload = canonical_json_bytes(document)
    digest = sha256_bytes(payload)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ReceiptError("could not create the launch-receipt directory") from error

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            try:
                existing = path.read_bytes()
            except OSError as error:
                raise ReceiptError(
                    "could not read an existing launch receipt"
                ) from error
            if existing != payload:
                raise ReceiptError(
                    "the run output already contains a different launch receipt"
                )
        except OSError as error:
            raise ReceiptError("could not create the launch receipt") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return digest


def read_receipt(path: Path) -> tuple[dict[str, Any], str]:
    """Load one strict JSON receipt and return its exact file digest."""

    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ReceiptError("missing or invalid launch receipt") from None
    if not isinstance(document, dict):
        raise ReceiptError("launch receipt root must be an object")
    return document, sha256_bytes(payload)


def find_receipt(run_directory: Path) -> Path:
    """Find the nearest receipt at or above a TensorBoard event directory."""

    try:
        current = run_directory.resolve(strict=True)
    except OSError:
        raise ReceiptError("could not resolve an arm input directory") from None
    for directory in (current, *current.parents):
        candidate = directory / RECEIPT_FILENAME
        if candidate.is_file():
            return candidate
    raise ReceiptError("no launch receipt was found for an arm input")


__all__ = [
    "RECEIPT_ARTIFACT_TYPE",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA_VERSION",
    "ReceiptError",
    "canonical_json_bytes",
    "find_receipt",
    "normalize_repository_path",
    "read_receipt",
    "sanitize_command",
    "sha256_bytes",
    "write_receipt_once",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
#
# Modified in 2026 for the standalone paper artifact: repository-path discovery
# and training-framework dependencies were removed; extensions are lazy imports.
"""Standalone activation adapters for the paper's compiled polynomial kernels."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F


PUBLIC_MLP_ACTIVATIONS = (
    "native_silu",
    "spline_silu",
    "spline_silu_compile",
    "native_gelu",
    "spline_gelu",
    "spline_gelu_compile",
)

_COEFF_SOURCE_TO_ID = {"current": 0, "sollya": 1}


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.lower() in ("1", "true", "yes", "on")


def _ensure_coeff_source_allowed(coeff_source: str) -> None:
    if coeff_source not in _COEFF_SOURCE_TO_ID:
        raise ValueError(
            f"unknown coeff_source {coeff_source!r}; expected one of "
            f"{tuple(_COEFF_SOURCE_TO_ID)}"
        )
    if coeff_source == "sollya" and not _env_flag("SFU_REPRO_ALLOW_SOLLYA_SWEEP"):
        raise ValueError(
            "Sollya coefficients are benchmark-only. Set "
            "SFU_REPRO_ALLOW_SOLLYA_SWEEP=1 for an explicit sweep."
        )


def _coeff_source_id(coeff_source: str) -> int:
    _ensure_coeff_source_allowed(coeff_source)
    return _COEFF_SOURCE_TO_ID[coeff_source]


@lru_cache(maxsize=1)
def load_spline_ops():
    """Import the compiled extension without assuming a repository layout."""

    try:
        import spline_ops
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "The compiled spline_ops extension is unavailable. Build/install the "
            "vendored autonumerics_zero/spline_ops package first."
        ) from error
    return spline_ops


@lru_cache(maxsize=1)
def _load_spline_compile_activation_functions():
    try:
        from spline_compile import (
            make_spline_gelu,
            make_spline_silu,
            make_spline_silu_mul,
            spline_gelu,
            spline_silu,
        )
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "spline_compile is unavailable. Build/install the vendored "
            "autonumerics_zero/spline_ops package first."
        ) from error
    return (
        spline_silu,
        spline_gelu,
        make_spline_silu,
        make_spline_gelu,
        make_spline_silu_mul,
    )


@lru_cache(maxsize=1)
def _load_spline_compile_packed_activation_function():
    try:
        from spline_compile import make_spline_silu_packed
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "packed polynomial SwiGLU support is unavailable. Rebuild/install "
            "the vendored spline_ops extension."
        ) from error
    return make_spline_silu_packed


@lru_cache(maxsize=1)
def _load_native_silu_packed_activation_function():
    try:
        from spline_compile import make_native_silu_packed
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "packed native SwiGLU support is unavailable. Rebuild/install the "
            "vendored spline_ops extension."
        ) from error
    return make_native_silu_packed


def _contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _require_fused_current_swiglu(
    name: str,
    coeff_source: str,
    function: Any,
    role: str,
    expected_op: str,
) -> None:
    if name != "spline_silu_compile" or coeff_source != "current":
        return
    callable_name = getattr(function, "__name__", "")
    resolved_op = getattr(function, "__spline_op__", None)
    if resolved_op != expected_op:
        raise RuntimeError(
            f"{role} requested the production fused path but resolved to "
            f"{callable_name or function!r} ({resolved_op=!r}) instead of "
            f"{expected_op!r}. Check SPLINE_COMPILE_DISABLE_FUSED_SWIGLU and "
            "rebuild spline_ops."
        )


def _make_direct_spline_silu(degree: int, coeff_source: str):
    spline_ops = load_spline_ops()
    coeff_source_id = _coeff_source_id(coeff_source)

    def activation(tensor: torch.Tensor) -> torch.Tensor:
        tensor = _contiguous(tensor)
        if torch.is_grad_enabled() and tensor.requires_grad:
            return spline_ops.swish_ag_variant(tensor, degree, coeff_source_id)
        return spline_ops.swish_fwd_variant(tensor, degree, coeff_source_id)

    activation.__name__ = f"spline_silu_d{degree}_{coeff_source}_direct"
    return activation


def _make_direct_spline_silu_mul(degree: int, coeff_source: str):
    spline_ops = load_spline_ops()
    coeff_source_id = _coeff_source_id(coeff_source)
    if coeff_source_id != 0:
        return None

    def activation_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate = _contiguous(gate)
        up = _contiguous(up)
        if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
            return spline_ops.swish_mul_ag_variant(gate, up, degree, coeff_source_id)
        return spline_ops.swish_mul_fwd_variant(gate, up, degree, coeff_source_id)

    activation_mul.__name__ = f"spline_silu_mul_d{degree}_{coeff_source}_direct"
    return activation_mul


def _make_direct_spline_gelu(degree: int, coeff_source: str):
    spline_ops = load_spline_ops()
    coeff_source_id = _coeff_source_id(coeff_source)

    def activation(tensor: torch.Tensor) -> torch.Tensor:
        tensor = _contiguous(tensor)
        if torch.is_grad_enabled() and tensor.requires_grad:
            return spline_ops.gelu_ag_variant(tensor, degree, coeff_source_id)
        return spline_ops.gelu_fwd_variant(tensor, degree, coeff_source_id)

    activation.__name__ = f"spline_gelu_d{degree}_{coeff_source}_direct"
    return activation


def resolve_mlp_activation_impl(
    name: str,
    *,
    degree: int | None = None,
    coeff_source: str = "current",
):
    """Resolve a native or compiled elementwise activation."""

    if name == "native_silu":
        return "native_silu", F.silu
    if name == "native_gelu":
        return "native_gelu", F.gelu
    if name == "spline_silu":
        resolved_degree = 3 if degree is None else int(degree)
        return (
            f"spline_silu[d{resolved_degree},{coeff_source}]",
            _make_direct_spline_silu(resolved_degree, coeff_source),
        )
    if name == "spline_silu_compile":
        _ensure_coeff_source_allowed(coeff_source)
        (
            spline_silu,
            _,
            make_spline_silu,
            _,
            _,
        ) = _load_spline_compile_activation_functions()
        resolved_degree = 3 if degree is None else int(degree)
        activation = (
            spline_silu
            if degree is None and coeff_source == "current"
            else make_spline_silu(degree=resolved_degree, coeff_source=coeff_source)
        )
        return (
            f"spline_silu_compile[d{resolved_degree},{coeff_source}]",
            activation,
        )
    if name == "spline_gelu":
        resolved_degree = 5 if degree is None else int(degree)
        return (
            f"spline_gelu[d{resolved_degree},{coeff_source}]",
            _make_direct_spline_gelu(resolved_degree, coeff_source),
        )
    if name == "spline_gelu_compile":
        _ensure_coeff_source_allowed(coeff_source)
        (
            _,
            spline_gelu,
            _,
            make_spline_gelu,
            _,
        ) = _load_spline_compile_activation_functions()
        resolved_degree = 5 if degree is None else int(degree)
        activation = (
            spline_gelu
            if degree is None and coeff_source == "current"
            else make_spline_gelu(degree=resolved_degree, coeff_source=coeff_source)
        )
        return (
            f"spline_gelu_compile[d{resolved_degree},{coeff_source}]",
            activation,
        )
    raise ValueError(
        f"unknown activation {name!r}; expected one of "
        + ", ".join(PUBLIC_MLP_ACTIVATIONS)
    )


def resolve_mlp_activation_mul_impl(
    name: str,
    *,
    degree: int | None = None,
    coeff_source: str = "current",
):
    """Resolve ``activation(gate) * up``, preferring the fused kernel."""

    if name == "native_silu":
        return "native_silu_mul", lambda gate, up: F.silu(gate) * up
    if name == "spline_silu":
        resolved_degree = 3 if degree is None else int(degree)
        activation_mul = _make_direct_spline_silu_mul(resolved_degree, coeff_source)
        if activation_mul is not None:
            return (
                f"spline_silu_mul[d{resolved_degree},{coeff_source}]",
                activation_mul,
            )
    if name == "spline_silu_compile":
        _ensure_coeff_source_allowed(coeff_source)
        *_, make_spline_silu_mul = _load_spline_compile_activation_functions()
        resolved_degree = 3 if degree is None else int(degree)
        activation_mul = make_spline_silu_mul(
            degree=resolved_degree, coeff_source=coeff_source
        )
        _require_fused_current_swiglu(
            name,
            coeff_source,
            activation_mul,
            "SwiGLU activation-multiply",
            "spline_ops::swish_mul_variant_fwd",
        )
        implementation = (
            "spline_ops::swish_mul_variant_fwd"
            if coeff_source == "current"
            else "spline_silu_compile_mul_unfused"
        )
        return (
            f"{implementation}[d{resolved_degree},{coeff_source}]",
            activation_mul,
        )
    activation_name, activation = resolve_mlp_activation_impl(
        name, degree=degree, coeff_source=coeff_source
    )
    return f"{activation_name}_mul", lambda gate, up: activation(gate) * up


def resolve_mlp_activation_packed_impl(
    name: str,
    *,
    degree: int | None = None,
    coeff_source: str = "current",
    backward_impl: str = "matched",
):
    """Resolve a packed ``[..., 2 * hidden]`` SwiGLU implementation."""

    if backward_impl not in ("matched", "native"):
        raise ValueError("backward_impl must be 'matched' or 'native'")
    if name == "native_silu":
        make_native_silu_packed = _load_native_silu_packed_activation_function()
        return "native_silu_packed", make_native_silu_packed()
    if name == "spline_silu_compile":
        _ensure_coeff_source_allowed(coeff_source)
        resolved_degree = 3 if degree is None else int(degree)
        make_spline_silu_packed = _load_spline_compile_packed_activation_function()
        activation = make_spline_silu_packed(
            degree=resolved_degree,
            coeff_source=coeff_source,
            backward_impl=backward_impl,
        )
        expected_op = (
            "spline_ops::swish_mul_packed_variant_fwd"
            if backward_impl == "matched"
            else "spline_ops::swish_mul_packed_native_bwd_fwd"
        )
        _require_fused_current_swiglu(
            name,
            coeff_source,
            activation,
            "packed SwiGLU",
            expected_op,
        )
        return (
            f"{expected_op}[d{resolved_degree},{coeff_source},{backward_impl}_bwd]",
            activation,
        )
    activation_name, activation_mul = resolve_mlp_activation_mul_impl(
        name, degree=degree, coeff_source=coeff_source
    )

    def activation_packed(packed: torch.Tensor) -> torch.Tensor:
        gate, up = packed.chunk(2, dim=-1)
        return activation_mul(gate, up)

    return f"{activation_name}_packed", activation_packed


__all__ = [
    "PUBLIC_MLP_ACTIVATIONS",
    "load_spline_ops",
    "resolve_mlp_activation_impl",
    "resolve_mlp_activation_mul_impl",
    "resolve_mlp_activation_packed_impl",
]

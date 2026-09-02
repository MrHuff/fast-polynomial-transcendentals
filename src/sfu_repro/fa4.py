# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
#
# Modified in 2026 for the standalone paper artifact: training-framework
# integration, job configuration, and repository-path discovery were removed.
"""Standalone configuration and PyTorch wrapper for the paper's FA4 kernels.

The module deliberately imports the patched :mod:`flash_attn` runtime lazily.
Configuration parsing and validation therefore work on a CPU-only machine,
while kernel construction fails with an actionable error when the optional
FA4 checkout has not been built and installed.

Inputs to :class:`FA4AttentionWrapper` use ``(batch, heads, sequence, dim)``.
The patched FA4 interface uses ``(batch, sequence, heads, dim)`` internally.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from torch import nn


LOGGER = logging.getLogger(__name__)

FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH = 4096

SUPPORTED_EXP2_BACKENDS = (
    "d3",
    "pwl2",
    "pwl1_safe",
    "pwl1_safe_f16",
    "pwl2_safe",
    "pwl2_safe_f16",
    "pwl2_safe_bf16",
    "d2_safe",
    "d2_safe_noclamp",
    "d2_safe_f16",
    "d2_safe_bf16",
    "pwl2_f16",
)


@dataclass(frozen=True)
class FA4Config:
    """Declarative FA4 configuration independent of a training framework."""

    mode: str = "softmax"
    exp2_emu_backend: str = "d3"
    exp2_emu_freq: int | None = None
    exp2_emu_freq_bwd: int = 0
    softcap: float = 30.0
    softcap_backend: str = "native"
    softcap_degree: int = 4
    softcap_coeff_source: str = "current"
    softcap_backward_mode: str = "analytical"
    sigmoid_variant: str = "poly"
    sigmoid_poly_backend: str = "cute"
    sigmoid_degree: int = 3
    sigmoid_degree_bwd: int | None = None
    sigmoid_coeff_source: str = "current"
    sigmoid_backward_mode: str = "algebraic"
    sigmoid_sfu_freq: int = 16
    sigmoid_sfu_res: int = 0
    sigmoid_sfu_freq_bwd: int | None = None
    sigmoid_sfu_res_bwd: int | None = None
    sigmoid_bias: float | None = None
    sigmoid_qk_norm: bool = True
    audit_coefficients: bool = True


@dataclass(frozen=True)
class ResolvedFA4Config:
    """A validated :class:`FA4Config` bound to the patched FA4 runtime."""

    mode: str
    flash_attn_fwd: Callable[..., Any]
    flash_attn_bwd: Callable[..., Any]
    exp2_emu_backend: str = "d3"
    exp2_emu_freq: int | None = None
    exp2_emu_freq_bwd: int = 0
    softcap: float = 0.0
    score_mod: Callable[..., Any] | None = None
    score_mod_bwd: Callable[..., Any] | None = None
    sigmoid_attention: bool = False
    sigmoid_sfu_freq: int = 16
    sigmoid_sfu_res: int = 0
    sigmoid_sfu_freq_bwd: int | None = None
    sigmoid_sfu_res_bwd: int | None = None
    sigmoid_use_direct_bwd_poly: bool = False
    sigmoid_bias: float | None = None
    sigmoid_poly_backend: str = "cute"
    sigmoid_degree: int = 3
    sigmoid_degree_bwd: int | None = None
    sigmoid_coeff_source: str = "current"
    sigmoid_qk_norm: bool = True
    implementation_summary: str = ""


@dataclass(frozen=True)
class Exp2Variant:
    """One mixed SFU/software exp2 route for the FA4 softmax benchmark."""

    name: str
    backend: str
    forward_frequency: int | None
    backward_frequency: int

    @property
    def nominal_forward_polynomial_fraction(self) -> float | None:
        if self.forward_frequency is None:
            return None
        if self.forward_frequency == 0:
            return 0.0
        return min(1.0, 4.0 / self.forward_frequency)

    @property
    def nominal_backward_polynomial_fraction(self) -> float:
        if self.backward_frequency == 0:
            return 0.0
        return min(1.0, 4.0 / self.backward_frequency)

    @property
    def nominal_forward_pwl_fraction(self) -> float | None:
        """Compatibility alias used by the original benchmark artifact."""

        return self.nominal_forward_polynomial_fraction

    @property
    def nominal_backward_pwl_fraction(self) -> float:
        """Compatibility alias used by the original benchmark artifact."""

        return self.nominal_backward_polynomial_fraction


DEFAULT_EXP2_VARIANTS = ",".join(
    (
        "sfu=d3:0:0",
        "d3_auto=d3:auto:0",
        "d2_safe_12p5=d2_safe:32:0",
        "d2_safe_16p7=d2_safe:24:0",
        "d2_safe_20=d2_safe:20:0",
        "d2_safe_25=d2_safe:16:0",
        "d2_safe_33p3=d2_safe:12:0",
        "d2_safe_40=d2_safe:10:0",
        "d2_safe_50=d2_safe:8:0",
        "d2_safe_66p7=d2_safe:6:0",
        "d2_safe_100=d2_safe:4:0",
    )
)


def parse_exp2_frequency(value: str, *, allow_auto: bool) -> int | None:
    """Parse and validate one FA4 exp2 routing frequency."""

    if allow_auto and value == "auto":
        return None
    try:
        frequency = int(value)
    except ValueError as error:
        allowed = (
            "'auto', 0, or an even integer >= 4"
            if allow_auto
            else "0 or an even integer >= 4"
        )
        raise ValueError(f"frequency must be {allowed}, got {value!r}") from error
    return _validate_exp2_route(frequency, "exp2")


def parse_exp2_variants(value: str) -> tuple[Exp2Variant, ...]:
    """Parse ``name=backend:forward:backward`` benchmark variants."""

    variants: list[Exp2Variant] = []
    for raw_spec in value.split(","):
        raw_spec = raw_spec.strip()
        if not raw_spec:
            continue
        try:
            name, route = raw_spec.split("=", maxsplit=1)
            backend, forward, backward = route.split(":")
        except ValueError as error:
            raise ValueError(
                "variant must have name=backend:forward_frequency:"
                f"backward_frequency syntax, got {raw_spec!r}"
            ) from error
        name = name.strip()
        backend = backend.strip()
        if not name:
            raise ValueError("variant name must not be empty")
        if backend not in SUPPORTED_EXP2_BACKENDS:
            raise ValueError(f"unsupported exp2 backend {backend!r}")
        variants.append(
            Exp2Variant(
                name=name,
                backend=backend,
                forward_frequency=parse_exp2_frequency(
                    forward.strip(), allow_auto=True
                ),
                backward_frequency=int(
                    parse_exp2_frequency(backward.strip(), allow_auto=False)
                ),
            )
        )
    if not variants:
        raise ValueError("at least one variant is required")
    if len({variant.name for variant in variants}) != len(variants):
        raise ValueError("variant names must be unique")
    return tuple(variants)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.lower() in ("1", "true", "yes", "on")


def _allow_sollya_sweep() -> bool:
    return _env_flag("SFU_REPRO_ALLOW_SOLLYA_SWEEP")


def _validate_exp2_route(frequency: int | None, label: str) -> int | None:
    if frequency is None:
        return None
    frequency = int(frequency)
    if frequency < 0 or (frequency != 0 and (frequency < 4 or frequency % 2)):
        raise ValueError(
            f"{label} frequency must be 0 or an even integer >= 4, got {frequency}"
        )
    return frequency


def _validate_sigmoid_sfu_route(
    frequency: int, residue_count: int, label: str
) -> tuple[int, int]:
    frequency = int(frequency)
    residue_count = int(residue_count)
    if frequency <= 0:
        raise ValueError(f"{label} sigmoid_sfu_freq must be positive, got {frequency}")
    if residue_count < 0 or residue_count > frequency:
        raise ValueError(
            f"{label} sigmoid_sfu_res must lie in [0, sigmoid_sfu_freq], "
            f"got {residue_count} for {frequency}"
        )
    return frequency, residue_count


def _uses_sollya(config: FA4Config) -> bool:
    return (config.mode == "softcap" and config.softcap_coeff_source == "sollya") or (
        config.mode == "sigmoid_attention"
        and config.sigmoid_variant == "poly"
        and config.sigmoid_coeff_source == "sollya"
    )


def _uses_direct_flash_sigmoid(config: FA4Config) -> bool:
    return (
        config.mode == "sigmoid_attention"
        and config.sigmoid_variant == "poly"
        and config.sigmoid_poly_backend == "device"
        and config.sigmoid_coeff_source == "current"
        and int(config.sigmoid_degree) == 3
        and config.sigmoid_bias is None
    )


def validate_fa4_config(
    config: FA4Config, *, sequence_length: int | None = None
) -> FA4Config:
    """Validate a declarative config without importing CUDA or FA4."""

    if config.mode not in ("softmax", "softcap", "sigmoid_attention"):
        raise ValueError(f"unsupported FA4 mode {config.mode!r}")
    if config.exp2_emu_backend not in SUPPORTED_EXP2_BACKENDS:
        raise ValueError(f"unsupported exp2 backend {config.exp2_emu_backend!r}")
    _validate_exp2_route(config.exp2_emu_freq, "forward exp2")
    _validate_exp2_route(config.exp2_emu_freq_bwd, "backward exp2")

    if _uses_sollya(config) and not _allow_sollya_sweep():
        raise ValueError(
            "Sollya coefficient sources are benchmark-only. Set "
            "SFU_REPRO_ALLOW_SOLLYA_SWEEP=1 for an explicit sweep."
        )

    if config.mode == "softcap":
        if config.softcap_backend not in ("native", "cute", "device"):
            raise ValueError(f"unsupported softcap backend {config.softcap_backend!r}")
        if config.softcap_backward_mode not in (
            "analytical",
            "ste",
            "exact_1mt2",
            "native",
        ):
            raise ValueError(
                f"unsupported softcap backward mode {config.softcap_backward_mode!r}"
            )
        if config.softcap_coeff_source not in ("current", "sollya"):
            raise ValueError(
                f"unsupported softcap coefficient source {config.softcap_coeff_source!r}"
            )
        if config.softcap <= 0:
            raise ValueError("softcap must be positive")
        if (
            config.softcap_backend == "native"
            and config.softcap_backward_mode != "native"
        ):
            raise ValueError("native softcap requires softcap_backward_mode='native'")
        if (
            config.softcap_backend != "native"
            and config.softcap_backward_mode == "native"
        ):
            raise ValueError(
                "polynomial softcap cannot use softcap_backward_mode='native'"
            )
        if (
            config.softcap_coeff_source == "sollya"
            and config.softcap_backend != "device"
        ):
            raise ValueError("Sollya softcap fits require softcap_backend='device'")

    if config.mode == "sigmoid_attention":
        if config.sigmoid_variant not in ("poly", "sfu"):
            raise ValueError(f"unsupported sigmoid variant {config.sigmoid_variant!r}")
        if config.sigmoid_poly_backend not in ("cute", "device"):
            raise ValueError(
                f"unsupported sigmoid polynomial backend {config.sigmoid_poly_backend!r}"
            )
        if config.sigmoid_backward_mode not in ("algebraic", "direct"):
            raise ValueError(
                f"unsupported sigmoid backward mode {config.sigmoid_backward_mode!r}"
            )
        if config.sigmoid_coeff_source not in ("current", "sollya"):
            raise ValueError(
                f"unsupported sigmoid coefficient source {config.sigmoid_coeff_source!r}"
            )
        if int(config.sigmoid_degree) not in (2, 3, 4, 5, 6):
            raise ValueError("sigmoid_degree must be one of 2, 3, 4, 5, or 6")
        if config.sigmoid_degree_bwd is not None and int(
            config.sigmoid_degree_bwd
        ) not in (2, 3, 4, 5, 6):
            raise ValueError("sigmoid_degree_bwd must be one of 2, 3, 4, 5, or 6")
        if config.sigmoid_variant == "poly" and config.sigmoid_poly_backend == "cute":
            raise ValueError(
                "causal polynomial sigmoid attention requires "
                "sigmoid_poly_backend='device'"
            )
        if (
            config.sigmoid_variant == "poly"
            and config.sigmoid_coeff_source == "sollya"
            and config.sigmoid_poly_backend != "device"
        ):
            raise ValueError(
                "Sollya sigmoid fits require sigmoid_poly_backend='device'"
            )

        forward_frequency, forward_residue = _validate_sigmoid_sfu_route(
            config.sigmoid_sfu_freq, config.sigmoid_sfu_res, "forward"
        )
        backward_frequency = (
            forward_frequency
            if config.sigmoid_sfu_freq_bwd is None
            else config.sigmoid_sfu_freq_bwd
        )
        backward_residue = (
            forward_residue
            if config.sigmoid_sfu_res_bwd is None
            else config.sigmoid_sfu_res_bwd
        )
        backward_frequency, backward_residue = _validate_sigmoid_sfu_route(
            backward_frequency, backward_residue, "backward"
        )
        if config.sigmoid_variant == "poly" and (
            forward_residue != 0 or backward_residue != 0
        ):
            raise ValueError(
                "sigmoid_variant='poly' must route every forward/backward lane "
                "through the polynomial"
            )
        if config.sigmoid_variant == "sfu" and (
            forward_residue != forward_frequency
            or backward_residue != backward_frequency
        ):
            raise ValueError(
                "sigmoid_variant='sfu' must route every forward/backward lane "
                "through the SFU"
            )

        uses_bias_aware_d2 = (
            config.sigmoid_variant == "poly"
            and config.sigmoid_poly_backend == "device"
            and config.sigmoid_coeff_source == "current"
            and int(config.sigmoid_degree) == 2
            and config.sigmoid_bias is None
        )
        if uses_bias_aware_d2 and config.sigmoid_backward_mode != "algebraic":
            raise ValueError(
                "bias-aware FlashSigmoid D2 requires "
                "sigmoid_backward_mode='algebraic'"
            )
        if _uses_direct_flash_sigmoid(config):
            if sequence_length is None:
                raise ValueError(
                    "direct FlashSigmoid D3 requires an explicit sequence_length"
                )
            if int(sequence_length) != FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH:
                raise ValueError(
                    "direct FlashSigmoid D3/D4 is fitted for sequence length "
                    f"{FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH}, got {sequence_length}"
                )

    return config


@lru_cache(maxsize=1)
def _load_fa4_runtime() -> Mapping[str, Any]:
    try:
        from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
        from flash_attn.cute.polynomial_manifest import (
            FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH as runtime_direct_length,
            audit_polynomial_selection,
            run_polynomial_coefficient_audit,
        )
        from flash_attn.cute.utils import (
            audit_attention_handwritten_fast_path,
            create_softcap_scoremod_backend,
            create_softcap_scoremod_bwd_backend,
        )
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "The patched FA4 runtime is unavailable. Initialize and build the "
            "pinned flash-attention checkout, then make `flash_attn` importable."
        ) from error
    if runtime_direct_length != FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH:
        raise RuntimeError(
            "The installed FA4 runtime uses direct FlashSigmoid sequence length "
            f"{runtime_direct_length}; this artifact expects "
            f"{FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH}."
        )
    return {
        "flash_attn_fwd": _flash_attn_fwd,
        "flash_attn_bwd": _flash_attn_bwd,
        "run_polynomial_coefficient_audit": run_polynomial_coefficient_audit,
        "audit_polynomial_selection": audit_polynomial_selection,
        "audit_attention_handwritten_fast_path": audit_attention_handwritten_fast_path,
        "create_softcap_scoremod_backend": create_softcap_scoremod_backend,
        "create_softcap_scoremod_bwd_backend": create_softcap_scoremod_bwd_backend,
    }


def resolve_fa4_config(
    config: FA4Config,
    *,
    sequence_length: int | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> ResolvedFA4Config:
    """Validate ``config`` and bind it to a patched FA4 runtime."""

    validate_fa4_config(config, sequence_length=sequence_length)
    runtime = _load_fa4_runtime() if runtime is None else runtime
    sigmoid_degree_bwd = (
        int(config.sigmoid_degree)
        if config.sigmoid_degree_bwd is None
        else int(config.sigmoid_degree_bwd)
    )
    uses_bias_aware_d2 = (
        config.mode == "sigmoid_attention"
        and config.sigmoid_variant == "poly"
        and config.sigmoid_poly_backend == "device"
        and config.sigmoid_coeff_source == "current"
        and int(config.sigmoid_degree) == 2
        and config.sigmoid_bias is None
    )

    if config.audit_coefficients:
        audited = list(runtime["run_polynomial_coefficient_audit"]())
        selections: list[tuple[str, int, str]] = []
        if config.mode == "softcap" and config.softcap_backend != "native":
            selections.append(
                (
                    "tanh_fwd",
                    int(config.softcap_degree),
                    config.softcap_coeff_source,
                )
            )
        if config.mode == "sigmoid_attention" and config.sigmoid_variant == "poly":
            selections.append(
                (
                    "flash_sigmoid_exp2" if uses_bias_aware_d2 else "sigmoid_fwd",
                    int(config.sigmoid_degree),
                    config.sigmoid_coeff_source,
                )
            )
            if config.sigmoid_backward_mode == "direct":
                selections.append(
                    (
                        "sigmoid_bwd",
                        sigmoid_degree_bwd,
                        config.sigmoid_coeff_source,
                    )
                )
            elif sigmoid_degree_bwd != int(config.sigmoid_degree):
                selections.append(
                    (
                        (
                            "flash_sigmoid_exp2"
                            if uses_bias_aware_d2 and sigmoid_degree_bwd == 2
                            else "sigmoid_fwd"
                        ),
                        sigmoid_degree_bwd,
                        config.sigmoid_coeff_source,
                    )
                )
        if selections:
            audited.extend(runtime["audit_polynomial_selection"](tuple(selections)))
        LOGGER.info("FA4 polynomial audit passed: %s", ", ".join(audited))

    if config.mode == "softmax":
        route = "automatic" if config.exp2_emu_freq is None else config.exp2_emu_freq
        return ResolvedFA4Config(
            mode="softmax",
            flash_attn_fwd=runtime["flash_attn_fwd"],
            flash_attn_bwd=runtime["flash_attn_bwd"],
            exp2_emu_backend=config.exp2_emu_backend,
            exp2_emu_freq=config.exp2_emu_freq,
            exp2_emu_freq_bwd=config.exp2_emu_freq_bwd,
            sigmoid_qk_norm=False,
            implementation_summary=(
                f"FA4 softmax exp2 backend={config.exp2_emu_backend}, "
                f"forward_freq={route}, backward_freq={config.exp2_emu_freq_bwd}"
            ),
        )

    if config.mode == "softcap":
        if config.softcap_backend == "native":
            return ResolvedFA4Config(
                mode="softcap",
                flash_attn_fwd=runtime["flash_attn_fwd"],
                flash_attn_bwd=runtime["flash_attn_bwd"],
                softcap=float(config.softcap),
                sigmoid_qk_norm=False,
                implementation_summary="FA4 native SFU tanh softcap",
            )
        paths: list[str] = []
        if config.softcap_backend == "device":
            paths.append(
                runtime["audit_attention_handwritten_fast_path"](
                    "tanh_fwd",
                    int(config.softcap_degree),
                    config.softcap_coeff_source,
                )
            )
            if config.softcap_backward_mode == "analytical":
                paths.append(
                    runtime["audit_attention_handwritten_fast_path"](
                        "tanh_grad_analytical",
                        int(config.softcap_degree),
                        config.softcap_coeff_source,
                    )
                )
        else:
            paths.append(
                f"CuTe tanh D{int(config.softcap_degree)} "
                f"{config.softcap_coeff_source}"
            )
        score_mod = runtime["create_softcap_scoremod_backend"](
            config.softcap,
            degree=config.softcap_degree,
            backend=config.softcap_backend,
            coeff_source=config.softcap_coeff_source,
        )
        score_mod_bwd = runtime["create_softcap_scoremod_bwd_backend"](
            config.softcap,
            degree=config.softcap_degree,
            backend=config.softcap_backend,
            backward_mode=config.softcap_backward_mode,
            coeff_source=config.softcap_coeff_source,
        )
        return ResolvedFA4Config(
            mode="softcap",
            flash_attn_fwd=runtime["flash_attn_fwd"],
            flash_attn_bwd=runtime["flash_attn_bwd"],
            score_mod=score_mod,
            score_mod_bwd=score_mod_bwd,
            sigmoid_qk_norm=False,
            implementation_summary="; ".join(paths),
        )

    forward_frequency, forward_residue = _validate_sigmoid_sfu_route(
        config.sigmoid_sfu_freq, config.sigmoid_sfu_res, "forward"
    )
    backward_frequency = config.sigmoid_sfu_freq_bwd
    backward_residue = config.sigmoid_sfu_res_bwd
    paths: list[str] = []
    if config.sigmoid_variant == "sfu":
        paths.append("FA4 native SFU sigmoid + algebraic sigmoid gradient")
    elif uses_bias_aware_d2:
        paths.append("FA4 bias-aware D2 exp2 polynomial")
    elif config.sigmoid_poly_backend == "device":
        paths.append(
            runtime["audit_attention_handwritten_fast_path"](
                (
                    "flash_sigmoid_direct"
                    if _uses_direct_flash_sigmoid(config)
                    else "sigmoid_fwd"
                ),
                int(config.sigmoid_degree),
                config.sigmoid_coeff_source,
            )
        )
        if config.sigmoid_backward_mode == "direct":
            paths.append(
                runtime["audit_attention_handwritten_fast_path"](
                    (
                        "flash_sigmoid_direct_with_grad"
                        if _uses_direct_flash_sigmoid(config)
                        and sigmoid_degree_bwd == 4
                        else "sigmoid_grad"
                    ),
                    sigmoid_degree_bwd,
                    config.sigmoid_coeff_source,
                )
            )

    return ResolvedFA4Config(
        mode="sigmoid_attention",
        flash_attn_fwd=runtime["flash_attn_fwd"],
        flash_attn_bwd=runtime["flash_attn_bwd"],
        sigmoid_attention=True,
        sigmoid_sfu_freq=forward_frequency,
        sigmoid_sfu_res=forward_residue,
        sigmoid_sfu_freq_bwd=backward_frequency,
        sigmoid_sfu_res_bwd=backward_residue,
        sigmoid_use_direct_bwd_poly=(config.sigmoid_backward_mode == "direct"),
        sigmoid_bias=config.sigmoid_bias,
        sigmoid_poly_backend=config.sigmoid_poly_backend,
        sigmoid_degree=int(config.sigmoid_degree),
        sigmoid_degree_bwd=(
            sigmoid_degree_bwd if config.sigmoid_degree_bwd is not None else None
        ),
        sigmoid_coeff_source=config.sigmoid_coeff_source,
        sigmoid_qk_norm=config.sigmoid_qk_norm,
        implementation_summary="; ".join(paths),
    )


def b2_component_configs() -> dict[str, FA4Config]:
    """Return the exact native and polynomial B2 component configurations."""

    return {
        "native": FA4Config(
            mode="softcap",
            softcap=30.0,
            softcap_backend="native",
            softcap_backward_mode="native",
        ),
        "polynomial": FA4Config(
            mode="softcap",
            softcap=30.0,
            softcap_backend="device",
            softcap_degree=4,
            softcap_coeff_source="current",
            softcap_backward_mode="analytical",
        ),
    }


def b3_component_configs(
    *, sequence_length: int = FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH
) -> dict[str, FA4Config]:
    """Return the exact B3 SFU and direct D3/D4 configurations.

    The fitted direct forward/gradient pair is specific to sequence length
    4096. Rejecting other lengths here prevents a benchmark label from
    silently diverging from the kernel that is actually selected.
    """

    if int(sequence_length) != FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH:
        raise ValueError(
            "B3 direct D3/D4 requires sequence length "
            f"{FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH}, got {sequence_length}"
        )
    configs = {
        "native": FA4Config(
            mode="sigmoid_attention",
            sigmoid_variant="sfu",
            sigmoid_sfu_freq=16,
            sigmoid_sfu_res=16,
            sigmoid_qk_norm=True,
        ),
        "polynomial": FA4Config(
            mode="sigmoid_attention",
            sigmoid_variant="poly",
            sigmoid_poly_backend="device",
            sigmoid_degree=3,
            sigmoid_degree_bwd=4,
            sigmoid_coeff_source="current",
            sigmoid_backward_mode="direct",
            sigmoid_sfu_freq=16,
            sigmoid_sfu_res=0,
            sigmoid_qk_norm=True,
        ),
    }
    for config in configs.values():
        validate_fa4_config(config, sequence_length=sequence_length)
    return configs


def softmax_exp2_config(variant: Exp2Variant) -> FA4Config:
    """Convert a parsed exp2 benchmark variant to a standalone FA4 config."""

    config = FA4Config(
        mode="softmax",
        exp2_emu_backend=variant.backend,
        exp2_emu_freq=variant.forward_frequency,
        exp2_emu_freq_bwd=variant.backward_frequency,
    )
    return validate_fa4_config(config)


class _FA4Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        softcap,
        sigmoid_attention,
        sigmoid_sfu_freq,
        sigmoid_sfu_res,
        sigmoid_sfu_freq_bwd,
        sigmoid_sfu_res_bwd,
        sigmoid_use_direct_bwd_poly,
        sigmoid_bias,
        sigmoid_poly_backend,
        sigmoid_degree,
        sigmoid_degree_bwd,
        sigmoid_coeff_source,
        exp2_emu_backend,
        exp2_emu_freq,
        exp2_emu_freq_bwd,
        score_mod,
        score_mod_bwd,
        flash_attn_fwd,
        flash_attn_bwd,
    ):
        if sigmoid_sfu_freq_bwd is None:
            sigmoid_sfu_freq_bwd = sigmoid_sfu_freq
        if sigmoid_sfu_res_bwd is None:
            sigmoid_sfu_res_bwd = sigmoid_sfu_res
        if sigmoid_degree_bwd is None:
            sigmoid_degree_bwd = sigmoid_degree
        out, lse = flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap if score_mod is None else 0.0,
            score_mod=score_mod,
            sigmoid_attention=sigmoid_attention,
            sigmoid_sfu_freq=sigmoid_sfu_freq,
            sigmoid_sfu_res=sigmoid_sfu_res,
            sigmoid_bias=sigmoid_bias,
            sigmoid_poly_backend=sigmoid_poly_backend,
            sigmoid_degree=sigmoid_degree,
            sigmoid_coeff_source=sigmoid_coeff_source,
            exp2_emu_backend=exp2_emu_backend,
            exp2_emu_freq=exp2_emu_freq,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.softcap = softcap if score_mod is None else 0.0
        ctx.score_mod = score_mod
        ctx.score_mod_bwd = score_mod_bwd
        ctx.sigmoid_attention = sigmoid_attention
        ctx.sigmoid_sfu_freq_bwd = sigmoid_sfu_freq_bwd
        ctx.sigmoid_sfu_res_bwd = sigmoid_sfu_res_bwd
        ctx.sigmoid_use_direct_bwd_poly = sigmoid_use_direct_bwd_poly
        ctx.sigmoid_bias = sigmoid_bias
        ctx.sigmoid_poly_backend = sigmoid_poly_backend
        ctx.sigmoid_degree = sigmoid_degree
        ctx.sigmoid_degree_bwd = sigmoid_degree_bwd
        ctx.sigmoid_coeff_source = sigmoid_coeff_source
        ctx.exp2_emu_backend = exp2_emu_backend
        ctx.exp2_emu_freq_bwd = exp2_emu_freq_bwd
        ctx.flash_attn_bwd = flash_attn_bwd
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = ctx.flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout.contiguous(),
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            score_mod=ctx.score_mod,
            score_mod_bwd=ctx.score_mod_bwd,
            sigmoid_attention=ctx.sigmoid_attention,
            sigmoid_bias=ctx.sigmoid_bias,
            sigmoid_sfu_freq=ctx.sigmoid_sfu_freq_bwd,
            sigmoid_sfu_res=ctx.sigmoid_sfu_res_bwd,
            sigmoid_use_direct_bwd_poly=ctx.sigmoid_use_direct_bwd_poly,
            sigmoid_poly_backend=ctx.sigmoid_poly_backend,
            sigmoid_degree=(
                ctx.sigmoid_degree
                if ctx.sigmoid_use_direct_bwd_poly
                else ctx.sigmoid_degree_bwd
            ),
            sigmoid_gradient_degree=ctx.sigmoid_degree_bwd,
            sigmoid_coeff_source=ctx.sigmoid_coeff_source,
            exp2_emu_backend=ctx.exp2_emu_backend,
            exp2_emu_freq=ctx.exp2_emu_freq_bwd,
        )
        return dq, dk, dv, *((None,) * 21)


def _fa4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None,
    causal: bool,
    config: ResolvedFA4Config,
) -> torch.Tensor:
    return _FA4Function.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        config.softcap,
        config.sigmoid_attention,
        config.sigmoid_sfu_freq,
        config.sigmoid_sfu_res,
        config.sigmoid_sfu_freq_bwd,
        config.sigmoid_sfu_res_bwd,
        config.sigmoid_use_direct_bwd_poly,
        config.sigmoid_bias,
        config.sigmoid_poly_backend,
        config.sigmoid_degree,
        config.sigmoid_degree_bwd,
        config.sigmoid_coeff_source,
        config.exp2_emu_backend,
        config.exp2_emu_freq,
        config.exp2_emu_freq_bwd,
        config.score_mod,
        config.score_mod_bwd,
        config.flash_attn_fwd,
        config.flash_attn_bwd,
    )


class FA4AttentionWrapper(nn.Module):
    """PyTorch module wrapping the patched FA4 forward and backward kernels."""

    def __init__(self, config: ResolvedFA4Config, head_dim: int):
        super().__init__()
        self.config = config
        self.head_dim = int(head_dim)
        if config.sigmoid_attention and config.sigmoid_qk_norm:
            self.q_norm: nn.Module | None = nn.RMSNorm(self.head_dim)
            self.k_norm: nn.Module | None = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        score_mod: Callable[..., Any] | None = None,
        scale: float | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if score_mod is not None:
            raise NotImplementedError(
                "external score_mod is unsupported; select it in FA4Config"
            )
        kwargs.pop("enable_gqa", None)
        if kwargs:
            raise TypeError(
                "unsupported FA4 attention kwargs: " + ", ".join(sorted(kwargs))
            )
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        target_dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else q.dtype
        )
        out = _fa4(
            q.transpose(1, 2).contiguous().to(target_dtype),
            k.transpose(1, 2).contiguous().to(target_dtype),
            v.transpose(1, 2).contiguous().to(target_dtype),
            softmax_scale=scale,
            causal=True,
            config=self.config,
        )
        return out.transpose(1, 2).contiguous()


def build_fa4_module(
    config: FA4Config,
    *,
    head_dim: int,
    sequence_length: int | None = None,
) -> FA4AttentionWrapper:
    """Resolve a declarative config and construct its attention module."""

    return FA4AttentionWrapper(
        resolve_fa4_config(config, sequence_length=sequence_length),
        head_dim=head_dim,
    )


def patch_attention_modules(model: nn.Module, config: ResolvedFA4Config) -> int:
    """Replace compatible ``inner_attention`` slots without a trainer adapter."""

    patched = 0
    for module in model.modules():
        if not hasattr(module, "inner_attention") or not hasattr(module, "head_dim"):
            continue
        if isinstance(module.inner_attention, FA4AttentionWrapper):
            continue
        wrapper = FA4AttentionWrapper(config=config, head_dim=module.head_dim)
        parameter = next(module.parameters(), None)
        if parameter is not None and parameter.device.type != "meta":
            wrapper = wrapper.to(device=parameter.device, dtype=parameter.dtype)
        elif parameter is not None:
            wrapper = wrapper.to(dtype=parameter.dtype)
        module.inner_attention = wrapper
        if hasattr(module, "use_flex_attn"):
            module.use_flex_attn = False
        if hasattr(module, "attn_score_modifier"):
            module.attn_score_modifier = None
        patched += 1
    return patched


__all__ = [
    "DEFAULT_EXP2_VARIANTS",
    "Exp2Variant",
    "FA4AttentionWrapper",
    "FA4Config",
    "FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH",
    "ResolvedFA4Config",
    "SUPPORTED_EXP2_BACKENDS",
    "b2_component_configs",
    "b3_component_configs",
    "build_fa4_module",
    "parse_exp2_frequency",
    "parse_exp2_variants",
    "patch_attention_modules",
    "resolve_fa4_config",
    "softmax_exp2_config",
    "validate_fa4_config",
]

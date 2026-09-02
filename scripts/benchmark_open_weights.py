#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified in 2026 for the standalone paper artifact: repository-layout and
# credential-command-line assumptions were removed. See NOTICE for provenance.
"""Benchmark open-weight Hugging Face models with local polynomial activations.

The main path patches SwiGLU/SiLU MLP activations by replacing ``module.act_fn``
on common decoder-only HF models such as Qwen2, Gemma2, and OLMo2. Gemma2
attention can be routed through FA4 with either native or polynomial tanh
softcapping for an apples-to-apples B2-style checkpoint experiment. The
Transformers grouped/batched MoE path is patched at its packed ``_apply_gate``
boundary so expert GEMMs and routing remain unchanged. The
GPT-OSS non-MXFP4 expert path is patched by replacing its gated-SwiGLU swish.
The GPT-OSS MXFP4 fused expert path is patched by swapping the Triton
``matmul_ogs`` fused activation to the selected current or Sollya odd/even
polynomial coefficients, keeping the activation fused into the expert matmul.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any, Callable, NamedTuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


SILU_RE = re.compile(r"^spline_silu_d(?P<degree>[0-9]+)_(?P<source>current|sollya)$")
GELU_RE = re.compile(r"^spline_gelu_d(?P<degree>[0-9]+)_(?P<source>current|sollya)$")
FUSED_SWIGLU_RE = re.compile(
    r"^fused_swiglu_d(?P<degree>[0-9]+)_(?P<source>current|sollya)$"
)
GEMMA_TANH_RE = re.compile(r"^gemma_tanh_(?P<source>current)$")
GEMMA_FA4_TANH_RE = re.compile(
    r"^gemma_fa4_tanh_(?:(?P<native>native)|d(?P<degree>[0-9]+)_(?P<source>current|sollya))$"
)
ROUTER_SIGMOID_RE = re.compile(
    r"^spline_router_sigmoid_d(?P<degree>[0-9]+)_(?P<source>current|sollya)$"
)
_URL_RE = re.compile(r"(?i)\b(?:https?|s3|gs|az)://[^\s<>'\"]+")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_TOKEN_RE = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{20,})\b"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth(?:orization)?|password|secret)"
    r"\s*[:=]\s*[^\s,;]+"
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    silu_degree: int | None = None
    silu_source: str = "current"
    gelu_degree: int | None = None
    gelu_source: str = "current"
    dense_swiglu_degree: int | None = None
    dense_swiglu_source: str = "current"
    gemma_tanh_source: str | None = None
    gemma_fa4_tanh_backend: str | None = None
    gemma_fa4_tanh_degree: int | None = None
    gemma_fa4_tanh_source: str = "current"
    router_sigmoid_degree: int | None = None
    router_sigmoid_source: str = "current"


@dataclass
class Result:
    model: str
    variant: str
    dtype: str
    device: str
    batch_size: int
    seq_len: int
    patched_silu_modules: int = 0
    patched_router_sigmoid_modules: int = 0
    patched_gemma_softcap: bool = False
    prefill_ms: float | None = None
    prefill_tokens_per_s: float | None = None
    decode_ms_per_token: float | None = None
    decode_tokens_per_s: float | None = None
    eval: dict[str, Any] | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


class ActivationSlot(NamedTuple):
    name: str
    module: torch.nn.Module
    attr: str
    fn: Callable[..., Any]


class DenseSwigluSlot(NamedTuple):
    name: str
    module: torch.nn.Module
    fn: Callable[..., Any]


class PackedMoeSwigluSlot(NamedTuple):
    name: str
    module: torch.nn.Module
    fn: Callable[..., Any]


class TensorExpertsSwigluSlot(NamedTuple):
    name: str
    module: torch.nn.Module
    fn: Callable[..., Any]


class PackedExpertsGateSlot(NamedTuple):
    name: str
    module: torch.nn.Module
    fn: Callable[..., Any]


class GptOssGateSlot(NamedTuple):
    name: str
    module: torch.nn.Module
    fn: Callable[..., Any]


class Mxfp4SwigluSlot(NamedTuple):
    name: str
    module: torch.nn.Module
    fn: Callable[..., Any]


class SigmoidRouterSlot(NamedTuple):
    name: str
    module: torch.nn.Module
    attribute: str
    fn: Callable[..., Any]


class FunctionActivation(torch.nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        return self.fn(x.contiguous())


def make_contiguous_activation(
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    def activation(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        return fn(x.contiguous())

    activation.__name__ = getattr(fn, "__name__", "contiguous_activation")
    return activation


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def parse_variant(raw: str) -> VariantSpec:
    silu_degree = None
    silu_source = "current"
    gelu_degree = None
    gelu_source = "current"
    dense_swiglu_degree = None
    dense_swiglu_source = "current"
    gemma_tanh_source = None
    gemma_fa4_tanh_backend = None
    gemma_fa4_tanh_degree = None
    gemma_fa4_tanh_source = "current"
    router_sigmoid_degree = None
    router_sigmoid_source = "current"
    for part in raw.split("+"):
        part = part.strip()
        if not part or part == "native":
            continue
        silu_match = SILU_RE.match(part)
        if silu_match:
            silu_degree = int(silu_match.group("degree"))
            silu_source = silu_match.group("source")
            continue
        gelu_match = GELU_RE.match(part)
        if gelu_match:
            gelu_degree = int(gelu_match.group("degree"))
            gelu_source = gelu_match.group("source")
            continue
        fused_swiglu_match = FUSED_SWIGLU_RE.match(part)
        if fused_swiglu_match:
            dense_swiglu_degree = int(fused_swiglu_match.group("degree"))
            dense_swiglu_source = fused_swiglu_match.group("source")
            continue
        tanh_match = GEMMA_TANH_RE.match(part)
        if tanh_match:
            gemma_tanh_source = tanh_match.group("source")
            continue
        fa4_tanh_match = GEMMA_FA4_TANH_RE.match(part)
        if fa4_tanh_match:
            if fa4_tanh_match.group("native"):
                gemma_fa4_tanh_backend = "native"
            else:
                gemma_fa4_tanh_backend = "device"
                gemma_fa4_tanh_degree = int(fa4_tanh_match.group("degree"))
                gemma_fa4_tanh_source = fa4_tanh_match.group("source")
            continue
        router_sigmoid_match = ROUTER_SIGMOID_RE.match(part)
        if router_sigmoid_match:
            router_sigmoid_degree = int(router_sigmoid_match.group("degree"))
            router_sigmoid_source = router_sigmoid_match.group("source")
            continue
        raise argparse.ArgumentTypeError(
            f"Unknown variant part {part!r}. Expected native, "
            "spline_silu_d{3,4,5,6}_{current,sollya}, "
            "spline_gelu_d{3,4,5,6}_{current,sollya}, "
            "fused_swiglu_d{3,4,5}_{current,sollya}, "
            "spline_router_sigmoid_d{3,4,5,6}_{current,sollya}, "
            "gemma_tanh_current, gemma_fa4_tanh_native, or "
            "gemma_fa4_tanh_d{3,4,5,6}_{current,sollya}."
        )
    return VariantSpec(
        name=raw,
        silu_degree=silu_degree,
        silu_source=silu_source,
        gelu_degree=gelu_degree,
        gelu_source=gelu_source,
        dense_swiglu_degree=dense_swiglu_degree,
        dense_swiglu_source=dense_swiglu_source,
        gemma_tanh_source=gemma_tanh_source,
        gemma_fa4_tanh_backend=gemma_fa4_tanh_backend,
        gemma_fa4_tanh_degree=gemma_fa4_tanh_degree,
        gemma_fa4_tanh_source=gemma_fa4_tanh_source,
        router_sigmoid_degree=router_sigmoid_degree,
        router_sigmoid_source=router_sigmoid_source,
    )


SOURCE_IDS = {
    "current": 0,
    "sollya": 1,
}

if triton is not None and tl is not None:

    @triton.jit
    def _poly_gated_swish_current_d3(gate, alpha):
        abs_gate = tl.abs(gate)
        t = tl.minimum(abs_gate * alpha, 6.0)
        h = t * ((0.0033893585 * t - 0.0533447266) * t + 0.2810058594)
        return 0.5 * gate + abs_gate * h

    @triton.jit
    def _poly_gated_swish_current_d4(gate, alpha):
        abs_gate = tl.abs(gate)
        t = tl.minimum(abs_gate * alpha, 5.25)
        h = (
            ((0.0005149841 * t - 0.0014419556) * t - 0.0402832031)
            * t
            + 0.2714843750
        )
        h = t * h
        return 0.5 * gate + abs_gate * h

    @triton.jit
    def _poly_gated_swish_current_d5(gate, alpha):
        abs_gate = tl.abs(gate)
        t = tl.minimum(abs_gate * alpha, 4.5)
        h = (
            (((-0.0001697540 * t + 0.0026550293) * t - 0.0107421875)
            * t
            - 0.0240478516)
            * t
            + 0.2617187500
        )
        h = t * h
        return 0.5 * gate + abs_gate * h

    @triton.jit
    def _poly_gated_swish_sollya_d3(gate, alpha):
        abs_gate = tl.abs(gate)
        t = tl.minimum(abs_gate * alpha, 6.0)
        h = t * ((0.0033874512 * t - 0.0534667969) * t + 0.28125)
        return 0.5 * gate + abs_gate * h

    @triton.jit
    def _poly_gated_swish_sollya_d4(gate, alpha):
        abs_gate = tl.abs(gate)
        t = tl.minimum(abs_gate * alpha, 5.28125)
        h = (
            ((0.0005722046 * t - 0.0020904541) * t - 0.0380859375)
            * t
            + 0.26953125
        )
        h = t * h
        return 0.5 * gate + abs_gate * h

    @triton.jit
    def _poly_gated_swish_sollya_d5(gate, alpha):
        abs_gate = tl.abs(gate)
        t = tl.minimum(abs_gate * alpha, 5.40625)
        h = (
            (((-0.0002956390 * t + 0.0042724609) * t - 0.0179443359)
            * t
            - 0.0115356445)
            * t
            + 0.255859375
        )
        h = t * h
        return 0.5 * gate + abs_gate * h

    @triton.jit
    def _gptoss_mxfp4_poly_swiglu_d3(input, alpha, limit):
        gate, up = tl.split(
            tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2))
        )
        gate = gate.to(tl.float32)
        up = up.to(tl.float32)
        if limit is not None:
            gate = tl.minimum(gate, limit)
            up = tl.minimum(tl.maximum(up, -limit), limit)
        glu = _poly_gated_swish_current_d3(gate, alpha)
        return glu * (up + 1.0)

    @triton.jit
    def _gptoss_mxfp4_poly_swiglu_d4(input, alpha, limit):
        gate, up = tl.split(
            tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2))
        )
        gate = gate.to(tl.float32)
        up = up.to(tl.float32)
        if limit is not None:
            gate = tl.minimum(gate, limit)
            up = tl.minimum(tl.maximum(up, -limit), limit)
        glu = _poly_gated_swish_current_d4(gate, alpha)
        return glu * (up + 1.0)

    @triton.jit
    def _gptoss_mxfp4_poly_swiglu_d5(input, alpha, limit):
        gate, up = tl.split(
            tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2))
        )
        gate = gate.to(tl.float32)
        up = up.to(tl.float32)
        if limit is not None:
            gate = tl.minimum(gate, limit)
            up = tl.minimum(tl.maximum(up, -limit), limit)
        glu = _poly_gated_swish_current_d5(gate, alpha)
        return glu * (up + 1.0)

    @triton.jit
    def _gptoss_mxfp4_poly_swiglu_sollya_d3(input, alpha, limit):
        gate, up = tl.split(
            tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2))
        )
        gate = gate.to(tl.float32)
        up = up.to(tl.float32)
        if limit is not None:
            gate = tl.minimum(gate, limit)
            up = tl.minimum(tl.maximum(up, -limit), limit)
        glu = _poly_gated_swish_sollya_d3(gate, alpha)
        return glu * (up + 1.0)

    @triton.jit
    def _gptoss_mxfp4_poly_swiglu_sollya_d4(input, alpha, limit):
        gate, up = tl.split(
            tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2))
        )
        gate = gate.to(tl.float32)
        up = up.to(tl.float32)
        if limit is not None:
            gate = tl.minimum(gate, limit)
            up = tl.minimum(tl.maximum(up, -limit), limit)
        glu = _poly_gated_swish_sollya_d4(gate, alpha)
        return glu * (up + 1.0)

    @triton.jit
    def _gptoss_mxfp4_poly_swiglu_sollya_d5(input, alpha, limit):
        gate, up = tl.split(
            tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2))
        )
        gate = gate.to(tl.float32)
        up = up.to(tl.float32)
        if limit is not None:
            gate = tl.minimum(gate, limit)
            up = tl.minimum(tl.maximum(up, -limit), limit)
        glu = _poly_gated_swish_sollya_d5(gate, alpha)
        return glu * (up + 1.0)

else:
    _gptoss_mxfp4_poly_swiglu_d3 = None
    _gptoss_mxfp4_poly_swiglu_d4 = None
    _gptoss_mxfp4_poly_swiglu_d5 = None
    _gptoss_mxfp4_poly_swiglu_sollya_d3 = None
    _gptoss_mxfp4_poly_swiglu_sollya_d4 = None
    _gptoss_mxfp4_poly_swiglu_sollya_d5 = None

GPTOSS_MXFP4_POLY_SWIGLU_FNS = {
    ("current", 3): _gptoss_mxfp4_poly_swiglu_d3,
    ("current", 4): _gptoss_mxfp4_poly_swiglu_d4,
    ("current", 5): _gptoss_mxfp4_poly_swiglu_d5,
    ("sollya", 3): _gptoss_mxfp4_poly_swiglu_sollya_d3,
    ("sollya", 4): _gptoss_mxfp4_poly_swiglu_sollya_d4,
    ("sollya", 5): _gptoss_mxfp4_poly_swiglu_sollya_d5,
}


def load_spline_ops_fast_swiglu() -> Any:
    try:
        import spline_ops
    except ImportError as exc:
        raise RuntimeError(
            "Dense fused polynomial SwiGLU requires spline_ops with "
            "swish_mul_fwd_variant; the fp32 Triton fallback was removed."
        ) from exc
    if not hasattr(spline_ops, "swish_mul_fwd_variant"):
        raise RuntimeError(
            "Dense fused polynomial SwiGLU requires rebuilt spline_ops with "
            "swish_mul_fwd_variant; the fp32 Triton fallback was removed."
        )
    return spline_ops


def load_spline_ops_fast_packed_swiglu() -> Any:
    spline_ops = load_spline_ops_fast_swiglu()
    if not hasattr(spline_ops, "swish_mul_packed_fwd_variant"):
        raise RuntimeError(
            "Packed fused polynomial SwiGLU requires rebuilt spline_ops with "
            "swish_mul_packed_fwd_variant."
        )
    return spline_ops


def dense_swiglu_poly(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    degree: int,
    coeff_source: str,
) -> torch.Tensor:
    if degree not in {3, 4, 5, 6}:
        raise ValueError(f"Dense fused polynomial SwiGLU only supports D3/D4/D5/D6, got D{degree}")
    if gate.numel() == 0:
        return torch.empty_like(up)
    gate = gate.contiguous()
    up = up.contiguous()
    if gate.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(
            "Dense fused polynomial SwiGLU only supports fp16/bf16 odd/even kernels."
        )
    spline_ops = load_spline_ops_fast_swiglu()
    return spline_ops.swish_mul_fwd_variant(
        gate, up, degree, SOURCE_IDS[coeff_source]
    )


def packed_swiglu_poly(
    packed_gate_up: torch.Tensor,
    *,
    degree: int,
    coeff_source: str,
) -> torch.Tensor:
    if degree not in {3, 4, 5, 6}:
        raise ValueError(f"Packed fused polynomial SwiGLU only supports D3/D4/D5/D6, got D{degree}")
    if packed_gate_up.numel() == 0:
        return packed_gate_up[..., : packed_gate_up.shape[-1] // 2].contiguous()
    packed_gate_up = packed_gate_up.contiguous()
    if packed_gate_up.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(
            "Packed fused polynomial SwiGLU only supports fp16/bf16 odd/even kernels."
        )
    spline_ops = load_spline_ops_fast_packed_swiglu()
    return spline_ops.swish_mul_packed_fwd_variant(
        packed_gate_up,
        degree,
        SOURCE_IDS[coeff_source],
    )


def load_spline_silu(
    degree: int,
    coeff_source: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    try:
        import spline_ops
    except ImportError as exc:
        raise RuntimeError(
            "Could not import spline_ops. Build the standalone extension with "
            "`python -m pip install -v ./autonumerics_zero/spline_ops` first."
        ) from exc
    coeff_source_id = SOURCE_IDS[coeff_source]

    def activation(x: torch.Tensor) -> torch.Tensor:
        return spline_ops.swish_fwd_variant(x, degree, coeff_source_id)

    activation.__name__ = f"spline_silu_d{degree}_{coeff_source}_direct"
    return activation


def load_spline_tanh() -> Callable[[torch.Tensor], torch.Tensor]:
    try:
        import spline_ops
    except ImportError as exc:
        raise RuntimeError(
            "Could not import spline_ops. Build the standalone extension with "
            "`python -m pip install -v ./autonumerics_zero/spline_ops` first."
        ) from exc

    def activation(x: torch.Tensor) -> torch.Tensor:
        return spline_ops.tanh_fwd(x.contiguous())

    activation.__name__ = "spline_tanh_direct"
    return activation


def load_spline_sigmoid(
    degree: int,
    coeff_source: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if degree not in {3, 4, 5, 6}:
        raise ValueError(
            f"Router sigmoid polynomial only supports D3/D4/D5/D6, got D{degree}"
        )
    try:
        import spline_ops
    except ImportError as exc:
        raise RuntimeError(
            "Could not import spline_ops. Build the standalone extension with "
            "`python -m pip install -v ./autonumerics_zero/spline_ops` first."
        ) from exc

    coeff_source_id = SOURCE_IDS[coeff_source]

    def activation(x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        kernel_input = x
        if input_dtype not in {torch.float16, torch.bfloat16}:
            kernel_input = x.to(torch.bfloat16)
        output = spline_ops.sigmoid_fwd_variant(
            kernel_input.contiguous(), degree, coeff_source_id
        )
        return output if output.dtype == input_dtype else output.to(input_dtype)

    activation.__name__ = f"spline_router_sigmoid_d{degree}_{coeff_source}"
    return activation


def load_spline_gelu(
    degree: int,
    coeff_source: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    try:
        import spline_ops
    except ImportError as exc:
        raise RuntimeError(
            "Could not import spline_ops. Build the standalone extension with "
            "`python -m pip install -v ./autonumerics_zero/spline_ops` first."
        ) from exc
    coeff_source_id = SOURCE_IDS[coeff_source]

    def activation(x: torch.Tensor) -> torch.Tensor:
        return spline_ops.gelu_fwd_variant(x, degree, coeff_source_id)

    activation.__name__ = f"spline_gelu_d{degree}_{coeff_source}_direct"
    return activation


def looks_like_silu_module(module: torch.nn.Module, fn: Any) -> bool:
    config = getattr(module, "config", None)
    config_names = [
        getattr(config, "hidden_act", None),
        getattr(config, "hidden_activation", None),
    ]
    for name in config_names:
        if isinstance(name, str) and name.lower() in {"silu", "swish"}:
            return True
    if isinstance(fn, torch.nn.SiLU):
        return True
    class_name = fn.__class__.__name__
    if isinstance(class_name, str) and class_name.lower() in {"silu", "swish"}:
        return True
    fn_name = getattr(fn, "__name__", "")
    return isinstance(fn_name, str) and fn_name.lower() in {"silu", "swish"}


def looks_like_gelu_module(module: torch.nn.Module, fn: Any) -> bool:
    config = getattr(module, "config", None)
    config_names = [
        getattr(config, "hidden_act", None),
        getattr(config, "hidden_activation", None),
        getattr(config, "activation_function", None),
    ]
    for name in config_names:
        if isinstance(name, str) and "gelu" in name.lower():
            return True
    if isinstance(fn, torch.nn.GELU):
        return True
    class_name = fn.__class__.__name__
    if isinstance(class_name, str) and "gelu" in class_name.lower():
        return True
    fn_name = getattr(fn, "__name__", "")
    return isinstance(fn_name, str) and "gelu" in fn_name.lower()


def collect_silu_act_fns(
    model: torch.nn.Module,
) -> list[ActivationSlot]:
    modules: list[ActivationSlot] = []
    for name, module in model.named_modules():
        for attr in ("act_fn", "activation", "act"):
            fn = getattr(module, attr, None)
            if fn is not None and looks_like_silu_module(module, fn):
                modules.append(ActivationSlot(name=name, module=module, attr=attr, fn=fn))
    return modules


def collect_gelu_act_fns(
    model: torch.nn.Module,
) -> list[ActivationSlot]:
    modules: list[ActivationSlot] = []
    for name, module in model.named_modules():
        for attr in ("act_fn", "activation", "act"):
            fn = getattr(module, attr, None)
            if fn is not None and looks_like_gelu_module(module, fn):
                modules.append(ActivationSlot(name=name, module=module, attr=attr, fn=fn))
    return modules


def collect_dense_swiglu_fns(model: torch.nn.Module) -> list[DenseSwigluSlot]:
    modules: list[DenseSwigluSlot] = []
    for name, module in model.named_modules():
        has_llama_names = all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj"))
        has_phi_names = all(hasattr(module, attr) for attr in ("w1", "w2", "w3"))
        if not has_llama_names and not has_phi_names:
            continue
        fn = getattr(module, "act_fn", None)
        if fn is not None and looks_like_silu_module(module, fn):
            modules.append(DenseSwigluSlot(name=name, module=module, fn=module.forward))
    return modules


def collect_packed_moe_swiglu_fns(model: torch.nn.Module) -> list[PackedMoeSwigluSlot]:
    modules: list[PackedMoeSwigluSlot] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "GraniteMoeMoE":
            continue
        if not all(hasattr(module, attr) for attr in ("input_linear", "output_linear", "router")):
            continue
        fn = getattr(module, "activation", None)
        if fn is not None and looks_like_silu_module(module, fn):
            modules.append(PackedMoeSwigluSlot(name=name, module=module, fn=module.forward))
    return modules


def collect_tensor_experts_swiglu_fns(model: torch.nn.Module) -> list[TensorExpertsSwigluSlot]:
    # HF tensorized MoE expert classes are usually decorated with a kernelized
    # experts implementation. Replacing their forward with a Python loop is much
    # slower, so do not patch this path without a lower-level fused kernel hook.
    return []

    modules: list[TensorExpertsSwigluSlot] = []
    for name, module in model.named_modules():
        if not module.__class__.__name__.endswith("Experts"):
            continue
        if not all(
            hasattr(module, attr)
            for attr in ("num_experts", "gate_up_proj", "down_proj", "act_fn")
        ):
            continue
        fn = getattr(module, "act_fn", None)
        if fn is not None and looks_like_silu_module(module, fn):
            modules.append(TensorExpertsSwigluSlot(name=name, module=module, fn=module.forward))
    return modules


def collect_packed_experts_gate_fns(
    model: torch.nn.Module,
) -> list[PackedExpertsGateSlot]:
    modules: list[PackedExpertsGateSlot] = []
    for name, module in model.named_modules():
        if not all(
            hasattr(module, attr)
            for attr in ("_apply_gate", "act_fn", "gate_up_proj", "down_proj")
        ):
            continue
        if not getattr(module, "has_gate", False):
            continue
        if not getattr(module, "is_concatenated", False):
            continue
        fn = getattr(module, "act_fn", None)
        if fn is not None and looks_like_silu_module(module, fn):
            modules.append(
                PackedExpertsGateSlot(
                    name=name,
                    module=module,
                    fn=module._apply_gate,
                )
            )
    return modules


@torch.inference_mode()
def capture_packed_moe_routes(
    model: torch.nn.Module,
    slots: list[PackedMoeSwigluSlot],
    tokens: torch.Tensor,
) -> int:
    if not slots:
        return 0

    from types import MethodType

    originals: list[tuple[torch.nn.Module, Callable[..., Any]]] = []
    caches: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], torch.Tensor]]] = {
        id(slot.module): [] for slot in slots
    }

    for slot in slots:
        router = slot.module.router
        original = router.forward
        originals.append((router, original))

        def router_forward(
            self: torch.nn.Module,
            hidden_states: torch.Tensor,
            *,
            original: Callable[..., Any] = original,
            module: torch.nn.Module = slot.module,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
            out = original(hidden_states)
            index_sorted_experts, batch_index, batch_gates, expert_size, router_logits = out
            caches[id(module)].append(
                (
                    index_sorted_experts.detach().clone(),
                    batch_index.detach().clone(),
                    batch_gates.detach().clone(),
                    list(expert_size),
                    router_logits.detach().clone(),
                )
            )
            return out

        router.forward = MethodType(router_forward, router)

    try:
        model(input_ids=tokens, use_cache=False)
        torch.cuda.synchronize()
    finally:
        for router, original in originals:
            router.forward = original

    captured = 0
    for slot in slots:
        cache = caches[id(slot.module)]
        if cache:
            slot.module._poly_frozen_route = cache[0]
            captured += 1
    return captured


def collect_gptoss_gate_fns(model: torch.nn.Module) -> list[GptOssGateSlot]:
    modules: list[GptOssGateSlot] = []
    for name, module in model.named_modules():
        fn = getattr(module, "_apply_gate", None)
        if fn is None:
            continue
        if not all(hasattr(module, attr) for attr in ("alpha", "limit")):
            continue
        if "gptoss" not in module.__class__.__name__.lower():
            continue
        modules.append(GptOssGateSlot(name=name, module=module, fn=fn))
    return modules


def collect_mxfp4_swiglu_fns(model: torch.nn.Module) -> list[Mxfp4SwigluSlot]:
    modules: list[Mxfp4SwigluSlot] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Mxfp4GptOssExperts":
            continue
        if not all(
            hasattr(module, attr)
            for attr in (
                "alpha",
                "limit",
                "gate_up_proj",
                "down_proj",
                "gate_up_proj_precision_config",
                "down_proj_precision_config",
            )
        ):
            continue
        modules.append(Mxfp4SwigluSlot(name=name, module=module, fn=module.forward))
    return modules


def collect_sigmoid_router_fns(model: torch.nn.Module) -> list[SigmoidRouterSlot]:
    modules: list[SigmoidRouterSlot] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Glm4MoeLiteMoE":
            modules.append(
                SigmoidRouterSlot(
                    name=name,
                    module=module,
                    attribute="route_tokens_to_experts",
                    fn=module.route_tokens_to_experts,
                )
            )
        elif module.__class__.__name__ == "KimiMoEGate":
            modules.append(
                SigmoidRouterSlot(
                    name=name,
                    module=module,
                    attribute="forward",
                    fn=module.forward,
                )
            )
    return modules


def _run_glm_sigmoid_router(
    module: torch.nn.Module,
    router_logits: torch.Tensor,
    sigmoid_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = sigmoid_fn(router_logits)
    scores_for_choice = scores + module.gate.e_score_correction_bias
    group_scores = (
        scores_for_choice.view(
            -1,
            module.n_group,
            module.n_routed_experts // module.n_group,
        )
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(
        group_scores,
        k=module.topk_group,
        dim=-1,
        sorted=False,
    )[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, module.n_group, module.n_routed_experts // module.n_group)
        .reshape(-1, module.n_routed_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )
    topk_indices = torch.topk(
        scores_for_choice,
        k=module.top_k,
        dim=-1,
        sorted=False,
    )[1]
    topk_weights = scores.gather(1, topk_indices)
    if module.norm_topk_prob:
        topk_weights = topk_weights / (
            topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        )
    topk_weights = topk_weights * module.routed_scaling_factor
    return topk_indices, topk_weights


def _run_kimi_sigmoid_router(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    sigmoid_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_size = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_size)
    logits = torch.nn.functional.linear(
        hidden_states.to(torch.float32), module.weight.to(torch.float32)
    )
    scores = sigmoid_fn(logits)
    scores_for_choice = scores.view(batch_size * sequence_length, -1)
    # Preserve the checkpoint's remote-code behavior: this in-place update also
    # affects the scores later gathered into top-k weights.
    scores_for_choice += module.e_score_correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(
            batch_size * sequence_length,
            module.num_expert_group,
            -1,
        )
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(
        group_scores,
        k=module.topk_group,
        dim=-1,
        sorted=False,
    )[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(
            batch_size * sequence_length,
            module.num_expert_group,
            module.num_experts // module.num_expert_group,
        )
        .reshape(batch_size * sequence_length, -1)
    )
    masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = torch.topk(
        masked_scores,
        k=module.top_k,
        dim=-1,
        sorted=False,
    )[1]
    topk_weights = scores.gather(1, topk_indices)
    if module.top_k > 1 and module.moe_renormalize:
        topk_weights = topk_weights / (
            topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        )
    topk_weights = topk_weights * module.routed_scaling_factor
    return topk_indices, topk_weights


def apply_sigmoid_router_patch(
    originals: list[SigmoidRouterSlot],
    spec: VariantSpec,
) -> int:
    if spec.router_sigmoid_degree is None:
        for slot in originals:
            setattr(slot.module, slot.attribute, slot.fn)
        return 0
    if not originals:
        return 0

    from types import MethodType

    sigmoid_fn = load_spline_sigmoid(
        spec.router_sigmoid_degree,
        spec.router_sigmoid_source,
    )
    for slot in originals:
        module = slot.module
        if module.__class__.__name__ == "Glm4MoeLiteMoE":

            def route_tokens_to_experts(
                self: torch.nn.Module,
                router_logits: torch.Tensor,
                *,
                sigmoid_fn: Callable[[torch.Tensor], torch.Tensor] = sigmoid_fn,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return _run_glm_sigmoid_router(self, router_logits, sigmoid_fn)

            replacement = route_tokens_to_experts

        elif module.__class__.__name__ == "KimiMoEGate":

            def forward(
                self: torch.nn.Module,
                hidden_states: torch.Tensor,
                *,
                sigmoid_fn: Callable[[torch.Tensor], torch.Tensor] = sigmoid_fn,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return _run_kimi_sigmoid_router(self, hidden_states, sigmoid_fn)

            replacement = forward

        else:
            raise RuntimeError(
                f"Unsupported sigmoid router class: {module.__class__.__name__}"
            )
        setattr(module, slot.attribute, MethodType(replacement, module))
    return len(originals)


def apply_silu_patch(
    originals: list[ActivationSlot],
    spec: VariantSpec,
) -> int:
    if spec.silu_degree is None:
        for slot in originals:
            setattr(slot.module, slot.attr, slot.fn)
        return 0

    activation = make_contiguous_activation(
        load_spline_silu(
            spec.silu_degree,
            spec.silu_source,
        )
    )
    for slot in originals:
        replacement: Callable[..., Any] | torch.nn.Module = activation
        if isinstance(slot.fn, torch.nn.Module):
            replacement = FunctionActivation(activation)
        setattr(slot.module, slot.attr, replacement)
    return len(originals)


def apply_gelu_patch(
    originals: list[ActivationSlot],
    spec: VariantSpec,
) -> int:
    if spec.gelu_degree is None:
        for slot in originals:
            setattr(slot.module, slot.attr, slot.fn)
        return 0

    activation = make_contiguous_activation(
        load_spline_gelu(
            spec.gelu_degree,
            spec.gelu_source,
        )
    )
    for slot in originals:
        replacement: Callable[..., Any] | torch.nn.Module = activation
        if isinstance(slot.fn, torch.nn.Module):
            replacement = FunctionActivation(activation)
        setattr(slot.module, slot.attr, replacement)
    return len(originals)


def apply_dense_swiglu_patch(
    originals: list[DenseSwigluSlot],
    spec: VariantSpec,
) -> int:
    if spec.dense_swiglu_degree is None:
        for slot in originals:
            slot.module.forward = slot.fn
        return 0
    if not originals:
        return 0
    load_spline_ops_fast_swiglu()

    from types import MethodType

    for slot in originals:
        module = slot.module

        def forward(
            self: torch.nn.Module,
            x: torch.Tensor,
            *,
            degree: int = spec.dense_swiglu_degree,
            coeff_source: str = spec.dense_swiglu_source,
        ) -> torch.Tensor:
            if hasattr(self, "gate_proj"):
                gate = self.gate_proj(x)
                up = self.up_proj(x)
                return self.down_proj(
                    dense_swiglu_poly(
                        gate, up, degree=degree, coeff_source=coeff_source
                    )
                )
            gate = self.w1(x)
            up = self.w3(x)
            return self.w2(
                dense_swiglu_poly(
                    gate, up, degree=degree, coeff_source=coeff_source
                )
            )

        module.forward = MethodType(forward, module)
    return len(originals)


def apply_packed_moe_swiglu_patch(
    originals: list[PackedMoeSwigluSlot],
    spec: VariantSpec,
    *,
    freeze_routing: bool,
) -> int:
    if spec.dense_swiglu_degree is None and not freeze_routing:
        for slot in originals:
            slot.module.forward = slot.fn
        return 0
    if not originals:
        return 0
    if spec.dense_swiglu_degree is not None:
        load_spline_ops_fast_packed_swiglu()

    from types import MethodType

    for slot in originals:
        module = slot.module

        def forward(
            self: torch.nn.Module,
            layer_input: torch.Tensor,
            *,
            degree: int = spec.dense_swiglu_degree,
            coeff_source: str = spec.dense_swiglu_source,
            freeze_routing: bool = freeze_routing,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            bsz, length, emb_size = layer_input.size()
            layer_input = layer_input.reshape(-1, emb_size)
            _, batch_index, batch_gates, expert_size, router_logits = self.router(layer_input)
            if freeze_routing:
                frozen_route = getattr(self, "_poly_frozen_route", None)
                if frozen_route is None:
                    raise RuntimeError("Granite MoE frozen routing was requested but no route cache exists.")
                _, batch_index, batch_gates, expert_size, _ = frozen_route

            expert_inputs = layer_input[batch_index]
            hidden_states = self.input_linear(expert_inputs, expert_size)
            if degree is None:
                gate, up = hidden_states.chunk(2, dim=-1)
                hidden_states = self.activation(gate) * up
            else:
                hidden_states = packed_swiglu_poly(
                    hidden_states, degree=degree, coeff_source=coeff_source
                )
            expert_outputs = self.output_linear(hidden_states, expert_size)
            expert_outputs = expert_outputs * batch_gates[:, None]

            zeros = torch.zeros(
                (bsz * length, self.input_size),
                dtype=expert_outputs.dtype,
                device=expert_outputs.device,
            )
            layer_output = zeros.index_add(0, batch_index, expert_outputs)
            layer_output = layer_output.view(bsz, length, self.input_size)
            return layer_output, router_logits

        module.forward = MethodType(forward, module)
    return len(originals) if spec.dense_swiglu_degree is not None else 0


def apply_tensor_experts_swiglu_patch(
    originals: list[TensorExpertsSwigluSlot],
    spec: VariantSpec,
) -> int:
    if spec.dense_swiglu_degree is None:
        for slot in originals:
            slot.module.forward = slot.fn
        return 0
    if not originals:
        return 0
    load_spline_ops_fast_packed_swiglu()

    from types import MethodType

    for slot in originals:
        module = slot.module

        def forward(
            self: torch.nn.Module,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
            *,
            degree: int = spec.dense_swiglu_degree,
            coeff_source: str = spec.dense_swiglu_source,
        ) -> torch.Tensor:
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(
                    top_k_index,
                    num_classes=self.num_experts,
                )
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate_up = torch.nn.functional.linear(
                    current_state,
                    self.gate_up_proj[expert_idx],
                )
                current_hidden_states = packed_swiglu_poly(
                    gate_up, degree=degree, coeff_source=coeff_source
                )
                current_hidden_states = torch.nn.functional.linear(
                    current_hidden_states,
                    self.down_proj[expert_idx],
                )
                current_hidden_states = (
                    current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                )
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    current_hidden_states.to(final_hidden_states.dtype),
                )

            return final_hidden_states

        module.forward = MethodType(forward, module)
    return len(originals)


def apply_packed_experts_gate_patch(
    originals: list[PackedExpertsGateSlot],
    spec: VariantSpec,
) -> int:
    if spec.dense_swiglu_degree is None:
        for slot in originals:
            slot.module._apply_gate = slot.fn
        return 0
    if not originals:
        return 0
    load_spline_ops_fast_packed_swiglu()

    from types import MethodType

    for slot in originals:
        module = slot.module

        def apply_gate(
            self: torch.nn.Module,
            packed_gate_up: torch.Tensor,
            *,
            degree: int = spec.dense_swiglu_degree,
            coeff_source: str = spec.dense_swiglu_source,
        ) -> torch.Tensor:
            return packed_swiglu_poly(
                packed_gate_up, degree=degree, coeff_source=coeff_source
            )

        module._apply_gate = MethodType(apply_gate, module)
    return len(originals)


def apply_gptoss_gate_patch(
    originals: list[GptOssGateSlot],
    spec: VariantSpec,
) -> int:
    if spec.silu_degree is None:
        for slot in originals:
            slot.module._apply_gate = slot.fn
        return 0
    for slot in originals:
        module = slot.module

        def apply_gate(gate_up: torch.Tensor, *, module=module) -> torch.Tensor:
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=module.limit)
            up = up.clamp(min=-module.limit, max=module.limit)
            return dense_swiglu_poly(
                gate * module.alpha,
                up + 1,
                degree=spec.silu_degree,
                coeff_source=spec.silu_source,
            ) / module.alpha

        module._apply_gate = apply_gate
    return len(originals)


def apply_mxfp4_swiglu_patch(
    originals: list[Mxfp4SwigluSlot],
    spec: VariantSpec,
) -> int:
    if spec.silu_degree is None:
        for slot in originals:
            slot.module.forward = slot.fn
        return 0
    if not originals:
        return 0
    poly_fn = GPTOSS_MXFP4_POLY_SWIGLU_FNS.get(
        (spec.silu_source, spec.silu_degree)
    )
    if poly_fn is None:
        raise RuntimeError("GPT-OSS MXFP4 fused patch requires Triton and supports D3/D4/D5.")

    import importlib
    from types import MethodType

    for slot in originals:
        module = slot.module
        mxfp4_module = importlib.import_module(module.__class__.__module__)
        hub = getattr(mxfp4_module, "triton_kernels_hub", None)
        on_device = getattr(mxfp4_module, "on_device", None)
        if hub is None or on_device is None:
            raise RuntimeError("Could not locate GPT-OSS MXFP4 Triton kernel hub.")

        FnSpecs = hub.matmul_ogs.FnSpecs
        FusedActivation = hub.matmul_ogs.FusedActivation
        matmul_ogs = hub.matmul_ogs.matmul_ogs

        def forward(
            self: torch.nn.Module,
            hidden_states: torch.Tensor,
            routing_data: Any,
            gather_idx: torch.Tensor,
            scatter_idx: torch.Tensor,
            *,
            degree: int = spec.silu_degree,
            coeff_source: str = spec.silu_source,
            poly_fn: Any = poly_fn,
            FnSpecs: Any = FnSpecs,
            FusedActivation: Any = FusedActivation,
            matmul_ogs: Any = matmul_ogs,
            on_device: Any = on_device,
        ) -> torch.Tensor:
            with on_device(hidden_states.device):
                act = FusedActivation(
                    FnSpecs(
                        f"poly_swiglu_d{degree}_{coeff_source}",
                        poly_fn,
                        ("alpha", "limit"),
                    ),
                    (self.alpha, self.limit),
                    2,
                )
                intermediate_cache1 = matmul_ogs(
                    hidden_states,
                    self.gate_up_proj,
                    self.gate_up_proj_bias.to(torch.float32),
                    routing_data,
                    gather_indx=gather_idx,
                    precision_config=self.gate_up_proj_precision_config,
                    gammas=None,
                    fused_activation=act,
                )
                intermediate_cache3 = matmul_ogs(
                    intermediate_cache1,
                    self.down_proj,
                    self.down_proj_bias.to(torch.float32),
                    routing_data,
                    scatter_indx=scatter_idx,
                    precision_config=self.down_proj_precision_config,
                    gammas=routing_data.gate_scal,
                )
            return intermediate_cache3

        module.forward = MethodType(forward, module)
    return len(originals)


_ORIGINAL_GEMMA_EAGER_ATTENTION: Callable[..., Any] | None = None


def restore_gemma_attention() -> None:
    global _ORIGINAL_GEMMA_EAGER_ATTENTION
    if _ORIGINAL_GEMMA_EAGER_ATTENTION is None:
        return
    try:
        from transformers.models.gemma2 import modeling_gemma2
    except ImportError:
        return
    modeling_gemma2.eager_attention_forward = _ORIGINAL_GEMMA_EAGER_ATTENTION


def load_gemma_fa4_runtime() -> dict[str, Any]:
    """Load FlashAttention 4 from the installed, pinned public dependency."""

    try:
        from flash_attn.cute.interface import _flash_attn_fwd
        from flash_attn.cute.utils import create_softcap_scoremod_backend
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "Could not import the FA4 runtime. Install this repository's pinned "
            "FlashAttention and CuTe DSL dependencies in the active environment. "
            f"Root error: {exc}"
        ) from exc
    return {
        "_flash_attn_fwd": _flash_attn_fwd,
        "create_softcap_scoremod_backend": create_softcap_scoremod_backend,
    }


def gemma_fa4_sequence_lengths(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 4:
        raise RuntimeError(
            "Gemma FA4 expects the eager 4D additive attention mask, got "
            f"shape {tuple(attention_mask.shape)}"
        )
    expected_prefix = (batch_size, 1, query_length)
    if tuple(attention_mask.shape[:3]) != expected_prefix:
        raise RuntimeError(
            "Gemma FA4 attention-mask shape does not match Q: "
            f"mask={tuple(attention_mask.shape)}, expected prefix={expected_prefix}"
        )
    if attention_mask.shape[-1] != key_length:
        raise RuntimeError(
            "Gemma FA4 attention-mask length does not match K/V: "
            f"mask={attention_mask.shape[-1]}, key_length={key_length}"
        )

    # The last query row has the furthest valid key for a causal mask. Inspecting
    # only that row avoids an O(query_length * key_length) reduction per layer.
    allowed = attention_mask[:, 0, -1] == 0
    key_positions = torch.arange(
        1,
        key_length + 1,
        dtype=torch.int32,
        device=attention_mask.device,
    )
    return torch.where(allowed, key_positions, 0).amax(dim=1)


def build_gemma_fa4_attention_forward(
    spec: VariantSpec,
) -> Callable[..., tuple[torch.Tensor, None]]:
    backend = spec.gemma_fa4_tanh_backend
    if backend not in {"native", "device"}:
        raise ValueError(f"Unsupported Gemma FA4 tanh backend: {backend!r}")
    if backend == "device" and spec.gemma_fa4_tanh_degree not in {3, 4, 5, 6}:
        raise ValueError(
            "Polynomial Gemma FA4 softcap requires degree D3/D4/D5/D6"
        )

    runtime = load_gemma_fa4_runtime()
    flash_attn_fwd = runtime["_flash_attn_fwd"]
    score_mod_cache: dict[float, Callable[..., Any]] = {}

    def fa4_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        softcap: float | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if module.training or dropout:
            raise RuntimeError("Gemma FA4 evaluation patch only supports inference")
        if kwargs.get("output_attentions", False):
            raise RuntimeError("Gemma FA4 evaluation patch does not materialize attention weights")
        if softcap is None or softcap <= 0:
            raise RuntimeError(f"Gemma FA4 requires a positive softcap, got {softcap!r}")
        if scaling is None:
            scaling = module.head_dim**-0.5

        batch_size, _, query_length, _ = query.shape
        key_length = key.shape[-2]
        sequence_lengths = gemma_fa4_sequence_lengths(
            attention_mask,
            batch_size=batch_size,
            query_length=query_length,
            key_length=key_length,
        )
        sliding_window = kwargs.get("sliding_window", module.sliding_window)
        window_size_left = None
        window_size_right = None
        if sliding_window is not None:
            # HF Gemma counts the current token in sliding_window; FA4's left
            # radius does not, so a 4096-token window maps to a radius of 4095.
            window_size_left = max(int(sliding_window) - 1, 0)
            window_size_right = 0

        score_mod = None
        native_softcap = float(softcap)
        if backend == "device":
            native_softcap = 0.0
            score_mod = score_mod_cache.get(float(softcap))
            if score_mod is None:
                score_mod = runtime["create_softcap_scoremod_backend"](
                    float(softcap),
                    degree=spec.gemma_fa4_tanh_degree,
                    backend="device",
                    coeff_source=spec.gemma_fa4_tanh_source,
                )
                score_mod_cache[float(softcap)] = score_mod

        output, _ = flash_attn_fwd(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            seqused_k=sequence_lengths,
            softmax_scale=scaling,
            causal=True,
            softcap=native_softcap,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            score_mod=score_mod,
        )
        return output, None

    return fa4_attention_forward


def apply_gemma_attention_patch(spec: VariantSpec) -> bool:
    restore_gemma_attention()
    if spec.gemma_tanh_source is None and spec.gemma_fa4_tanh_backend is None:
        return False

    try:
        from transformers.models.gemma2 import modeling_gemma2
    except ImportError:
        return False

    global _ORIGINAL_GEMMA_EAGER_ATTENTION
    if _ORIGINAL_GEMMA_EAGER_ATTENTION is None:
        _ORIGINAL_GEMMA_EAGER_ATTENTION = modeling_gemma2.eager_attention_forward

    if spec.gemma_fa4_tanh_backend is not None:
        modeling_gemma2.eager_attention_forward = build_gemma_fa4_attention_forward(
            spec
        )
        return True

    tanh_fn = load_spline_tanh()

    def poly_eager_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        softcap: float | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        if scaling is None:
            scaling = module.head_dim**-0.5

        key_states = modeling_gemma2.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_gemma2.repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if softcap is not None:
            attn_weights = tanh_fn(attn_weights / softcap) * softcap
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    modeling_gemma2.eager_attention_forward = poly_eager_attention_forward
    return True


def cuda_elapsed_ms(start: torch.cuda.Event, stop: torch.cuda.Event) -> float:
    stop.synchronize()
    return float(start.elapsed_time(stop))


def make_tokens(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(
        low=1,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        generator=generator,
    )


def model_vocab_size(model: torch.nn.Module) -> int:
    embeddings = model.get_input_embeddings()
    if embeddings is not None and hasattr(embeddings, "num_embeddings"):
        return int(embeddings.num_embeddings)
    return int(model.config.vocab_size)


def make_decode_cache(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    decode_steps: int,
) -> Any | None:
    """Preallocate fixed-size caches used by direct Gemma2 decoding."""
    if getattr(model.config, "model_type", None) != "gemma2":
        return None

    try:
        from transformers.cache_utils import HybridCache
    except ImportError as exc:
        raise RuntimeError(
            "Gemma2 decode benchmarking requires transformers.cache_utils.HybridCache"
        ) from exc

    try:
        cache_dtype = next(model.parameters()).dtype
    except StopIteration:
        cache_dtype = torch.get_default_dtype()
    return HybridCache(
        config=model.config,
        max_batch_size=tokens.shape[0],
        max_cache_len=tokens.shape[1] + decode_steps,
        device=tokens.device,
        dtype=cache_dtype,
    )


@torch.inference_mode()
def benchmark_prefill(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    warmup: int,
    steps: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        model(input_ids=tokens, use_cache=False)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        model(input_ids=tokens, use_cache=False)
    stop.record()
    mean_ms = cuda_elapsed_ms(start, stop) / steps
    tokens_per_s = tokens.numel() * 1000.0 / mean_ms
    return mean_ms, tokens_per_s


@torch.inference_mode()
def benchmark_decode(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
    decode_steps: int,
    seed: int,
) -> tuple[float, float]:
    batch_size, prompt_length = tokens.shape
    vocab_size = model_vocab_size(model)
    generator = torch.Generator(device=tokens.device).manual_seed(seed + 17)
    cache_positions = None
    if getattr(model.config, "model_type", None) == "gemma2":
        cache_positions = torch.arange(
            prompt_length + decode_steps,
            device=tokens.device,
        )

    def run_loop(timed: bool) -> float:
        decode_cache = make_decode_cache(model, tokens, decode_steps=decode_steps)
        prefill_kwargs: dict[str, Any] = {}
        if decode_cache is not None:
            assert cache_positions is not None
            prefill_kwargs.update(
                past_key_values=decode_cache,
                cache_position=cache_positions[:prompt_length],
            )
        out = model(input_ids=tokens, use_cache=True, **prefill_kwargs)
        past_key_values = out.past_key_values
        next_token = torch.randint(
            1,
            vocab_size,
            (batch_size, 1),
            device=tokens.device,
            generator=generator,
        )
        if not timed:
            for decode_idx in range(decode_steps):
                decode_kwargs: dict[str, Any] = {}
                if decode_cache is not None:
                    assert cache_positions is not None
                    position = prompt_length + decode_idx
                    decode_kwargs["cache_position"] = cache_positions[position : position + 1]
                out = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    **decode_kwargs,
                )
                past_key_values = out.past_key_values
            return 0.0

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for decode_idx in range(decode_steps):
            decode_kwargs = {}
            if decode_cache is not None:
                assert cache_positions is not None
                position = prompt_length + decode_idx
                decode_kwargs["cache_position"] = cache_positions[position : position + 1]
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                **decode_kwargs,
            )
            past_key_values = out.past_key_values
        stop.record()
        return cuda_elapsed_ms(start, stop)

    for _ in range(warmup):
        run_loop(timed=False)

    total_ms = 0.0
    for _ in range(repeats):
        total_ms += run_loop(timed=True)
    ms_per_token = total_ms / (repeats * decode_steps)
    tokens_per_s = batch_size * 1000.0 / ms_per_token
    return ms_per_token, tokens_per_s


def run_lm_eval(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: list[str],
    num_fewshot: int,
    task_num_fewshot: dict[str, int],
    limit: int | None,
    batch_size: str,
    device: str,
    log_samples: bool,
) -> dict[str, Any]:
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise RuntimeError(
            "lm_eval is not installed. Install this artifact's evaluation "
            "dependencies or install lm-eval directly."
        ) from exc

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )
    tasks_by_fewshot: dict[int, list[str]] = {}
    for task in tasks:
        shots = task_num_fewshot.get(task, num_fewshot)
        tasks_by_fewshot.setdefault(shots, []).append(task)

    evaluations: list[dict[str, Any]] = []
    for shots, grouped_tasks in tasks_by_fewshot.items():
        evaluation = evaluator.simple_evaluate(
            model=lm,
            tasks=grouped_tasks,
            num_fewshot=shots,
            limit=limit,
            batch_size=batch_size,
            log_samples=log_samples,
        )
        if evaluation is None:
            raise RuntimeError(f"lm-eval returned no results for tasks {grouped_tasks}")
        evaluations.append(evaluation)

    merged = merge_lm_eval_results(evaluations)
    merged["task_num_fewshot"] = {
        task: task_num_fewshot.get(task, num_fewshot) for task in tasks
    }
    return merged


def merge_lm_eval_results(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    if not evaluations:
        return {}

    merged = dict(evaluations[0])
    for evaluation in evaluations[1:]:
        for key, value in evaluation.items():
            if key == "config":
                continue
            if isinstance(value, dict):
                existing = merged.get(key)
                if isinstance(existing, dict):
                    existing.update(value)
                else:
                    merged[key] = dict(value)
            elif key not in merged:
                merged[key] = value
    return merged


def parse_task_num_fewshot(values: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        task, separator, raw_shots = value.partition("=")
        task = task.strip()
        if not separator or not task:
            raise ValueError(
                f"Invalid --eval-task-fewshot value {value!r}; expected TASK=SHOTS"
            )
        try:
            shots = int(raw_shots)
        except ValueError as exc:
            raise ValueError(
                f"Invalid few-shot count in --eval-task-fewshot {value!r}"
            ) from exc
        if shots < 0:
            raise ValueError("Few-shot counts must be non-negative")
        parsed[task] = shots
    return parsed


def load_model_and_tokenizer(args: argparse.Namespace, model_id: str, dtype: torch.dtype):
    attn_implementation = args.attn_implementation
    variants = [parse_variant(raw) for raw in args.variant]
    if any(
        spec.gemma_tanh_source or spec.gemma_fa4_tanh_backend
        for spec in variants
    ) and attn_implementation != "eager":
        print(
            "[info] Gemma tanh patch installs through the eager attention hook; "
            "loading with attn_implementation=eager",
            flush=True,
        )
        attn_implementation = "eager"

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "attn_implementation": attn_implementation,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.mxfp4_dequantize:
        try:
            from transformers import Mxfp4Config
        except ImportError as exc:
            raise RuntimeError(
                "--mxfp4-dequantize requires a Transformers build with Mxfp4Config"
            ) from exc
        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    if args.experts_implementation:
        load_kwargs["experts_implementation"] = args.experts_implementation
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir
    if args.revision:
        load_kwargs["revision"] = args.revision
    # Authentication is accepted only through the process environment so it
    # cannot leak through command histories or dry-run output.
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        load_kwargs["token"] = hf_token

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.to(args.device)
    model.eval()

    tokenizer = None
    if args.eval_tasks:
        tok_kwargs: dict[str, Any] = {
            "trust_remote_code": args.trust_remote_code,
        }
        if args.cache_dir:
            tok_kwargs["cache_dir"] = args.cache_dir
        if args.revision:
            tok_kwargs["revision"] = args.revision
        if hf_token:
            tok_kwargs["token"] = hf_token
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
        probe_tokens = tokenizer.encode(
            "Tokenizer smoke test",
            add_special_tokens=False,
        )
        if len(tokenizer) <= 1 or not probe_tokens:
            raise RuntimeError(
                f"Tokenizer assets for {model_id} are incomplete: vocabulary size "
                f"is {len(tokenizer)} and the smoke probe produced "
                f"{len(probe_tokens)} tokens. "
                "Fetch the tokenizer files before running evaluation."
            )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_variant(
    *,
    model_id: str,
    model: torch.nn.Module,
    tokenizer: Any,
    originals: list[ActivationSlot],
    gelu_originals: list[ActivationSlot],
    dense_swiglus: list[DenseSwigluSlot],
    packed_moe_swiglus: list[PackedMoeSwigluSlot],
    tensor_expert_swiglus: list[TensorExpertsSwigluSlot],
    packed_expert_gates: list[PackedExpertsGateSlot],
    gptoss_gates: list[GptOssGateSlot],
    mxfp4_swiglus: list[Mxfp4SwigluSlot],
    sigmoid_routers: list[SigmoidRouterSlot],
    spec: VariantSpec,
    tokens: torch.Tensor,
    args: argparse.Namespace,
) -> Result:
    result = Result(
        model=model_id,
        variant=spec.name,
        dtype=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    result.patched_silu_modules = apply_silu_patch(
        originals,
        spec,
    )
    result.patched_silu_modules += apply_gelu_patch(
        gelu_originals,
        spec,
    )
    dense_swiglu_count = apply_dense_swiglu_patch(dense_swiglus, spec)
    result.patched_silu_modules += dense_swiglu_count
    if dense_swiglu_count:
        result.notes.append(
            "Patched dense MLP forward with spline_ops.swish_mul_fwd_variant "
            "odd/even CUDA fast path."
        )
    packed_moe_swiglu_count = apply_packed_moe_swiglu_patch(
        packed_moe_swiglus,
        spec,
        freeze_routing=args.freeze_granite_moe_routing,
    )
    result.patched_silu_modules += packed_moe_swiglu_count
    if packed_moe_swiglu_count:
        result.notes.append(
            "Patched packed MoE SwiGLU with spline_ops.swish_mul_packed_fwd_variant "
            "odd/even CUDA fast path."
        )
    tensor_expert_swiglu_count = apply_tensor_experts_swiglu_patch(
        tensor_expert_swiglus,
        spec,
    )
    result.patched_silu_modules += tensor_expert_swiglu_count
    if tensor_expert_swiglu_count:
        result.notes.append(
            "Patched tensorized HF MoE experts with "
            "spline_ops.swish_mul_packed_fwd_variant odd/even CUDA fast path."
        )
    packed_expert_gate_count = apply_packed_experts_gate_patch(
        packed_expert_gates,
        spec,
    )
    result.patched_silu_modules += packed_expert_gate_count
    if packed_expert_gate_count:
        result.notes.append(
            "Patched Transformers grouped/batched MoE expert gates with "
            "spline_ops.swish_mul_packed_fwd_variant."
        )
    gptoss_gate_count = apply_gptoss_gate_patch(gptoss_gates, spec)
    result.patched_silu_modules += gptoss_gate_count
    if gptoss_gate_count:
        result.notes.append(
            "Patched GPT-OSS gated-SwiGLU swish inside expert activation."
        )
    mxfp4_swiglu_count = apply_mxfp4_swiglu_patch(mxfp4_swiglus, spec)
    result.patched_silu_modules += mxfp4_swiglu_count
    if mxfp4_swiglu_count:
        result.notes.append(
            "Patched GPT-OSS MXFP4 matmul_ogs fused activation with "
            f"{spec.silu_source} polynomial SwiGLU coefficients."
        )
    result.patched_router_sigmoid_modules = apply_sigmoid_router_patch(
        sigmoid_routers,
        spec,
    )
    if result.patched_router_sigmoid_modules:
        result.notes.append(
            "Patched GLM/Kimi sigmoid MoE routing with the spline_ops "
            f"D{spec.router_sigmoid_degree} {spec.router_sigmoid_source} "
            "sigmoid kernel."
        )
    result.patched_gemma_softcap = apply_gemma_attention_patch(spec)
    if spec.gemma_tanh_source is not None:
        result.notes.append("Gemma2 softcap patch only affects eager attention.")
    if spec.gemma_fa4_tanh_backend == "native":
        result.notes.append("Gemma2 attention uses FA4 with native tanh softcapping.")
    elif spec.gemma_fa4_tanh_backend == "device":
        result.notes.append(
            "Gemma2 attention uses FA4 with the in-kernel CuTe DSL polynomial "
            f"tanh D{spec.gemma_fa4_tanh_degree} "
            f"{spec.gemma_fa4_tanh_source} score modifier."
        )

    if result.patched_silu_modules == 0 and spec.silu_degree is not None:
        result.notes.append("No supported SiLU/SwiGLU modules were found to patch.")
    if result.patched_silu_modules == 0 and spec.gelu_degree is not None:
        result.notes.append("No supported GeLU modules were found to patch.")
    if result.patched_silu_modules == 0 and spec.dense_swiglu_degree is not None:
        result.notes.append("No supported dense or packed MoE SwiGLU MLPs were found to patch.")
    if (
        result.patched_router_sigmoid_modules == 0
        and spec.router_sigmoid_degree is not None
    ):
        result.notes.append("No supported GLM/Kimi sigmoid routers were found to patch.")

    if args.mode in {"prefill", "both"}:
        result.prefill_ms, result.prefill_tokens_per_s = benchmark_prefill(
            model,
            tokens,
            warmup=args.warmup,
            steps=args.steps,
        )
    if args.mode in {"decode", "both"}:
        result.decode_ms_per_token, result.decode_tokens_per_s = benchmark_decode(
            model,
            tokens,
            warmup=args.decode_warmup,
            repeats=args.decode_repeats,
            decode_steps=args.decode_steps,
            seed=args.seed,
        )
    if args.eval_tasks:
        if tokenizer is None:
            raise RuntimeError("Tokenizer was not loaded for evaluation.")
        result.eval = run_lm_eval(
            model=model,
            tokenizer=tokenizer,
            tasks=[task.strip() for task in args.eval_tasks.split(",") if task.strip()],
            num_fewshot=args.eval_num_fewshot,
            task_num_fewshot=parse_task_num_fewshot(args.eval_task_fewshot),
            limit=args.eval_limit,
            batch_size=args.eval_batch_size,
            device=args.device,
            log_samples=args.eval_log_samples,
        )
    return result


def print_results(results: list[Result]) -> None:
    print(
        f"{'model':<28} {'variant':<42} {'acts':>5} {'rtrs':>5} {'prefill tok/s':>14} "
        f"{'decode tok/s':>13} {'status':>8}"
    )
    print("-" * 124)
    for result in results:
        prefill = (
            "-"
            if result.prefill_tokens_per_s is None
            else f"{result.prefill_tokens_per_s:.0f}"
        )
        decode = (
            "-"
            if result.decode_tokens_per_s is None
            else f"{result.decode_tokens_per_s:.0f}"
        )
        status = "error" if result.error else "ok"
        print(
            f"{result.model:<28} {result.variant:<42} "
            f"{result.patched_silu_modules:>5} "
            f"{result.patched_router_sigmoid_modules:>5} "
            f"{prefill:>14} {decode:>13} "
            f"{status:>8}"
        )
        for note in result.notes:
            print(f"  note: {note}")
        if result.error:
            print(f"  {result.error}")


def sanitize_exception(error: BaseException) -> str:
    """Return a useful, single-line error without credentials or provider URLs."""

    message = str(error)
    environment_token = os.environ.get("HF_TOKEN")
    if environment_token:
        message = message.replace(environment_token, "[redacted-secret]")
    message = _URL_RE.sub("[redacted-url]", message)
    message = _BEARER_RE.sub("Bearer [redacted-secret]", message)
    message = _TOKEN_RE.sub("[redacted-secret]", message)
    message = _SECRET_ASSIGNMENT_RE.sub(r"\1=[redacted-secret]", message)
    message = " ".join(message.split())
    if not message:
        message = "message unavailable"
    if len(message) > 500:
        message = message[:497] + "..."
    return f"{type(error).__name__}: {message}"


def git_worktree_state(repository: Path) -> tuple[bool | None, int | None]:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=repository,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None, None
    if completed.returncode:
        return None, None
    entries = tuple(entry for entry in completed.stdout.split("\0") if entry)
    return bool(entries), sum(entry.startswith("?? ") for entry in entries)


def package_version(distribution: str) -> str | None:
    try:
        return distribution_version(distribution)
    except PackageNotFoundError:
        return None


def repository_state(
    repository: Path | None = None,
) -> dict[str, Any]:
    repository = repository or Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    dirty, untracked_files = git_worktree_state(repository)
    return {
        "repository": "https://github.com/MrHuff/fast-polynomial-transcendentals",
        "revision": revision if revision and len(revision) == 40 else None,
        "dirty": dirty,
        "untracked_files": untracked_files,
        "legacy_source": {
            "repository": "graphcore-research/low-bits-training",
            "revision": "393b69e2993ef00c812dfc87ac5f93c146159f45",
        },
    }


def result_document(results: list[Result], args: argparse.Namespace) -> dict[str, Any]:
    device: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": {
            "sfu-repro": package_version("sfu-repro"),
            "transformers": package_version("transformers"),
            "lm-eval": package_version("lm-eval"),
            "accelerate": package_version("accelerate"),
            "safetensors": package_version("safetensors"),
        },
    }
    if args.device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(args.device).index
        resolved_index = torch.cuda.current_device() if index is None else index
        properties = torch.cuda.get_device_properties(resolved_index)
        device.update(
            {
                "name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(resolved_index)),
                "total_memory_bytes": properties.total_memory,
            }
        )
    return {
        "schema_version": 1,
        "experiment": {
            "id": "open-weight-evaluation",
            "provenance_class": "new-measurement",
            "models": list(args.model),
            "variants": list(args.variant),
            "requested_model_revisions": {
                model: args.revision for model in args.model
            },
            "requested_tokenizer_revisions": {
                model: args.revision for model in args.model
            },
        },
        "source": repository_state(),
        "environment": device,
        "measurement": {
            "mode": args.mode,
            "dtype": args.dtype,
            "device": args.device,
            "batch_size": args.batch_size,
            "sequence_length_argument": args.seq_len,
            "prefill_measurements": args.steps,
            "prefill_warmups": args.warmup,
            "decode_steps": args.decode_steps,
            "decode_measurements": args.decode_repeats,
            "decode_warmups": args.decode_warmup,
            "seed": args.seed,
            "eval_tasks": args.eval_tasks,
        },
        "results": [asdict(result) for result in results],
    }


def write_json_results(
    path: str | None,
    results: list[Result],
    args: argparse.Namespace,
) -> None:
    if not path:
        return
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = json_path.with_suffix(f"{json_path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(result_document(results, args), f, indent=2, default=str)
    temporary_path.replace(json_path)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="HF model id. Repeat for a size sweep.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help=(
            "Variant name. Repeat as needed. Supported parts: native, "
            "spline_silu_d{3,4,5,6}_{current,sollya}, "
            "spline_gelu_d{3,4,5,6}_{current,sollya}, "
            "fused_swiglu_d{3,4,5}_{current,sollya}, "
            "spline_router_sigmoid_d{3,4,5,6}_{current,sollya}, "
            "gemma_tanh_current, gemma_fa4_tanh_native, "
            "gemma_fa4_tanh_d{3,4,5,6}_{current,sollya}. "
            "The spline_silu variants also patch GPT-OSS non-MXFP4 expert gates "
            "and the default MXFP4 matmul_ogs fused activation for D3/D4/D5. "
            "Parts can be combined with '+'."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["prefill", "decode", "both", "eval"],
        default="prefill",
        help="Benchmark phase; eval skips synthetic throughput and requires --eval-tasks.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--decode-repeats", type=int, default=10)
    parser.add_argument("--decode-warmup", type=int, default=2)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=[
            "eager",
            "sdpa",
            "flash_attention_2",
            "flash_attention_3",
            "flash_attention_4",
            "flex_attention",
        ],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--experts-implementation",
        choices=["eager", "batched_mm", "grouped_mm"],
        default=None,
        help=(
            "Transformers MoE expert backend; grouped_mm exposes the packed "
            "fused gate hook."
        ),
    )
    parser.add_argument(
        "--freeze-granite-moe-routing",
        action="store_true",
        help=(
            "For Granite MoE throughput only: pay router cost but reuse native "
            "expert assignments for every variant so activation timing is not "
            "confounded by polynomial-induced routing changes."
        ),
    )
    parser.add_argument(
        "--mxfp4-dequantize",
        action="store_true",
        help=(
            "Load MXFP4 checkpoints as dequantized BF16 so fallback expert "
            "modules can be patched with the odd/even CUDA spline kernels. "
            "By default, GPT-OSS MXFP4 stays quantized and patches matmul_ogs "
            "with Triton fused polynomial activation."
        ),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--eval-tasks",
        default="",
        help="Optional lm-eval task list, for example mmlu or mmlu,hellaswag.",
    )
    parser.add_argument("--eval-num-fewshot", type=int, default=5)
    parser.add_argument(
        "--eval-task-fewshot",
        action="append",
        default=[],
        metavar="TASK=SHOTS",
        help=(
            "Override the few-shot count for one task. Repeat for mixed-shot "
            "paper protocols."
        ),
    )
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-batch-size", default="auto")
    parser.add_argument("--eval-log-samples", action="store_true")
    parser.add_argument("--json-out", default=None)
    return parser


def main() -> int:
    args = get_parser().parse_args()
    if args.mode == "eval" and not args.eval_tasks:
        raise ValueError("--mode eval requires --eval-tasks")
    if not args.model:
        args.model = ["Qwen/Qwen2.5-1.5B"]
    if args.variant is None:
        args.variant = ["native", "spline_silu_d3_current"]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")

    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)
    all_results: list[Result] = []

    for model_id in args.model:
        model: torch.nn.Module | None = None
        try:
            model, tokenizer = load_model_and_tokenizer(args, model_id, dtype)
            originals = collect_silu_act_fns(model)
            gelu_originals = collect_gelu_act_fns(model)
            dense_swiglus = collect_dense_swiglu_fns(model)
            packed_moe_swiglus = collect_packed_moe_swiglu_fns(model)
            tensor_expert_swiglus = collect_tensor_experts_swiglu_fns(model)
            packed_expert_gates = collect_packed_experts_gate_fns(model)
            gptoss_gates = collect_gptoss_gate_fns(model)
            mxfp4_swiglus = collect_mxfp4_swiglu_fns(model)
            sigmoid_routers = collect_sigmoid_router_fns(model)
            tokens = make_tokens(
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                vocab_size=model_vocab_size(model),
                device=device,
                seed=args.seed,
            )
            if args.freeze_granite_moe_routing:
                captured = capture_packed_moe_routes(model, packed_moe_swiglus, tokens)
                if captured:
                    print(
                        f"[info] Captured frozen Granite MoE routes for {captured} modules.",
                        flush=True,
                    )
            for raw_variant in args.variant:
                spec = parse_variant(raw_variant)
                try:
                    all_results.append(
                        run_variant(
                            model_id=model_id,
                            model=model,
                            tokenizer=tokenizer,
                            originals=originals,
                            gelu_originals=gelu_originals,
                            dense_swiglus=dense_swiglus,
                            packed_moe_swiglus=packed_moe_swiglus,
                            tensor_expert_swiglus=tensor_expert_swiglus,
                            packed_expert_gates=packed_expert_gates,
                            gptoss_gates=gptoss_gates,
                            mxfp4_swiglus=mxfp4_swiglus,
                            sigmoid_routers=sigmoid_routers,
                            spec=spec,
                            tokens=tokens,
                            args=args,
                        )
                    )
                except Exception as exc:
                    sanitized_error = sanitize_exception(exc)
                    print(
                        f"[error] {model_id} {raw_variant} failed: "
                        f"{sanitized_error}",
                        file=sys.stderr,
                    )
                    all_results.append(
                        Result(
                            model=model_id,
                            variant=raw_variant,
                            dtype=args.dtype,
                            device=args.device,
                            batch_size=args.batch_size,
                            seq_len=args.seq_len,
                            error=sanitized_error,
                        )
                    )
                finally:
                    write_json_results(args.json_out, all_results, args)
                    torch.cuda.empty_cache()
        except Exception as exc:
            sanitized_error = sanitize_exception(exc)
            for raw_variant in args.variant:
                all_results.append(
                    Result(
                        model=model_id,
                        variant=raw_variant,
                        dtype=args.dtype,
                        device=args.device,
                        batch_size=args.batch_size,
                        seq_len=args.seq_len,
                        error=sanitized_error,
                    )
                )
        finally:
            restore_gemma_attention()
            del model
            torch.cuda.empty_cache()

    print_results(all_results)
    write_json_results(args.json_out, all_results, args)
    return 1 if any(result.error for result in all_results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

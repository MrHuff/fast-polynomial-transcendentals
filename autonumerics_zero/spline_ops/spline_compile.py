# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified in 2026 for the standalone fast-polynomial-transcendentals release.

"""
torch.compile compatibility layer for spline_ops.

Uses torch.library.custom_op to create proper opaque custom ops that
torch.compile can handle without tracing into CUDA kernels.

Usage:
    from spline_compile import spline_silu, spline_sigmoid, spline_tanh, spline_gelu

    model.act_fn = spline_silu  # drop-in replacement for F.silu
    compiled = torch.compile(model, mode="reduce-overhead")
"""
import torch
import torch.autograd
from functools import lru_cache

# Ensure the C++ extension is loaded first
import os
import spline_ops as _spline_ops  # noqa: F401


_COEFF_SOURCE_TO_ID = {
    "current": 0,
    "sollya": 1,
}


# The extension registers these operators directly with the PyTorch dispatcher.
# Unlike the Python-backed custom ops below, they stay entirely in C++ at runtime.
_swish_variant_fwd_cpp = torch.ops.spline_ops.swish_variant_fwd.default
_swish_variant_bwd_cpp = torch.ops.spline_ops.swish_variant_bwd.default
_swish_mul_variant_fwd_cpp = torch.ops.spline_ops.swish_mul_variant_fwd.default
_swish_mul_variant_bwd_cpp = torch.ops.spline_ops.swish_mul_variant_bwd.default
_swish_mul_packed_variant_fwd_cpp = (
    torch.ops.spline_ops.swish_mul_packed_variant_fwd.default
)
_swish_mul_packed_native_fwd_cpp = (
    torch.ops.spline_ops.swish_mul_packed_native_fwd.default
)
_swish_mul_packed_native_bwd_fwd_cpp = (
    torch.ops.spline_ops.swish_mul_packed_native_bwd_fwd.default
)
_swish_mul_packed_variant_bwd_cpp = (
    torch.ops.spline_ops.swish_mul_packed_variant_bwd.default
)


@torch.library.register_fake("spline_ops::swish_variant_fwd")
def _swish_variant_fwd_cpp_fake(x, degree, coeff_source_id):
    del degree, coeff_source_id
    return torch.empty_like(x)


@torch.library.register_fake("spline_ops::swish_variant_bwd")
def _swish_variant_bwd_cpp_fake(grad_out, x, degree, coeff_source_id):
    del grad_out, degree, coeff_source_id
    return torch.empty_like(x)


@torch.library.register_fake("spline_ops::swish_mul_variant_fwd")
def _swish_mul_variant_fwd_cpp_fake(gate, up, degree, coeff_source_id):
    del gate, degree, coeff_source_id
    return torch.empty_like(up)


@torch.library.register_fake("spline_ops::swish_mul_variant_bwd")
def _swish_mul_variant_bwd_cpp_fake(grad_out, gate, up, degree, coeff_source_id):
    del grad_out, degree, coeff_source_id
    return [torch.empty_like(gate), torch.empty_like(up)]


@torch.library.register_fake("spline_ops::swish_mul_packed_variant_fwd")
def _swish_mul_packed_variant_fwd_cpp_fake(packed, degree, coeff_source_id):
    del degree, coeff_source_id
    return packed.new_empty((*packed.shape[:-1], packed.shape[-1] // 2))


@torch.library.register_fake("spline_ops::swish_mul_packed_variant_bwd")
def _swish_mul_packed_variant_bwd_cpp_fake(grad_out, packed, degree, coeff_source_id):
    del grad_out, degree, coeff_source_id
    return torch.empty_like(packed)


@torch.library.register_fake("spline_ops::swish_mul_packed_native_fwd")
def _swish_mul_packed_native_fwd_cpp_fake(packed, coeff_source_id):
    del coeff_source_id
    return packed.new_empty((*packed.shape[:-1], packed.shape[-1] // 2))


@torch.library.register_fake("spline_ops::swish_mul_packed_native_bwd_fwd")
def _swish_mul_packed_native_bwd_fwd_cpp_fake(packed, degree, coeff_source_id):
    del degree, coeff_source_id
    return packed.new_empty((*packed.shape[:-1], packed.shape[-1] // 2))


def _swish_variant_fwd_cpp_setup_context(ctx, inputs, output):
    del output
    x, degree, coeff_source_id = inputs
    ctx.save_for_backward(x)
    ctx.degree = degree
    ctx.coeff_source_id = coeff_source_id


def _swish_variant_fwd_cpp_backward(ctx, grad_out):
    (x,) = ctx.saved_tensors
    return (
        _swish_variant_bwd_cpp(
            grad_out.contiguous(),
            x,
            ctx.degree,
            ctx.coeff_source_id,
        ),
        None,
        None,
    )


torch.library.register_autograd(
    "spline_ops::swish_variant_fwd",
    _swish_variant_fwd_cpp_backward,
    setup_context=_swish_variant_fwd_cpp_setup_context,
)


def _swish_mul_variant_fwd_cpp_setup_context(ctx, inputs, output):
    del output
    gate, up, degree, coeff_source_id = inputs
    ctx.save_for_backward(gate, up)
    ctx.degree = degree
    ctx.coeff_source_id = coeff_source_id


def _swish_mul_variant_fwd_cpp_backward(ctx, grad_out):
    gate, up = ctx.saved_tensors
    grad_gate, grad_up = _swish_mul_variant_bwd_cpp(
        grad_out.contiguous(),
        gate,
        up,
        ctx.degree,
        ctx.coeff_source_id,
    )
    return grad_gate, grad_up, None, None


torch.library.register_autograd(
    "spline_ops::swish_mul_variant_fwd",
    _swish_mul_variant_fwd_cpp_backward,
    setup_context=_swish_mul_variant_fwd_cpp_setup_context,
)


def _swish_mul_packed_variant_fwd_cpp_setup_context(ctx, inputs, output):
    del output
    packed, degree, coeff_source_id = inputs
    ctx.save_for_backward(packed)
    ctx.degree = degree
    ctx.coeff_source_id = coeff_source_id


def _swish_mul_packed_variant_fwd_cpp_backward(ctx, grad_out):
    (packed,) = ctx.saved_tensors
    return (
        _swish_mul_packed_variant_bwd_cpp(
            grad_out.contiguous(),
            packed,
            ctx.degree,
            ctx.coeff_source_id,
        ),
        None,
        None,
    )


torch.library.register_autograd(
    "spline_ops::swish_mul_packed_variant_fwd",
    _swish_mul_packed_variant_fwd_cpp_backward,
    setup_context=_swish_mul_packed_variant_fwd_cpp_setup_context,
)


def _swish_mul_packed_native_fwd_cpp_setup_context(ctx, inputs, output):
    del output
    packed, coeff_source_id = inputs
    ctx.save_for_backward(packed)
    ctx.coeff_source_id = coeff_source_id


def _swish_mul_packed_native_fwd_cpp_backward(ctx, grad_out):
    (packed,) = ctx.saved_tensors
    return (
        _swish_mul_packed_variant_bwd_cpp(
            grad_out.contiguous(),
            packed,
            0,
            ctx.coeff_source_id,
        ),
        None,
    )


torch.library.register_autograd(
    "spline_ops::swish_mul_packed_native_fwd",
    _swish_mul_packed_native_fwd_cpp_backward,
    setup_context=_swish_mul_packed_native_fwd_cpp_setup_context,
)


def _swish_mul_packed_native_bwd_fwd_cpp_setup_context(ctx, inputs, output):
    del output
    packed, degree, coeff_source_id = inputs
    del degree
    ctx.save_for_backward(packed)
    ctx.coeff_source_id = coeff_source_id


def _swish_mul_packed_native_bwd_fwd_cpp_backward(ctx, grad_out):
    (packed,) = ctx.saved_tensors
    return (
        _swish_mul_packed_variant_bwd_cpp(
            grad_out.contiguous(),
            packed,
            0,
            ctx.coeff_source_id,
        ),
        None,
        None,
    )


torch.library.register_autograd(
    "spline_ops::swish_mul_packed_native_bwd_fwd",
    _swish_mul_packed_native_bwd_fwd_cpp_backward,
    setup_context=_swish_mul_packed_native_bwd_fwd_cpp_setup_context,
)


def _coeff_source_id(coeff_source: str) -> int:
    try:
        return _COEFF_SOURCE_TO_ID[coeff_source]
    except KeyError as exc:
        raise ValueError(
            f"Unknown coeff_source '{coeff_source}'. Expected one of: {tuple(_COEFF_SOURCE_TO_ID)}"
        ) from exc


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


# =============================================================================
# Custom ops via torch.library.custom_op — opaque to dynamo
# =============================================================================


@torch.library.custom_op("spline_compile::swish_fwd", mutates_args=())
def _swish_fwd(x: torch.Tensor) -> torch.Tensor:
    return _spline_ops.swish_fwd(x)


@_swish_fwd.register_fake
def _swish_fwd_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("spline_compile::swish_bwd", mutates_args=())
def _swish_bwd(grad_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _spline_ops.swish_bwd(grad_out, x)


@_swish_bwd.register_fake
def _swish_bwd_fake(grad_out, x):
    return torch.empty_like(x)


@torch.library.custom_op("spline_compile::sigmoid_fwd", mutates_args=())
def _sigmoid_fwd(x: torch.Tensor) -> torch.Tensor:
    return _spline_ops.sigmoid_fwd(x)


@_sigmoid_fwd.register_fake
def _sigmoid_fwd_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("spline_compile::sigmoid_bwd_alg", mutates_args=())
def _sigmoid_bwd_alg(grad_out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _spline_ops.sigmoid_bwd_alg(grad_out, y)


@_sigmoid_bwd_alg.register_fake
def _sigmoid_bwd_alg_fake(grad_out, y):
    return torch.empty_like(y)


@torch.library.custom_op("spline_compile::tanh_fwd", mutates_args=())
def _tanh_fwd(x: torch.Tensor) -> torch.Tensor:
    return _spline_ops.tanh_fwd(x)


@_tanh_fwd.register_fake
def _tanh_fwd_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("spline_compile::tanh_bwd_alg", mutates_args=())
def _tanh_bwd_alg(grad_out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _spline_ops.tanh_bwd_alg(grad_out, y)


@_tanh_bwd_alg.register_fake
def _tanh_bwd_alg_fake(grad_out, y):
    return torch.empty_like(y)


@torch.library.custom_op("spline_compile::gelu_fwd", mutates_args=())
def _gelu_fwd(x: torch.Tensor) -> torch.Tensor:
    return _spline_ops.gelu_fwd(x)


@_gelu_fwd.register_fake
def _gelu_fwd_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("spline_compile::gelu_bwd", mutates_args=())
def _gelu_bwd(grad_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _spline_ops.gelu_bwd(grad_out, x)


@_gelu_bwd.register_fake
def _gelu_bwd_fake(grad_out, x):
    return torch.empty_like(x)


@torch.library.custom_op("spline_compile::swish_variant_fwd", mutates_args=())
def _swish_variant_fwd(
    x: torch.Tensor, degree: int, coeff_source_id: int
) -> torch.Tensor:
    return _spline_ops.swish_fwd_variant(x, degree, coeff_source_id)


@_swish_variant_fwd.register_fake
def _swish_variant_fwd_fake(x, degree, coeff_source_id):
    del degree, coeff_source_id
    return torch.empty_like(x)


@torch.library.custom_op("spline_compile::swish_variant_bwd", mutates_args=())
def _swish_variant_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    degree: int,
    coeff_source_id: int,
) -> torch.Tensor:
    return _spline_ops.swish_bwd_variant(grad_out, x, degree, coeff_source_id)


@_swish_variant_bwd.register_fake
def _swish_variant_bwd_fake(grad_out, x, degree, coeff_source_id):
    del degree, coeff_source_id
    return torch.empty_like(x)


@torch.library.custom_op("spline_compile::swish_mul_variant_fwd", mutates_args=())
def _swish_mul_variant_fwd(
    gate: torch.Tensor,
    up: torch.Tensor,
    degree: int,
    coeff_source_id: int,
) -> torch.Tensor:
    return _spline_ops.swish_mul_fwd_variant(gate, up, degree, coeff_source_id)


@_swish_mul_variant_fwd.register_fake
def _swish_mul_variant_fwd_fake(gate, up, degree, coeff_source_id):
    del gate, degree, coeff_source_id
    return torch.empty_like(up)


@torch.library.custom_op("spline_compile::swish_mul_variant_bwd", mutates_args=())
def _swish_mul_variant_bwd(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    degree: int,
    coeff_source_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_gate, grad_up = _spline_ops.swish_mul_bwd_variant(
        grad_out,
        gate,
        up,
        degree,
        coeff_source_id,
    )
    return grad_gate, grad_up


@_swish_mul_variant_bwd.register_fake
def _swish_mul_variant_bwd_fake(grad_out, gate, up, degree, coeff_source_id):
    del grad_out, degree, coeff_source_id
    return torch.empty_like(gate), torch.empty_like(up)


@torch.library.custom_op("spline_compile::gelu_variant_fwd", mutates_args=())
def _gelu_variant_fwd(
    x: torch.Tensor, degree: int, coeff_source_id: int
) -> torch.Tensor:
    return _spline_ops.gelu_fwd_variant(x, degree, coeff_source_id)


@_gelu_variant_fwd.register_fake
def _gelu_variant_fwd_fake(x, degree, coeff_source_id):
    del degree, coeff_source_id
    return torch.empty_like(x)


@torch.library.custom_op("spline_compile::gelu_variant_bwd", mutates_args=())
def _gelu_variant_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    degree: int,
    coeff_source_id: int,
) -> torch.Tensor:
    return _spline_ops.gelu_bwd_variant(grad_out, x, degree, coeff_source_id)


@_gelu_variant_bwd.register_fake
def _gelu_variant_bwd_fake(grad_out, x, degree, coeff_source_id):
    del degree, coeff_source_id
    return torch.empty_like(x)


# =============================================================================
# Autograd setup via setup_context (torch.compile-compatible pattern)
# =============================================================================


def _swish_fwd_setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)


def _swish_fwd_backward(ctx, grad_out):
    (x,) = ctx.saved_tensors
    return _swish_bwd(grad_out.contiguous(), x)


_swish_fwd.register_autograd(
    _swish_fwd_backward, setup_context=_swish_fwd_setup_context
)


def _sigmoid_fwd_setup_context(ctx, inputs, output):
    ctx.save_for_backward(output)  # save y, not x


def _sigmoid_fwd_backward(ctx, grad_out):
    (y,) = ctx.saved_tensors
    return _sigmoid_bwd_alg(grad_out.contiguous(), y)


_sigmoid_fwd.register_autograd(
    _sigmoid_fwd_backward, setup_context=_sigmoid_fwd_setup_context
)


def _tanh_fwd_setup_context(ctx, inputs, output):
    ctx.save_for_backward(output)  # save y, not x


def _tanh_fwd_backward(ctx, grad_out):
    (y,) = ctx.saved_tensors
    return _tanh_bwd_alg(grad_out.contiguous(), y)


_tanh_fwd.register_autograd(_tanh_fwd_backward, setup_context=_tanh_fwd_setup_context)


def _gelu_fwd_setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)


def _gelu_fwd_backward(ctx, grad_out):
    (x,) = ctx.saved_tensors
    return _gelu_bwd(grad_out.contiguous(), x)


_gelu_fwd.register_autograd(_gelu_fwd_backward, setup_context=_gelu_fwd_setup_context)


def _swish_variant_fwd_setup_context(ctx, inputs, output):
    x, degree, coeff_source_id = inputs
    ctx.save_for_backward(x)
    ctx.degree = degree
    ctx.coeff_source_id = coeff_source_id


def _swish_variant_fwd_backward(ctx, grad_out):
    (x,) = ctx.saved_tensors
    return (
        _swish_variant_bwd(
            grad_out.contiguous(),
            x,
            ctx.degree,
            ctx.coeff_source_id,
        ),
        None,
        None,
    )


_swish_variant_fwd.register_autograd(
    _swish_variant_fwd_backward,
    setup_context=_swish_variant_fwd_setup_context,
)


def _swish_mul_variant_fwd_setup_context(ctx, inputs, output):
    gate, up, degree, coeff_source_id = inputs
    ctx.save_for_backward(gate, up)
    ctx.degree = degree
    ctx.coeff_source_id = coeff_source_id


def _swish_mul_variant_fwd_backward(ctx, grad_out):
    gate, up = ctx.saved_tensors
    grad_gate, grad_up = _swish_mul_variant_bwd(
        grad_out.contiguous(),
        gate,
        up,
        ctx.degree,
        ctx.coeff_source_id,
    )
    return grad_gate, grad_up, None, None


_swish_mul_variant_fwd.register_autograd(
    _swish_mul_variant_fwd_backward,
    setup_context=_swish_mul_variant_fwd_setup_context,
)


def _gelu_variant_fwd_setup_context(ctx, inputs, output):
    x, degree, coeff_source_id = inputs
    ctx.save_for_backward(x)
    ctx.degree = degree
    ctx.coeff_source_id = coeff_source_id


def _gelu_variant_fwd_backward(ctx, grad_out):
    (x,) = ctx.saved_tensors
    return (
        _gelu_variant_bwd(
            grad_out.contiguous(),
            x,
            ctx.degree,
            ctx.coeff_source_id,
        ),
        None,
        None,
    )


_gelu_variant_fwd.register_autograd(
    _gelu_variant_fwd_backward,
    setup_context=_gelu_variant_fwd_setup_context,
)


# =============================================================================
# Public API — drop-in replacements
# =============================================================================


def spline_silu(x):
    """Spline SiLU activation, torch.compile-compatible."""
    return _swish_fwd(x)


def spline_sigmoid(x):
    """Spline sigmoid activation, torch.compile-compatible."""
    return _sigmoid_fwd(x)


def spline_tanh(x):
    """Spline tanh activation, torch.compile-compatible."""
    return _tanh_fwd(x)


def spline_gelu(x):
    """Spline GeLU activation, torch.compile-compatible."""
    return _gelu_fwd(x)


@lru_cache(maxsize=None)
def make_spline_silu(degree: int | None = None, coeff_source: str = "current"):
    resolved_degree = 3 if degree is None else int(degree)
    coeff_source_id = _coeff_source_id(coeff_source)
    if resolved_degree == 3 and coeff_source_id == 0:

        def activation(x):
            return _swish_variant_fwd_cpp(x, resolved_degree, coeff_source_id)

        activation.__name__ = "spline_silu_d3_current_cpp"
        return activation

    def activation(x):
        return _swish_variant_fwd_cpp(x, resolved_degree, coeff_source_id)

    activation.__name__ = f"spline_silu_d{resolved_degree}_{coeff_source}"
    return activation


@lru_cache(maxsize=None)
def make_spline_silu_mul(degree: int | None = None, coeff_source: str = "current"):
    resolved_degree = 3 if degree is None else int(degree)
    coeff_source_id = _coeff_source_id(coeff_source)
    if coeff_source_id != 0 or _env_flag("SPLINE_COMPILE_DISABLE_FUSED_SWIGLU"):
        activation = make_spline_silu(resolved_degree, coeff_source)

        def activation_mul(gate, up):
            return activation(gate.contiguous()) * up.contiguous()

        activation_mul.__name__ = f"spline_silu_mul_d{resolved_degree}_{coeff_source}"
        return activation_mul

    def activation_mul(gate, up):
        return _swish_mul_variant_fwd_cpp(
            gate.contiguous(),
            up.contiguous(),
            resolved_degree,
            coeff_source_id,
        )

    activation_mul.__name__ = f"spline_silu_mul_d{resolved_degree}_{coeff_source}_fused"
    activation_mul.__spline_op__ = "spline_ops::swish_mul_variant_fwd"
    return activation_mul


@lru_cache(maxsize=None)
def make_spline_silu_packed(
    degree: int | None = None,
    coeff_source: str = "current",
    backward_impl: str = "matched",
):
    resolved_degree = 3 if degree is None else int(degree)
    coeff_source_id = _coeff_source_id(coeff_source)
    if backward_impl not in ("matched", "native"):
        raise ValueError(
            f"Unknown packed SwiGLU backward_impl '{backward_impl}'. "
            "Expected one of: matched, native"
        )
    if coeff_source_id != 0 or _env_flag("SPLINE_COMPILE_DISABLE_FUSED_SWIGLU"):
        activation_mul = make_spline_silu_mul(resolved_degree, coeff_source)

        def activation_packed(packed):
            gate, up = packed.chunk(2, dim=-1)
            poly_out = activation_mul(gate, up)
            if backward_impl == "matched":
                return poly_out
            native_out = torch.nn.functional.silu(gate) * up
            return poly_out.detach() + (native_out - native_out.detach())

        activation_packed.__name__ = (
            f"spline_silu_packed_d{resolved_degree}_{coeff_source}"
        )
        return activation_packed

    packed_op = (
        _swish_mul_packed_variant_fwd_cpp
        if backward_impl == "matched"
        else _swish_mul_packed_native_bwd_fwd_cpp
    )

    def activation_packed(packed):
        return packed_op(
            packed.contiguous(),
            resolved_degree,
            coeff_source_id,
        )

    activation_packed.__name__ = f"spline_silu_packed_d{resolved_degree}_{coeff_source}_{backward_impl}_bwd_fused"
    activation_packed.__spline_op__ = (
        "spline_ops::swish_mul_packed_variant_fwd"
        if backward_impl == "matched"
        else "spline_ops::swish_mul_packed_native_bwd_fwd"
    )
    return activation_packed


@lru_cache(maxsize=None)
def make_native_silu_packed():
    def activation_packed(packed):
        return _swish_mul_packed_native_fwd_cpp(packed.contiguous(), 0)

    activation_packed.__name__ = "native_silu_packed_fused"
    activation_packed.__spline_op__ = "spline_ops::swish_mul_packed_native_fwd"
    return activation_packed


@lru_cache(maxsize=None)
def make_spline_gelu(degree: int | None = None, coeff_source: str = "current"):
    resolved_degree = 5 if degree is None else int(degree)
    coeff_source_id = _coeff_source_id(coeff_source)
    if resolved_degree == 5 and coeff_source_id == 0:
        return spline_gelu

    def activation(x):
        return _gelu_variant_fwd(x, resolved_degree, coeff_source_id)

    activation.__name__ = f"spline_gelu_d{resolved_degree}_{coeff_source}"
    return activation

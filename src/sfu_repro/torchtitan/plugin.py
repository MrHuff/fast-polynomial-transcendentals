# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# Portions follow the TorchTitan grouped-expert control flow:
# Copyright (c) Meta Platforms, Inc. and affiliates, BSD-3-Clause.
# See THIRD_PARTY_NOTICES.md and licenses/TORCHTITAN-BSD-3-CLAUSE.txt.
#
# Written in 2026 as a standalone adapter to the public TorchTitan APIs.  It
# contains no cluster launcher, service integration, credential handling, or
# dependency on the original training repository.
"""Register the B1--B5 kernels with the pinned public TorchTitan checkout.

TorchTitan imports this module before it constructs the model.  The registered
model converter changes only the operation named by ``[sfu]``.  Native and
polynomial configurations therefore use the same model definition,
initialization, data loader, optimizer, and distributed layout.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any

import torch
from datasets import load_dataset
from torch import nn
from torch.distributed.tensor import DTensor

from sfu_repro.activations import (
    resolve_mlp_activation_mul_impl,
    resolve_mlp_activation_packed_impl,
)
from sfu_repro.fa4 import (
    FA4Config,
    b2_component_configs,
    b3_component_configs,
    b5_component_configs,
    patch_attention_modules,
    resolve_fa4_config,
)
from sfu_repro.torchtitan.pins import (
    OLMO_MIX_REPOSITORY,
    OLMO_MIX_REVISION,
    SLIMPAJAMA_REPOSITORY,
    SLIMPAJAMA_REVISION,
)
from torchtitan.components.validate import build_validator
from torchtitan.distributed import ParallelDims
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.deepseek_v3 import get_train_spec as get_deepseek_train_spec
from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
from torchtitan.models.llama3 import get_train_spec as get_llama_train_spec
from torchtitan.models.llama3.model.args import TransformerModelArgs
from torchtitan.models.llama3.model.model import Attention as LlamaAttention
from torchtitan.models.llama3.model.model import FeedForward as LlamaFeedForward
from torchtitan.models.moe import MoEArgs
from torchtitan.models.moe.moe import GroupedExperts, indices_padding_wrapper
from torchtitan.protocols.model_converter import (
    ModelConverter,
    register_model_converter,
)
from torchtitan.protocols.train_spec import register_train_spec
from torchtitan.tools.logging import logger


# The original Cerebras Hub repository is no longer anonymously available.
# The public rerun protocol uses this immutable snapshot of a community
# re-upload.  It is a new input selection, not proof of the historical bytes.
def _load_text_dataset(
    path: str,
    *,
    split: str,
    name: str = "default",
    revision: str | None = None,
):
    kwargs: dict[str, Any] = {
        "name": name,
        "split": split,
        "streaming": True,
    }
    # Apply the immutable revision to the declared Hub repositories. A caller
    # may replace ``dataset_path`` with a local materialization, for which the
    # Hub-only revision argument is invalid.
    if revision is not None and path in {
        SLIMPAJAMA_REPOSITORY,
        OLMO_MIX_REPOSITORY,
    }:
        kwargs["revision"] = revision
    return load_dataset(path, **kwargs)


def _sample_text(sample: dict[str, Any]) -> str:
    return sample["text"]


def _register_datasets() -> None:
    """Add public corpus adapters without modifying the TorchTitan submodule."""

    registrations = {
        "sfu_slimpajama": DatasetConfig(
            path=SLIMPAJAMA_REPOSITORY,
            loader=partial(
                _load_text_dataset,
                split="train",
                revision=SLIMPAJAMA_REVISION,
            ),
            sample_processor=_sample_text,
        ),
        "sfu_slimpajama_validation": DatasetConfig(
            path=SLIMPAJAMA_REPOSITORY,
            loader=partial(
                _load_text_dataset,
                split="validation",
                revision=SLIMPAJAMA_REVISION,
            ),
            sample_processor=_sample_text,
        ),
        "sfu_olmo_mix_1124": DatasetConfig(
            path=OLMO_MIX_REPOSITORY,
            loader=partial(
                _load_text_dataset,
                split="train",
                revision=OLMO_MIX_REVISION,
            ),
            sample_processor=_sample_text,
        ),
    }
    conflicts = sorted(set(registrations) & set(DATASETS))
    if conflicts:
        raise RuntimeError(
            "refusing to replace pre-registered public-protocol datasets: "
            + ", ".join(conflicts)
        )
    DATASETS.update(registrations)


def _register_model_specs() -> None:
    """Register the paper's Llama 1B and DeepSeek 27A4B model flavors."""

    llama_spec = get_llama_train_spec()
    llama_args = dict(llama_spec.model_args)
    llama_args["1B"] = TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=8192 / 4 / 2048 * 3 / 2,
        multiple_of=256,
        rope_theta=500000,
    )
    register_train_spec(
        "sfu_llama3",
        replace(llama_spec, model_args=llama_args),
    )

    deepseek_spec = get_deepseek_train_spec()
    deepseek_args = dict(deepseek_spec.model_args)
    deepseek_args["27A4B"] = DeepSeekV3ModelArgs(
        vocab_size=129280,
        dim=2560,
        inter_dim=12800,
        moe_inter_dim=1536,
        n_layers=30,
        n_dense_layers=1,
        n_heads=32,
        moe_args=MoEArgs(
            num_experts=72,
            num_shared_experts=2,
            top_k=6,
            score_func="sigmoid",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
        attn_type="flex",
        attn_mask_type="block_causal",
    )
    register_train_spec(
        "sfu_deepseek_v3",
        replace(
            deepseek_spec,
            model_args=deepseek_args,
            # TorchTitan v0.2.2 does not wire its stock validator into the
            # DeepSeek spec.  The public validator supports the same model
            # protocol and distributed contexts.
            build_validator_fn=build_validator,
        ),
    )


class ConfiguredFusedLlamaFeedForward(LlamaFeedForward):
    """Llama MLP with the paper's fused gate/up projection and packed SwiGLU."""

    sfu_activation_packed: Any

    @classmethod
    @torch.no_grad()
    def from_unfused(
        cls,
        module: LlamaFeedForward,
        activation_packed: Any,
    ) -> "ConfiguredFusedLlamaFeedForward":
        dim, hidden_dim = module.w2.weight.shape
        device = module.w1.weight.device
        dtype = module.w1.weight.dtype

        w_in = nn.Linear(
            dim,
            2 * hidden_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        w_out = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if device.type != "meta":
            w_in.weight.copy_(torch.cat((module.w1.weight, module.w3.weight), dim=0))
            w_out.weight.copy_(module.w2.weight)

        del module.w1, module.w2, module.w3
        module.w_in = w_in
        module.w_out = w_out
        module.sfu_activation_packed = activation_packed
        module.__class__ = cls
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.sfu_activation_packed(self.w_in(x)))

    def init_weights(self, init_std: float) -> None:
        hidden_dim = self.w_out.weight.shape[-1]
        nn.init.trunc_normal_(self.w_in.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.w_in.weight[hidden_dim:].mul_(init_std / 0.02)
        nn.init.trunc_normal_(self.w_out.weight, mean=0.0, std=init_std)
        if hasattr(self.w_in, "norm_weight"):
            self.w_in.norm_weight.data.fill_(1.0)


def _patch_llama_mlp(model: nn.Module, variant: str) -> int:
    implementation = "native_silu" if variant == "native" else "spline_silu_compile"
    _, activation_packed = resolve_mlp_activation_packed_impl(
        implementation,
        degree=3,
        coeff_source="current",
        backward_impl="matched",
    )
    patched = 0
    for module in list(model.modules()):
        if not isinstance(module, LlamaFeedForward):
            continue
        ConfiguredFusedLlamaFeedForward.from_unfused(module, activation_packed)
        patched += 1
    return patched


class LlamaAttentionWithFA4Initialization(LlamaAttention):
    """Initialize affine norms inserted by the B3 FA4 wrapper."""

    def init_weights(self, init_std: float) -> None:
        super().init_weights(init_std)
        for norm in (
            getattr(self.inner_attention, "q_norm", None),
            getattr(self.inner_attention, "k_norm", None),
        ):
            if norm is not None:
                norm.reset_parameters()


def _install_fa4_initializers(model: nn.Module) -> int:
    patched = 0
    for module in model.modules():
        if not isinstance(module, LlamaAttention):
            continue
        if not hasattr(module.inner_attention, "config"):
            continue
        if module.__class__ is not LlamaAttentionWithFA4Initialization:
            module.__class__ = LlamaAttentionWithFA4Initialization
        patched += 1
    return patched


def _run_grouped_experts_polynomial(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    activation_mul: Any,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    gate = torch._grouped_mm(
        x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets
    )
    up = torch._grouped_mm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets)
    hidden = activation_mul(gate, up)
    return torch._grouped_mm(
        hidden, w2.bfloat16().transpose(-2, -1), offs=offsets
    ).type_as(x)


class ConfiguredGroupedExperts(GroupedExperts):
    """Routed experts with an explicit matched activation boundary."""

    sfu_activation_mul: Any

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_grouped_mm:
            raise RuntimeError(
                "B4 requires TorchTitan grouped GEMM on an SM90-or-newer GPU"
            )
        if isinstance(self.w1, DTensor):
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1, w2, w3 = self.w1, self.w2, self.w3

        def run_grouped(
            local_w1: torch.Tensor,
            local_w2: torch.Tensor,
            local_w3: torch.Tensor,
            local_x: torch.Tensor,
            local_counts: torch.Tensor,
        ) -> torch.Tensor:
            return _run_grouped_experts_polynomial(
                local_w1,
                local_w2,
                local_w3,
                local_x,
                local_counts,
                self.sfu_activation_mul,
            )

        if isinstance(self.w1, DTensor) and "ep" in self.w1.device_mesh.mesh_dim_names:
            runner = run_grouped
        else:
            runner = indices_padding_wrapper(run_grouped)
        return runner(w1, w2, w3, x, num_tokens_per_expert)


def _patch_grouped_experts(model: nn.Module, variant: str) -> int:
    implementation = "native_silu" if variant == "native" else "spline_silu_compile"
    _, activation_mul = resolve_mlp_activation_mul_impl(
        implementation,
        degree=3,
        coeff_source="current",
    )
    patched = 0
    for module in model.modules():
        if not isinstance(module, GroupedExperts):
            continue
        module.sfu_activation_mul = activation_mul
        if module.__class__ is not ConfiguredGroupedExperts:
            module.__class__ = ConfiguredGroupedExperts
        patched += 1
    return patched


def _attention_config(case: str, variant: str) -> FA4Config | None:
    if case == "b1":
        return FA4Config(mode="softmax")
    if case == "b2":
        return b2_component_configs()[variant]
    if case == "b3":
        return b3_component_configs(sequence_length=4096)[variant]
    if case == "b5":
        return b5_component_configs()[variant]
    return None


class SFUReproConverter(ModelConverter):
    """Apply one B1--B5 intervention through TorchTitan's public API."""

    def __init__(self, job_config: Any, parallel_dims: ParallelDims):
        self.job_config = job_config
        self.parallel_dims = parallel_dims
        self.case = str(job_config.sfu.case).lower()
        self.variant = str(job_config.sfu.variant).lower()
        self.strict = bool(job_config.sfu.strict)
        if self.case not in {"b1", "b2", "b3", "b4", "b5"}:
            raise ValueError(f"unknown SFU case {self.case!r}")
        allowed_variants = (
            set(b5_component_configs())
            if self.case == "b5"
            else {"native", "polynomial"}
        )
        if self.variant not in allowed_variants:
            raise ValueError(f"unknown SFU variant {self.variant!r}")
        expected_model = "sfu_deepseek_v3" if self.case == "b4" else "sfu_llama3"
        if self.strict and job_config.model.name != expected_model:
            raise ValueError(
                f"{self.case.upper()} requires model.name={expected_model!r}, "
                f"got {job_config.model.name!r}"
            )
        if self.case == "b3" and int(job_config.training.seq_len) != 4096:
            raise ValueError("B3's direct D3/D4 fit requires training.seq_len=4096")
        if self.case == "b5" and int(job_config.training.seq_len) != 4096:
            raise ValueError("B5's routed-exp2 probe requires training.seq_len=4096")
        if self.case == "b5" and bool(job_config.compile.enable):
            raise ValueError("B5's public model probe requires compile.enable=false")
        if self.case == "b5" and self.strict:
            probe_shape = (
                int(job_config.training.local_batch_size),
                int(job_config.training.global_batch_size),
                int(job_config.training.steps),
            )
            if probe_shape != (1, 1, 80):
                raise ValueError(
                    "B5's public model probe requires local/global batch 1 and "
                    f"80 steps, got {probe_shape}"
                )
        if self.case == "b1" and parallel_dims.tp != 1:
            raise ValueError(
                "B1's fused gate/up projection requires tensor_parallel_degree=1"
            )

    def convert(self, model: nn.Module) -> None:
        patched_attention = 0
        patched_activation = 0

        attention_config = _attention_config(self.case, self.variant)
        if attention_config is not None:
            incompatible = [
                str(getattr(module, "attn_type", "unknown"))
                for module in model.modules()
                if isinstance(module, LlamaAttention)
                and getattr(module, "attn_type", None) != "sdpa"
            ]
            if incompatible:
                raise ValueError(
                    f"{self.case.upper()} requires TorchTitan SDPA attention before "
                    f"the FA4 adapter; observed {sorted(set(incompatible))}"
                )
            resolved = resolve_fa4_config(
                attention_config,
                sequence_length=int(self.job_config.training.seq_len),
            )
            patched_attention = patch_attention_modules(model, resolved)
            initialized_attention = _install_fa4_initializers(model)
            if initialized_attention != patched_attention:
                raise RuntimeError(
                    "FA4 patch count and initialization-hook count disagree: "
                    f"{patched_attention} != {initialized_attention}"
                )

        if self.case == "b1":
            patched_activation = _patch_llama_mlp(model, self.variant)
        elif self.case == "b4":
            patched_activation = _patch_grouped_experts(model, self.variant)

        expected_layers = 29 if self.case == "b4" else int(model.n_layers)
        observed = (
            patched_activation if self.case in {"b1", "b4"} else patched_attention
        )
        if self.strict and observed != expected_layers:
            raise RuntimeError(
                f"{self.case.upper()} {self.variant} expected {expected_layers} "
                f"patched modules, observed {observed}"
            )
        logger.info(
            "Configured %s %s: attention modules=%d, activation modules=%d",
            self.case.upper(),
            self.variant,
            patched_attention,
            patched_activation,
        )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        del model


_register_datasets()
_register_model_specs()
register_model_converter(SFUReproConverter, "sfu_repro")


__all__ = [
    "OLMO_MIX_REPOSITORY",
    "OLMO_MIX_REVISION",
    "ConfiguredFusedLlamaFeedForward",
    "ConfiguredGroupedExperts",
    "SFUReproConverter",
    "SLIMPAJAMA_REPOSITORY",
]

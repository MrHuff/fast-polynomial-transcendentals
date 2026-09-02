from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sfu_repro.fa4 import (
    DEFAULT_EXP2_VARIANTS,
    FA4Config,
    FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH,
    b2_component_configs,
    b3_component_configs,
    parse_exp2_frequency,
    parse_exp2_variants,
    resolve_fa4_config,
    softmax_exp2_config,
    validate_fa4_config,
)


def _fake_runtime() -> dict[str, Callable[..., Any]]:
    def forward(*args: Any, **kwargs: Any) -> tuple[object, object]:
        return object(), object()

    def backward(*args: Any, **kwargs: Any) -> tuple[object, object, object]:
        return object(), object(), object()

    def audit_all() -> tuple[str, ...]:
        return ("manifest",)

    def audit_selection(selections: object) -> tuple[str, ...]:
        return (f"selected:{selections!r}",)

    def audit_path(family: str, degree: int, source: str) -> str:
        return f"device {family} D{degree} {source}"

    def score_mod(*args: Any, **kwargs: Any):
        return lambda value: value

    return {
        "flash_attn_fwd": forward,
        "flash_attn_bwd": backward,
        "run_polynomial_coefficient_audit": audit_all,
        "audit_polynomial_selection": audit_selection,
        "audit_attention_handwritten_fast_path": audit_path,
        "create_softcap_scoremod_backend": score_mod,
        "create_softcap_scoremod_bwd_backend": score_mod,
    }


def test_b2_configs_match_reported_native_and_device_d4_routes() -> None:
    configs = b2_component_configs()
    native = configs["native"]
    polynomial = configs["polynomial"]
    assert (native.mode, native.softcap, native.softcap_backend) == (
        "softcap",
        30.0,
        "native",
    )
    assert native.softcap_backward_mode == "native"
    assert (
        polynomial.softcap_backend,
        polynomial.softcap_degree,
        polynomial.softcap_coeff_source,
        polynomial.softcap_backward_mode,
    ) == ("device", 4, "current", "analytical")

    runtime = _fake_runtime()
    resolved_native = resolve_fa4_config(native, runtime=runtime)
    resolved_polynomial = resolve_fa4_config(polynomial, runtime=runtime)
    assert resolved_native.softcap == 30.0
    assert resolved_native.score_mod is None
    assert resolved_polynomial.softcap == 0.0
    assert resolved_polynomial.score_mod is not None
    assert "tanh_fwd D4 current" in resolved_polynomial.implementation_summary
    assert "tanh_grad_analytical D4 current" in (
        resolved_polynomial.implementation_summary
    )


def test_b3_configs_match_native_sfu_and_direct_device_d3_d4() -> None:
    configs = b3_component_configs(sequence_length=4096)
    native = configs["native"]
    polynomial = configs["polynomial"]
    assert (
        native.sigmoid_variant,
        native.sigmoid_sfu_freq,
        native.sigmoid_sfu_res,
        native.sigmoid_qk_norm,
    ) == ("sfu", 16, 16, True)
    assert (
        polynomial.sigmoid_variant,
        polynomial.sigmoid_poly_backend,
        polynomial.sigmoid_degree,
        polynomial.sigmoid_degree_bwd,
        polynomial.sigmoid_backward_mode,
        polynomial.sigmoid_sfu_res,
        polynomial.sigmoid_qk_norm,
    ) == ("poly", "device", 3, 4, "direct", 0, True)

    resolved = resolve_fa4_config(
        polynomial, sequence_length=4096, runtime=_fake_runtime()
    )
    assert resolved.sigmoid_attention
    assert resolved.sigmoid_use_direct_bwd_poly
    assert resolved.sigmoid_degree == 3
    assert resolved.sigmoid_degree_bwd == 4
    assert "flash_sigmoid_direct D3 current" in resolved.implementation_summary
    assert "flash_sigmoid_direct_with_grad D4 current" in (
        resolved.implementation_summary
    )


@pytest.mark.parametrize("sequence_length", (1, 2048, 4095, 4097, 8192))
def test_b3_rejects_every_non_fitted_sequence_length(sequence_length: int) -> None:
    with pytest.raises(ValueError, match="requires sequence length 4096"):
        b3_component_configs(sequence_length=sequence_length)


def test_direct_b3_config_requires_explicit_fitted_length() -> None:
    polynomial = b3_component_configs()["polynomial"]
    with pytest.raises(ValueError, match="explicit sequence_length"):
        validate_fa4_config(polynomial)
    assert (
        validate_fa4_config(
            polynomial,
            sequence_length=FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH,
        )
        is polynomial
    )


def test_exp2_variant_parser_preserves_auto_and_lane_fractions() -> None:
    variants = parse_exp2_variants("sfu=d3:0:0,mixed=d2_safe:8:12,auto=d3:auto:0")
    assert [variant.name for variant in variants] == ["sfu", "mixed", "auto"]
    assert variants[0].nominal_forward_polynomial_fraction == 0.0
    assert variants[1].nominal_forward_polynomial_fraction == 0.5
    assert variants[1].nominal_backward_polynomial_fraction == pytest.approx(1 / 3)
    assert variants[2].forward_frequency is None
    assert variants[2].nominal_forward_pwl_fraction is None
    assert softmax_exp2_config(variants[1]) == FA4Config(
        mode="softmax",
        exp2_emu_backend="d2_safe",
        exp2_emu_freq=8,
        exp2_emu_freq_bwd=12,
    )
    assert len(parse_exp2_variants(DEFAULT_EXP2_VARIANTS)) == 11


@pytest.mark.parametrize(
    "spec,match",
    (
        ("bad=d3:5:0", "even integer"),
        ("bad=d3:4:auto", "frequency"),
        ("bad=unknown:4:0", "unsupported exp2 backend"),
        ("same=d3:0:0,same=d2_safe:4:0", "names must be unique"),
        ("missing", "variant must have"),
        ("", "at least one variant"),
    ),
)
def test_exp2_variant_parser_rejects_ambiguous_routes(spec: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_exp2_variants(spec)


def test_exp2_frequency_parser_rejects_auto_backward_and_odd_values() -> None:
    assert parse_exp2_frequency("auto", allow_auto=True) is None
    with pytest.raises(ValueError, match="frequency"):
        parse_exp2_frequency("auto", allow_auto=False)
    with pytest.raises(ValueError, match="even integer"):
        parse_exp2_frequency("7", allow_auto=True)


def test_standalone_sources_have_no_training_framework_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    owned_files = (
        root / "src/sfu_repro/fa4.py",
        root / "src/sfu_repro/activations.py",
        root / "scripts/benchmark_components.py",
        root / "scripts/benchmark_fa4_exp2_mix.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in owned_files)
    assert "low_bits_training" not in text
    assert "torchtitan" not in text.lower()

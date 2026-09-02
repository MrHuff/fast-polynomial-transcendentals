from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import benchmark_open_weights as benchmark


def _result(variant: str) -> benchmark.Result:
    return benchmark.Result(
        model="example/model",
        variant=variant,
        dtype="bf16",
        device="cpu",
        batch_size=1,
        seq_len=4,
    )


def test_intervention_contract_rejects_each_missing_patch_family() -> None:
    activation = benchmark.parse_variant("fused_swiglu_d4_current")
    with pytest.raises(RuntimeError, match="activation/MLP"):
        benchmark.require_applied_interventions(activation, _result(activation.name))

    combined = benchmark.parse_variant(
        "fused_swiglu_d4_current+spline_router_sigmoid_d4_current"
    )
    result = _result(combined.name)
    result.patched_silu_modules = 16
    with pytest.raises(RuntimeError, match="router sigmoid"):
        benchmark.require_applied_interventions(combined, result)


def test_intervention_contract_accepts_router_only_patch() -> None:
    spec = benchmark.parse_variant("spline_router_sigmoid_d4_current")
    result = _result(spec.name)
    result.patched_router_sigmoid_modules = 8

    benchmark.require_applied_interventions(spec, result)


class _FakeEvent:
    clock = 0.0

    def __init__(self, *, enable_timing: bool) -> None:
        assert enable_timing
        self.timestamp = 0.0

    def record(self) -> None:
        self.timestamp = self.clock

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, stop: "_FakeEvent") -> float:
        return stop.timestamp - self.timestamp


class _TimedModel:
    def __init__(self, durations: list[float]) -> None:
        self.durations = iter(durations)
        self.config = SimpleNamespace(model_type="test", vocab_size=100)

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        _FakeEvent.clock += next(self.durations)
        return SimpleNamespace(past_key_values=object())

    def get_input_embeddings(self) -> SimpleNamespace:
        return SimpleNamespace(num_embeddings=100)


def test_benchmarks_retain_each_prefill_and_decode_repetition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: None)
    tokens = torch.ones((1, 4), dtype=torch.long)

    _FakeEvent.clock = 0.0
    prefill = benchmark.benchmark_prefill(
        _TimedModel([100.0, 1.0, 2.0, 3.0]),
        tokens,
        warmup=1,
        steps=3,
    )
    assert prefill[0] == pytest.approx(2.0)
    assert prefill[2] == pytest.approx([1.0, 2.0, 3.0])

    _FakeEvent.clock = 0.0
    decode = benchmark.benchmark_decode(
        _TimedModel([100.0, 2.0, 3.0, 100.0, 4.0, 5.0]),
        tokens,
        warmup=0,
        repeats=2,
        decode_steps=2,
        seed=1234,
    )
    assert decode[0] == pytest.approx(3.5)
    assert decode[2] == pytest.approx([5.0, 9.0])


def test_module_attestation_binds_loaded_binary_through_direct_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "autonumerics_zero" / "spline_ops"
    source.mkdir(parents=True)
    module_file = tmp_path / "environment" / "spline_ops.so"
    module_file.parent.mkdir()
    module_file.write_bytes(b"test extension")
    monkeypatch.setitem(
        sys.modules,
        "spline_ops",
        SimpleNamespace(__file__=str(module_file)),
    )

    class _Distribution:
        def read_text(self, name: str) -> str | None:
            assert name == "direct_url.json"
            return '{"url": "' + source.as_uri() + '"}'

    revision = "a" * 40
    monkeypatch.setattr(
        benchmark, "distribution_metadata", lambda name: _Distribution()
    )
    monkeypatch.setattr(benchmark, "distribution_version", lambda name: "0.1.0")
    monkeypatch.setattr(benchmark, "git_revision", lambda path: revision)
    monkeypatch.setattr(benchmark, "git_worktree_state", lambda path: (False, 0))

    attestation = benchmark.module_origin_attestation(
        "spline_ops",
        {
            "module": "spline_ops",
            "distribution": "sfu-spline-ops",
            "version": "0.1.0",
            "path": "autonumerics_zero/spline_ops",
            "revision_source": "repository",
        },
        repository_revision=revision,
        repository_root=tmp_path,
    )

    assert attestation["module_loaded"] is True
    assert attestation["binding_method"] == "pep610-direct-url"
    assert attestation["source_revision"] == revision
    assert attestation["source_dirty"] is False
    assert attestation["module_sha256"]
    assert attestation["bound"] is True

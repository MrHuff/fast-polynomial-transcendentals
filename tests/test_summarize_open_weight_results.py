from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import summarize_open_weight_results as summary


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPEN_WEIGHT_CONFIG = REPOSITORY_ROOT / "configs/open_weight_paper.json"
TASK_CONFIG = REPOSITORY_ROOT / "configs/lm_eval_paper_tasks.json"
ENVIRONMENT_CONFIG = REPOSITORY_ROOT / "configs/eval_environments/profiles.json"
EXPERIMENT_MANIFEST = REPOSITORY_ROOT / "repro/experiments.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol() -> summary.TaskExpectations:
    return summary.load_task_expectations(TASK_CONFIG)


def _external_source_pins() -> dict[str, str]:
    return summary.load_external_source_pins(EXPERIMENT_MANIFEST)


def _environment_packages(case: summary.CaseSpec) -> dict[str, str]:
    cases = summary.load_cases(OPEN_WEIGHT_CONFIG)
    expectations = summary.load_environment_expectations(
        ENVIRONMENT_CONFIG,
        cases,
    )
    packages = dict(expectations[case.key])
    for module in summary.load_local_module_expectations(ENVIRONMENT_CONFIG, (case,))[
        case.key
    ]:
        packages[module.distribution] = module.version
    return packages


def _module_origins(case: summary.CaseSpec) -> dict[str, dict[str, object]]:
    origins: dict[str, dict[str, object]] = {}
    source_revision = "1" * 40
    for module in summary.load_local_module_expectations(ENVIRONMENT_CONFIG, (case,))[
        case.key
    ]:
        expected_revision = (
            source_revision
            if module.revision_source == "repository"
            else module.revision
        )
        origins[module.name] = {
            "name": module.name,
            "module": module.module,
            "module_loaded": True,
            "module_file": f"/environment/{module.module}.so",
            "module_sha256": "a" * 64,
            "native_binary_required": module.require_native_binary,
            "native_binary_sha256": (
                {f"<external>/{module.module}.so": "b" * 64}
                if module.require_native_binary
                else {}
            ),
            "distribution": module.distribution,
            "package_version": module.version,
            "expected_package_version": module.version,
            "source_path": module.source_path,
            "direct_url_source": module.source_path,
            "origin_matches_source": True,
            "binding_method": "pep610-direct-url",
            "source_revision": expected_revision,
            "expected_source_revision": expected_revision,
            "source_revision_matches": True,
            "source_dirty": False,
            "source_untracked_files": 0,
            "bound": True,
        }
    return origins


def _evaluation(score_offset: float = 0.0) -> dict[str, object]:
    protocol = _protocol()
    task_results = {
        task: {metric: (8.0 if task == "wikitext" else 0.5 + score_offset)}
        for task, metric in summary.TASK_METRICS.items()
    }
    leaf_configs = {task: {"task": task} for task in summary.TASK_METRICS}
    return {
        "results": task_results,
        "configs": leaf_configs,
        "samples": {task: [{"doc_id": 0}] for task in leaf_configs},
        "n-samples": {task: {"original": 1, "effective": 1} for task in leaf_configs},
        "task_num_fewshot": summary.load_fewshot_expectations(OPEN_WEIGHT_CONFIG),
        "public_task_protocol": {
            "class": protocol.protocol_class,
            "selection_date": protocol.selection_date,
            "lm_eval_version": protocol.harness_version,
            "lm_eval_source_revision": protocol.harness_revision,
            "lm_eval_source_path": protocol.harness_submodule_path,
            "lm_eval_source_clean": True,
            "lm_eval_module_file": "lm_eval/__init__.py",
            "log_samples": True,
            "dataset_pins": protocol.task_pins,
            "evaluator_seeds": protocol.evaluator_seeds,
        },
    }


def _recorded_command(
    case: summary.CaseSpec,
    *,
    quality: bool,
    protocol: summary.MeasurementProtocol,
) -> list[str]:
    sequence_length = (
        case.quality_sequence_length_argument
        if quality
        else case.throughput_sequence_length
    )
    command = [
        "python",
        "scripts/benchmark_open_weights.py",
        "--model",
        case.model_id,
        "--attn-implementation",
        case.attn_implementation,
        "--mode",
        protocol.quality_mode if quality else protocol.throughput_mode,
        "--batch-size",
        str(protocol.batch_size),
        "--seq-len",
        str(sequence_length),
        "--steps",
        str(protocol.prefill_measurements),
        "--warmup",
        str(protocol.prefill_warmups),
        "--decode-steps",
        str(protocol.decode_steps),
        "--decode-repeats",
        str(protocol.decode_measurements),
        "--decode-warmup",
        str(protocol.decode_warmups),
        "--dtype",
        protocol.dtype,
        "--device",
        protocol.device,
        "--environment-profiles",
        "configs/eval_environments/profiles.json",
        "--suite-config",
        "configs/open_weight_paper.json",
        "--seed",
        str(protocol.seed),
        "--suite-environment-preflight",
        "passed",
        "--json-out",
        f"outputs/{case.key}.json",
        "--revision",
        case.revision,
        "--revision-provenance",
        case.revision_provenance,
    ]
    if case.experts_implementation is not None:
        command.extend(("--experts-implementation", case.experts_implementation))
    if case.trust_remote_code:
        command.append("--trust-remote-code")
    if protocol.mxfp4_dequantize:
        command.append("--mxfp4-dequantize")
    if protocol.freeze_granite_moe_routing:
        command.append("--freeze-granite-moe-routing")
    if quality:
        command.extend(
            (
                "--eval-tasks",
                ",".join(protocol.quality_tasks),
                "--eval-num-fewshot",
                str(protocol.quality_default_num_fewshot),
                "--eval-batch-size",
                case.eval_batch_size,
            )
        )
        fewshot = summary.load_fewshot_expectations(OPEN_WEIGHT_CONFIG)
        for task in protocol.quality_tasks:
            command.extend(("--eval-task-fewshot", f"{task}={fewshot[task]}"))
        command.extend(
            ("--eval-task-config", protocol.quality_task_config, "--eval-log-samples")
        )
    for variant in case.variants:
        command.extend(("--variant", variant))
    return command


def _document(case: summary.CaseSpec, *, kind: str = "quality") -> dict[str, object]:
    if kind not in {"quality", "throughput"}:
        raise ValueError(kind)
    protocol = summary.load_measurement_protocol(OPEN_WEIGHT_CONFIG)
    quality = kind == "quality"
    sequence_length = (
        case.quality_sequence_length_argument
        if quality
        else case.throughput_sequence_length
    )
    rows = []
    for index, variant in enumerate(case.variants):
        patch_requirements = summary.variant_patch_requirements(variant)
        rows.append(
            {
                "model": case.model_id,
                "variant": variant,
                "dtype": protocol.dtype,
                "device": protocol.device,
                "batch_size": protocol.batch_size,
                "seq_len": sequence_length,
                "error": None,
                "eval": _evaluation(index * 0.001) if quality else None,
                "prefill_tokens_per_s": None if quality else 100.0 + index,
                "prefill_ms": None if quality else 10.0,
                "prefill_repetition_ms": (
                    [] if quality else [10.0] * protocol.prefill_measurements
                ),
                "decode_tokens_per_s": None if quality else 10.0 + index,
                "decode_ms_per_token": None if quality else 2.0,
                "decode_repetition_ms": (
                    []
                    if quality
                    else [2.0 * protocol.decode_steps] * protocol.decode_measurements
                ),
                "patched_silu_modules": (
                    32 if "activation" in patch_requirements else 0
                ),
                "patched_router_sigmoid_modules": (
                    32 if "router" in patch_requirements else 0
                ),
                "patched_gemma_softcap": "softcap" in patch_requirements,
            }
        )
    return {
        "schema_version": 1,
        "experiment": {
            "id": "open-weight-evaluation",
            "command": _recorded_command(case, quality=quality, protocol=protocol),
            "models": [case.model_id],
            "variants": list(case.variants),
            "requested_model_revisions": {case.model_id: case.revision},
            "requested_tokenizer_revisions": {case.model_id: case.revision},
            "revision_provenance": case.revision_provenance,
        },
        "source": {
            "revision": "1" * 40,
            "dirty": False,
            "external_components": _external_source_pins(),
            "external_component_dirty": {
                name: False for name in _external_source_pins()
            },
        },
        "environment": {
            "python": "3.12.0",
            "packages": _environment_packages(case),
            "module_origins": _module_origins(case),
        },
        "measurement": {
            "mode": protocol.quality_mode if quality else protocol.throughput_mode,
            "dtype": protocol.dtype,
            "device": protocol.device,
            "batch_size": protocol.batch_size,
            "sequence_length_argument": sequence_length,
            "prefill_measurements": protocol.prefill_measurements,
            "prefill_warmups": protocol.prefill_warmups,
            "decode_steps": protocol.decode_steps,
            "decode_measurements": protocol.decode_measurements,
            "decode_warmups": protocol.decode_warmups,
            "seed": protocol.seed,
            "eval_tasks": ",".join(protocol.quality_tasks) if quality else "",
            "eval_num_fewshot": (
                protocol.quality_default_num_fewshot if quality else 5
            ),
            "eval_task_fewshot": (
                [
                    f"{task}={shots}"
                    for task, shots in summary.load_fewshot_expectations(
                        OPEN_WEIGHT_CONFIG
                    ).items()
                ]
                if quality
                else []
            ),
            "eval_limit": None,
            "eval_batch_size": case.eval_batch_size if quality else "auto",
            "eval_task_config": protocol.quality_task_config if quality else None,
            "eval_task_config_sha256": _sha256(TASK_CONFIG) if quality else None,
            "eval_log_samples": protocol.quality_log_samples if quality else False,
            "attention_implementation": case.attn_implementation,
            "experts_implementation": case.experts_implementation,
            "trust_remote_code": case.trust_remote_code,
            "mxfp4_dequantize": protocol.mxfp4_dequantize,
            "freeze_granite_moe_routing": protocol.freeze_granite_moe_routing,
            "suite_environment_preflight": "passed",
            "suite_config": "configs/open_weight_paper.json",
            "suite_config_sha256": _sha256(OPEN_WEIGHT_CONFIG),
            "environment_profiles": "configs/eval_environments/profiles.json",
            "environment_profiles_sha256": _sha256(ENVIRONMENT_CONFIG),
        },
        "results": rows,
    }


def _write_complete_set(tmp_path: Path) -> tuple[summary.CaseSpec, ...]:
    cases = summary.load_cases(OPEN_WEIGHT_CONFIG)
    for case in cases:
        for kind in ("quality", "throughput"):
            (tmp_path / f"{case.key}_{kind}.json").write_text(
                json.dumps(_document(case, kind=kind)), encoding="utf-8"
            )
    return cases


def _collect_one(
    tmp_path: Path,
    document: dict[str, object],
    *,
    task_config: Path = TASK_CONFIG,
    external_source_pins: dict[str, str] | None = None,
) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    summary.collect_records(
        [path],
        summary.load_cases(OPEN_WEIGHT_CONFIG),
        task_config=task_config,
        open_weight_config=OPEN_WEIGHT_CONFIG,
        external_source_pins=(
            _external_source_pins()
            if external_source_pins is None
            else external_source_pins
        ),
    )


def test_complete_results_preserve_exact_metric_names_and_compute_deltas(
    tmp_path: Path,
) -> None:
    cases = _write_complete_set(tmp_path)
    paths = summary.expand_inputs([tmp_path])
    records = summary.collect_records(
        paths,
        cases,
        task_config=TASK_CONFIG,
        open_weight_config=OPEN_WEIGHT_CONFIG,
        external_source_pins=_external_source_pins(),
    )
    environment_expectations = summary.load_environment_expectations(
        ENVIRONMENT_CONFIG,
        cases,
    )
    summary.require_complete(
        records,
        cases,
        "combined",
        environment_expectations,
    )
    rows = summary.build_rows(records, cases)

    assert len(rows) == 15
    assert "gsm8k/exact_match,flexible-extract" in rows[0]
    assert "wikitext/word_perplexity,none" in rows[0]
    assert rows[0]["quality_mean_delta_pp"] == pytest.approx(0.0)
    assert rows[1]["quality_mean_delta_pp"] == pytest.approx(0.1)
    assert rows[1]["prefill_speedup"] == pytest.approx(1.01)
    assert rows[1]["decode_speedup"] == pytest.approx(1.1)


def test_summary_rejects_a_missing_selected_metric(tmp_path: Path) -> None:
    cases = summary.load_cases(OPEN_WEIGHT_CONFIG)
    document = _document(cases[0])
    document["results"][0]["eval"]["results"]["gsm8k"].pop(
        "exact_match,flexible-extract"
    )
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="gsm8k/exact_match,flexible-extract"):
        summary.collect_records(
            [path],
            cases,
            task_config=TASK_CONFIG,
            open_weight_config=OPEN_WEIGHT_CONFIG,
            external_source_pins=_external_source_pins(),
        )


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("class", "historical", "protocol class"),
        ("selection_date", "1970-01-01", "selection date"),
        ("lm_eval_version", "0.4.11", "lm-eval version"),
        ("lm_eval_source_path", "another-checkout", "source path"),
        ("lm_eval_source_clean", False, "clean lm-eval checkout"),
        ("lm_eval_module_file", "../lm_eval.py", "safe lm-eval module path"),
    ),
)
def test_summary_rejects_mismatched_public_task_protocol(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    cases = summary.load_cases(OPEN_WEIGHT_CONFIG)
    document = _document(cases[0])
    document["results"][0]["eval"]["public_task_protocol"][field] = value

    with pytest.raises(ValueError, match=match):
        _collect_one(tmp_path, document)


def test_summary_rejects_truncated_quality_samples(tmp_path: Path) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case)
    evaluation = document["results"][0]["eval"]
    evaluation["n-samples"]["piqa"] = {"original": 2, "effective": 2}

    with pytest.raises(ValueError, match="retained 1 samples for piqa, expected 2"):
        _collect_one(tmp_path, document)


def test_summary_rejects_limited_effective_sample_count(tmp_path: Path) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case)
    evaluation = document["results"][0]["eval"]
    evaluation["n-samples"]["piqa"] = {"original": 2, "effective": 1}

    with pytest.raises(ValueError, match="effective differs from n-samples.original"):
        _collect_one(tmp_path, document)


@pytest.mark.parametrize("value", (0, 1.0, True))
def test_summary_rejects_non_integral_or_nonpositive_sample_counts(
    tmp_path: Path, value: object
) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case)
    evaluation = document["results"][0]["eval"]
    evaluation["n-samples"]["piqa"]["effective"] = value

    with pytest.raises(ValueError, match="positive integer"):
        _collect_one(tmp_path, document)


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("suite_environment_preflight", "skipped", "preflight"),
        ("eval_limit", 1, "limited evaluation"),
        ("batch_size", 2, "batch_size"),
        ("dtype", "fp16", "dtype"),
        ("prefill_measurements", 1, "prefill_measurements"),
        ("decode_measurements", 1, "decode_measurements"),
        ("eval_batch_size", "1", "eval_batch_size"),
        ("eval_task_config", "configs/other.json", "eval_task_config"),
        ("eval_task_config_sha256", "0" * 64, "eval_task_config_sha256"),
        ("suite_config", "configs/other.json", "suite_config"),
        ("suite_config_sha256", "0" * 64, "suite_config_sha256"),
        (
            "environment_profiles",
            "configs/eval_environments/other.json",
            "environment_profiles",
        ),
        (
            "environment_profiles_sha256",
            "0" * 64,
            "environment_profiles_sha256",
        ),
        ("mode", "both", "mode"),
    ),
)
def test_summary_rejects_non_protocol_quality_measurements(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case)
    document["measurement"][field] = value

    with pytest.raises(ValueError, match=match):
        _collect_one(tmp_path, document)


def test_quality_rejects_an_alternate_task_config_path(tmp_path: Path) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case)
    alternate = tmp_path / "tasks.json"
    alternate.write_bytes(TASK_CONFIG.read_bytes())

    with pytest.raises(ValueError, match="path declared by the open-weight protocol"):
        _collect_one(tmp_path, document, task_config=alternate)


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("sequence_length_argument", 1, "sequence length"),
        ("prefill_warmups", 1, "prefill_warmups"),
        ("decode_steps", 1, "decode_steps"),
        ("decode_warmups", 1, "decode_warmups"),
    ),
)
def test_summary_rejects_non_protocol_throughput_measurements(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case, kind="throughput")
    document["measurement"][field] = value

    with pytest.raises(ValueError, match=match):
        _collect_one(tmp_path, document)


@pytest.mark.parametrize(
    "field,value",
    (
        ("batch_size", 2),
        ("dtype", "fp16"),
        ("device", "cpu"),
        ("seq_len", 1),
    ),
)
def test_summary_rejects_result_shape_that_disagrees_with_protocol(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case, kind="throughput")
    document["results"][0][field] = value

    with pytest.raises(ValueError, match=field):
        _collect_one(tmp_path, document)


def test_summary_requires_recorded_benchmark_command(tmp_path: Path) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case)
    document["experiment"].pop("command")

    with pytest.raises(ValueError, match="complete benchmark command"):
        _collect_one(tmp_path, document)


def test_summary_rejects_an_incomplete_recorded_command(tmp_path: Path) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case)
    command = document["experiment"]["command"]
    steps_index = command.index("--steps")
    del command[steps_index : steps_index + 2]

    with pytest.raises(ValueError, match="inconsistent --steps"):
        _collect_one(tmp_path, document)


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("protocol_class", "historical", "protocol_class"),
        ("harness.version", "0.4.11", "harness version"),
        ("harness.submodule_path", "../outside", "stay inside"),
    ),
)
def test_task_config_requires_public_class_version_and_safe_path(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    document = json.loads(TASK_CONFIG.read_text(encoding="utf-8"))
    if field == "protocol_class":
        document[field] = value
    else:
        _, child = field.split(".", 1)
        document["harness"][child] = value
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        summary.load_task_expectations(path)


def test_summary_requires_clean_source_and_external_checkouts(
    tmp_path: Path,
) -> None:
    cases = summary.load_cases(OPEN_WEIGHT_CONFIG)

    dirty_root = _document(cases[0])
    dirty_root["source"]["dirty"] = True
    with pytest.raises(ValueError, match="dirty source tree"):
        _collect_one(tmp_path, dirty_root)

    dirty_external = _document(cases[0])
    dirty_external["source"]["external_component_dirty"]["lm-evaluation-harness"] = True
    with pytest.raises(ValueError, match="dirty or unavailable external sources"):
        _collect_one(tmp_path, dirty_external)

    wrong_external = _document(cases[0])
    wrong_external["source"]["external_components"]["lm-evaluation-harness"] = "0" * 40
    with pytest.raises(ValueError, match="external source pins differ"):
        _collect_one(tmp_path, wrong_external)


def test_external_pin_manifest_does_not_require_unrelated_components(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "external_source_pins": {"torchtitan": None},
            }
        ),
        encoding="utf-8",
    )

    assert summary.load_external_source_pins(manifest) == {}


def test_non_flash_throughput_does_not_require_external_checkouts(
    tmp_path: Path,
) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case, kind="throughput")
    document["source"]["external_components"] = {}
    document["source"]["external_component_dirty"] = {}

    _collect_one(
        tmp_path,
        document,
        task_config=tmp_path / "absent-task-config.json",
        external_source_pins={"lm-evaluation-harness": None},
    )


def test_quality_requires_only_the_lm_eval_external_source(
    tmp_path: Path,
) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case)
    lm_eval_revision = _protocol().harness_revision
    document["source"]["external_components"] = {
        "lm-evaluation-harness": lm_eval_revision,
        "torchtitan": "0" * 40,
    }
    document["source"]["external_component_dirty"] = {
        "lm-evaluation-harness": False,
        "torchtitan": True,
    }

    _collect_one(
        tmp_path,
        document,
        external_source_pins={
            "lm-evaluation-harness": lm_eval_revision,
            "torchtitan": "f" * 40,
        },
    )

    with pytest.raises(ValueError, match="required external source pins"):
        _collect_one(tmp_path, document, external_source_pins={})


def test_kimi_throughput_requires_flash_attention_but_not_lm_eval(
    tmp_path: Path,
) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[-1]
    assert case.attn_implementation.startswith("flash_attention_")
    document = _document(case, kind="throughput")
    flash_revision = _external_source_pins()["flash-attention"]
    document["source"]["external_components"] = {"flash-attention": flash_revision}
    document["source"]["external_component_dirty"] = {"flash-attention": False}

    _collect_one(
        tmp_path,
        document,
        task_config=tmp_path / "absent-task-config.json",
        external_source_pins={"flash-attention": flash_revision},
    )

    with pytest.raises(ValueError, match="required external source pins"):
        _collect_one(tmp_path, document, external_source_pins={})


def test_throughput_completion_does_not_require_lm_eval_package(
    tmp_path: Path,
) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case, kind="throughput")
    document["source"]["external_components"] = {}
    document["source"]["external_component_dirty"] = {}
    document["environment"]["packages"].pop("lm-eval")
    path = tmp_path / "throughput.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    records = summary.collect_records(
        [path],
        (case,),
        task_config=tmp_path / "absent-task-config.json",
        open_weight_config=OPEN_WEIGHT_CONFIG,
        external_source_pins={},
    )
    expectations = summary.load_environment_expectations(
        ENVIRONMENT_CONFIG, summary.load_cases(OPEN_WEIGHT_CONFIG)
    )

    summary.require_complete(
        records,
        (case,),
        "throughput",
        {case.key: expectations[case.key]},
    )


def test_summary_rejects_unbound_local_module_origin(tmp_path: Path) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case)
    document["environment"]["module_origins"]["spline_ops"]["bound"] = False

    with pytest.raises(ValueError, match="spline_ops attestation is unbound"):
        _collect_one(tmp_path, document)


def test_summary_rejects_zero_patch_non_native_row(tmp_path: Path) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case)
    document["results"][1]["patched_silu_modules"] = 0
    path = tmp_path / "result.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    records = summary.collect_records(
        [path],
        (case,),
        task_config=TASK_CONFIG,
        open_weight_config=OPEN_WEIGHT_CONFIG,
        external_source_pins=_external_source_pins(),
    )
    expectations = summary.load_environment_expectations(
        ENVIRONMENT_CONFIG, summary.load_cases(OPEN_WEIGHT_CONFIG)
    )

    with pytest.raises(ValueError, match="activation patch scope is unexpected"):
        summary.require_complete(
            records,
            (case,),
            "combined",
            {case.key: expectations[case.key]},
        )


def test_summary_requires_declared_variant_order_and_repetition_timings(
    tmp_path: Path,
) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    reordered = _document(case)
    reordered["results"][0], reordered["results"][1] = (
        reordered["results"][1],
        reordered["results"][0],
    )
    with pytest.raises(ValueError, match="result rows do not preserve variant order"):
        _collect_one(tmp_path, reordered)

    missing_samples = _document(case, kind="throughput")
    missing_samples["results"][0]["prefill_repetition_ms"] = []
    with pytest.raises(ValueError, match="contains 0 samples, expected 20"):
        _collect_one(tmp_path, missing_samples)


def test_task_protocol_revision_must_match_external_source_pin(
    tmp_path: Path,
) -> None:
    cases = summary.load_cases(OPEN_WEIGHT_CONFIG)
    pins = _external_source_pins()
    pins["lm-evaluation-harness"] = "0" * 40

    with pytest.raises(ValueError, match="task protocol lm-eval revision"):
        _collect_one(tmp_path, _document(cases[0]), external_source_pins=pins)


def test_environment_inventory_must_match_declared_profile(
    tmp_path: Path,
) -> None:
    case = summary.load_cases(OPEN_WEIGHT_CONFIG)[0]
    document = _document(case)
    document["environment"]["packages"]["lm-eval"] = "0.4.11"
    path = tmp_path / "result.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    records = summary.collect_records(
        [path],
        (case,),
        task_config=TASK_CONFIG,
        open_weight_config=OPEN_WEIGHT_CONFIG,
        external_source_pins=_external_source_pins(),
    )
    all_expectations = summary.load_environment_expectations(
        ENVIRONMENT_CONFIG,
        summary.load_cases(OPEN_WEIGHT_CONFIG),
    )
    expectations = {case.key: all_expectations[case.key]}

    with pytest.raises(ValueError, match="lm-eval does not match"):
        summary.require_complete(records, (case,), "combined", expectations)


def test_summary_cli_writes_csv_and_markdown(tmp_path: Path) -> None:
    _write_complete_set(tmp_path)
    csv_path = tmp_path / "summary.csv"
    markdown_path = tmp_path / "summary.md"

    assert (
        summary.main(
            [
                str(tmp_path),
                "--csv-out",
                str(csv_path),
                "--markdown-out",
                str(markdown_path),
            ]
        )
        == 0
    )
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 16
    assert "| Qwen2.5 7B | current |" in markdown_path.read_text(encoding="utf-8")

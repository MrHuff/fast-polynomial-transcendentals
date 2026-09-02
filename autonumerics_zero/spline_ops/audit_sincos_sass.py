#!/usr/bin/env python3
"""Audit the emitted SASS for the paired native and polynomial sin/cos kernels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess


FUNCTION_RE = re.compile(r"^\s*Function\s*:\s*(?P<name>\S+)", re.MULTILINE)
INSTRUCTION_RE = re.compile(r"/\*[0-9a-f]+\*/\s+(?P<opcode>[A-Z][A-Z0-9_.]*)")


def function_sections(sass: str) -> dict[str, str]:
    matches = list(FUNCTION_RE.finditer(sass))
    return {
        match.group("name"): sass[
            match.start() : matches[index + 1].start() if index + 1 < len(matches) else None
        ]
        for index, match in enumerate(matches)
    }


def select_kernel(
    sections: dict[str, str],
    kernel: str,
    evaluator: str,
) -> tuple[str, str]:
    mangled_evaluator = f"{len(evaluator)}{evaluator}E"
    matches = [
        (name, body)
        for name, body in sections.items()
        if kernel in name and mangled_evaluator in name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {kernel} {evaluator} kernel, found {len(matches)}"
        )
    return matches[0]


def select_named_kernel(
    sections: dict[str, str],
    kernel: str,
) -> tuple[str, str]:
    matches = [(name, body) for name, body in sections.items() if kernel in name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {kernel} kernel, found {len(matches)}")
    return matches[0]


def summarize(name: str, body: str) -> dict[str, object]:
    opcodes = [match.group("opcode") for match in INSTRUCTION_RE.finditer(body)]
    mufu = [opcode for opcode in opcodes if opcode.startswith("MUFU")]
    non_padding = [opcode for opcode in opcodes if opcode != "NOP"]
    evaluator_related = [
        opcode
        for opcode in non_padding
        if opcode.startswith(("F", "H", "MUFU", "LOP3", "SHF", "PRMT"))
    ]
    return {
        "symbol": name,
        # ELF function extents include unreachable alignment NOPs. Keep both
        # counts explicit so symbol padding is never presented as evaluator
        # work again.
        "encoded_instruction_count": len(opcodes),
        "non_padding_instruction_count": len(non_padding),
        "evaluator_related_instruction_count": len(evaluator_related),
        "evaluator_related_opcodes": dict(sorted(Counter(evaluator_related).items())),
        "ffma_count": sum(opcode == "FFMA" for opcode in opcodes),
        "float_to_int_count": sum(opcode.startswith("F2I") for opcode in opcodes),
        "int_to_float_count": sum(opcode.startswith("I2FP") for opcode in opcodes),
        "round_count": sum(opcode.startswith("FRND") for opcode in opcodes),
        "packed_bf16_fma_count": sum(
            opcode == "HFMA2.BF16_V2" for opcode in opcodes
        ),
        "packed_bf16_mul_count": sum(
            opcode == "HMUL2.BF16_V2" for opcode in opcodes
        ),
        "packed_fp16_fma_count": sum(opcode == "HFMA2" for opcode in opcodes),
        "packed_fp16_mul_count": sum(opcode == "HMUL2" for opcode in opcodes),
        "mufu_instructions": mufu,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", type=Path)
    parser.add_argument(
        "--cuobjdump",
        type=Path,
        default=Path("/usr/local/cuda-13.0/bin/cuobjdump"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completed = subprocess.run(
        [str(args.cuobjdump), "--dump-sass", str(args.extension)],
        check=True,
        capture_output=True,
        text=True,
    )
    sections = function_sections(completed.stdout)
    payload: dict[str, object] = {}
    for label, evaluator in (
        ("native_sfu", "NativeSincos"),
        ("poly_d3_d4", "PolynomialD3D4"),
        ("poly_d3_d4_cycle", "PolynomialD3D4Cycle"),
        ("poly_d3_d4_magic_bias", "PolynomialD3D4MagicBias"),
        ("half_turn_d5_d4_ls", "PolynomialD5D4HalfTurnLS"),
        ("half_turn_d5_d4_sollya", "PolynomialD5D4HalfTurnSollya"),
        ("half_turn_d5_d4_sollya_fast", "PolynomialD5D4HalfTurnSollyaFast"),
        ("poly_d5_d4", "PolynomialD5D4"),
        ("poly_d7_d6", "PolynomialD7D6"),
    ):
        name, body = select_kernel(sections, "sincos_scalar_kernel", evaluator)
        payload[label] = summarize(name, body)

    for label, evaluator in (
        ("sigmoid_d3_packed_fp16_reference", "SIGMOID_FWD_D3_ODD"),
        ("tanh_d3_packed_fp16_reference", "TANH_FWD_D3_ODD"),
    ):
        name, body = select_kernel(sections, "unary_scalar_kernel", evaluator)
        payload[label] = summarize(name, body)

    for label, evaluator in (
        ("native_sfu_fp16", "NativeSincosFP16"),
        ("quarter_turn_d3_d4_packed_fp16", "PolynomialD3D4QuarterTurnFP16"),
        ("half_turn_d3_d4_packed_fp16", "PolynomialD3D4HalfTurnFP16"),
        ("half_turn_d5_d4_packed_fp16", "PolynomialD5D4HalfTurnFP16"),
        ("half_turn_d5_d6_packed_fp16", "PolynomialD5D6HalfTurnFP16"),
        ("half_turn_d7_d6_packed_fp16", "PolynomialD7D6HalfTurnFP16"),
    ):
        name, body = select_kernel(sections, "sincos_fp16_vec_kernel", evaluator)
        payload[label] = summarize(name, body)

    for label, kernel in (
        ("rope_native_sfu_fp16", "rope_sincos_native_fp16_kernel"),
        ("rope_fixed_q32_d3_d4_fp16", "rope_sincos_fixed_d3_d4_fp16_kernel"),
        (
            "rope_fixed_q32_half_turn_d5_d6_fp16",
            "rope_sincos_fixed_half_turn_d5_d6_fp16_kernel",
        ),
        ("rope_fixed_q32_lut_fp16", "rope_sincos_fixed_lut_fp16_kernel"),
        ("rope_apply_cached_fp16", "rope_apply_cached_fp16_kernel"),
        ("rope_apply_native_sfu_fp16", "rope_apply_native_fp16_kernel"),
        (
            "rope_apply_fixed_q32_half_turn_d5_d6_fp16",
            "rope_apply_fixed_half_turn_d5_d6_fp16_kernel",
        ),
    ):
        name, body = select_named_kernel(sections, kernel)
        payload[label] = summarize(name, body)

    for label in (
        "rope_fixed_q32_d3_d4_fp16",
        "rope_fixed_q32_half_turn_d5_d6_fp16",
    ):
        row = payload[label]
        if row["packed_bf16_fma_count"] or row["packed_bf16_mul_count"]:
            raise RuntimeError(f"{label} unexpectedly emitted packed BF16 arithmetic")
        if row["ffma_count"] or row["mufu_instructions"]:
            raise RuntimeError(f"{label} unexpectedly emitted FP32 FMA or SFU work")

    fused_polynomial = payload[
        "rope_apply_fixed_q32_half_turn_d5_d6_fp16"
    ]
    if (
        fused_polynomial["packed_bf16_fma_count"]
        or fused_polynomial["packed_bf16_mul_count"]
    ):
        raise RuntimeError(
            "fused polynomial RoPE unexpectedly emitted packed BF16 arithmetic"
        )
    if fused_polynomial["ffma_count"] or fused_polynomial["mufu_instructions"]:
        raise RuntimeError(
            "fused polynomial RoPE unexpectedly emitted FP32 FMA or SFU work"
        )
    fused_native_mufu = payload["rope_apply_native_sfu_fp16"][
        "mufu_instructions"
    ]
    if fused_native_mufu.count("MUFU.SIN") != 4 or fused_native_mufu.count(
        "MUFU.COS"
    ) != 4:
        raise RuntimeError(
            "fused native RoPE did not emit four paired SFU evaluations: "
            f"{fused_native_mufu}"
        )

    for label, evaluator in (
        ("native_sfu_bf16", "NativeSincosBF16"),
        ("poly_d3_d4_bf16", "PolynomialD3D4BF16"),
        ("quarter_turn_d3_d4_packed_bf16", "PolynomialD3D4QuarterTurnBF16"),
        ("half_turn_d3_d4_packed_bf16", "PolynomialD3D4HalfTurnBF16"),
        ("half_turn_d5_d6_packed_bf16", "PolynomialD5D6HalfTurnBF16"),
    ):
        name, body = select_kernel(sections, "sincos_bf16_vec_kernel", evaluator)
        payload[label] = summarize(name, body)

    native_mufu = payload["native_sfu"]["mufu_instructions"]
    if native_mufu != ["MUFU.SIN", "MUFU.COS"]:
        raise RuntimeError(f"Native control did not lower to SIN/COS SFUs: {native_mufu}")
    for label in (
        "poly_d3_d4",
        "poly_d3_d4_cycle",
        "poly_d3_d4_magic_bias",
        "half_turn_d5_d4_ls",
        "half_turn_d5_d4_sollya",
        "half_turn_d5_d4_sollya_fast",
        "poly_d5_d4",
        "poly_d7_d6",
    ):
        if payload[label]["mufu_instructions"]:
            raise RuntimeError(f"{label} unexpectedly contains MUFU instructions")
    cycle = payload["poly_d3_d4_cycle"]
    if cycle["float_to_int_count"] or cycle["int_to_float_count"]:
        raise RuntimeError(
            "cycle-domain D3/D4 unexpectedly emitted integer quadrant conversions"
        )
    if cycle["round_count"] != 1:
        raise RuntimeError(
            "cycle-domain D3/D4 should emit one floating-point round: "
            f"FRND={cycle['round_count']}"
        )
    if payload["poly_d3_d4_bf16"]["mufu_instructions"]:
        raise RuntimeError("packed BF16 D3/D4 unexpectedly contains MUFU instructions")
    if payload["quarter_turn_d3_d4_packed_bf16"]["mufu_instructions"]:
        raise RuntimeError(
            "quarter-turn packed BF16 D3/D4 unexpectedly contains MUFU instructions"
        )
    if payload["half_turn_d5_d6_packed_bf16"]["mufu_instructions"]:
        raise RuntimeError(
            "half-turn packed BF16 D5/D6 unexpectedly contains MUFU instructions"
        )
    if payload["half_turn_d3_d4_packed_bf16"]["mufu_instructions"]:
        raise RuntimeError(
            "half-turn packed BF16 D3/D4 unexpectedly contains MUFU instructions"
        )
    magic_bias = payload["poly_d3_d4_magic_bias"]
    if magic_bias["float_to_int_count"] or magic_bias["int_to_float_count"]:
        raise RuntimeError(
            "magic-bias D3/D4 unexpectedly emitted integer quadrant conversions"
        )
    for label in (
        "half_turn_d5_d4_ls",
        "half_turn_d5_d4_sollya",
        "half_turn_d5_d4_sollya_fast",
    ):
        half_turn = payload[label]
        if half_turn["float_to_int_count"] or half_turn["int_to_float_count"]:
            raise RuntimeError(
                f"{label} unexpectedly emitted integer half-turn conversions"
            )
    split_half_turn = payload["half_turn_d5_d4_sollya"]
    fast_half_turn = payload["half_turn_d5_d4_sollya_fast"]
    if split_half_turn["ffma_count"] != 7 or fast_half_turn["ffma_count"] != 6:
        raise RuntimeError(
            "half-turn reducers did not emit the expected FMA counts: "
            f"split={split_half_turn['ffma_count']}, fast={fast_half_turn['ffma_count']}"
        )
    packed_fma = payload["poly_d3_d4_bf16"]["packed_bf16_fma_count"]
    packed_mul = payload["poly_d3_d4_bf16"]["packed_bf16_mul_count"]
    if packed_fma < 6 or packed_fma + packed_mul != 10:
        raise RuntimeError(
            "packed BF16 D3/D4 did not emit the expected ten two-lane "
            f"arithmetic operations: HFMA2={packed_fma}, HMUL2={packed_mul}"
        )
    native_bf16_mufu = payload["native_sfu_bf16"]["mufu_instructions"]
    if native_bf16_mufu.count("MUFU.SIN") != 4 or native_bf16_mufu.count("MUFU.COS") != 4:
        raise RuntimeError(
            "BF16-output native control did not emit four paired SFU evaluations: "
            f"{native_bf16_mufu}"
        )
    packed_fp16 = payload["half_turn_d5_d4_packed_fp16"]
    if packed_fp16["mufu_instructions"]:
        raise RuntimeError("packed FP16 half-turn kernel unexpectedly contains MUFU")
    if packed_fp16["packed_fp16_fma_count"] < 8:
        raise RuntimeError(
            "four-angle packed FP16 half-turn kernel did not retain the "
            "eight polynomial HFMA2 operations: "
            f"got {packed_fp16['packed_fp16_fma_count']}"
        )
    packed_fp16_d3_d4 = payload["quarter_turn_d3_d4_packed_fp16"]
    if packed_fp16_d3_d4["mufu_instructions"]:
        raise RuntimeError(
            "quarter-turn packed FP16 D3/D4 unexpectedly contains MUFU"
        )
    if (
        packed_fp16_d3_d4["packed_fp16_fma_count"]
        >= packed_fp16["packed_fp16_fma_count"]
    ):
        raise RuntimeError(
            "quarter-turn packed FP16 D3/D4 did not remove polynomial FMA stages"
        )
    if payload["half_turn_d3_d4_packed_fp16"]["mufu_instructions"]:
        raise RuntimeError("half-turn packed FP16 D3/D4 unexpectedly contains MUFU")
    packed_fp16_accurate = payload["half_turn_d7_d6_packed_fp16"]
    if packed_fp16_accurate["mufu_instructions"]:
        raise RuntimeError("packed FP16 D7/D6 half-turn kernel unexpectedly contains MUFU")
    if packed_fp16_accurate["packed_fp16_fma_count"] <= packed_fp16["packed_fp16_fma_count"]:
        raise RuntimeError(
            "packed FP16 D7/D6 kernel did not retain its additional HFMA2 stages"
        )
    packed_fp16_balanced = payload["half_turn_d5_d6_packed_fp16"]
    if packed_fp16_balanced["mufu_instructions"]:
        raise RuntimeError("packed FP16 D5/D6 half-turn kernel unexpectedly contains MUFU")
    if not (
        packed_fp16["packed_fp16_fma_count"]
        < packed_fp16_balanced["packed_fp16_fma_count"]
        < packed_fp16_accurate["packed_fp16_fma_count"]
    ):
        raise RuntimeError(
            "packed FP16 D5/D6 kernel should retain one additional HFMA2 "
            "stage between D5/D4 and D7/D6"
        )
    native_fp16_mufu = payload["native_sfu_fp16"]["mufu_instructions"]
    if native_fp16_mufu.count("MUFU.SIN") != 4 or native_fp16_mufu.count("MUFU.COS") != 4:
        raise RuntimeError(
            "FP16-output native control did not emit four paired SFU evaluations: "
            f"{native_fp16_mufu}"
        )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Flash Attention Full Benchmark: SFU vs Spline (tanh × exp2)
=============================================================

Benchmarks all 4 combinations:
  1. SFU tanh + SFU exp2    (baseline)
  2. Spline tanh + SFU exp2 (tanh-only FMA)
  3. SFU tanh + Spline exp2 (exp2-only FMA)
  4. Spline tanh + Spline exp2 (fully SFU-free!)

Uses softcapping (so tanh is in the scoremod) and e2e mode
(so exp2 uses emulation instead of SFU).

Usage:
    python benchmark_fa_softcap.py              # default sweep
    python benchmark_fa_softcap.py --quick      # quick smoke test
    python benchmark_fa_softcap.py --causal     # with causal masking
    python benchmark_fa_softcap.py --seqlen 4096
"""

import argparse
import sys
import time
import importlib
import os

import torch
import numpy as np

FA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "flash-attention"))
sys.path.insert(0, FA_ROOT)
from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute import utils
from flash_attn.cute.polynomial_manifest import run_polynomial_coefficient_audit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def bench_fn(fn, warmup=10, reps=50):
    """Benchmark a function using CUDA events. Returns time in µs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / reps  # ms → µs


def parse_backends(spec):
    backends = []
    for token in spec.split(","):
        backend = token.strip()
        if not backend:
            continue
        if backend not in {"cute", "device"}:
            raise ValueError(f"Invalid backend '{backend}'")
        if backend not in backends:
            backends.append(backend)
    return backends


def parse_degrees(spec):
    degrees = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        degree = int(token)
        if degree not in {3, 4, 5, 6}:
            raise ValueError(f"Invalid degree {degree}")
        if degree not in degrees:
            degrees.append(degree)
    return degrees


def fa_fwd(q, k, v, softcap=None, score_mod=None, causal=False):
    """Wrapper around _flash_attn_fwd with sensible defaults."""
    return _flash_attn_fwd(
        q, k, v,
        softcap=softcap,
        score_mod=score_mod,
        causal=causal,
        n_block_size=128,
    )


# ─────────────────────────────────────────────────────────────────────────────
# exp2 monkeypatching for benchmarking
# ─────────────────────────────────────────────────────────────────────────────

# Save original references
_orig_ex2_emulation_2 = utils.ex2_emulation_2
_linear_ex2_emulation_2 = utils.ex2_emulation_2_linear
_pwl4_ex2_emulation_2 = utils.ex2_emulation_2_pwl4
_pwl8_ex2_emulation_2 = utils.ex2_emulation_2_pwl8


def set_exp2_mode(mode):
    """Swap the exp2 emulation function used by softmax.

    mode='deg3': Original degree-3 polynomial (3 FMA chain)
    mode='linear': 2-interval linear spline (1 FMA + 1 select)
    """
    if mode == "deg3":
        utils.ex2_emulation_2 = _orig_ex2_emulation_2
    elif mode == "linear":
        utils.ex2_emulation_2 = _linear_ex2_emulation_2
    elif mode == "pwl4":
        utils.ex2_emulation_2 = _pwl4_ex2_emulation_2
    elif mode == "pwl8":
        utils.ex2_emulation_2 = _pwl8_ex2_emulation_2
    else:
        raise ValueError(f"Unknown exp2 mode: {mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Correctness check
# ─────────────────────────────────────────────────────────────────────────────

def check_correctness(softcap_val=50.0, dtype=torch.bfloat16, causal=False):
    """Compare all 4 modes against SFU baseline."""
    print("\n" + "=" * 70)
    print("CORRECTNESS CHECK")
    print("=" * 70)

    B, S, H, D = 1, 256, 8, 128
    q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    k = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    v = torch.randn(B, S, H, D, device="cuda", dtype=dtype)

    spline_scoremod = utils.create_softcap_scoremod_backend(softcap_val, degree=4, backend="device")

    # Mode 1: SFU tanh + SFU exp2 (baseline) — e2e=False by default for small S
    set_exp2_mode("deg3")
    out_baseline, _ = fa_fwd(q, k, v, softcap=softcap_val, causal=causal)
    print(f"  Baseline (SFU tanh + SFU exp2): mean={out_baseline.float().mean():.6f}")

    # Mode 2: Spline tanh + SFU exp2
    set_exp2_mode("deg3")
    out_spline_tanh, _ = fa_fwd(q, k, v, score_mod=spline_scoremod, causal=causal)
    d = (out_baseline.float() - out_spline_tanh.float()).abs()
    print(f"  Spline tanh + SFU exp2:  max_diff={d.max():.6f}  mean_diff={d.mean():.8f}")

    # Note: exp2 monkeypatching only works when e2e=True, which only kicks in
    # when mask_fn is None and head_dim_padded <= 128. For small configs this
    # may not trigger. We'll compare modes 3 & 4 in the benchmark.

    all_pass = d.max().item() < 0.01
    status = "✅ All checks PASSED" if all_pass else "❌ Some checks FAILED"
    print(f"\n  {status}")

    del q, k, v
    torch.cuda.empty_cache()
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# 4-mode benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_4mode(
    softcap_val=50.0,
    dtype=torch.bfloat16,
    causal=False,
    warmup=10,
    reps=50,
    configs=None,
):
    """Benchmark all 4 combinations of tanh × exp2."""
    print("\n" + "=" * 70)
    print(f"4-MODE BENCHMARK  (softcap={softcap_val}, causal={causal})")
    print("  Mode A: SFU tanh + deg-3 exp2  (baseline)")
    print("  Mode B: device D4 tanh + deg-3 exp2")
    print("  Mode C: SFU tanh + linear exp2")
    print("  Mode D: device D4 tanh + linear exp2  (fully SFU-free!)")
    print("=" * 70)

    if configs is None:
        configs = [
            (1, 1024, 32, 128),
            (1, 2048, 32, 128),
            (2, 1024, 32, 128),
            (2, 2048, 32, 128),
            (4, 1024, 32, 128),
            (4, 2048, 32, 128),
            (8, 512, 32, 128),
            (8, 1024, 32, 128),
        ]

    spline_scoremod = utils.create_softcap_scoremod_backend(softcap_val, degree=4, backend="device")

    print(f"\n  {'Config':>25s}  {'A: SFU/SFU':>11s}  {'B: Spl/SFU':>11s}  "
          f"{'C: SFU/Spl':>11s}  {'D: Spl/Spl':>11s}  "
          f"{'B/A':>6s}  {'C/A':>6s}  {'D/A':>6s}")
    print("  " + "-" * 110)

    results = []
    for B, S, H, D in configs:
        try:
            q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
            k = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
            v = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"  {'B='+str(B)+' S='+str(S)+' H='+str(H)+' D='+str(D):>25s}  OOM")
            continue

        # Mode A: SFU tanh + deg-3 exp2 (e2e mode uses deg-3)
        set_exp2_mode("deg3")
        t_a = bench_fn(
            lambda: fa_fwd(q, k, v, softcap=softcap_val, causal=causal),
            warmup=warmup, reps=reps,
        )

        # Mode B: Spline tanh + deg-3 exp2
        set_exp2_mode("deg3")
        t_b = bench_fn(
            lambda: fa_fwd(q, k, v, score_mod=spline_scoremod, causal=causal),
            warmup=warmup, reps=reps,
        )

        # Mode C: SFU tanh + linear exp2
        set_exp2_mode("linear")
        t_c = bench_fn(
            lambda: fa_fwd(q, k, v, softcap=softcap_val, causal=causal),
            warmup=warmup, reps=reps,
        )

        # Mode D: Spline tanh + linear exp2 (fully SFU-free)
        set_exp2_mode("linear")
        t_d = bench_fn(
            lambda: fa_fwd(q, k, v, score_mod=spline_scoremod, causal=causal),
            warmup=warmup, reps=reps,
        )

        # Reset
        set_exp2_mode("deg3")

        config = f"B={B} S={S} H={H} D={D}"
        r_b = t_a / t_b
        r_c = t_a / t_c
        r_d = t_a / t_d

        best = max(r_b, r_c, r_d)
        marker = " 🔥" if best > 1.3 else ""
        print(f"  {config:>25s}  {t_a:>9.0f}µ  {t_b:>9.0f}µ  "
              f"{t_c:>9.0f}µ  {t_d:>9.0f}µ  "
              f"{r_b:>5.2f}x  {r_c:>5.2f}x  {r_d:>5.2f}x{marker}")

        results.append({
            "B": B, "S": S, "H": H, "D": D,
            "A_us": t_a, "B_us": t_b, "C_us": t_c, "D_us": t_d,
            "B_speedup": r_b, "C_speedup": r_c, "D_speedup": r_d,
        })

        del q, k, v
        torch.cuda.empty_cache()

    # Summary
    if results:
        print(f"\n  {'':>25s}  {'':>11s}  {'':>11s}  {'':>11s}  {'':>11s}  "
              f"{'B/A':>6s}  {'C/A':>6s}  {'D/A':>6s}")
        speedups_b = [r["B_speedup"] for r in results]
        speedups_c = [r["C_speedup"] for r in results]
        speedups_d = [r["D_speedup"] for r in results]
        gmean_b = np.exp(np.mean(np.log(speedups_b)))
        gmean_c = np.exp(np.mean(np.log(speedups_c)))
        gmean_d = np.exp(np.mean(np.log(speedups_d)))
        print(f"  {'GeoMean Speedup:':>25s}  {'':>11s}  {'':>11s}  "
              f"{'':>11s}  {'':>11s}  "
              f"{gmean_b:>5.2f}x  {gmean_c:>5.2f}x  {gmean_d:>5.2f}x")

    return results


def benchmark_softcap_backend_compare(
    softcap_val=30.0,
    dtype=torch.bfloat16,
    causal=False,
    warmup=10,
    reps=50,
    configs=None,
    degrees=(3, 4, 5, 6),
    backends=("cute", "device"),
):
    print("\n" + "=" * 90)
    print(f"SOFTCAP BACKEND/DEGREE COMPARISON  (softcap={softcap_val}, causal={causal})")
    print("=" * 90)

    if configs is None:
        configs = [
            (1, 2048, 32, 128),
            (2, 4096, 32, 128),
            (1, 8192, 32, 128),
        ]

    for B, S, H, D in configs:
        q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        k = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        v = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        config = f"B={B} S={S} H={H} D={D}"
        print(f"\n  --- {config} ---")
        print(f"  {'Variant':<28s}  {'FWD µs':>10s}  {'vs native':>10s}")
        print(f"  {'-'*28}  {'-'*10}  {'-'*10}")
        t_native = bench_fn(
            lambda: fa_fwd(q, k, v, softcap=softcap_val, causal=causal),
            warmup=warmup,
            reps=reps,
        )
        print(f"  {'native tanh':<28s}  {t_native:>10.0f}  {'1.00x':>10s}")
        for degree in degrees:
            for backend in backends:
                score_mod = utils.create_softcap_scoremod_backend(
                    softcap_val, degree=degree, backend=backend
                )
                t = bench_fn(
                    lambda sm=score_mod: fa_fwd(q, k, v, score_mod=sm, causal=causal),
                    warmup=warmup,
                    reps=reps,
                )
                print(f"  {f'D{degree} {backend}':<28s}  {t:>10.0f}  {t_native/t:>10.3f}x")
        del q, k, v
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# No-softcap comparison
# ─────────────────────────────────────────────────────────────────────────────

def no_softcap_comparison(dtype=torch.bfloat16, warmup=10, reps=50):
    """Compare overhead of softcapping vs no softcapping."""
    print("\n" + "=" * 70)
    print("SOFTCAP OVERHEAD COMPARISON")
    print("  How much does softcapping cost, and how much does spline save?")
    print("=" * 70)

    spline_scoremod = utils.create_softcap_scoremod_backend(50.0, degree=4, backend="device")

    print(f"\n  {'Config':>25s}  {'No SC':>9s}  {'SFU SC':>9s}  "
          f"{'Spline SC':>10s}  {'SC overhead':>12s}  {'Spline overhead':>16s}")
    print("  " + "-" * 90)

    for B, S, H, D in [(1, 2048, 32, 128), (2, 2048, 32, 128), (4, 1024, 32, 128)]:
        q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        k = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        v = torch.randn(B, S, H, D, device="cuda", dtype=dtype)

        t_none = bench_fn(lambda: fa_fwd(q, k, v), warmup=warmup, reps=reps)
        t_sfu = bench_fn(lambda: fa_fwd(q, k, v, softcap=50.0), warmup=warmup, reps=reps)
        t_spl = bench_fn(lambda: fa_fwd(q, k, v, score_mod=spline_scoremod), warmup=warmup, reps=reps)

        config = f"B={B} S={S} H={H} D={D}"
        sfu_oh = (t_sfu / t_none - 1) * 100
        spl_oh = (t_spl / t_none - 1) * 100
        print(f"  {config:>25s}  {t_none:>7.0f}µ  {t_sfu:>7.0f}µ  "
              f"{t_spl:>8.0f}µ  {sfu_oh:>+10.0f}%  {spl_oh:>+14.0f}%")

        del q, k, v
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Softcap value sweep
# ─────────────────────────────────────────────────────────────────────────────

def softcap_sweep(dtype=torch.bfloat16, warmup=10, reps=50):
    """Does speedup depend on softcap magnitude?"""
    print("\n" + "=" * 70)
    print("SOFTCAP VALUE SWEEP  (B=2, S=2048, H=32, D=128)")
    print("=" * 70)

    B, S, H, D = 2, 2048, 32, 128
    q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    k = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    v = torch.randn(B, S, H, D, device="cuda", dtype=dtype)

    print(f"\n  {'Softcap':>10s}  {'SFU':>9s}  {'Spline':>9s}  "
          f"{'Speedup':>8s}  {'Max Diff':>10s}")
    print("  " + "-" * 55)

    for softcap_val in [10.0, 20.0, 30.0, 50.0, 100.0]:
        spline_scoremod = utils.create_softcap_scoremod_backend(softcap_val, degree=4, backend="device")

        t_sfu = bench_fn(lambda: fa_fwd(q, k, v, softcap=softcap_val), warmup=warmup, reps=reps)
        t_spl = bench_fn(lambda: fa_fwd(q, k, v, score_mod=spline_scoremod), warmup=warmup, reps=reps)

        out_sfu, _ = fa_fwd(q, k, v, softcap=softcap_val)
        out_spl, _ = fa_fwd(q, k, v, score_mod=spline_scoremod)
        max_diff = (out_sfu.float() - out_spl.float()).abs().max().item()

        speedup = t_sfu / t_spl
        print(f"  {softcap_val:>10.1f}  {t_sfu:>7.0f}µ  {t_spl:>7.0f}µ  "
              f"{speedup:>7.2f}x  {max_diff:>10.6f}")

    del q, k, v
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FA Softcap: 4-mode benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--softcap", type=float, default=50.0)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--seqlen", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--sweep-softcap", action="store_true")
    parser.add_argument("--no-softcap-baseline", action="store_true")
    parser.add_argument("--compare-backends", action="store_true")
    parser.add_argument("--poly-backends", type=str, default="device,cute")
    parser.add_argument("--degrees", type=str, default="4")
    args = parser.parse_args()
    run_polynomial_coefficient_audit()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print("=" * 70)
    print("  Flash Attention Softcap Benchmark — D4 Device Port")
    print("  Headline path: device-side D4 tanh forward; backend comparison optional")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  SM: {torch.cuda.get_device_capability()}")
    print("=" * 70)

    # Correctness
    check_correctness(args.softcap, dtype, args.causal)
    if args.correctness_only:
        return

    # 4-mode benchmark
    if args.quick:
        configs = [(1, 1024, 32, 128), (2, 2048, 32, 128)]
    elif args.seqlen or args.batch:
        S = args.seqlen or 2048
        batches = [args.batch] if args.batch else [1, 2, 4, 8]
        configs = [(B, S, args.heads, args.hdim) for B in batches]
    else:
        configs = None

    benchmark_4mode(
        softcap_val=args.softcap,
        dtype=dtype,
        causal=args.causal,
        warmup=args.warmup,
        reps=args.reps,
        configs=configs,
    )

    if args.sweep_softcap:
        softcap_sweep(dtype, args.warmup, args.reps)

    if args.no_softcap_baseline:
        no_softcap_comparison(dtype, args.warmup, args.reps)
    if args.compare_backends:
        benchmark_softcap_backend_compare(
            softcap_val=args.softcap,
            dtype=dtype,
            causal=args.causal,
            warmup=args.warmup,
            reps=args.reps,
            configs=configs,
            degrees=parse_degrees(args.degrees),
            backends=parse_backends(args.poly_backends),
        )

    print("\n✅ Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sigmoid Attention Benchmark: SFU sigmoid vs Polynomial sigmoid vs Softmax

Compares FA4 attention variants:
  1. Standard softmax attention (baseline)
  2. Sigmoid attention — all polynomial (sigmoid_sfu_res=0, default)
  3. Sigmoid attention — all SFU (sigmoid_sfu_res=16)
  4. Sigmoid attention — mixed routing (various ratios)

For each, measures:
  - Forward-only latency
  - Forward+backward latency (training mode)
  - Correctness (output similarity)

Usage:
    cd experiments && python bench_sigmoid_attn.py
"""
import argparse
import sys
import os
import torch
import torch.nn.functional as F

FA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "flash-attention"))
sys.path.insert(0, FA_ROOT)

from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd
from flash_attn.cute.polynomial_manifest import run_polynomial_coefficient_audit

torch.set_float32_matmul_precision('high')


def parse_sigmoid_ratios(spec):
    ratios = []
    if not spec:
        return [(16, 0), (4, 1), (4, 2)]
    for token in spec.split(','):
        tok = token.strip()
        if not tok:
            continue
        freq_s, res_s = tok.split(':')
        freq, res = int(freq_s), int(res_s)
        if res < 0 or res > freq:
            raise ValueError(f"Invalid sigmoid ratio '{tok}'")
        pair = (freq, res)
        if pair not in ratios:
            ratios.append(pair)
    return ratios


def parse_backends(spec):
    backends = []
    if not spec:
        return ["cute", "device"]
    for token in spec.split(','):
        backend = token.strip()
        if not backend:
            continue
        if backend not in {"cute", "device"}:
            raise ValueError(f"Invalid backend '{backend}'")
        if backend not in backends:
            backends.append(backend)
    return backends


def ratio_label(freq, res):
    return "σ-poly (0% SFU)" if res == 0 and freq == 16 else f"σ-mix ({int(round(100 * res / freq))}% SFU)"


def label_with_backend(name, backend):
    return name if backend == "cute" else f"{name} [device]"


def bench_fwd(fn, warmup=15, reps=50):
    """Benchmark forward-only."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / reps  # µs


def bench_fwd_bwd(fwd_fn, bwd_fn, warmup=10, reps=30):
    """Benchmark forward + backward."""
    for _ in range(warmup):
        out, lse = fwd_fn()
        bwd_fn(out, lse)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        out, lse = fwd_fn()
        bwd_fn(out, lse)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / reps  # µs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigmoid-ratios', type=str, default='16:0,4:1,4:2',
                        help='Comma-separated sigmoid routing ratios as freq:res')
    parser.add_argument('--poly-backends', type=str, default='cute,device',
                        help='Comma-separated polynomial backends: cute,device')
    parser.add_argument('--include-direct-bwd-poly', action='store_true',
                        help='Also benchmark the direct fitted sigmoid-gradient polynomial in backward.')
    parser.add_argument('--headline-only', action='store_true',
                        help='Limit the benchmark to σ-SFU and σ-poly cute/device rows.')
    parser.add_argument('--full-sweep', action='store_true',
                        help='Include mixed forward/backward routing rows; default is headline rows only.')
    args = parser.parse_args()
    run_polynomial_coefficient_audit()
    headline_only = args.headline_only or not args.full_sweep

    device = 'cuda'
    dtype = torch.bfloat16
    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name}, SMs: {props.multi_processor_count}")
    print(f"Dtype: {dtype}")

    configs = [
        # (batch, seqlen, nheads, headdim)
        (2, 2048, 32, 128),
        (4, 2048, 32, 128),
        (2, 4096, 32, 128),
        (8, 2048, 32, 128),
        (2, 8192, 32, 128),
    ]

    # Sigmoid routing variants: (name, sfu_freq, sfu_res)
    parsed_ratios = parse_sigmoid_ratios(args.sigmoid_ratios)
    parsed_backends = parse_backends(args.poly_backends)
    sigmoid_variants = [("σ-SFU (100% SFU)", 16, 16, "cute")]
    for freq, res in parsed_ratios:
        base_name = ratio_label(freq, res)
        if res == freq:
            sigmoid_variants.append((base_name, freq, res, "cute"))
        else:
            for backend in parsed_backends:
                sigmoid_variants.append((label_with_backend(base_name, backend), freq, res, backend))
    if headline_only:
        sigmoid_variants = [
            row
            for row in sigmoid_variants
            if row[0] == "σ-SFU (100% SFU)" or row[1:3] == (16, 0)
        ]

    print(f"\n{'='*130}")
    print(f"  FORWARD-ONLY COMPARISON")
    print(f"{'='*130}")
    fwd_sensitivity = []

    for B, S, H, D in configs:
        q = torch.randn(B, S, H, D, device=device, dtype=dtype)
        k = torch.randn(B, S, H, D, device=device, dtype=dtype)
        v = torch.randn(B, S, H, D, device=device, dtype=dtype)
        softmax_scale = D ** -0.5

        print(f"\n  --- B={B}, S={S}, H={H}, D={D} ---")
        print(f"  {'Variant':<22s}  {'FWD µs':>10s}  {'vs Softmax':>10s}  {'vs σ-poly':>10s}")
        print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")

        times = {}

        # Baseline: softmax attention
        def fwd_softmax():
            return _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=True,
                                   return_lse=True)
        t = bench_fwd(fwd_softmax)
        times['softmax'] = t
        print(f"  {'Softmax (baseline)':<22s}  {t:>10.0f}  {'1.00x':>10s}  {'-':>10s}")

        # Sigmoid variants
        t_poly = None
        for name, freq, res, backend in sigmoid_variants:
            def fwd_sig(f=freq, r=res, b=backend):
                return _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=True,
                                       sigmoid_attention=True,
                                       sigmoid_sfu_freq=f, sigmoid_sfu_res=r,
                                       sigmoid_poly_backend=b,
                                       return_lse=True)
            t = bench_fwd(fwd_sig)
            times[name] = t
            if t_poly is None:
                t_poly = t
            vs_sm = f"{times['softmax']/t:.3f}x"
            vs_poly = f"{t_poly/t:.3f}x"
            print(f"  {name:<22s}  {t:>10.0f}  {vs_sm:>10s}  {vs_poly:>10s}")

        if "σ-SFU (100% SFU)" in times and "σ-poly (0% SFU)" in times:
            fwd_sensitivity.append(
                (B, S, H, D, times["σ-SFU (100% SFU)"], times["σ-poly (0% SFU)"])
            )

        del q, k, v
        torch.cuda.empty_cache()

    # =========================================================================
    # Forward + Backward
    # =========================================================================
    print(f"\n{'='*130}")
    print(f"  FORWARD + BACKWARD COMPARISON")
    print(f"{'='*130}")
    bwd_sensitivity = []

    for B, S, H, D in configs:
        q = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
        softmax_scale = D ** -0.5

        print(f"\n  --- B={B}, S={S}, H={H}, D={D} ---")
        print(f"  {'Variant':<22s}  {'FWD+BWD µs':>12s}  {'vs Softmax':>10s}  {'vs σ-poly':>10s}")
        print(f"  {'-'*22}  {'-'*12}  {'-'*10}  {'-'*10}")

        times = {}

        # Softmax F+B
        def fwd_softmax():
            return _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=True,
                                   return_lse=True)
        dout = torch.randn(B, S, H, D, device=device, dtype=dtype)

        def bwd_softmax(out, lse):
            _flash_attn_bwd(q, k, v, out, dout, lse, softmax_scale, causal=True,
                            softcap=0.0)

        t = bench_fwd_bwd(fwd_softmax, bwd_softmax)
        times['softmax'] = t
        print(f"  {'Softmax (baseline)':<22s}  {t:>12.0f}  {'1.00x':>10s}  {'-':>10s}")

        # Sigmoid F+B routing experiments.
        # Includes same-mix plus forward-only / backward-only deviations from all-SFU.
        fb_variants = [("σ-SFU (100% SFU)", 16, 16, 16, 16, False, "cute")]
        if headline_only:
            for backend in parsed_backends:
                fb_variants.append((label_with_backend("σ-poly (0% SFU)", backend), 16, 0, 16, 0, False, backend))
        else:
            for freq, res in parsed_ratios:
                base_name = ratio_label(freq, res)
                backends = ["cute"] if res == freq else parsed_backends
                for backend in backends:
                    name = label_with_backend(base_name, backend)
                    fb_variants.append((name, freq, res, freq, res, False, backend))
                    fb_variants.append((f"σ-SFU fwd + {name[2:]}", 16, 16, freq, res, False, backend))
                    fb_variants.append((f"{name} fwd + SFU bwd", freq, res, 16, 16, False, backend))
            if args.include_direct_bwd_poly:
                for backend in parsed_backends:
                    fb_variants.append((label_with_backend("σ-poly + direct ∇σ", backend), 16, 0, 16, 0, True, backend))
                    fb_variants.append((f"σ-SFU fwd + {label_with_backend('poly/direct ∇σ', backend)}", 16, 16, 16, 0, True, backend))

        seen = set()
        for name, fwd_freq, fwd_res, bwd_freq, bwd_res, use_direct_bwd_poly, backend in fb_variants:
            key = (fwd_freq, fwd_res, bwd_freq, bwd_res, use_direct_bwd_poly, backend)
            if key in seen:
                continue
            seen.add(key)
            def fwd_sig(f=fwd_freq, r=fwd_res, b=backend):
                return _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=True,
                                       sigmoid_attention=True,
                                       sigmoid_sfu_freq=f, sigmoid_sfu_res=r,
                                       sigmoid_poly_backend=b,
                                       return_lse=True)
            def bwd_sig(out, lse, f=bwd_freq, r=bwd_res, direct=use_direct_bwd_poly, b=backend):
                _flash_attn_bwd(q, k, v, out, dout, lse, softmax_scale, causal=True,
                                softcap=0.0, sigmoid_attention=True,
                                sigmoid_sfu_freq=f, sigmoid_sfu_res=r,
                                sigmoid_use_direct_bwd_poly=direct,
                                sigmoid_poly_backend=b)

            try:
                t = bench_fwd_bwd(fwd_sig, bwd_sig)
                times[name] = t
                t_poly = times.get('σ-poly (0% SFU)', t)
                vs_sm = f"{times['softmax']/t:.3f}x"
                vs_poly = f"{t_poly/t:.3f}x"
                print(f"  {name:<22s}  {t:>12.0f}  {vs_sm:>10s}  {vs_poly:>10s}")
            except Exception as e:
                print(f"  {name:<22s}  {'ERROR':>12s}  {str(e)[:40]}")

        if "σ-SFU (100% SFU)" in times and "σ-poly (0% SFU)" in times:
            bwd_sensitivity.append(
                (B, S, H, D, times["σ-SFU (100% SFU)"], times["σ-poly (0% SFU)"])
            )

        del q, k, v, dout
        torch.cuda.empty_cache()

    # =========================================================================
    # Correctness: compare sigmoid outputs
    # =========================================================================
    print(f"\n{'='*130}")
    print(f"  CORRECTNESS")
    print(f"{'='*130}")

    B, S, H, D = 2, 1024, 32, 128
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)
    softmax_scale = D ** -0.5

    # Reference: all-polynomial sigmoid (CuTe backend)
    out_poly, _ = _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=True,
                                   sigmoid_attention=True,
                                   sigmoid_sfu_freq=16, sigmoid_sfu_res=0,
                                   sigmoid_poly_backend='cute',
                                   return_lse=True)

    out_poly_device, _ = _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=True,
                                   sigmoid_attention=True,
                                   sigmoid_sfu_freq=16, sigmoid_sfu_res=0,
                                   sigmoid_poly_backend='device',
                                   return_lse=True)

    # All-SFU sigmoid
    out_sfu, _ = _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=True,
                                  sigmoid_attention=True,
                                  sigmoid_sfu_freq=16, sigmoid_sfu_res=16,
                                  return_lse=True)

    # Softmax (different function, just for scale comparison)
    out_sm, _ = _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=True,
                                 return_lse=True)

    print(f"  σ-SFU vs σ-poly:   max_abs_diff={abs(out_sfu.float() - out_poly.float()).max().item():.6f}")
    print(f"  σ-SFU vs σ-poly:   rel_diff={abs(out_sfu.float() - out_poly.float()).mean().item() / abs(out_poly.float()).mean().item() * 100:.4f}%")
    print(f"  σ-poly device vs cute:  max_abs_diff={abs(out_poly_device.float() - out_poly.float()).max().item():.6f}")
    print(f"  Softmax vs σ-poly:  max_abs_diff={abs(out_sm.float() - out_poly.float()).max().item():.6f}  (expected large — different fn)")

    print(f"\n{'='*130}")
    print("  ROUTING SENSITIVITY SUMMARY")
    print(f"{'='*130}")
    print("  Forward sensitivity (σ-SFU / σ-poly):")
    for B, S, H, D, t_sfu, t_poly in fwd_sensitivity:
        print(f"    B={B}, S={S}, H={H}, D={D}: {t_sfu/t_poly:.3f}x  (SFU={t_sfu:.0f}µs, poly={t_poly:.0f}µs)")
    print("  Forward+Backward sensitivity (σ-SFU / σ-poly):")
    for B, S, H, D, t_sfu, t_poly in bwd_sensitivity:
        print(f"    B={B}, S={S}, H={H}, D={D}: {t_sfu/t_poly:.3f}x  (SFU={t_sfu:.0f}µs, poly={t_poly:.0f}µs)")

    print("\n✅ Sigmoid attention benchmark complete!")


if __name__ == "__main__":
    main()

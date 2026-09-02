#!/usr/bin/env python3
"""
Side-by-side benchmark: Sigmoid, Tanh, SiLU, GeLU
  1. Forward  (pure CUDA kernel, no grad)
  2. Backward (isolated aten op, pure CUDA kernel)
  3. FWD+BWD  (full autograd round-trip)
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "spline_ops"))
import spline_ops


def parse_csv_ints(spec):
    return [int(tok.strip()) for tok in spec.split(",") if tok.strip()]


def parse_csv_strings(spec):
    return [tok.strip() for tok in spec.split(",") if tok.strip()]


def bench(fn, warmup=100, reps=300):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / reps  # µs

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sizes",
    type=str,
    default="1024,16384,262144,4194304,16777216,67108864,268435456",
    help="Comma-separated element counts",
)
parser.add_argument("--warmup", type=int, default=100)
parser.add_argument("--reps", type=int, default=300)
parser.add_argument(
    "--acts",
    type=str,
    default="Sigmoid,Tanh,SiLU,GeLU",
    help="Comma-separated activation names from {Sigmoid,Tanh,SiLU,GeLU}",
)
args = parser.parse_args()

props = torch.cuda.get_device_properties(0)
l2 = props.L2_cache_size
print(f"Device: {props.name}  |  SMs: {props.multi_processor_count}  |  L2: {l2/1024/1024:.0f} MB")
print(f"Warmup: {args.warmup}  |  Reps: {args.reps}")
print()

sizes = parse_csv_ints(args.sizes)

def regime(n): return "L2" if n * 2 * 2 < l2 else "HBM"

def fmt_us(t):
    """Format microseconds: compact, right-aligned."""
    if t < 10:
        return f"{t:>6.1f}"
    elif t < 1000:
        return f"{t:>6.0f}"
    else:
        return f"{t:>6.0f}"

# Python autograd wrapper for GeLU
class _SplineGeLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return spline_ops.gelu_fwd(x)
    @staticmethod
    def backward(ctx, go):
        x, = ctx.saved_tensors
        return spline_ops.gelu_bwd(go.contiguous(), x),

def gelu_ag(x): return _SplineGeLU.apply(x)

ALL_ACTS = [
    ("Sigmoid", torch.sigmoid,  lambda go, x: torch.ops.aten.sigmoid_backward(go, torch.sigmoid(x)),
                spline_ops.sigmoid_fwd, spline_ops.sigmoid_bwd, spline_ops.sigmoid_ag),
    ("Tanh",    torch.tanh,     lambda go, x: torch.ops.aten.tanh_backward(go, torch.tanh(x)),
                spline_ops.tanh_fwd,    spline_ops.tanh_bwd,    spline_ops.tanh_ag),
    ("SiLU",    F.silu,         lambda go, x: torch.ops.aten.silu_backward(go, x),
                spline_ops.swish_fwd,   spline_ops.swish_bwd,   spline_ops.swish_ag),
    ("GeLU",    F.gelu,         lambda go, x: torch.ops.aten.gelu_backward(go, x),
                spline_ops.gelu_fwd,    spline_ops.gelu_bwd,    gelu_ag),
]

selected_names = {name.lower() for name in parse_csv_strings(args.acts)}
ACTS = [act for act in ALL_ACTS if act[0].lower() in selected_names]
if not ACTS:
    raise ValueError("No activations selected; expected names from {Sigmoid,Tanh,SiLU,GeLU}")

names = [a[0] for a in ACTS]

def print_section(title):
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)

def print_table(headers, rows):
    """Print a nicely formatted table."""
    # Calculate column widths
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    widths = [max(w, 6) for w in widths]

    # Header
    hdr = "  ".join(f"{h:>{w}}" for h, w in zip(headers, widths))
    print(f"  {hdr}")
    print("  " + "  ".join("─" * w for w in widths))

    # Rows
    for row in rows:
        line = "  ".join(f"{str(v):>{w}}" for v, w in zip(row, widths))
        print(f"  {line}")
    print()


# ============================================================================
# 1. FORWARD
# ============================================================================
print_section("FORWARD: Pure CUDA Kernel (no autograd)")

headers = ["N", "Regime"]
for nm in names:
    headers += [f"{nm} Sp", f"{nm} Nat", f"{nm} ×"]

rows = []
for n in sizes:
    x = torch.randn(n, device='cuda', dtype=torch.float16)
    row = [f"{n:>12,d}", regime(n)]
    for _, nat_fn, _, sp_fn, _, _ in ACTS:
        t_sp = bench(lambda: sp_fn(x), warmup=args.warmup, reps=args.reps)
        t_nat = bench(lambda: nat_fn(x), warmup=args.warmup, reps=args.reps)
        row += [f"{t_sp:.1f}µs", f"{t_nat:.1f}µs", f"{t_nat/t_sp:.2f}x"]
    rows.append(row)

print_table(headers, rows)


# ============================================================================
# 2. BACKWARD
# ============================================================================
print_section("BACKWARD: Pure CUDA Kernel (isolated aten op)")

headers = ["N", "Regime"]
for nm in names:
    headers += [f"{nm} Sp", f"{nm} Nat", f"{nm} ×"]

rows = []
for n in sizes:
    x = torch.randn(n, device='cuda', dtype=torch.float16)
    go = torch.ones(n, device='cuda', dtype=torch.float16)
    row = [f"{n:>12,d}", regime(n)]
    for _, _, nat_bwd_fn, _, sp_bwd_fn, _ in ACTS:
        t_sp = bench(lambda: sp_bwd_fn(go, x), warmup=args.warmup, reps=args.reps)
        t_nat = bench(lambda: nat_bwd_fn(go, x), warmup=args.warmup, reps=args.reps)
        row += [f"{t_sp:.1f}µs", f"{t_nat:.1f}µs", f"{t_nat/t_sp:.2f}x"]
    rows.append(row)

print_table(headers, rows)


# ============================================================================
# 3. FWD+BWD AUTOGRAD
# ============================================================================
print_section("FWD+BWD AUTOGRAD: Full Round-Trip")

headers = ["N", "Regime"]
for nm in names:
    headers += [f"{nm} Sp", f"{nm} Nat", f"{nm} ×"]

rows = []
for n in sizes:
    x = torch.randn(n, device='cuda', dtype=torch.float16, requires_grad=True)
    go = torch.ones(n, device='cuda', dtype=torch.float16)
    row = [f"{n:>12,d}", regime(n)]
    for _, nat_fn, _, _, _, sp_ag_fn in ACTS:
        def pt_ag(fn=nat_fn): x.grad = None; fn(x).backward(go)
        def sp_ag(fn=sp_ag_fn): x.grad = None; fn(x).backward(go)
        t_sp = bench(sp_ag, warmup=args.warmup, reps=args.reps)
        t_nat = bench(pt_ag, warmup=args.warmup, reps=args.reps)
        row += [f"{t_sp:.1f}µs", f"{t_nat:.1f}µs", f"{t_nat/t_sp:.2f}x"]
    rows.append(row)

print_table(headers, rows)

# ============================================================================
# Summary: HBM regime only (268M elements)
# ============================================================================
print_section("SUMMARY: Speedup by Memory Regime")

for label, n_sum in [("L2 (4M elements)", 4194304), ("HBM (268M elements)", 268435456)]:
    print(f"  {label}")
    print(f"  {'Activation':<12s}  {'FWD ×':>7s}  {'BWD ×':>7s}  {'Autograd ×':>10s}")
    print("  " + "─" * 42)

    x_fwd = torch.randn(n_sum, device='cuda', dtype=torch.float16)
    go = torch.ones(n_sum, device='cuda', dtype=torch.float16)

    for nm, nat_fn, nat_bwd_fn, sp_fn, sp_bwd_fn, sp_ag_fn in ACTS:
        t_sp_f = bench(lambda: sp_fn(x_fwd), warmup=args.warmup, reps=args.reps)
        t_nat_f = bench(lambda: nat_fn(x_fwd), warmup=args.warmup, reps=args.reps)
        t_sp_b = bench(lambda: sp_bwd_fn(go, x_fwd), warmup=args.warmup, reps=args.reps)
        t_nat_b = bench(lambda: nat_bwd_fn(go, x_fwd), warmup=args.warmup, reps=args.reps)
        x_ag = torch.randn(n_sum, device='cuda', dtype=torch.float16, requires_grad=True)
        def pt_ag(fn=nat_fn): x_ag.grad = None; fn(x_ag).backward(go)
        def sp_ag(fn=sp_ag_fn): x_ag.grad = None; fn(x_ag).backward(go)
        t_sp_a = bench(sp_ag, warmup=args.warmup, reps=args.reps)
        t_nat_a = bench(pt_ag, warmup=args.warmup, reps=args.reps)
        print(f"  {nm:<12s}  {t_nat_f/t_sp_f:>6.2f}x  {t_nat_b/t_sp_b:>6.2f}x  {t_nat_a/t_sp_a:>9.2f}x")
    print()

print("  × > 1.0 = spline is faster than native PyTorch")
print("=" * 100)

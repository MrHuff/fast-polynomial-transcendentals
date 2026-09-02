#!/usr/bin/env python3
"""
Plot BF16 accuracy of all spline activation functions across D3-D6 degrees.
Generates vector comparison plots for FWD and BWD sigmoid, tanh, and SiLU.

Usage:
    python plot_accuracy.py [--outdir DIR] [--format pdf|png]
"""
import argparse
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'spline_ops'))
import spline_ops


def compute_errors(spline_fn, ref_fn, x):
    """Compare a BF16 kernel output with one FP32 reference definition."""
    with torch.no_grad():
        y_spline = spline_fn(x)
        y_ref = ref_fn(x)
    torch.cuda.synchronize()
    err = (y_spline.float() - y_ref.float()).abs()
    return y_spline.float().cpu().numpy(), y_ref.float().cpu().numpy(), err.cpu().numpy()


def plot_function_degrees(func_name, direction, x_np, ref_np,
                          degree_data, outdir, output_format):
    """
    Plot a function across D3-D6 in a single figure with 2 rows:
    Row 1: function value overlaid per degree
    Row 2: abs error per degree
    """
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(4.25, 3.0),
        sharex=True,
        gridspec_kw={'height_ratios': [1.7, 1]},
    )

    ax1.plot(x_np, ref_np, 'k-', linewidth=2, label='Reference', alpha=0.4)

    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
    for i, (deg, y_spline, err) in enumerate(degree_data):
        ax1.plot(x_np, y_spline, color=colors[i], linewidth=1.2,
                 label=f'D{deg}', alpha=0.8)
        max_err = np.max(err)
        ax2.plot(x_np, err, color=colors[i], linewidth=1,
                 label=f'D{deg} (max={max_err:.4f})', alpha=0.8)

    ax1.set_ylabel('Value', fontsize=8.5)
    ax1.legend(loc='best', fontsize=7.5, ncol=2)
    ax1.tick_params(labelsize=7.5)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('BF16-representable input', fontsize=8.5)
    ax2.set_ylabel('Absolute error', fontsize=8.5)
    ax2.legend(loc='best', fontsize=7.5, ncol=2)
    ax2.tick_params(labelsize=7.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    fname = f'{func_name.lower()}_{direction.lower()}_accuracy.{output_format}'
    path = os.path.join(outdir, fname)
    save_args = {'bbox_inches': 'tight'}
    if output_format == 'png':
        save_args['dpi'] = 200
    else:
        save_args['metadata'] = {'CreationDate': None, 'ModDate': None}
    plt.savefig(path, **save_args)
    plt.close()
    print(f'Saved {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=os.path.dirname(__file__))
    parser.add_argument('--format', choices=('pdf', 'png'), default='pdf')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    N = 30000
    x = torch.linspace(-15, 15, N, device='cuda', dtype=torch.bfloat16)
    x_np = x.float().cpu().numpy()
    go = torch.ones_like(x)

    degrees = [3, 4, 5, 6]

    # ===== SIGMOID FWD =====
    ref_np = torch.sigmoid(x.float()).cpu().numpy()
    degree_data = []
    for d in degrees:
        fn = getattr(spline_ops, f'sigmoid_fwd_d{d}')
        y, _, err = compute_errors(fn, lambda t: torch.sigmoid(t.float()), x)
        degree_data.append((d, y, err))
    plot_function_degrees(
        'Sigmoid', 'FWD', x_np, ref_np, degree_data, args.outdir, args.format
    )

    # ===== SIGMOID BWD =====
    ref_bwd = torch.sigmoid(x.float()) * (1 - torch.sigmoid(x.float()))
    ref_np = ref_bwd.cpu().numpy()
    degree_data = []
    for d in degrees:
        fn = getattr(spline_ops, f'sigmoid_bwd_d{d}')
        gi = fn(go, x)
        torch.cuda.synchronize()
        gi_np = gi.float().cpu().numpy()
        err = np.abs(gi_np - ref_np)
        degree_data.append((d, gi_np, err))
    plot_function_degrees(
        'Sigmoid', 'BWD', x_np, ref_np, degree_data, args.outdir, args.format
    )

    # ===== TANH FWD =====
    ref_np = torch.tanh(x.float()).cpu().numpy()
    degree_data = []
    for d in degrees:
        fn = getattr(spline_ops, f'tanh_fwd_d{d}')
        y, _, err = compute_errors(fn, lambda t: torch.tanh(t.float()), x)
        degree_data.append((d, y, err))
    plot_function_degrees(
        'Tanh', 'FWD', x_np, ref_np, degree_data, args.outdir, args.format
    )

    # ===== TANH BWD =====
    ref_bwd = 1 - torch.tanh(x.float())**2
    ref_np = ref_bwd.cpu().numpy()
    degree_data = []
    for d in degrees:
        fn = getattr(spline_ops, f'tanh_bwd_d{d}')
        gi = fn(go, x)
        torch.cuda.synchronize()
        gi_np = gi.float().cpu().numpy()
        err = np.abs(gi_np - ref_np)
        degree_data.append((d, gi_np, err))
    plot_function_degrees(
        'Tanh', 'BWD', x_np, ref_np, degree_data, args.outdir, args.format
    )

    # ===== SWISH FWD =====
    ref_np = (x.float() * torch.sigmoid(x.float())).cpu().numpy()
    degree_data = []
    for d in degrees:
        fn = getattr(spline_ops, f'swish_fwd_d{d}')
        y, _, err = compute_errors(
            fn, lambda t: t.float() * torch.sigmoid(t.float()), x
        )
        degree_data.append((d, y, err))
    plot_function_degrees(
        'Swish', 'FWD', x_np, ref_np, degree_data, args.outdir, args.format
    )

    # ===== SWISH BWD =====
    s = torch.sigmoid(x.float())
    ref_bwd = s * (1 + x.float() * (1 - s))
    ref_np = ref_bwd.cpu().numpy()
    degree_data = []
    for d in degrees:
        fn = getattr(spline_ops, f'swish_bwd_d{d}')
        gi = fn(go, x)
        torch.cuda.synchronize()
        gi_np = gi.float().cpu().numpy()
        err = np.abs(gi_np - ref_np)
        degree_data.append((d, gi_np, err))
    plot_function_degrees(
        'Swish', 'BWD', x_np, ref_np, degree_data, args.outdir, args.format
    )

    print('\nAll plots generated.')


if __name__ == '__main__':
    main()


import os
import time

import torch
import spline_ops


def verify_compiled_swiglu_d3():
    """Exercise the dispatcher-backed B4 activation through compiled autograd."""
    from spline_compile import make_spline_silu_mul

    activation = make_spline_silu_mul(3, "current")
    gate = torch.linspace(
        -12.0, 12.0, 4097, device="cuda", dtype=torch.bfloat16
    )
    up = torch.linspace(-1.0, 1.0, gate.numel(), device="cuda", dtype=torch.bfloat16)
    grad_out = torch.linspace(
        1.0, -1.0, gate.numel(), device="cuda", dtype=torch.bfloat16
    )

    expected = spline_ops.swish_mul_fwd_variant(gate, up, 3, 0)
    expected_grad_gate, expected_grad_up = spline_ops.swish_mul_bwd_variant(
        grad_out, gate, up, 3, 0
    )

    compiled = torch.compile(activation, fullgraph=True)
    compiled_gate = gate.detach().requires_grad_(True)
    compiled_up = up.detach().requires_grad_(True)
    actual = compiled(compiled_gate, compiled_up)
    actual.backward(grad_out)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        compiled_gate.grad, expected_grad_gate, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        compiled_up.grad, expected_grad_up, rtol=0.0, atol=0.0
    )

    native_gate = gate.float().requires_grad_(True)
    native_up = up.float().requires_grad_(True)
    native = torch.nn.functional.silu(native_gate) * native_up
    native.backward(grad_out.float())
    torch.testing.assert_close(actual.float(), native, rtol=0.0, atol=0.07)
    torch.testing.assert_close(
        compiled_gate.grad.float(), native_gate.grad, rtol=0.0, atol=0.07
    )
    torch.testing.assert_close(
        compiled_up.grad.float(), native_up.grad, rtol=0.0, atol=0.07
    )


def verify_compiled_packed_swiglu_d3():
    """Check the copy-free fused-linear activation and its packed gradient."""
    from spline_compile import make_spline_silu_packed

    packed = torch.linspace(
        -12.0, 12.0, 8192, device="cuda", dtype=torch.bfloat16
    ).reshape(2, 4096).requires_grad_(True)
    grad_out = torch.linspace(
        1.0, -1.0, 4096, device="cuda", dtype=torch.bfloat16
    ).reshape(2, 2048)
    gate, up = (part.contiguous() for part in packed.detach().chunk(2, dim=-1))
    expected = spline_ops.swish_mul_fwd_variant(gate, up, 3, 0)
    expected_grad_gate, expected_grad_up = spline_ops.swish_mul_bwd_variant(
        grad_out, gate, up, 3, 0
    )

    activation = torch.compile(make_spline_silu_packed(3, "current"), fullgraph=True)
    actual = activation(packed)
    actual.backward(grad_out)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        packed.grad,
        torch.cat((expected_grad_gate, expected_grad_up), dim=-1),
        rtol=0.0,
        atol=0.0,
    )


def verify_compiled_packed_swiglu_d3_native_backward():
    """Check D3 forward with native SiLU gradients in one packed kernel."""
    from spline_compile import make_spline_silu_packed

    packed = torch.linspace(
        -12.0, 12.0, 8192, device="cuda", dtype=torch.bfloat16
    ).reshape(2, 4096).requires_grad_(True)
    grad_out = torch.linspace(
        1.0, -1.0, 4096, device="cuda", dtype=torch.bfloat16
    ).reshape(2, 2048)
    gate, up = packed.detach().float().chunk(2, dim=-1)
    gate.requires_grad_(True)
    up.requires_grad_(True)
    native = torch.nn.functional.silu(gate) * up
    native.backward(grad_out.float())

    activation = torch.compile(
        make_spline_silu_packed(3, "current", backward_impl="native"),
        fullgraph=True,
    )
    actual = activation(packed)
    actual.backward(grad_out)

    expected = spline_ops.swish_mul_fwd_variant(
        gate.bfloat16(), up.bfloat16(), 3, 0
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    expected_grad = torch.cat((gate.grad, up.grad), dim=-1).bfloat16()
    torch.testing.assert_close(packed.grad, expected_grad, rtol=0.01, atol=0.0078125)


def verify_compiled_packed_native_swiglu():
    """Check the apples-to-apples packed SFU baseline and its gradients."""
    from spline_compile import make_native_silu_packed

    packed = torch.linspace(
        -12.0, 12.0, 8192, device="cuda", dtype=torch.bfloat16
    ).reshape(2, 4096).requires_grad_(True)
    grad_out = torch.linspace(
        1.0, -1.0, 4096, device="cuda", dtype=torch.bfloat16
    ).reshape(2, 2048)

    native_packed = packed.detach().clone().requires_grad_(True)
    native_gate, native_up = native_packed.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(native_gate) * native_up
    expected.backward(grad_out)

    activation = torch.compile(make_native_silu_packed(), fullgraph=True)
    actual = activation(packed)
    actual.backward(grad_out)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(packed.grad, native_packed.grad, rtol=0.0, atol=0.0)


def benchmark(name, func, args, n_warmup=10, n_repeats=100):
    # Warmup
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_repeats):
        func(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / n_repeats


def verify_bf16_swish_tails():
    x = torch.tensor([-100.0, -20.0, 20.0, 100.0], device="cuda", dtype=torch.bfloat16)
    expected = torch.tensor([0.0, 0.0, 20.0, 100.0], device="cuda", dtype=torch.bfloat16)
    for degree in (3, 4, 5, 6):
        y = spline_ops.swish_mul_fwd_variant(x, torch.ones_like(x), degree, 0)
        torch.testing.assert_close(y, expected, rtol=0.0, atol=0.0)


def verify_bf16_sollya_variants():
    """Exercise Sollya coefficients through unary, fused, and packed paths."""
    x = torch.linspace(-8.0, 8.0, 16384, device="cuda", dtype=torch.bfloat16)
    up = torch.linspace(-1.0, 1.0, x.numel(), device="cuda", dtype=torch.bfloat16)
    grad_out = torch.linspace(
        1.0, -1.0, x.numel(), device="cuda", dtype=torch.bfloat16
    )

    sigmoid = spline_ops.sigmoid_fwd_variant(x, 4, 1)
    torch.testing.assert_close(
        sigmoid.float(), torch.sigmoid(x).float(), rtol=0.0, atol=0.0078125
    )
    sigmoid_grad = spline_ops.sigmoid_bwd_variant(grad_out, x, 4, 1)
    sigmoid_ref = torch.sigmoid(x.float())
    sigmoid_grad_ref = grad_out.float() * sigmoid_ref * (1.0 - sigmoid_ref)
    torch.testing.assert_close(
        sigmoid_grad.float(), sigmoid_grad_ref, rtol=0.0, atol=0.0082
    )

    fused = spline_ops.swish_mul_fwd_variant(x, up, 4, 1)
    expected = spline_ops.swish_fwd_variant(x, 4, 1) * up
    torch.testing.assert_close(fused, expected, rtol=0.0, atol=0.0)

    packed = torch.cat((x.reshape(8, -1), up.reshape(8, -1)), dim=-1)
    packed_output = spline_ops.swish_mul_packed_fwd_variant(packed, 4, 1)
    torch.testing.assert_close(packed_output.reshape(-1), fused, rtol=0.0, atol=0.0)

    grad_gate, grad_up = spline_ops.swish_mul_bwd_variant(
        grad_out, x, up, 4, 1
    )
    packed_grad = spline_ops.swish_mul_packed_bwd_variant(
        grad_out.reshape(8, -1), packed, 4, 1
    )
    split = packed.shape[-1] // 2
    torch.testing.assert_close(
        packed_grad[..., :split].reshape(-1), grad_gate, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        packed_grad[..., split:].reshape(-1), grad_up, rtol=0.0, atol=0.0
    )


def verify_packed_swish_large_row_offsets():
    """Cross the old signed-int row-offset boundary in both packed fast paths."""
    expected = spline_ops.swish_mul_fwd_variant(
        torch.ones(1, device="cuda", dtype=torch.bfloat16),
        torch.ones(1, device="cuda", dtype=torch.bfloat16),
        4,
        0,
    ).item()
    for hidden_size in (512, 768):
        boundary = 2**31 // (2 * hidden_size)
        rows = boundary + 8
        packed = torch.ones(
            (rows, hidden_size * 2), device="cuda", dtype=torch.bfloat16
        )
        output = spline_ops.swish_mul_packed_fwd_variant(packed, 4, 0)
        probe_rows = torch.tensor(
            [0, boundary - 1, boundary, rows - 1], device="cuda", dtype=torch.long
        )
        expected_values = torch.full(
            (probe_rows.numel(), hidden_size),
            expected,
            device="cuda",
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(
            output.index_select(0, probe_rows), expected_values, rtol=0.0, atol=0.0
        )
        del packed, output, probe_rows, expected_values
        torch.cuda.empty_cache()


def verify_and_bench():
    print("=== Spline Ops Verification & Benchmark ===")
    verify_bf16_swish_tails()
    verify_bf16_sollya_variants()
    verify_compiled_swiglu_d3()
    verify_compiled_packed_swiglu_d3()
    verify_compiled_packed_swiglu_d3_native_backward()
    verify_compiled_packed_native_swiglu()
    if os.environ.get("SPLINE_OPS_TEST_LARGE_PACKED") == "1":
        verify_packed_swish_large_row_offsets()
    device = torch.device("cuda")
    N = 1024 * 1024 * 16 # 16M elements
    x = torch.randn(N, device=device, dtype=torch.half) * 3.0 # Range approx [-9, 9]
    grad_out = torch.randn(N, device=device, dtype=torch.half)

    # --- SIGMOID ---
    print("\n--- Sigmoid ---")
    y_ref = torch.sigmoid(x)
    y_test = spline_ops.sigmoid_fwd(x)
    max_err = (y_ref - y_test).abs().max().item()
    print(f"Fwd Max Err: {max_err:.6f}")
    assert max_err < 0.02, "Sigmoid Fwd Error too high"

    fwd_time_ref = benchmark("Sigmoid Native", torch.sigmoid, (x,))
    fwd_time_test = benchmark("Sigmoid Spline", spline_ops.sigmoid_fwd, (x,))
    print(f"Fwd Time: Native {fwd_time_ref:.3f} ms, Spline {fwd_time_test:.3f} ms -> Speedup {fwd_time_ref/fwd_time_test:.2f}x")


    # Bwd
    # Gradients depend on implementation. Native: sig * (1-sig)
    # My spline bwd kernel takes (grad_out, x).
    # Native:
    x.requires_grad_(True)
    y = torch.sigmoid(x)
    y.backward(grad_out, retain_graph=False)
    grad_ref = x.grad.clone()
    x.grad = None
    x.requires_grad_(False)

    grad_test = spline_ops.sigmoid_bwd(grad_out, x)
    max_err_grad = (grad_ref - grad_test).abs().max().item()
    print(f"Bwd Max Err: {max_err_grad:.6f}")
    if max_err_grad > 0.1:
        print("WARNING: Sigmoid Bwd Error high")

    # Check if we can beat simple python JIT or native backward
    def native_sigmoid_bwd(g, x):
        s = torch.sigmoid(x)
        return g * s * (1 - s)

    bwd_time_ref = benchmark("Sigmoid Bwd Native", native_sigmoid_bwd, (grad_out, x))
    bwd_time_test = benchmark("Sigmoid Bwd Spline", spline_ops.sigmoid_bwd, (grad_out, x))
    print(f"Bwd Time: Native {bwd_time_ref:.3f} ms, Spline {bwd_time_test:.3f} ms -> Speedup {bwd_time_ref/bwd_time_test:.2f}x")

    # --- TANH ---
    print("\n--- Tanh ---")
    y_ref = torch.tanh(x)
    y_test = spline_ops.tanh_fwd(x)
    max_err = (y_ref - y_test).abs().max().item()
    print(f"Fwd Max Err: {max_err:.6f}")
    if max_err > 0.05:
        print("WARNING: Tanh Fwd Error high")

    fwd_time_ref = benchmark("Tanh Native", torch.tanh, (x,))
    fwd_time_test = benchmark("Tanh Spline", spline_ops.tanh_fwd, (x,))
    print(f"Fwd Time: Native {fwd_time_ref:.3f} ms, Spline {fwd_time_test:.3f} ms -> Speedup {fwd_time_ref/fwd_time_test:.2f}x")

    # Bwd: 1 - tanh^2
    def native_tanh_bwd(g, x):
        t = torch.tanh(x)
        return g * (1 - t*t)

    grad_test = spline_ops.tanh_bwd(grad_out, x)
    grad_ref = native_tanh_bwd(grad_out, x)
    max_err_grad = (grad_ref - grad_test).abs().max().item()
    print(f"Bwd Max Err: {max_err_grad:.6f}")
    if max_err_grad > 0.1:
        print("WARNING: Tanh Bwd Error high")

    bwd_time_ref = benchmark("Tanh Bwd Native", native_tanh_bwd, (grad_out, x))
    bwd_time_test = benchmark("Tanh Bwd Spline", spline_ops.tanh_bwd, (grad_out, x))
    print(f"Bwd Time: Native {bwd_time_ref:.3f} ms, Spline {bwd_time_test:.3f} ms -> Speedup {bwd_time_ref/bwd_time_test:.2f}x")

    # --- SWISH ---
    print("\n--- Swish ---")
    y_ref = torch.nn.functional.silu(x)
    y_test = spline_ops.swish_fwd(x)
    max_err = (y_ref - y_test).abs().max().item()
    print(f"Fwd Max Err: {max_err:.6f}")
    if max_err > 0.05:
        print("WARNING: Swish Fwd Error high")

    fwd_time_ref = benchmark("Swish Native", torch.nn.functional.silu, (x,))
    fwd_time_test = benchmark("Swish Spline", spline_ops.swish_fwd, (x,))
    print(f"Fwd Time: Native {fwd_time_ref:.3f} ms, Spline {fwd_time_test:.3f} ms -> Speedup {fwd_time_ref/fwd_time_test:.2f}x")

    # Bwd
    def native_swish_bwd(g, x):
        s = torch.sigmoid(x)
        return g * (s * (1 + x * (1 - s)))

    grad_test = spline_ops.swish_bwd(grad_out, x)
    grad_ref = native_swish_bwd(grad_out, x)
    max_err_grad = (grad_ref - grad_test).abs().max().item()
    print(f"Bwd Max Err: {max_err_grad:.6f}")
    if max_err_grad > 0.1:
        print("WARNING: Swish Bwd Error high")

    bwd_time_ref = benchmark("Swish Bwd Native", native_swish_bwd, (grad_out, x))
    bwd_time_test = benchmark("Swish Bwd Spline", spline_ops.swish_bwd, (grad_out, x))
    print(f"Bwd Time: Native {bwd_time_ref:.3f} ms, Spline {bwd_time_test:.3f} ms -> Speedup {bwd_time_ref/bwd_time_test:.2f}x")

if __name__ == "__main__":
    verify_and_bench()

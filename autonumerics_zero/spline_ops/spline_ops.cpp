
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

// =============================================================================
// CUDA launcher declarations — FP16 (from spline_kernels.cu)
// =============================================================================
extern "C" {
    // Paired FP32 sin/cos controls and polynomial kernels.
    void launch_sincos_native_f32(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d3_d4_f32(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d3_d4_cycle_f32(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d3_d4_magic_bias_f32(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d5_d4_half_turn_ls_f32(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d5_d4_half_turn_sollya_f32(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d5_d4_half_turn_sollya_fast_f32(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d5_d4_f32(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d7_d6_f32(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_native_compute_f32(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d3_d4_compute_f32(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d3_d4_cycle_compute_f32(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d3_d4_magic_bias_compute_f32(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d5_d4_half_turn_ls_compute_f32(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d5_d4_half_turn_sollya_compute_f32(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d5_d4_half_turn_sollya_fast_compute_f32(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d5_d4_compute_f32(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d7_d6_compute_f32(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_native_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d3_d4_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d3_d4_quarter_turn_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d3_d4_half_turn_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d5_d6_half_turn_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d5_d4_half_turn_fp16_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_native_fp16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d3_d4_quarter_turn_fp16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d3_d4_half_turn_fp16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d5_d4_half_turn_fp16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d5_d6_half_turn_fp16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_d7_d6_half_turn_fp16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sincos_native_bf16_compute(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d3_d4_bf16_compute(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d3_d4_quarter_turn_bf16_compute(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d3_d4_half_turn_bf16_compute(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d5_d6_half_turn_bf16_compute(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d5_d4_half_turn_fp16_bf16_compute(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_native_fp16_compute(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d3_d4_quarter_turn_fp16_compute(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d3_d4_half_turn_fp16_compute(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d5_d4_half_turn_fp16_compute(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d5_d6_half_turn_fp16_compute(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_sincos_d7_d6_half_turn_fp16_compute(void* out, const void* in, int size, int iterations, cudaStream_t s);
    void launch_rope_sincos_native_fp16(void* out, const void* frequencies, int sequence_length, int frequency_count, cudaStream_t s);
    void launch_rope_sincos_fixed_d3_d4_fp16(void* out, const void* phase_increments, int sequence_length, int frequency_count, cudaStream_t s);
    void launch_rope_sincos_fixed_half_turn_d5_d6_fp16(void* out, const void* phase_increments, int sequence_length, int frequency_count, cudaStream_t s);
    void launch_rope_sincos_fixed_lut_fp16(void* out, const void* phase_increments, const void* phase_table, int sequence_length, int frequency_count, cudaStream_t s);
    void launch_rope_sincos_native_fp16_compute(void* out, const void* frequencies, int sequence_length, int frequency_count, int iterations, cudaStream_t s);
    void launch_rope_sincos_fixed_d3_d4_fp16_compute(void* out, const void* phase_increments, int sequence_length, int frequency_count, int iterations, cudaStream_t s);
    void launch_rope_sincos_fixed_half_turn_d5_d6_fp16_compute(void* out, const void* phase_increments, int sequence_length, int frequency_count, int iterations, cudaStream_t s);
    void launch_rope_sincos_fixed_lut_fp16_compute(void* out, const void* phase_increments, const void* phase_table, int sequence_length, int frequency_count, int iterations, cudaStream_t s);
    void launch_rope_apply_cached_fp16(void* q_out, void* k_out, const void* q, const void* k, const void* rope_table, int batch_size, int sequence_length, int q_head_count, int k_head_count, int head_dim, cudaStream_t s);
    void launch_rope_apply_native_fp16(void* q_out, void* k_out, const void* q, const void* k, const void* frequencies, int batch_size, int sequence_length, int q_head_count, int k_head_count, int head_dim, cudaStream_t s);
    void launch_rope_apply_fixed_half_turn_d5_d6_fp16(void* q_out, void* k_out, const void* q, const void* k, const void* phase_increments, int batch_size, int sequence_length, int q_head_count, int k_head_count, int head_dim, cudaStream_t s);

    // Sigmoid FWD D3-D6
    void launch_sigmoid_fwd_d3_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_fwd_d4_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_fwd_d5_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_fwd_d6_kernel(void* out, const void* in, int size, cudaStream_t s);

    // Sigmoid BWD D3-D6
    void launch_sigmoid_bwd_d3_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_bwd_d4_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_bwd_d5_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_bwd_d6_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);

    // Tanh FWD D3-D6
    void launch_tanh_fwd_d3_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_tanh_fwd_d4_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_tanh_fwd_d5_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_tanh_fwd_d6_kernel(void* out, const void* in, int size, cudaStream_t s);

    // Tanh BWD D3-D6
    void launch_tanh_bwd_d3_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_tanh_bwd_d4_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_tanh_bwd_d5_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_tanh_bwd_d6_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);

    // Swish FWD native and D3-D6
    void launch_swish_fwd_native_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_fwd_d3_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_fwd_d4_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_fwd_d5_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_fwd_d6_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_mul_fwd_native_kernel(void* out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_fwd_d3_kernel(void* out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_fwd_d4_kernel(void* out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_fwd_d5_kernel(void* out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_fwd_d6_kernel(void* out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_bwd_native_kernel(void* grad_gate, void* grad_up, const void* grad_out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_bwd_d3_kernel(void* grad_gate, void* grad_up, const void* grad_out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_bwd_d4_kernel(void* grad_gate, void* grad_up, const void* grad_out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_bwd_d5_kernel(void* grad_gate, void* grad_up, const void* grad_out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_bwd_d6_kernel(void* grad_gate, void* grad_up, const void* grad_out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_packed_fwd_d3_kernel(void* out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_fwd_d4_kernel(void* out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_fwd_d5_kernel(void* out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_fwd_d6_kernel(void* out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_fwd_native_kernel(void* out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_bwd_d3_kernel(void* grad_packed, const void* grad_out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_bwd_d4_kernel(void* grad_packed, const void* grad_out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_bwd_d5_kernel(void* grad_packed, const void* grad_out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_bwd_d6_kernel(void* grad_packed, const void* grad_out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_bwd_native_kernel(void* grad_packed, const void* grad_out, const void* packed, int rows, int hidden_size, cudaStream_t s);

    // Swish BWD native and D3-D6
    void launch_swish_bwd_native_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_swish_bwd_d3_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_swish_bwd_d4_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_swish_bwd_d5_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_swish_bwd_d6_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);

    // GeLU FWD D3-D6
    void launch_gelu_fwd_d3_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_gelu_fwd_d4_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_gelu_fwd_d5_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_gelu_fwd_d6_kernel(void* out, const void* in, int size, cudaStream_t s);

    // GeLU BWD D3-D6
    void launch_gelu_bwd_d3_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_gelu_bwd_d4_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_gelu_bwd_d5_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_gelu_bwd_d6_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);

    // Algebraic backward (uses cached forward output y)
    void launch_sigmoid_bwd_alg_kernel(void* gi, const void* go, const void* y, int size, cudaStream_t s);
    void launch_tanh_bwd_alg_kernel(void* gi, const void* go, const void* y, int size, cudaStream_t s);

    // Hybrid FWD: 1 SFU + 3 Polynomial
    void launch_sigmoid_fwd_hybrid_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_tanh_fwd_hybrid_kernel(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_fwd_hybrid_kernel(void* out, const void* in, int size, cudaStream_t s);

    // Hybrid BWD: 1 SFU + 3 Polynomial
    void launch_sigmoid_bwd_hybrid_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_tanh_bwd_hybrid_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_swish_bwd_hybrid_kernel(void* gi, const void* go, const void* in, int size, cudaStream_t s);

    // Fused FWD+BWD: single pass (needs go available)
    void launch_sigmoid_fused_kernel(void* y_out, void* gi_out, const void* x_in, const void* go_in, int size, cudaStream_t s);
    void launch_tanh_fused_kernel(void* y_out, void* gi_out, const void* x_in, const void* go_in, int size, cudaStream_t s);
    void launch_swish_fused_kernel(void* y_out, void* gi_out, const void* x_in, const void* go_in, int size, cudaStream_t s);

    // FWD with derivative: standard autograd (forward writes y AND dy)
    void launch_sigmoid_fwd_deriv_alg_kernel(void* y, void* dy, const void* x, int size, cudaStream_t s);
    void launch_sigmoid_fwd_deriv_poly_kernel(void* y, void* dy, const void* x, int size, cudaStream_t s);
    void launch_tanh_fwd_deriv_alg_kernel(void* y, void* dy, const void* x, int size, cudaStream_t s);
    void launch_tanh_fwd_deriv_poly_kernel(void* y, void* dy, const void* x, int size, cudaStream_t s);
    void launch_swish_fwd_deriv_kernel(void* y, void* dy, const void* x, int size, cudaStream_t s);
    void launch_swish_fwd_deriv_poly_kernel(void* y, void* dy, const void* x, int size, cudaStream_t s);

    // Trivial multiply: gi = a * b
    void launch_multiply_kernel(void* out, const void* a, const void* b, int size, cudaStream_t s);
}

// =============================================================================
// CUDA launcher declarations — BF16 (from spline_kernels_bf16.cu)
// =============================================================================
extern "C" {
    // Sigmoid FWD D3-D6 (BF16)
    void launch_sigmoid_fwd_d3_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_fwd_d4_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_fwd_d5_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_fwd_d6_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_fwd_d3_kernel_bf16_sollya(void* out, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_fwd_d4_kernel_bf16_sollya(void* out, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_fwd_d5_kernel_bf16_sollya(void* out, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_fwd_d6_kernel_bf16_sollya(void* out, const void* in, int size, cudaStream_t s);

    // Sigmoid BWD D3-D6 (BF16)
    void launch_sigmoid_bwd_d3_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_bwd_d4_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_bwd_d5_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_bwd_d6_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_bwd_d3_kernel_bf16_sollya(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_bwd_d4_kernel_bf16_sollya(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_bwd_d5_kernel_bf16_sollya(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_sigmoid_bwd_d6_kernel_bf16_sollya(void* gi, const void* go, const void* in, int size, cudaStream_t s);

    // Tanh FWD D3-D6 (BF16)
    void launch_tanh_fwd_d3_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_tanh_fwd_d4_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_tanh_fwd_d5_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_tanh_fwd_d6_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);

    // Tanh BWD D3-D6 (BF16)
    void launch_tanh_bwd_d3_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_tanh_bwd_d4_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_tanh_bwd_d5_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_tanh_bwd_d6_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);

    // Swish FWD native and D3-D6 (BF16)
    void launch_swish_fwd_native_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_fwd_d3_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_fwd_d4_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_fwd_d5_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_fwd_d6_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_mul_fwd_native_kernel_bf16(void* out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_fwd_d3_kernel_bf16(void* out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_fwd_d4_kernel_bf16(void* out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_fwd_d5_kernel_bf16(void* out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_fwd_d6_kernel_bf16(void* out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_fwd_d3_kernel_bf16_sollya(void* out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_fwd_d4_kernel_bf16_sollya(void* out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_fwd_d5_kernel_bf16_sollya(void* out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_fwd_d6_kernel_bf16_sollya(void* out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_bwd_native_kernel_bf16(void* grad_gate, void* grad_up, const void* grad_out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_bwd_d3_kernel_bf16(void* grad_gate, void* grad_up, const void* grad_out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_bwd_d4_kernel_bf16(void* grad_gate, void* grad_up, const void* grad_out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_bwd_d5_kernel_bf16(void* grad_gate, void* grad_up, const void* grad_out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_bwd_d6_kernel_bf16(void* grad_gate, void* grad_up, const void* grad_out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_bwd_d3_kernel_bf16_sollya(void* grad_gate, void* grad_up, const void* grad_out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_bwd_d4_kernel_bf16_sollya(void* grad_gate, void* grad_up, const void* grad_out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_bwd_d5_kernel_bf16_sollya(void* grad_gate, void* grad_up, const void* grad_out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_bwd_d6_kernel_bf16_sollya(void* grad_gate, void* grad_up, const void* grad_out, const void* gate, const void* up, int size, cudaStream_t s);
    void launch_swish_mul_packed_fwd_d3_kernel_bf16(void* out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_fwd_d4_kernel_bf16(void* out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_fwd_d5_kernel_bf16(void* out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_fwd_d6_kernel_bf16(void* out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_fwd_d3_kernel_bf16_sollya(void* out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_fwd_d4_kernel_bf16_sollya(void* out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_fwd_d5_kernel_bf16_sollya(void* out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_fwd_d6_kernel_bf16_sollya(void* out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_fwd_native_kernel_bf16(void* out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_bwd_d3_kernel_bf16(void* grad_packed, const void* grad_out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_bwd_d4_kernel_bf16(void* grad_packed, const void* grad_out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_bwd_d5_kernel_bf16(void* grad_packed, const void* grad_out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_bwd_d6_kernel_bf16(void* grad_packed, const void* grad_out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_bwd_d3_kernel_bf16_sollya(void* grad_packed, const void* grad_out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_bwd_d4_kernel_bf16_sollya(void* grad_packed, const void* grad_out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_bwd_d5_kernel_bf16_sollya(void* grad_packed, const void* grad_out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_bwd_d6_kernel_bf16_sollya(void* grad_packed, const void* grad_out, const void* packed, int rows, int hidden_size, cudaStream_t s);
    void launch_swish_mul_packed_bwd_native_kernel_bf16(void* grad_packed, const void* grad_out, const void* packed, int rows, int hidden_size, cudaStream_t s);

    // Swish BWD native and D3-D6 (BF16)
    void launch_swish_bwd_native_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_swish_bwd_d3_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_swish_bwd_d4_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_swish_bwd_d5_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_swish_bwd_d6_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_swish_fwd_d3_kernel_bf16_sollya(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_fwd_d4_kernel_bf16_sollya(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_fwd_d5_kernel_bf16_sollya(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_fwd_d6_kernel_bf16_sollya(void* out, const void* in, int size, cudaStream_t s);
    void launch_swish_bwd_d3_kernel_bf16_sollya(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_swish_bwd_d4_kernel_bf16_sollya(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_swish_bwd_d5_kernel_bf16_sollya(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_swish_bwd_d6_kernel_bf16_sollya(void* gi, const void* go, const void* in, int size, cudaStream_t s);

    // GeLU FWD D3-D6 (BF16)
    void launch_gelu_fwd_d3_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_gelu_fwd_d4_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_gelu_fwd_d5_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);
    void launch_gelu_fwd_d6_kernel_bf16(void* out, const void* in, int size, cudaStream_t s);

    // GeLU BWD D3-D6 (BF16)
    void launch_gelu_bwd_d3_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_gelu_bwd_d4_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_gelu_bwd_d5_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_gelu_bwd_d6_kernel_bf16(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_gelu_fwd_d3_kernel_bf16_sollya(void* out, const void* in, int size, cudaStream_t s);
    void launch_gelu_fwd_d4_kernel_bf16_sollya(void* out, const void* in, int size, cudaStream_t s);
    void launch_gelu_fwd_d5_kernel_bf16_sollya(void* out, const void* in, int size, cudaStream_t s);
    void launch_gelu_fwd_d6_kernel_bf16_sollya(void* out, const void* in, int size, cudaStream_t s);
    void launch_gelu_bwd_d3_kernel_bf16_sollya(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_gelu_bwd_d4_kernel_bf16_sollya(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_gelu_bwd_d5_kernel_bf16_sollya(void* gi, const void* go, const void* in, int size, cudaStream_t s);
    void launch_gelu_bwd_d6_kernel_bf16_sollya(void* gi, const void* go, const void* in, int size, cudaStream_t s);

    // Algebraic backward (BF16)
    void launch_sigmoid_bwd_alg_kernel_bf16(void* gi, const void* go, const void* y, int size, cudaStream_t s);
    void launch_tanh_bwd_alg_kernel_bf16(void* gi, const void* go, const void* y, int size, cudaStream_t s);
}

// =============================================================================
// Input checking macros
// =============================================================================
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_HALF_OR_BF16(x) TORCH_CHECK(                                 \
    x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,      \
    #x " must be float16 or bfloat16")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_HALF_OR_BF16(x)

static bool spline_env_flag(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "1") == 0
        || std::strcmp(value, "true") == 0
        || std::strcmp(value, "TRUE") == 0
        || std::strcmp(value, "yes") == 0
        || std::strcmp(value, "YES") == 0
        || std::strcmp(value, "on") == 0
        || std::strcmp(value, "ON") == 0;
}

static void check_spline_cuda_launch(cudaStream_t stream) {
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (spline_env_flag("SPLINE_OPS_SYNC_DEBUG")) {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
        if (capture_status == cudaStreamCaptureStatusNone) {
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
}

using SincosLauncher = void (*)(void*, const void*, int, cudaStream_t);

static at::Tensor sincos_f32(at::Tensor angles, SincosLauncher launcher) {
    CHECK_CUDA(angles);
    CHECK_CONTIGUOUS(angles);
    TORCH_CHECK(angles.scalar_type() == at::kFloat, "angles must be float32");
    TORCH_CHECK(
        angles.numel() <= std::numeric_limits<int>::max(),
        "angles has too many elements for the CUDA launcher"
    );

    std::vector<int64_t> output_sizes = angles.sizes().vec();
    output_sizes.insert(output_sizes.begin(), 2);
    at::Tensor output = at::empty(output_sizes, angles.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        output.data_ptr(),
        angles.data_ptr(),
        static_cast<int>(angles.numel()),
        stream
    );
    check_spline_cuda_launch(stream);
    return output;
}

static at::Tensor sincos_native_f32(at::Tensor angles) {
    return sincos_f32(angles, launch_sincos_native_f32);
}

static at::Tensor sincos_d3_d4_f32(at::Tensor angles) {
    return sincos_f32(angles, launch_sincos_d3_d4_f32);
}

static at::Tensor sincos_d3_d4_cycle_f32(at::Tensor angles) {
    return sincos_f32(angles, launch_sincos_d3_d4_cycle_f32);
}

static at::Tensor sincos_d3_d4_magic_bias_f32(at::Tensor angles) {
    return sincos_f32(angles, launch_sincos_d3_d4_magic_bias_f32);
}

static at::Tensor sincos_d5_d4_half_turn_ls_f32(at::Tensor angles) {
    return sincos_f32(angles, launch_sincos_d5_d4_half_turn_ls_f32);
}

static at::Tensor sincos_d5_d4_half_turn_sollya_f32(at::Tensor angles) {
    return sincos_f32(angles, launch_sincos_d5_d4_half_turn_sollya_f32);
}

static at::Tensor sincos_d5_d4_half_turn_sollya_fast_f32(at::Tensor angles) {
    return sincos_f32(angles, launch_sincos_d5_d4_half_turn_sollya_fast_f32);
}

static at::Tensor sincos_d5_d4_f32(at::Tensor angles) {
    return sincos_f32(angles, launch_sincos_d5_d4_f32);
}

static at::Tensor sincos_d7_d6_f32(at::Tensor angles) {
    return sincos_f32(angles, launch_sincos_d7_d6_f32);
}

using SincosComputeLauncher = void (*)(
    void*, const void*, int, int, cudaStream_t);

static at::Tensor sincos_compute_f32(
    at::Tensor angles,
    int64_t iterations,
    SincosComputeLauncher launcher) {
    CHECK_CUDA(angles);
    CHECK_CONTIGUOUS(angles);
    TORCH_CHECK(angles.scalar_type() == at::kFloat, "angles must be float32");
    TORCH_CHECK(iterations > 0, "iterations must be positive");
    TORCH_CHECK(
        iterations <= std::numeric_limits<int>::max(),
        "iterations is too large for the CUDA launcher"
    );
    TORCH_CHECK(
        angles.numel() <= std::numeric_limits<int>::max(),
        "angles has too many elements for the CUDA launcher"
    );

    std::vector<int64_t> output_sizes = angles.sizes().vec();
    output_sizes.insert(output_sizes.begin(), 2);
    at::Tensor output = at::empty(output_sizes, angles.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        output.data_ptr(),
        angles.data_ptr(),
        static_cast<int>(angles.numel()),
        static_cast<int>(iterations),
        stream
    );
    check_spline_cuda_launch(stream);
    return output;
}

static at::Tensor sincos_native_compute_f32(at::Tensor angles, int64_t iterations) {
    return sincos_compute_f32(
        angles, iterations, launch_sincos_native_compute_f32);
}

static at::Tensor sincos_d3_d4_compute_f32(at::Tensor angles, int64_t iterations) {
    return sincos_compute_f32(
        angles, iterations, launch_sincos_d3_d4_compute_f32);
}

static at::Tensor sincos_d3_d4_cycle_compute_f32(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_f32(
        angles, iterations, launch_sincos_d3_d4_cycle_compute_f32);
}

static at::Tensor sincos_d3_d4_magic_bias_compute_f32(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_f32(
        angles, iterations, launch_sincos_d3_d4_magic_bias_compute_f32);
}

static at::Tensor sincos_d5_d4_half_turn_ls_compute_f32(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_f32(
        angles, iterations, launch_sincos_d5_d4_half_turn_ls_compute_f32);
}

static at::Tensor sincos_d5_d4_half_turn_sollya_compute_f32(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_f32(
        angles, iterations, launch_sincos_d5_d4_half_turn_sollya_compute_f32);
}

static at::Tensor sincos_d5_d4_half_turn_sollya_fast_compute_f32(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_f32(
        angles, iterations, launch_sincos_d5_d4_half_turn_sollya_fast_compute_f32);
}

static at::Tensor sincos_d5_d4_compute_f32(at::Tensor angles, int64_t iterations) {
    return sincos_compute_f32(
        angles, iterations, launch_sincos_d5_d4_compute_f32);
}

static at::Tensor sincos_d7_d6_compute_f32(at::Tensor angles, int64_t iterations) {
    return sincos_compute_f32(
        angles, iterations, launch_sincos_d7_d6_compute_f32);
}

static at::Tensor sincos_bf16(
    at::Tensor angles,
    SincosLauncher launcher) {
    CHECK_CUDA(angles);
    CHECK_CONTIGUOUS(angles);
    TORCH_CHECK(angles.scalar_type() == at::kFloat, "angles must be float32");
    TORCH_CHECK(
        (angles.numel() % 2) == 0,
        "angles must contain an even number of elements"
    );
    TORCH_CHECK(
        angles.numel() <= std::numeric_limits<int>::max(),
        "angles has too many elements for the CUDA launcher"
    );

    std::vector<int64_t> output_sizes = angles.sizes().vec();
    output_sizes.insert(output_sizes.begin(), 2);
    at::Tensor output = at::empty(
        output_sizes, angles.options().dtype(at::kBFloat16));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        output.data_ptr(),
        angles.data_ptr(),
        static_cast<int>(angles.numel()),
        stream
    );
    check_spline_cuda_launch(stream);
    return output;
}

static at::Tensor sincos_native_bf16(at::Tensor angles) {
    return sincos_bf16(angles, launch_sincos_native_bf16);
}

static at::Tensor sincos_d3_d4_bf16(at::Tensor angles) {
    return sincos_bf16(angles, launch_sincos_d3_d4_bf16);
}

static at::Tensor sincos_d3_d4_quarter_turn_bf16(at::Tensor angles) {
    return sincos_bf16(angles, launch_sincos_d3_d4_quarter_turn_bf16);
}

static at::Tensor sincos_d3_d4_half_turn_bf16(at::Tensor angles) {
    return sincos_bf16(angles, launch_sincos_d3_d4_half_turn_bf16);
}

static at::Tensor sincos_d5_d6_half_turn_bf16(at::Tensor angles) {
    return sincos_bf16(angles, launch_sincos_d5_d6_half_turn_bf16);
}

static at::Tensor sincos_d5_d4_half_turn_fp16_bf16(at::Tensor angles) {
    return sincos_bf16(
        angles, launch_sincos_d5_d4_half_turn_fp16_bf16);
}

static at::Tensor sincos_fp16(
    at::Tensor angles,
    SincosLauncher launcher) {
    CHECK_CUDA(angles);
    CHECK_CONTIGUOUS(angles);
    TORCH_CHECK(angles.scalar_type() == at::kFloat, "angles must be float32");
    TORCH_CHECK(
        (angles.numel() % 2) == 0,
        "angles must contain an even number of elements"
    );
    TORCH_CHECK(
        angles.numel() <= std::numeric_limits<int>::max(),
        "angles has too many elements for the CUDA launcher"
    );

    std::vector<int64_t> output_sizes = angles.sizes().vec();
    output_sizes.insert(output_sizes.begin(), 2);
    at::Tensor output = at::empty(
        output_sizes, angles.options().dtype(at::kHalf));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        output.data_ptr(),
        angles.data_ptr(),
        static_cast<int>(angles.numel()),
        stream
    );
    check_spline_cuda_launch(stream);
    return output;
}

static at::Tensor sincos_native_fp16(at::Tensor angles) {
    return sincos_fp16(angles, launch_sincos_native_fp16);
}

static at::Tensor sincos_d3_d4_quarter_turn_fp16(at::Tensor angles) {
    return sincos_fp16(angles, launch_sincos_d3_d4_quarter_turn_fp16);
}

static at::Tensor sincos_d3_d4_half_turn_fp16(at::Tensor angles) {
    return sincos_fp16(angles, launch_sincos_d3_d4_half_turn_fp16);
}

static at::Tensor sincos_d5_d4_half_turn_fp16(at::Tensor angles) {
    return sincos_fp16(angles, launch_sincos_d5_d4_half_turn_fp16);
}

static at::Tensor sincos_d5_d6_half_turn_fp16(at::Tensor angles) {
    return sincos_fp16(angles, launch_sincos_d5_d6_half_turn_fp16);
}

static at::Tensor sincos_d7_d6_half_turn_fp16(at::Tensor angles) {
    return sincos_fp16(angles, launch_sincos_d7_d6_half_turn_fp16);
}

using RopeSincosLauncher = void (*)(
    void*, const void*, int, int, cudaStream_t);
using RopeSincosComputeLauncher = void (*)(
    void*, const void*, int, int, int, cudaStream_t);

static at::Tensor allocate_rope_sincos_fp16_output(
    at::Tensor parameters,
    int64_t sequence_length,
    at::ScalarType parameter_type,
    const char* parameter_name) {
    CHECK_CUDA(parameters);
    CHECK_CONTIGUOUS(parameters);
    TORCH_CHECK(parameters.dim() == 1, parameter_name, " must be one-dimensional");
    TORCH_CHECK(
        parameters.scalar_type() == parameter_type,
        parameter_name,
        " has the wrong dtype");
    TORCH_CHECK(
        (parameters.numel() % 2) == 0,
        parameter_name,
        " must contain an even number of elements");
    TORCH_CHECK(sequence_length >= 0, "sequence_length must be non-negative");
    TORCH_CHECK(
        sequence_length <= std::numeric_limits<int>::max(),
        "sequence_length is too large for the CUDA launcher");
    TORCH_CHECK(
        parameters.numel() <= std::numeric_limits<int>::max(),
        parameter_name,
        " has too many elements for the CUDA launcher");
    TORCH_CHECK(
        sequence_length == 0
            || parameters.numel()
                <= std::numeric_limits<int>::max() / sequence_length,
        "RoPE table has too many elements for the CUDA launcher");

    return at::empty(
        {2, sequence_length, parameters.numel()},
        parameters.options().dtype(at::kHalf));
}

static at::Tensor rope_sincos_fp16(
    at::Tensor parameters,
    int64_t sequence_length,
    at::ScalarType parameter_type,
    const char* parameter_name,
    RopeSincosLauncher launcher) {
    at::Tensor output = allocate_rope_sincos_fp16_output(
        parameters, sequence_length, parameter_type, parameter_name);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        output.data_ptr(),
        parameters.data_ptr(),
        static_cast<int>(sequence_length),
        static_cast<int>(parameters.numel()),
        stream);
    check_spline_cuda_launch(stream);
    return output;
}

static at::Tensor rope_sincos_native_fp16(
    at::Tensor frequencies,
    int64_t sequence_length) {
    return rope_sincos_fp16(
        frequencies,
        sequence_length,
        at::kFloat,
        "frequencies",
        launch_rope_sincos_native_fp16);
}

static at::Tensor rope_sincos_fixed_d3_d4_fp16(
    at::Tensor phase_increments,
    int64_t sequence_length) {
    return rope_sincos_fp16(
        phase_increments,
        sequence_length,
        at::kInt,
        "phase_increments",
        launch_rope_sincos_fixed_d3_d4_fp16);
}

static at::Tensor rope_sincos_fixed_half_turn_d5_d6_fp16(
    at::Tensor phase_increments,
    int64_t sequence_length) {
    TORCH_CHECK(
        (phase_increments.numel() % 4) == 0,
        "phase_increments must contain a multiple of four elements");
    return rope_sincos_fp16(
        phase_increments,
        sequence_length,
        at::kInt,
        "phase_increments",
        launch_rope_sincos_fixed_half_turn_d5_d6_fp16);
}

static void check_rope_phase_table(
    const at::Tensor& phase_increments,
    const at::Tensor& phase_table) {
    CHECK_CUDA(phase_table);
    CHECK_CONTIGUOUS(phase_table);
    TORCH_CHECK(
        phase_table.scalar_type() == at::kHalf,
        "phase_table must be float16");
    TORCH_CHECK(
        phase_table.dim() == 2
            && phase_table.size(0) == 128
            && phase_table.size(1) == 2,
        "phase_table must have shape [128, 2]");
    TORCH_CHECK(
        phase_table.get_device() == phase_increments.get_device(),
        "phase_table and phase_increments must be on the same CUDA device");
}

static at::Tensor rope_sincos_fixed_lut_fp16(
    at::Tensor phase_increments,
    at::Tensor phase_table,
    int64_t sequence_length) {
    at::Tensor output = allocate_rope_sincos_fp16_output(
        phase_increments,
        sequence_length,
        at::kInt,
        "phase_increments");
    check_rope_phase_table(phase_increments, phase_table);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_rope_sincos_fixed_lut_fp16(
        output.data_ptr(),
        phase_increments.data_ptr(),
        phase_table.data_ptr(),
        static_cast<int>(sequence_length),
        static_cast<int>(phase_increments.numel()),
        stream);
    check_spline_cuda_launch(stream);
    return output;
}

static at::Tensor rope_sincos_compute_fp16(
    at::Tensor parameters,
    int64_t sequence_length,
    int64_t iterations,
    at::ScalarType parameter_type,
    const char* parameter_name,
    RopeSincosComputeLauncher launcher) {
    at::Tensor output = allocate_rope_sincos_fp16_output(
        parameters, sequence_length, parameter_type, parameter_name);
    TORCH_CHECK(iterations > 0, "iterations must be positive");
    TORCH_CHECK(
        iterations <= std::numeric_limits<int>::max(),
        "iterations is too large for the CUDA launcher");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        output.data_ptr(),
        parameters.data_ptr(),
        static_cast<int>(sequence_length),
        static_cast<int>(parameters.numel()),
        static_cast<int>(iterations),
        stream);
    check_spline_cuda_launch(stream);
    return output;
}

static at::Tensor rope_sincos_native_fp16_compute(
    at::Tensor frequencies,
    int64_t sequence_length,
    int64_t iterations) {
    return rope_sincos_compute_fp16(
        frequencies,
        sequence_length,
        iterations,
        at::kFloat,
        "frequencies",
        launch_rope_sincos_native_fp16_compute);
}

static at::Tensor rope_sincos_fixed_d3_d4_fp16_compute(
    at::Tensor phase_increments,
    int64_t sequence_length,
    int64_t iterations) {
    return rope_sincos_compute_fp16(
        phase_increments,
        sequence_length,
        iterations,
        at::kInt,
        "phase_increments",
        launch_rope_sincos_fixed_d3_d4_fp16_compute);
}

static at::Tensor rope_sincos_fixed_half_turn_d5_d6_fp16_compute(
    at::Tensor phase_increments,
    int64_t sequence_length,
    int64_t iterations) {
    TORCH_CHECK(
        (phase_increments.numel() % 4) == 0,
        "phase_increments must contain a multiple of four elements");
    return rope_sincos_compute_fp16(
        phase_increments,
        sequence_length,
        iterations,
        at::kInt,
        "phase_increments",
        launch_rope_sincos_fixed_half_turn_d5_d6_fp16_compute);
}

static at::Tensor rope_sincos_fixed_lut_fp16_compute(
    at::Tensor phase_increments,
    at::Tensor phase_table,
    int64_t sequence_length,
    int64_t iterations) {
    at::Tensor output = allocate_rope_sincos_fp16_output(
        phase_increments,
        sequence_length,
        at::kInt,
        "phase_increments");
    check_rope_phase_table(phase_increments, phase_table);
    TORCH_CHECK(iterations > 0, "iterations must be positive");
    TORCH_CHECK(
        iterations <= std::numeric_limits<int>::max(),
        "iterations is too large for the CUDA launcher");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_rope_sincos_fixed_lut_fp16_compute(
        output.data_ptr(),
        phase_increments.data_ptr(),
        phase_table.data_ptr(),
        static_cast<int>(sequence_length),
        static_cast<int>(phase_increments.numel()),
        static_cast<int>(iterations),
        stream);
    check_spline_cuda_launch(stream);
    return output;
}

using RopeApplyLauncher = void (*)(
    void*,
    void*,
    const void*,
    const void*,
    const void*,
    int,
    int,
    int,
    int,
    int,
    cudaStream_t);

struct RopeApplyShape {
    int batch_size;
    int sequence_length;
    int q_head_count;
    int k_head_count;
    int head_dim;
};

static RopeApplyShape check_rope_apply_fp16_inputs(
    const at::Tensor& q,
    const at::Tensor& k) {
    CHECK_CUDA(q);
    CHECK_CUDA(k);
    CHECK_CONTIGUOUS(q);
    CHECK_CONTIGUOUS(k);
    TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be float16");
    TORCH_CHECK(k.scalar_type() == at::kHalf, "k must be float16");
    TORCH_CHECK(q.dim() == 4, "q must have shape [batch, sequence, heads, head_dim]");
    TORCH_CHECK(k.dim() == 4, "k must have shape [batch, sequence, heads, head_dim]");
    TORCH_CHECK(q.get_device() == k.get_device(), "q and k must be on the same CUDA device");
    TORCH_CHECK(q.size(0) == k.size(0), "q and k batch sizes must match");
    TORCH_CHECK(q.size(1) == k.size(1), "q and k sequence lengths must match");
    TORCH_CHECK(q.size(3) == k.size(3), "q and k head dimensions must match");
    TORCH_CHECK(q.size(2) > 0 && k.size(2) > 0, "q and k must contain at least one head");
    TORCH_CHECK(q.size(3) > 0 && (q.size(3) % 8) == 0, "head_dim must be a positive multiple of eight");
    TORCH_CHECK(q.size(0) <= 65535, "batch size is too large for the CUDA grid");
    TORCH_CHECK(q.size(1) <= std::numeric_limits<int>::max(), "sequence length is too large for the CUDA launcher");
    TORCH_CHECK(q.size(2) <= std::numeric_limits<int>::max(), "q head count is too large for the CUDA launcher");
    TORCH_CHECK(k.size(2) <= std::numeric_limits<int>::max(), "k head count is too large for the CUDA launcher");
    TORCH_CHECK(q.size(3) <= std::numeric_limits<int>::max(), "head_dim is too large for the CUDA launcher");
    constexpr int64_t kValuesPerVector = 8;
    constexpr int64_t kMaxVectorCount = std::numeric_limits<int>::max();
    TORCH_CHECK(q.numel() / kValuesPerVector <= kMaxVectorCount, "q is too large for 32-bit CUDA indexing");
    TORCH_CHECK(k.numel() / kValuesPerVector <= kMaxVectorCount, "k is too large for 32-bit CUDA indexing");
    return {
        static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(2)),
        static_cast<int>(k.size(2)),
        static_cast<int>(q.size(3)),
    };
}

static std::vector<at::Tensor> allocate_rope_apply_fp16_outputs(
    const at::Tensor& q,
    const at::Tensor& k,
    const RopeApplyShape& shape) {
    return {
        at::empty(
            {shape.batch_size,
             shape.q_head_count,
             shape.sequence_length,
             shape.head_dim},
            q.options()),
        at::empty(
            {shape.batch_size,
             shape.k_head_count,
             shape.sequence_length,
             shape.head_dim},
            k.options()),
    };
}

static std::vector<at::Tensor> rope_apply_parameter_fp16(
    at::Tensor q,
    at::Tensor k,
    at::Tensor parameters,
    at::ScalarType parameter_type,
    const char* parameter_name,
    RopeApplyLauncher launcher) {
    const RopeApplyShape shape = check_rope_apply_fp16_inputs(q, k);
    CHECK_CUDA(parameters);
    CHECK_CONTIGUOUS(parameters);
    TORCH_CHECK(parameters.dim() == 1, parameter_name, " must be one-dimensional");
    TORCH_CHECK(parameters.scalar_type() == parameter_type, parameter_name, " has the wrong dtype");
    TORCH_CHECK(parameters.get_device() == q.get_device(), parameter_name, " must be on the same CUDA device as q and k");
    TORCH_CHECK(parameters.numel() == shape.head_dim / 2, parameter_name, " must contain head_dim / 2 elements");

    std::vector<at::Tensor> outputs =
        allocate_rope_apply_fp16_outputs(q, k, shape);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        outputs[0].data_ptr(),
        outputs[1].data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        parameters.data_ptr(),
        shape.batch_size,
        shape.sequence_length,
        shape.q_head_count,
        shape.k_head_count,
        shape.head_dim,
        stream);
    check_spline_cuda_launch(stream);
    return outputs;
}

static std::vector<at::Tensor> rope_apply_cached_fp16(
    at::Tensor q,
    at::Tensor k,
    at::Tensor rope_table) {
    const RopeApplyShape shape = check_rope_apply_fp16_inputs(q, k);
    CHECK_CUDA(rope_table);
    CHECK_CONTIGUOUS(rope_table);
    TORCH_CHECK(rope_table.scalar_type() == at::kHalf, "rope_table must be float16");
    TORCH_CHECK(rope_table.get_device() == q.get_device(), "rope_table must be on the same CUDA device as q and k");
    TORCH_CHECK(
        rope_table.dim() == 3
            && rope_table.size(0) == 2
            && rope_table.size(1) == shape.sequence_length
            && rope_table.size(2) == shape.head_dim / 2,
        "rope_table must have shape [2, sequence_length, head_dim / 2]");

    std::vector<at::Tensor> outputs =
        allocate_rope_apply_fp16_outputs(q, k, shape);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_rope_apply_cached_fp16(
        outputs[0].data_ptr(),
        outputs[1].data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        rope_table.data_ptr(),
        shape.batch_size,
        shape.sequence_length,
        shape.q_head_count,
        shape.k_head_count,
        shape.head_dim,
        stream);
    check_spline_cuda_launch(stream);
    return outputs;
}

static std::vector<at::Tensor> rope_apply_native_fp16(
    at::Tensor q,
    at::Tensor k,
    at::Tensor frequencies) {
    return rope_apply_parameter_fp16(
        q,
        k,
        frequencies,
        at::kFloat,
        "frequencies",
        launch_rope_apply_native_fp16);
}

static std::vector<at::Tensor> rope_apply_fixed_half_turn_d5_d6_fp16(
    at::Tensor q,
    at::Tensor k,
    at::Tensor phase_increments) {
    return rope_apply_parameter_fp16(
        q,
        k,
        phase_increments,
        at::kInt,
        "phase_increments",
        launch_rope_apply_fixed_half_turn_d5_d6_fp16);
}

static at::Tensor sincos_compute_bf16(
    at::Tensor angles,
    int64_t iterations,
    SincosComputeLauncher launcher) {
    CHECK_CUDA(angles);
    CHECK_CONTIGUOUS(angles);
    TORCH_CHECK(angles.scalar_type() == at::kFloat, "angles must be float32");
    TORCH_CHECK(
        (angles.numel() % 4) == 0,
        "angles must contain a multiple of four elements"
    );
    TORCH_CHECK(iterations > 0, "iterations must be positive");
    TORCH_CHECK(
        iterations <= std::numeric_limits<int>::max(),
        "iterations is too large for the CUDA launcher"
    );
    TORCH_CHECK(
        angles.numel() <= std::numeric_limits<int>::max(),
        "angles has too many elements for the CUDA launcher"
    );

    std::vector<int64_t> output_sizes = angles.sizes().vec();
    output_sizes.insert(output_sizes.begin(), 2);
    at::Tensor output = at::empty(
        output_sizes, angles.options().dtype(at::kBFloat16));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        output.data_ptr(),
        angles.data_ptr(),
        static_cast<int>(angles.numel()),
        static_cast<int>(iterations),
        stream
    );
    check_spline_cuda_launch(stream);
    return output;
}

static at::Tensor sincos_native_bf16_compute(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_bf16(
        angles, iterations, launch_sincos_native_bf16_compute);
}

static at::Tensor sincos_d3_d4_bf16_compute(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_bf16(
        angles, iterations, launch_sincos_d3_d4_bf16_compute);
}

static at::Tensor sincos_d3_d4_quarter_turn_bf16_compute(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_bf16(
        angles,
        iterations,
        launch_sincos_d3_d4_quarter_turn_bf16_compute);
}

static at::Tensor sincos_d3_d4_half_turn_bf16_compute(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_bf16(
        angles,
        iterations,
        launch_sincos_d3_d4_half_turn_bf16_compute);
}

static at::Tensor sincos_d5_d6_half_turn_bf16_compute(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_bf16(
        angles,
        iterations,
        launch_sincos_d5_d6_half_turn_bf16_compute);
}

static at::Tensor sincos_d5_d4_half_turn_fp16_bf16_compute(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_bf16(
        angles,
        iterations,
        launch_sincos_d5_d4_half_turn_fp16_bf16_compute);
}

static at::Tensor sincos_compute_fp16(
    at::Tensor angles,
    int64_t iterations,
    SincosComputeLauncher launcher) {
    CHECK_CUDA(angles);
    CHECK_CONTIGUOUS(angles);
    TORCH_CHECK(angles.scalar_type() == at::kFloat, "angles must be float32");
    TORCH_CHECK(
        (angles.numel() % 4) == 0,
        "angles must contain a multiple of four elements"
    );
    TORCH_CHECK(iterations > 0, "iterations must be positive");
    TORCH_CHECK(
        iterations <= std::numeric_limits<int>::max(),
        "iterations is too large for the CUDA launcher"
    );
    TORCH_CHECK(
        angles.numel() <= std::numeric_limits<int>::max(),
        "angles has too many elements for the CUDA launcher"
    );

    std::vector<int64_t> output_sizes = angles.sizes().vec();
    output_sizes.insert(output_sizes.begin(), 2);
    at::Tensor output = at::empty(
        output_sizes, angles.options().dtype(at::kHalf));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        output.data_ptr(),
        angles.data_ptr(),
        static_cast<int>(angles.numel()),
        static_cast<int>(iterations),
        stream
    );
    check_spline_cuda_launch(stream);
    return output;
}

static at::Tensor sincos_native_fp16_compute(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_fp16(
        angles, iterations, launch_sincos_native_fp16_compute);
}

static at::Tensor sincos_d3_d4_quarter_turn_fp16_compute(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_fp16(
        angles,
        iterations,
        launch_sincos_d3_d4_quarter_turn_fp16_compute);
}

static at::Tensor sincos_d3_d4_half_turn_fp16_compute(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_fp16(
        angles,
        iterations,
        launch_sincos_d3_d4_half_turn_fp16_compute);
}

static at::Tensor sincos_d5_d4_half_turn_fp16_compute(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_fp16(
        angles, iterations, launch_sincos_d5_d4_half_turn_fp16_compute);
}

static at::Tensor sincos_d5_d6_half_turn_fp16_compute(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_fp16(
        angles, iterations, launch_sincos_d5_d6_half_turn_fp16_compute);
}

static at::Tensor sincos_d7_d6_half_turn_fp16_compute(
    at::Tensor angles, int64_t iterations) {
    return sincos_compute_fp16(
        angles, iterations, launch_sincos_d7_d6_half_turn_fp16_compute);
}

// =============================================================================
// Generic FWD/BWD wrappers with FP16/BF16 dtype dispatch
// =============================================================================

#define DEF_FWD(NAME, LAUNCHER_FP16)                                       \
at::Tensor NAME(at::Tensor x) {                                            \
    CHECK_INPUT(x);                                                         \
    at::Tensor out = at::empty_like(x);                                     \
    auto s = at::cuda::getCurrentCUDAStream().stream();                     \
    if (x.scalar_type() == at::kBFloat16)                                   \
        LAUNCHER_FP16##_bf16(out.data_ptr(), x.data_ptr(), x.numel(), s);   \
    else                                                                    \
        LAUNCHER_FP16(out.data_ptr(), x.data_ptr(), x.numel(), s);          \
    check_spline_cuda_launch(s);                                            \
    return out;                                                             \
}

#define DEF_BWD(NAME, LAUNCHER_FP16)                                       \
at::Tensor NAME(at::Tensor grad_out, at::Tensor x) {                       \
    CHECK_INPUT(grad_out); CHECK_INPUT(x);                                  \
    at::Tensor grad_in = at::empty_like(x);                                 \
    auto s = at::cuda::getCurrentCUDAStream().stream();                     \
    if (x.scalar_type() == at::kBFloat16)                                   \
        LAUNCHER_FP16##_bf16(grad_in.data_ptr(), grad_out.data_ptr(),       \
                              x.data_ptr(), x.numel(), s);                  \
    else                                                                    \
        LAUNCHER_FP16(grad_in.data_ptr(), grad_out.data_ptr(),              \
                      x.data_ptr(), x.numel(), s);                          \
    check_spline_cuda_launch(s);                                            \
    return grad_in;                                                         \
}

using UnaryLauncher = void(*)(void*, const void*, int, cudaStream_t);
using BinaryLauncher = void(*)(void*, const void*, const void*, int, cudaStream_t);
using PackedBinaryLauncher = void(*)(void*, const void*, int, int, cudaStream_t);
using PackedBwdLauncher = void(*)(
    void*, const void*, const void*, int, int, cudaStream_t);
using SwishMulBwdLauncher = void(*)(
    void*, void*, const void*, const void*, const void*, int, cudaStream_t);

static int normalize_coeff_source_id(int64_t coeff_source_id) {
    TORCH_CHECK(
        coeff_source_id == 0 || coeff_source_id == 1,
        "coeff_source_id must be 0 (current) or 1 (sollya)"
    );
    return static_cast<int>(coeff_source_id);
}

static UnaryLauncher resolve_sigmoid_fwd_launcher(
        at::ScalarType dtype, int degree, int coeff_source_id) {
    if (dtype == at::kBFloat16) {
        if (coeff_source_id == 0) {
            switch (degree) {
                case 3: return launch_sigmoid_fwd_d3_kernel_bf16;
                case 4: return launch_sigmoid_fwd_d4_kernel_bf16;
                case 5: return launch_sigmoid_fwd_d5_kernel_bf16;
                case 6: return launch_sigmoid_fwd_d6_kernel_bf16;
            }
        } else {
            switch (degree) {
                case 3: return launch_sigmoid_fwd_d3_kernel_bf16_sollya;
                case 4: return launch_sigmoid_fwd_d4_kernel_bf16_sollya;
                case 5: return launch_sigmoid_fwd_d5_kernel_bf16_sollya;
                case 6: return launch_sigmoid_fwd_d6_kernel_bf16_sollya;
            }
        }
    } else {
        TORCH_CHECK(coeff_source_id == 0, "FP16 spline kernels only support coeff_source=current");
        switch (degree) {
            case 3: return launch_sigmoid_fwd_d3_kernel;
            case 4: return launch_sigmoid_fwd_d4_kernel;
            case 5: return launch_sigmoid_fwd_d5_kernel;
            case 6: return launch_sigmoid_fwd_d6_kernel;
        }
    }
    TORCH_CHECK(false, "Unsupported sigmoid forward degree: ", degree);
    return nullptr;
}

static BinaryLauncher resolve_sigmoid_bwd_launcher(
        at::ScalarType dtype, int degree, int coeff_source_id) {
    if (dtype == at::kBFloat16) {
        if (coeff_source_id == 0) {
            switch (degree) {
                case 3: return launch_sigmoid_bwd_d3_kernel_bf16;
                case 4: return launch_sigmoid_bwd_d4_kernel_bf16;
                case 5: return launch_sigmoid_bwd_d5_kernel_bf16;
                case 6: return launch_sigmoid_bwd_d6_kernel_bf16;
            }
        } else {
            switch (degree) {
                case 3: return launch_sigmoid_bwd_d3_kernel_bf16_sollya;
                case 4: return launch_sigmoid_bwd_d4_kernel_bf16_sollya;
                case 5: return launch_sigmoid_bwd_d5_kernel_bf16_sollya;
                case 6: return launch_sigmoid_bwd_d6_kernel_bf16_sollya;
            }
        }
    } else {
        TORCH_CHECK(coeff_source_id == 0, "FP16 spline kernels only support coeff_source=current");
        switch (degree) {
            case 3: return launch_sigmoid_bwd_d3_kernel;
            case 4: return launch_sigmoid_bwd_d4_kernel;
            case 5: return launch_sigmoid_bwd_d5_kernel;
            case 6: return launch_sigmoid_bwd_d6_kernel;
        }
    }
    TORCH_CHECK(false, "Unsupported sigmoid backward degree: ", degree);
    return nullptr;
}

static UnaryLauncher resolve_swish_fwd_launcher(at::ScalarType dtype, int degree, int coeff_source_id) {
    if (dtype == at::kBFloat16) {
        if (coeff_source_id == 0) {
            switch (degree) {
                case 0: return launch_swish_fwd_native_kernel_bf16;
                case 3: return launch_swish_fwd_d3_kernel_bf16;
                case 4: return launch_swish_fwd_d4_kernel_bf16;
                case 5: return launch_swish_fwd_d5_kernel_bf16;
                case 6: return launch_swish_fwd_d6_kernel_bf16;
            }
        } else {
            switch (degree) {
                case 3: return launch_swish_fwd_d3_kernel_bf16_sollya;
                case 4: return launch_swish_fwd_d4_kernel_bf16_sollya;
                case 5: return launch_swish_fwd_d5_kernel_bf16_sollya;
                case 6: return launch_swish_fwd_d6_kernel_bf16_sollya;
            }
        }
    } else {
        TORCH_CHECK(coeff_source_id == 0, "FP16 spline kernels only support coeff_source=current");
        switch (degree) {
            case 0: return launch_swish_fwd_native_kernel;
            case 3: return launch_swish_fwd_d3_kernel;
            case 4: return launch_swish_fwd_d4_kernel;
            case 5: return launch_swish_fwd_d5_kernel;
            case 6: return launch_swish_fwd_d6_kernel;
        }
    }
    TORCH_CHECK(false, "Unsupported swish forward degree: ", degree);
    return nullptr;
}

static BinaryLauncher resolve_swish_mul_fwd_launcher(at::ScalarType dtype, int degree, int coeff_source_id) {
    if (dtype == at::kBFloat16) {
        if (coeff_source_id == 0) {
            switch (degree) {
                case 0: return launch_swish_mul_fwd_native_kernel_bf16;
                case 3: return launch_swish_mul_fwd_d3_kernel_bf16;
                case 4: return launch_swish_mul_fwd_d4_kernel_bf16;
                case 5: return launch_swish_mul_fwd_d5_kernel_bf16;
                case 6: return launch_swish_mul_fwd_d6_kernel_bf16;
            }
        } else {
            switch (degree) {
                case 3: return launch_swish_mul_fwd_d3_kernel_bf16_sollya;
                case 4: return launch_swish_mul_fwd_d4_kernel_bf16_sollya;
                case 5: return launch_swish_mul_fwd_d5_kernel_bf16_sollya;
                case 6: return launch_swish_mul_fwd_d6_kernel_bf16_sollya;
            }
        }
    } else {
        TORCH_CHECK(coeff_source_id == 0, "FP16 spline kernels only support coeff_source=current");
        switch (degree) {
            case 0: return launch_swish_mul_fwd_native_kernel;
            case 3: return launch_swish_mul_fwd_d3_kernel;
            case 4: return launch_swish_mul_fwd_d4_kernel;
            case 5: return launch_swish_mul_fwd_d5_kernel;
            case 6: return launch_swish_mul_fwd_d6_kernel;
        }
    }
    TORCH_CHECK(false, "Unsupported swish_mul forward degree: ", degree);
    return nullptr;
}

static PackedBinaryLauncher resolve_swish_mul_packed_fwd_launcher(
        at::ScalarType dtype, int degree, int coeff_source_id) {
    if (dtype == at::kBFloat16) {
        if (coeff_source_id == 0) {
            switch (degree) {
                case 0: return launch_swish_mul_packed_fwd_native_kernel_bf16;
                case 3: return launch_swish_mul_packed_fwd_d3_kernel_bf16;
                case 4: return launch_swish_mul_packed_fwd_d4_kernel_bf16;
                case 5: return launch_swish_mul_packed_fwd_d5_kernel_bf16;
                case 6: return launch_swish_mul_packed_fwd_d6_kernel_bf16;
            }
        } else {
            switch (degree) {
                case 3: return launch_swish_mul_packed_fwd_d3_kernel_bf16_sollya;
                case 4: return launch_swish_mul_packed_fwd_d4_kernel_bf16_sollya;
                case 5: return launch_swish_mul_packed_fwd_d5_kernel_bf16_sollya;
                case 6: return launch_swish_mul_packed_fwd_d6_kernel_bf16_sollya;
            }
        }
    } else {
        TORCH_CHECK(coeff_source_id == 0, "FP16 spline kernels only support coeff_source=current");
        switch (degree) {
            case 0: return launch_swish_mul_packed_fwd_native_kernel;
            case 3: return launch_swish_mul_packed_fwd_d3_kernel;
            case 4: return launch_swish_mul_packed_fwd_d4_kernel;
            case 5: return launch_swish_mul_packed_fwd_d5_kernel;
            case 6: return launch_swish_mul_packed_fwd_d6_kernel;
        }
    }
    TORCH_CHECK(false, "Unsupported swish_mul packed forward degree: ", degree);
    return nullptr;
}

static PackedBwdLauncher resolve_swish_mul_packed_bwd_launcher(
        at::ScalarType dtype, int degree, int coeff_source_id) {
    if (dtype == at::kBFloat16) {
        if (coeff_source_id == 0) {
            switch (degree) {
                case 0: return launch_swish_mul_packed_bwd_native_kernel_bf16;
                case 3: return launch_swish_mul_packed_bwd_d3_kernel_bf16;
                case 4: return launch_swish_mul_packed_bwd_d4_kernel_bf16;
                case 5: return launch_swish_mul_packed_bwd_d5_kernel_bf16;
                case 6: return launch_swish_mul_packed_bwd_d6_kernel_bf16;
            }
        } else {
            switch (degree) {
                case 3: return launch_swish_mul_packed_bwd_d3_kernel_bf16_sollya;
                case 4: return launch_swish_mul_packed_bwd_d4_kernel_bf16_sollya;
                case 5: return launch_swish_mul_packed_bwd_d5_kernel_bf16_sollya;
                case 6: return launch_swish_mul_packed_bwd_d6_kernel_bf16_sollya;
            }
        }
    } else {
        TORCH_CHECK(coeff_source_id == 0, "FP16 spline kernels only support coeff_source=current");
        switch (degree) {
            case 0: return launch_swish_mul_packed_bwd_native_kernel;
            case 3: return launch_swish_mul_packed_bwd_d3_kernel;
            case 4: return launch_swish_mul_packed_bwd_d4_kernel;
            case 5: return launch_swish_mul_packed_bwd_d5_kernel;
            case 6: return launch_swish_mul_packed_bwd_d6_kernel;
        }
    }
    TORCH_CHECK(false, "Unsupported swish_mul packed backward degree: ", degree);
    return nullptr;
}

static SwishMulBwdLauncher resolve_swish_mul_bwd_launcher(
        at::ScalarType dtype, int degree, int coeff_source_id) {
    if (dtype == at::kBFloat16) {
        if (coeff_source_id == 0) {
            switch (degree) {
                case 0: return launch_swish_mul_bwd_native_kernel_bf16;
                case 3: return launch_swish_mul_bwd_d3_kernel_bf16;
                case 4: return launch_swish_mul_bwd_d4_kernel_bf16;
                case 5: return launch_swish_mul_bwd_d5_kernel_bf16;
                case 6: return launch_swish_mul_bwd_d6_kernel_bf16;
            }
        } else {
            switch (degree) {
                case 3: return launch_swish_mul_bwd_d3_kernel_bf16_sollya;
                case 4: return launch_swish_mul_bwd_d4_kernel_bf16_sollya;
                case 5: return launch_swish_mul_bwd_d5_kernel_bf16_sollya;
                case 6: return launch_swish_mul_bwd_d6_kernel_bf16_sollya;
            }
        }
    } else {
        TORCH_CHECK(coeff_source_id == 0, "FP16 spline kernels only support coeff_source=current");
        switch (degree) {
            case 0: return launch_swish_mul_bwd_native_kernel;
            case 3: return launch_swish_mul_bwd_d3_kernel;
            case 4: return launch_swish_mul_bwd_d4_kernel;
            case 5: return launch_swish_mul_bwd_d5_kernel;
            case 6: return launch_swish_mul_bwd_d6_kernel;
        }
    }
    TORCH_CHECK(false, "Unsupported swish_mul backward degree: ", degree);
    return nullptr;
}

static BinaryLauncher resolve_swish_bwd_launcher(at::ScalarType dtype, int degree, int coeff_source_id) {
    if (dtype == at::kBFloat16) {
        if (coeff_source_id == 0) {
            switch (degree) {
                case 0: return launch_swish_bwd_native_kernel_bf16;
                case 3: return launch_swish_bwd_d3_kernel_bf16;
                case 4: return launch_swish_bwd_d4_kernel_bf16;
                case 5: return launch_swish_bwd_d5_kernel_bf16;
                case 6: return launch_swish_bwd_d6_kernel_bf16;
            }
        } else {
            switch (degree) {
                case 3: return launch_swish_bwd_d3_kernel_bf16_sollya;
                case 4: return launch_swish_bwd_d4_kernel_bf16_sollya;
                case 5: return launch_swish_bwd_d5_kernel_bf16_sollya;
                case 6: return launch_swish_bwd_d6_kernel_bf16_sollya;
            }
        }
    } else {
        TORCH_CHECK(coeff_source_id == 0, "FP16 spline kernels only support coeff_source=current");
        switch (degree) {
            case 0: return launch_swish_bwd_native_kernel;
            case 3: return launch_swish_bwd_d3_kernel;
            case 4: return launch_swish_bwd_d4_kernel;
            case 5: return launch_swish_bwd_d5_kernel;
            case 6: return launch_swish_bwd_d6_kernel;
        }
    }
    TORCH_CHECK(false, "Unsupported swish backward degree: ", degree);
    return nullptr;
}

static UnaryLauncher resolve_gelu_fwd_launcher(at::ScalarType dtype, int degree, int coeff_source_id) {
    if (dtype == at::kBFloat16) {
        if (coeff_source_id == 0) {
            switch (degree) {
                case 3: return launch_gelu_fwd_d3_kernel_bf16;
                case 4: return launch_gelu_fwd_d4_kernel_bf16;
                case 5: return launch_gelu_fwd_d5_kernel_bf16;
                case 6: return launch_gelu_fwd_d6_kernel_bf16;
            }
        } else {
            switch (degree) {
                case 3: return launch_gelu_fwd_d3_kernel_bf16_sollya;
                case 4: return launch_gelu_fwd_d4_kernel_bf16_sollya;
                case 5: return launch_gelu_fwd_d5_kernel_bf16_sollya;
                case 6: return launch_gelu_fwd_d6_kernel_bf16_sollya;
            }
        }
    } else {
        TORCH_CHECK(coeff_source_id == 0, "FP16 spline kernels only support coeff_source=current");
        switch (degree) {
            case 3: return launch_gelu_fwd_d3_kernel;
            case 4: return launch_gelu_fwd_d4_kernel;
            case 5: return launch_gelu_fwd_d5_kernel;
            case 6: return launch_gelu_fwd_d6_kernel;
        }
    }
    TORCH_CHECK(false, "Unsupported GeLU forward degree: ", degree);
    return nullptr;
}

static BinaryLauncher resolve_gelu_bwd_launcher(at::ScalarType dtype, int degree, int coeff_source_id) {
    if (dtype == at::kBFloat16) {
        if (coeff_source_id == 0) {
            switch (degree) {
                case 3: return launch_gelu_bwd_d3_kernel_bf16;
                case 4: return launch_gelu_bwd_d4_kernel_bf16;
                case 5: return launch_gelu_bwd_d5_kernel_bf16;
                case 6: return launch_gelu_bwd_d6_kernel_bf16;
            }
        } else {
            switch (degree) {
                case 3: return launch_gelu_bwd_d3_kernel_bf16_sollya;
                case 4: return launch_gelu_bwd_d4_kernel_bf16_sollya;
                case 5: return launch_gelu_bwd_d5_kernel_bf16_sollya;
                case 6: return launch_gelu_bwd_d6_kernel_bf16_sollya;
            }
        }
    } else {
        TORCH_CHECK(coeff_source_id == 0, "FP16 spline kernels only support coeff_source=current");
        switch (degree) {
            case 3: return launch_gelu_bwd_d3_kernel;
            case 4: return launch_gelu_bwd_d4_kernel;
            case 5: return launch_gelu_bwd_d5_kernel;
            case 6: return launch_gelu_bwd_d6_kernel;
        }
    }
    TORCH_CHECK(false, "Unsupported GeLU backward degree: ", degree);
    return nullptr;
}

// --- SIGMOID --- (FP16 name used as base; _bf16 suffix appended automatically for BF16)
DEF_FWD(sigmoid_fwd,     launch_sigmoid_fwd_d3_kernel)    // default = D3
DEF_FWD(sigmoid_fwd_d3,  launch_sigmoid_fwd_d3_kernel)
DEF_FWD(sigmoid_fwd_d4,  launch_sigmoid_fwd_d4_kernel)
DEF_FWD(sigmoid_fwd_d5,  launch_sigmoid_fwd_d5_kernel)
DEF_FWD(sigmoid_fwd_d6,  launch_sigmoid_fwd_d6_kernel)

DEF_BWD(sigmoid_bwd,     launch_sigmoid_bwd_d4_kernel)    // default = D4
DEF_BWD(sigmoid_bwd_d3,  launch_sigmoid_bwd_d3_kernel)
DEF_BWD(sigmoid_bwd_d4,  launch_sigmoid_bwd_d4_kernel)
DEF_BWD(sigmoid_bwd_d5,  launch_sigmoid_bwd_d5_kernel)
DEF_BWD(sigmoid_bwd_d6,  launch_sigmoid_bwd_d6_kernel)

// --- TANH ---
DEF_FWD(tanh_fwd,        launch_tanh_fwd_d3_kernel)       // default = D3
DEF_FWD(tanh_fwd_d3,     launch_tanh_fwd_d3_kernel)
DEF_FWD(tanh_fwd_d4,     launch_tanh_fwd_d4_kernel)
DEF_FWD(tanh_fwd_d5,     launch_tanh_fwd_d5_kernel)
DEF_FWD(tanh_fwd_d6,     launch_tanh_fwd_d6_kernel)

DEF_BWD(tanh_bwd,        launch_tanh_bwd_d4_kernel)       // default = D4
DEF_BWD(tanh_bwd_d3,     launch_tanh_bwd_d3_kernel)
DEF_BWD(tanh_bwd_d4,     launch_tanh_bwd_d4_kernel)
DEF_BWD(tanh_bwd_d5,     launch_tanh_bwd_d5_kernel)
DEF_BWD(tanh_bwd_d6,     launch_tanh_bwd_d6_kernel)

// --- SWISH ---
DEF_FWD(swish_fwd,       launch_swish_fwd_d3_kernel)      // default = D3
DEF_FWD(swish_fwd_d3,    launch_swish_fwd_d3_kernel)
DEF_FWD(swish_fwd_d4,    launch_swish_fwd_d4_kernel)
DEF_FWD(swish_fwd_d5,    launch_swish_fwd_d5_kernel)
DEF_FWD(swish_fwd_d6,    launch_swish_fwd_d6_kernel)

DEF_BWD(swish_bwd,       launch_swish_bwd_d4_kernel)      // default = D4
DEF_BWD(swish_bwd_d3,    launch_swish_bwd_d3_kernel)
DEF_BWD(swish_bwd_d4,    launch_swish_bwd_d4_kernel)
DEF_BWD(swish_bwd_d5,    launch_swish_bwd_d5_kernel)
DEF_BWD(swish_bwd_d6,    launch_swish_bwd_d6_kernel)

// --- ALGEBRAIC BACKWARD (takes cached forward output y, not raw input x) ---
DEF_BWD(sigmoid_bwd_alg, launch_sigmoid_bwd_alg_kernel)
DEF_BWD(tanh_bwd_alg,    launch_tanh_bwd_alg_kernel)

// --- FP16-ONLY macros for ops without BF16 kernels (hybrid, fused, deriv) ---
#define CHECK_HALF(x) TORCH_CHECK(x.scalar_type() == at::kHalf, #x " must be float16 (no BF16 support for this op)")
#define CHECK_INPUT_FP16(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_HALF(x)

#define DEF_FWD_FP16(NAME, LAUNCHER)                                       \
at::Tensor NAME(at::Tensor x) {                                            \
    CHECK_INPUT_FP16(x);                                                    \
    at::Tensor out = at::empty_like(x);                                     \
    auto s = at::cuda::getCurrentCUDAStream().stream();                     \
    LAUNCHER(out.data_ptr(), x.data_ptr(), x.numel(), s);                   \
    check_spline_cuda_launch(s);                                            \
    return out;                                                             \
}

#define DEF_BWD_FP16(NAME, LAUNCHER)                                       \
at::Tensor NAME(at::Tensor grad_out, at::Tensor x) {                       \
    CHECK_INPUT_FP16(grad_out); CHECK_INPUT_FP16(x);                        \
    at::Tensor grad_in = at::empty_like(x);                                 \
    auto s = at::cuda::getCurrentCUDAStream().stream();                     \
    LAUNCHER(grad_in.data_ptr(), grad_out.data_ptr(), x.data_ptr(),         \
             x.numel(), s);                                                 \
    check_spline_cuda_launch(s);                                            \
    return grad_in;                                                         \
}

// --- HYBRID FWD: 1 SFU + 3 Polynomial (FP16 only) ---
DEF_FWD_FP16(sigmoid_fwd_hybrid, launch_sigmoid_fwd_hybrid_kernel)
DEF_FWD_FP16(tanh_fwd_hybrid,    launch_tanh_fwd_hybrid_kernel)
DEF_FWD_FP16(swish_fwd_hybrid,   launch_swish_fwd_hybrid_kernel)

// --- GELU ---
DEF_FWD(gelu_fwd,       launch_gelu_fwd_d5_kernel)      // default = D5
DEF_FWD(gelu_fwd_d3,    launch_gelu_fwd_d3_kernel)
DEF_FWD(gelu_fwd_d4,    launch_gelu_fwd_d4_kernel)
DEF_FWD(gelu_fwd_d5,    launch_gelu_fwd_d5_kernel)
DEF_FWD(gelu_fwd_d6,    launch_gelu_fwd_d6_kernel)

DEF_BWD(gelu_bwd,       launch_gelu_bwd_d5_kernel)      // default = D5
DEF_BWD(gelu_bwd_d3,    launch_gelu_bwd_d3_kernel)
DEF_BWD(gelu_bwd_d4,    launch_gelu_bwd_d4_kernel)
DEF_BWD(gelu_bwd_d5,    launch_gelu_bwd_d5_kernel)
DEF_BWD(gelu_bwd_d6,    launch_gelu_bwd_d6_kernel)

at::Tensor sigmoid_fwd_variant(
        at::Tensor x, int64_t degree, int64_t coeff_source_id) {
    CHECK_INPUT(x);
    auto launcher = resolve_sigmoid_fwd_launcher(
        x.scalar_type(),
        static_cast<int>(degree),
        normalize_coeff_source_id(coeff_source_id)
    );
    at::Tensor out = at::empty_like(x);
    auto s = at::cuda::getCurrentCUDAStream().stream();
    launcher(out.data_ptr(), x.data_ptr(), x.numel(), s);
    check_spline_cuda_launch(s);
    return out;
}

at::Tensor sigmoid_bwd_variant(
        at::Tensor grad_out,
        at::Tensor x,
        int64_t degree,
        int64_t coeff_source_id) {
    CHECK_INPUT(grad_out); CHECK_INPUT(x);
    TORCH_CHECK(
        grad_out.scalar_type() == x.scalar_type(),
        "grad_out and x must have the same dtype"
    );
    TORCH_CHECK(
        grad_out.numel() == x.numel(),
        "grad_out and x must have the same number of elements"
    );
    auto launcher = resolve_sigmoid_bwd_launcher(
        x.scalar_type(),
        static_cast<int>(degree),
        normalize_coeff_source_id(coeff_source_id)
    );
    at::Tensor grad_in = at::empty_like(x);
    auto s = at::cuda::getCurrentCUDAStream().stream();
    launcher(grad_in.data_ptr(), grad_out.data_ptr(), x.data_ptr(), x.numel(), s);
    check_spline_cuda_launch(s);
    return grad_in;
}

at::Tensor swish_fwd_variant(at::Tensor x, int64_t degree, int64_t coeff_source_id) {
    CHECK_INPUT(x);
    auto launcher = resolve_swish_fwd_launcher(
        x.scalar_type(),
        static_cast<int>(degree),
        normalize_coeff_source_id(coeff_source_id)
    );
    at::Tensor out = at::empty_like(x);
    auto s = at::cuda::getCurrentCUDAStream().stream();
    launcher(out.data_ptr(), x.data_ptr(), x.numel(), s);
    check_spline_cuda_launch(s);
    return out;
}

at::Tensor swish_mul_fwd_variant(
        at::Tensor gate, at::Tensor up, int64_t degree, int64_t coeff_source_id) {
    CHECK_INPUT(gate); CHECK_INPUT(up);
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "gate and up must have the same dtype");
    TORCH_CHECK(gate.numel() == up.numel(), "gate and up must have the same number of elements");
    auto launcher = resolve_swish_mul_fwd_launcher(
        gate.scalar_type(),
        static_cast<int>(degree),
        normalize_coeff_source_id(coeff_source_id)
    );
    at::Tensor out = at::empty_like(up);
    auto s = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        out.data_ptr(),
        gate.data_ptr(),
        up.data_ptr(),
        gate.numel(),
        s
    );
    check_spline_cuda_launch(s);
    return out;
}

at::Tensor swish_mul_packed_fwd_variant(
        at::Tensor packed, int64_t degree, int64_t coeff_source_id) {
    CHECK_INPUT(packed);
    TORCH_CHECK(packed.dim() >= 1, "packed gate/up tensor must have at least one dimension");
    int64_t packed_hidden_size = packed.size(-1);
    TORCH_CHECK((packed_hidden_size % 2) == 0, "packed gate/up last dimension must be even");
    int64_t hidden_size = packed_hidden_size / 2;
    TORCH_CHECK((hidden_size % 2) == 0, "packed SwiGLU hidden size must be even");

    std::vector<int64_t> out_sizes(packed.sizes().begin(), packed.sizes().end());
    out_sizes.back() = hidden_size;
    at::Tensor out = at::empty(out_sizes, packed.options());
    if (packed.numel() == 0) {
        return out;
    }

    int64_t rows = packed.numel() / packed_hidden_size;
    TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "packed SwiGLU row count exceeds int32");
    TORCH_CHECK(hidden_size <= std::numeric_limits<int>::max(), "packed SwiGLU hidden size exceeds int32");
    auto launcher = resolve_swish_mul_packed_fwd_launcher(
        packed.scalar_type(),
        static_cast<int>(degree),
        normalize_coeff_source_id(coeff_source_id)
    );
    auto s = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        out.data_ptr(),
        packed.data_ptr(),
        static_cast<int>(rows),
        static_cast<int>(hidden_size),
        s
    );
    check_spline_cuda_launch(s);
    return out;
}

at::Tensor swish_mul_packed_bwd_variant(
        at::Tensor grad_out,
        at::Tensor packed,
        int64_t degree,
        int64_t coeff_source_id) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(packed);
    TORCH_CHECK(
        packed.dim() >= 1,
        "packed gate/up tensor must have at least one dimension"
    );
    TORCH_CHECK(
        grad_out.dim() == packed.dim(),
        "grad_out and packed must have the same rank"
    );
    TORCH_CHECK(
        grad_out.scalar_type() == packed.scalar_type(),
        "grad_out and packed must have the same dtype"
    );
    const int64_t packed_hidden_size = packed.size(-1);
    TORCH_CHECK(
        (packed_hidden_size % 2) == 0,
        "packed gate/up last dimension must be even"
    );
    const int64_t hidden_size = packed_hidden_size / 2;
    TORCH_CHECK(
        (hidden_size % 2) == 0,
        "packed SwiGLU hidden size must be even"
    );
    TORCH_CHECK(
        grad_out.size(-1) == hidden_size,
        "grad_out last dimension must be half the packed last dimension"
    );
    for (int64_t dim = 0; dim + 1 < packed.dim(); ++dim) {
        TORCH_CHECK(
            grad_out.size(dim) == packed.size(dim),
            "grad_out and packed leading dimensions must match"
        );
    }

    at::Tensor grad_packed = at::empty_like(packed);
    if (packed.numel() == 0) {
        return grad_packed;
    }
    const int64_t rows = packed.numel() / packed_hidden_size;
    TORCH_CHECK(
        rows <= std::numeric_limits<int>::max(),
        "packed SwiGLU row count exceeds int32"
    );
    TORCH_CHECK(
        hidden_size <= std::numeric_limits<int>::max(),
        "packed SwiGLU hidden size exceeds int32"
    );
    auto launcher = resolve_swish_mul_packed_bwd_launcher(
        packed.scalar_type(),
        static_cast<int>(degree),
        normalize_coeff_source_id(coeff_source_id)
    );
    auto s = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        grad_packed.data_ptr(),
        grad_out.data_ptr(),
        packed.data_ptr(),
        static_cast<int>(rows),
        static_cast<int>(hidden_size),
        s
    );
    check_spline_cuda_launch(s);
    return grad_packed;
}

std::vector<at::Tensor> swish_mul_bwd_variant(
        at::Tensor grad_out,
        at::Tensor gate,
        at::Tensor up,
        int64_t degree,
        int64_t coeff_source_id) {
    CHECK_INPUT(grad_out); CHECK_INPUT(gate); CHECK_INPUT(up);
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "gate and up must have the same dtype");
    TORCH_CHECK(grad_out.scalar_type() == gate.scalar_type(), "grad_out and gate must have the same dtype");
    TORCH_CHECK(gate.numel() == up.numel(), "gate and up must have the same number of elements");
    TORCH_CHECK(grad_out.numel() == gate.numel(), "grad_out and gate must have the same number of elements");
    auto launcher = resolve_swish_mul_bwd_launcher(
        gate.scalar_type(),
        static_cast<int>(degree),
        normalize_coeff_source_id(coeff_source_id)
    );
    at::Tensor grad_gate = at::empty_like(gate);
    at::Tensor grad_up = at::empty_like(up);
    auto s = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        grad_gate.data_ptr(),
        grad_up.data_ptr(),
        grad_out.data_ptr(),
        gate.data_ptr(),
        up.data_ptr(),
        gate.numel(),
        s
    );
    check_spline_cuda_launch(s);
    return {grad_gate, grad_up};
}

at::Tensor swish_bwd_variant(at::Tensor grad_out, at::Tensor x, int64_t degree, int64_t coeff_source_id) {
    CHECK_INPUT(grad_out); CHECK_INPUT(x);
    auto launcher = resolve_swish_bwd_launcher(
        x.scalar_type(),
        static_cast<int>(degree),
        normalize_coeff_source_id(coeff_source_id)
    );
    at::Tensor grad_in = at::empty_like(x);
    auto s = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        grad_in.data_ptr(),
        grad_out.data_ptr(),
        x.data_ptr(),
        x.numel(),
        s
    );
    check_spline_cuda_launch(s);
    return grad_in;
}

at::Tensor gelu_fwd_variant(at::Tensor x, int64_t degree, int64_t coeff_source_id) {
    CHECK_INPUT(x);
    auto launcher = resolve_gelu_fwd_launcher(
        x.scalar_type(),
        static_cast<int>(degree),
        normalize_coeff_source_id(coeff_source_id)
    );
    at::Tensor out = at::empty_like(x);
    auto s = at::cuda::getCurrentCUDAStream().stream();
    launcher(out.data_ptr(), x.data_ptr(), x.numel(), s);
    check_spline_cuda_launch(s);
    return out;
}

at::Tensor gelu_bwd_variant(at::Tensor grad_out, at::Tensor x, int64_t degree, int64_t coeff_source_id) {
    CHECK_INPUT(grad_out); CHECK_INPUT(x);
    auto launcher = resolve_gelu_bwd_launcher(
        x.scalar_type(),
        static_cast<int>(degree),
        normalize_coeff_source_id(coeff_source_id)
    );
    at::Tensor grad_in = at::empty_like(x);
    auto s = at::cuda::getCurrentCUDAStream().stream();
    launcher(
        grad_in.data_ptr(),
        grad_out.data_ptr(),
        x.data_ptr(),
        x.numel(),
        s
    );
    check_spline_cuda_launch(s);
    return grad_in;
}

// --- HYBRID BWD: 1 SFU + 3 Polynomial (FP16 only) ---
DEF_BWD_FP16(sigmoid_bwd_hybrid, launch_sigmoid_bwd_hybrid_kernel)
DEF_BWD_FP16(tanh_bwd_hybrid,    launch_tanh_bwd_hybrid_kernel)
DEF_BWD_FP16(swish_bwd_hybrid,   launch_swish_bwd_hybrid_kernel)

// --- FUSED FWD+BWD: single pass, returns (y, grad_input) (FP16 only) ---
std::vector<at::Tensor> sigmoid_fused(at::Tensor x, at::Tensor grad_out) {
    CHECK_INPUT_FP16(x); CHECK_INPUT_FP16(grad_out);
    at::Tensor y = at::empty_like(x);
    at::Tensor gi = at::empty_like(x);
    auto s = at::cuda::getCurrentCUDAStream().stream();
    launch_sigmoid_fused_kernel(y.data_ptr(), gi.data_ptr(),
                                x.data_ptr(), grad_out.data_ptr(), x.numel(), s);
    check_spline_cuda_launch(s);
    return {y, gi};
}
std::vector<at::Tensor> tanh_fused(at::Tensor x, at::Tensor grad_out) {
    CHECK_INPUT_FP16(x); CHECK_INPUT_FP16(grad_out);
    at::Tensor y = at::empty_like(x);
    at::Tensor gi = at::empty_like(x);
    auto s = at::cuda::getCurrentCUDAStream().stream();
    launch_tanh_fused_kernel(y.data_ptr(), gi.data_ptr(),
                             x.data_ptr(), grad_out.data_ptr(), x.numel(), s);
    check_spline_cuda_launch(s);
    return {y, gi};
}
std::vector<at::Tensor> swish_fused(at::Tensor x, at::Tensor grad_out) {
    CHECK_INPUT_FP16(x); CHECK_INPUT_FP16(grad_out);
    at::Tensor y = at::empty_like(x);
    at::Tensor gi = at::empty_like(x);
    auto s = at::cuda::getCurrentCUDAStream().stream();
    launch_swish_fused_kernel(y.data_ptr(), gi.data_ptr(),
                              x.data_ptr(), grad_out.data_ptr(), x.numel(), s);
    check_spline_cuda_launch(s);
    return {y, gi};
}

// --- FWD WITH DERIVATIVE: returns (y, f'(x)) for standard autograd (FP16 only) ---
#define DEF_FWD_DERIV(NAME, LAUNCHER)                                       \
std::vector<at::Tensor> NAME(at::Tensor x) {                               \
    CHECK_INPUT_FP16(x);                                                    \
    at::Tensor y = at::empty_like(x);                                       \
    at::Tensor dy = at::empty_like(x);                                      \
    auto s = at::cuda::getCurrentCUDAStream().stream();                     \
    LAUNCHER(y.data_ptr(), dy.data_ptr(), x.data_ptr(),                     \
             x.numel(), s);                                                 \
    check_spline_cuda_launch(s);                                            \
    return {y, dy};                                                         \
}

DEF_FWD_DERIV(sigmoid_fwd_deriv_alg,   launch_sigmoid_fwd_deriv_alg_kernel)
DEF_FWD_DERIV(sigmoid_fwd_deriv_poly,  launch_sigmoid_fwd_deriv_poly_kernel)
DEF_FWD_DERIV(tanh_fwd_deriv_alg,      launch_tanh_fwd_deriv_alg_kernel)
DEF_FWD_DERIV(tanh_fwd_deriv_poly,     launch_tanh_fwd_deriv_poly_kernel)
DEF_FWD_DERIV(swish_fwd_deriv,         launch_swish_fwd_deriv_kernel)
DEF_FWD_DERIV(swish_fwd_deriv_poly,    launch_swish_fwd_deriv_poly_kernel)

// Trivial multiply: gi = go * saved_dy
at::Tensor multiply(at::Tensor a, at::Tensor b) {
    CHECK_INPUT(a); CHECK_INPUT(b);
    at::Tensor out = at::empty_like(a);
    auto s = at::cuda::getCurrentCUDAStream().stream();
    launch_multiply_kernel(out.data_ptr(), a.data_ptr(), b.data_ptr(), a.numel(), s);
    check_spline_cuda_launch(s);
    return out;
}

// =============================================================================
// C++ Autograd Functions — same mechanism as PyTorch's built-in ops
// Registered on AutogradCUDA dispatch key, zero Python overhead.
// =============================================================================

class SplineSigmoidFwd : public torch::autograd::Function<SplineSigmoidFwd> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, at::Tensor x) {
        at::Tensor out = at::empty_like(x);
        auto s = at::cuda::getCurrentCUDAStream().stream();
        if (x.scalar_type() == at::kBFloat16)
            launch_sigmoid_fwd_d3_kernel_bf16(out.data_ptr(), x.data_ptr(), x.numel(), s);
        else
            launch_sigmoid_fwd_d3_kernel(out.data_ptr(), x.data_ptr(), x.numel(), s);
        ctx->save_for_backward({out});  // save y, not x — algebraic BWD
        return out;
    }
    static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto y = saved[0];  // cached forward output
        auto go = grad_outputs[0].contiguous();
        at::Tensor gi = at::empty_like(y);
        auto s = at::cuda::getCurrentCUDAStream().stream();
        if (y.scalar_type() == at::kBFloat16)
            launch_sigmoid_bwd_alg_kernel_bf16(gi.data_ptr(), go.data_ptr(), y.data_ptr(), y.numel(), s);
        else
            launch_sigmoid_bwd_alg_kernel(gi.data_ptr(), go.data_ptr(), y.data_ptr(), y.numel(), s);
        return {gi};
    }
};

class SplineTanhFwd : public torch::autograd::Function<SplineTanhFwd> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, at::Tensor x) {
        at::Tensor out = at::empty_like(x);
        auto s = at::cuda::getCurrentCUDAStream().stream();
        if (x.scalar_type() == at::kBFloat16)
            launch_tanh_fwd_d3_kernel_bf16(out.data_ptr(), x.data_ptr(), x.numel(), s);
        else
            launch_tanh_fwd_d3_kernel(out.data_ptr(), x.data_ptr(), x.numel(), s);
        ctx->save_for_backward({out});  // save y, not x — algebraic BWD
        return out;
    }
    static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto y = saved[0];  // cached forward output
        auto go = grad_outputs[0].contiguous();
        at::Tensor gi = at::empty_like(y);
        auto s = at::cuda::getCurrentCUDAStream().stream();
        if (y.scalar_type() == at::kBFloat16)
            launch_tanh_bwd_alg_kernel_bf16(gi.data_ptr(), go.data_ptr(), y.data_ptr(), y.numel(), s);
        else
            launch_tanh_bwd_alg_kernel(gi.data_ptr(), go.data_ptr(), y.data_ptr(), y.numel(), s);
        return {gi};
    }
};

class SplineSwishFwd : public torch::autograd::Function<SplineSwishFwd> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, at::Tensor x) {
        ctx->save_for_backward({x});
        at::Tensor out = at::empty_like(x);
        auto s = at::cuda::getCurrentCUDAStream().stream();
        if (x.scalar_type() == at::kBFloat16)
            launch_swish_fwd_d3_kernel_bf16(out.data_ptr(), x.data_ptr(), x.numel(), s);
        else
            launch_swish_fwd_d3_kernel(out.data_ptr(), x.data_ptr(), x.numel(), s);
        return out;
    }
    static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto go = grad_outputs[0].contiguous();
        at::Tensor gi = at::empty_like(x);
        auto s = at::cuda::getCurrentCUDAStream().stream();
        if (x.scalar_type() == at::kBFloat16)
            launch_swish_bwd_d4_kernel_bf16(gi.data_ptr(), go.data_ptr(), x.data_ptr(), x.numel(), s);
        else
            launch_swish_bwd_d4_kernel(gi.data_ptr(), go.data_ptr(), x.data_ptr(), x.numel(), s);
        return {gi};
    }
};

class SplineSwishVariantFwd : public torch::autograd::Function<SplineSwishVariantFwd> {
public:
    static at::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            at::Tensor x,
            int64_t degree,
            int64_t coeff_source_id) {
        int coeff_source = normalize_coeff_source_id(coeff_source_id);
        ctx->save_for_backward({x});
        ctx->saved_data["degree"] = degree;
        ctx->saved_data["coeff_source_id"] = coeff_source;
        auto launcher = resolve_swish_fwd_launcher(
            x.scalar_type(),
            static_cast<int>(degree),
            coeff_source
        );
        at::Tensor out = at::empty_like(x);
        launcher(
            out.data_ptr(),
            x.data_ptr(),
            x.numel(),
            at::cuda::getCurrentCUDAStream().stream()
        );
        return out;
    }

    static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto go = grad_outputs[0].contiguous();
        int degree = static_cast<int>(ctx->saved_data["degree"].toInt());
        int coeff_source = static_cast<int>(
            ctx->saved_data["coeff_source_id"].toInt()
        );
        auto launcher = resolve_swish_bwd_launcher(
            x.scalar_type(),
            degree,
            coeff_source
        );
        at::Tensor gi = at::empty_like(x);
        launcher(
            gi.data_ptr(),
            go.data_ptr(),
            x.data_ptr(),
            x.numel(),
            at::cuda::getCurrentCUDAStream().stream()
        );
        return {gi, at::Tensor(), at::Tensor()};
    }
};

class SplineSwishMulVariantFwd : public torch::autograd::Function<SplineSwishMulVariantFwd> {
public:
    static at::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            at::Tensor gate,
            at::Tensor up,
            int64_t degree,
            int64_t coeff_source_id) {
        CHECK_INPUT(gate); CHECK_INPUT(up);
        TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "gate and up must have the same dtype");
        TORCH_CHECK(gate.numel() == up.numel(), "gate and up must have the same number of elements");
        int coeff_source = normalize_coeff_source_id(coeff_source_id);
        ctx->save_for_backward({gate, up});
        ctx->saved_data["degree"] = degree;
        ctx->saved_data["coeff_source_id"] = coeff_source;
        auto launcher = resolve_swish_mul_fwd_launcher(
            gate.scalar_type(),
            static_cast<int>(degree),
            coeff_source
        );
        at::Tensor out = at::empty_like(up);
        launcher(
            out.data_ptr(),
            gate.data_ptr(),
            up.data_ptr(),
            gate.numel(),
            at::cuda::getCurrentCUDAStream().stream()
        );
        return out;
    }

    static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto gate = saved[0];
        auto up = saved[1];
        auto go = grad_outputs[0].contiguous();
        int degree = static_cast<int>(ctx->saved_data["degree"].toInt());
        int coeff_source = static_cast<int>(
            ctx->saved_data["coeff_source_id"].toInt()
        );
        auto launcher = resolve_swish_mul_bwd_launcher(
            gate.scalar_type(),
            degree,
            coeff_source
        );
        at::Tensor grad_gate = at::empty_like(gate);
        at::Tensor grad_up = at::empty_like(up);
        launcher(
            grad_gate.data_ptr(),
            grad_up.data_ptr(),
            go.data_ptr(),
            gate.data_ptr(),
            up.data_ptr(),
            gate.numel(),
            at::cuda::getCurrentCUDAStream().stream()
        );
        return {grad_gate, grad_up, at::Tensor(), at::Tensor()};
    }
};

class SplineGeLUVariantFwd : public torch::autograd::Function<SplineGeLUVariantFwd> {
public:
    static at::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            at::Tensor x,
            int64_t degree,
            int64_t coeff_source_id) {
        int coeff_source = normalize_coeff_source_id(coeff_source_id);
        ctx->save_for_backward({x});
        ctx->saved_data["degree"] = degree;
        ctx->saved_data["coeff_source_id"] = coeff_source;
        auto launcher = resolve_gelu_fwd_launcher(
            x.scalar_type(),
            static_cast<int>(degree),
            coeff_source
        );
        at::Tensor out = at::empty_like(x);
        launcher(
            out.data_ptr(),
            x.data_ptr(),
            x.numel(),
            at::cuda::getCurrentCUDAStream().stream()
        );
        return out;
    }

    static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto go = grad_outputs[0].contiguous();
        int degree = static_cast<int>(ctx->saved_data["degree"].toInt());
        int coeff_source = static_cast<int>(
            ctx->saved_data["coeff_source_id"].toInt()
        );
        auto launcher = resolve_gelu_bwd_launcher(
            x.scalar_type(),
            degree,
            coeff_source
        );
        at::Tensor gi = at::empty_like(x);
        launcher(
            gi.data_ptr(),
            go.data_ptr(),
            x.data_ptr(),
            x.numel(),
            at::cuda::getCurrentCUDAStream().stream()
        );
        return {gi, at::Tensor(), at::Tensor()};
    }
};

// Thin wrappers that go through C++ autograd
static at::Tensor sigmoid_fwd_autograd(at::Tensor x) {
    return SplineSigmoidFwd::apply(x);
}
static at::Tensor tanh_fwd_autograd(at::Tensor x) {
    return SplineTanhFwd::apply(x);
}
static at::Tensor swish_fwd_autograd(at::Tensor x) {
    return SplineSwishFwd::apply(x);
}
static at::Tensor swish_fwd_variant_autograd(
        at::Tensor x, int64_t degree, int64_t coeff_source_id) {
    return SplineSwishVariantFwd::apply(x, degree, coeff_source_id);
}
static at::Tensor swish_mul_fwd_variant_autograd(
        at::Tensor gate, at::Tensor up, int64_t degree, int64_t coeff_source_id) {
    return SplineSwishMulVariantFwd::apply(gate, up, degree, coeff_source_id);
}
static at::Tensor gelu_fwd_variant_autograd(
        at::Tensor x, int64_t degree, int64_t coeff_source_id) {
    return SplineGeLUVariantFwd::apply(x, degree, coeff_source_id);
}

// =============================================================================
// TORCH_LIBRARY: schema + dispatch
// =============================================================================

TORCH_LIBRARY(spline_ops, m) {
    // FWD (autograd-enabled)
    m.def("sigmoid_fwd(Tensor x) -> Tensor");
    m.def("tanh_fwd(Tensor x) -> Tensor");
    m.def("swish_fwd(Tensor x) -> Tensor");
    // BWD (explicit, for direct calling)
    m.def("sigmoid_bwd(Tensor grad_out, Tensor x) -> Tensor");
    m.def("tanh_bwd(Tensor grad_out, Tensor x) -> Tensor");
    m.def("swish_bwd(Tensor grad_out, Tensor x) -> Tensor");
    // BWD algebraic (from cached y)
    m.def("sigmoid_bwd_alg(Tensor grad_out, Tensor y) -> Tensor");
    m.def("tanh_bwd_alg(Tensor grad_out, Tensor y) -> Tensor");
    // Degree-selectable SwiGLU ops used from torch.compile. Registering these
    // directly in C++ avoids a Python custom-op callback on every expert MLP.
    m.def(
        "swish_variant_fwd(Tensor x, int degree, int coeff_source_id) -> Tensor"
    );
    m.def(
        "swish_variant_bwd(Tensor grad_out, Tensor x, int degree, int coeff_source_id) -> Tensor"
    );
    m.def(
        "swish_mul_variant_fwd(Tensor gate, Tensor up, int degree, int coeff_source_id) -> Tensor"
    );
    m.def(
        "swish_mul_variant_bwd(Tensor grad_out, Tensor gate, Tensor up, int degree, int coeff_source_id) -> Tensor[]"
    );
    m.def(
        "swish_mul_packed_variant_fwd(Tensor packed, int degree, int coeff_source_id) -> Tensor"
    );
    m.def(
        "swish_mul_packed_native_fwd(Tensor packed, int coeff_source_id) -> Tensor"
    );
    m.def(
        "swish_mul_packed_native_bwd_fwd(Tensor packed, int degree, int coeff_source_id) -> Tensor"
    );
    m.def(
        "swish_mul_packed_variant_bwd(Tensor grad_out, Tensor packed, int degree, int coeff_source_id) -> Tensor"
    );
}

// AutogradCUDA: intercepts calls, applies C++ autograd, dispatches to CUDA
TORCH_LIBRARY_IMPL(spline_ops, AutogradCUDA, m) {
    m.impl("sigmoid_fwd", &sigmoid_fwd_autograd);
    m.impl("tanh_fwd",    &tanh_fwd_autograd);
    m.impl("swish_fwd",   &swish_fwd_autograd);
}

// CUDA: raw kernel dispatch (for BWD ops and inference)
TORCH_LIBRARY_IMPL(spline_ops, CUDA, m) {
    m.impl("sigmoid_bwd", &sigmoid_bwd);
    m.impl("tanh_bwd",    &tanh_bwd);
    m.impl("swish_bwd",   &swish_bwd);
    m.impl("sigmoid_bwd_alg", &sigmoid_bwd_alg);
    m.impl("tanh_bwd_alg",    &tanh_bwd_alg);
    m.impl("swish_variant_fwd", &swish_fwd_variant);
    m.impl("swish_variant_bwd", &swish_bwd_variant);
    m.impl("swish_mul_variant_fwd", &swish_mul_fwd_variant);
    m.impl("swish_mul_variant_bwd", &swish_mul_bwd_variant);
    m.impl("swish_mul_packed_variant_fwd", &swish_mul_packed_fwd_variant);
    m.impl(
        "swish_mul_packed_native_fwd",
        [](at::Tensor packed, int64_t coeff_source_id) {
            return swish_mul_packed_fwd_variant(packed, 0, coeff_source_id);
        }
    );
    m.impl("swish_mul_packed_native_bwd_fwd", &swish_mul_packed_fwd_variant);
    m.impl("swish_mul_packed_variant_bwd", &swish_mul_packed_bwd_variant);
}

// =============================================================================
// PYBIND11 module (direct pybind11 path — lowest overhead)
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Paired FP32 sin/cos kernels used by the RoPE microbenchmarks.
    m.def("sincos_native_f32", &sincos_native_f32, "Paired native FP32 sin/cos");
    m.def("sincos_d3_d4_f32", &sincos_d3_d4_f32, "Paired polynomial FP32 sin D3/cos D4");
    m.def("sincos_d3_d4_cycle_f32", &sincos_d3_d4_cycle_f32, "Paired polynomial FP32 sin D3/cos D4 with cycle-domain symmetry reduction");
    m.def("sincos_d3_d4_magic_bias_f32", &sincos_d3_d4_magic_bias_f32, "Paired polynomial FP32 sin D3/cos D4 with conversion-free magic-bias reduction");
    m.def("sincos_d5_d4_half_turn_ls_f32", &sincos_d5_d4_half_turn_ls_f32, "Paired half-turn polynomial FP32 sin D5/cos D4 with least-squares coefficients");
    m.def("sincos_d5_d4_half_turn_sollya_f32", &sincos_d5_d4_half_turn_sollya_f32, "Paired half-turn polynomial FP32 sin D5/cos D4 with Sollya coefficients");
    m.def("sincos_d5_d4_half_turn_sollya_fast_f32", &sincos_d5_d4_half_turn_sollya_fast_f32, "Paired half-turn polynomial FP32 sin D5/cos D4 with Sollya coefficients and one-term pi reduction");
    m.def("sincos_d5_d4_f32", &sincos_d5_d4_f32, "Paired polynomial FP32 sin D5/cos D4");
    m.def("sincos_d7_d6_f32", &sincos_d7_d6_f32, "Paired polynomial FP32 sin D7/cos D6");
    m.def("sincos_native_compute_f32", &sincos_native_compute_f32, "Compute-saturated paired native FP32 sin/cos");
    m.def("sincos_d3_d4_compute_f32", &sincos_d3_d4_compute_f32, "Compute-saturated polynomial FP32 sin D3/cos D4");
    m.def("sincos_d3_d4_cycle_compute_f32", &sincos_d3_d4_cycle_compute_f32, "Compute-saturated polynomial FP32 sin D3/cos D4 with cycle-domain symmetry reduction");
    m.def("sincos_d3_d4_magic_bias_compute_f32", &sincos_d3_d4_magic_bias_compute_f32, "Compute-saturated polynomial FP32 sin D3/cos D4 with conversion-free magic-bias reduction");
    m.def("sincos_d5_d4_half_turn_ls_compute_f32", &sincos_d5_d4_half_turn_ls_compute_f32, "Compute-saturated half-turn polynomial FP32 sin D5/cos D4 with least-squares coefficients");
    m.def("sincos_d5_d4_half_turn_sollya_compute_f32", &sincos_d5_d4_half_turn_sollya_compute_f32, "Compute-saturated half-turn polynomial FP32 sin D5/cos D4 with Sollya coefficients");
    m.def("sincos_d5_d4_half_turn_sollya_fast_compute_f32", &sincos_d5_d4_half_turn_sollya_fast_compute_f32, "Compute-saturated half-turn polynomial FP32 sin D5/cos D4 with Sollya coefficients and one-term pi reduction");
    m.def("sincos_d5_d4_compute_f32", &sincos_d5_d4_compute_f32, "Compute-saturated polynomial FP32 sin D5/cos D4");
    m.def("sincos_d7_d6_compute_f32", &sincos_d7_d6_compute_f32, "Compute-saturated polynomial FP32 sin D7/cos D6");
    m.def("sincos_native_bf16", &sincos_native_bf16, "Paired native sin/cos with BF16 output");
    m.def("sincos_d3_d4_bf16", &sincos_d3_d4_bf16, "Paired packed-BF16 polynomial sin D3/cos D4");
    m.def("sincos_d3_d4_quarter_turn_bf16", &sincos_d3_d4_quarter_turn_bf16, "Paired quarter-turn polynomial sin D3/cos D4 with packed BF16 evaluation and output");
    m.def("sincos_d3_d4_half_turn_bf16", &sincos_d3_d4_half_turn_bf16, "Paired half-turn polynomial sin D3/cos D4 with packed BF16 evaluation and output");
    m.def("sincos_d5_d6_half_turn_bf16", &sincos_d5_d6_half_turn_bf16, "Paired half-turn polynomial sin D5/cos D6 with packed BF16 evaluation and output");
    m.def("sincos_d5_d4_half_turn_fp16_bf16", &sincos_d5_d4_half_turn_fp16_bf16, "Paired half-turn polynomial sin D5/cos D4 with packed FP16 evaluation and BF16 output");
    m.def("sincos_native_fp16", &sincos_native_fp16, "Paired native sin/cos with FP16 output");
    m.def("sincos_d3_d4_quarter_turn_fp16", &sincos_d3_d4_quarter_turn_fp16, "Paired quarter-turn polynomial sin D3/cos D4 with packed FP16 evaluation and output");
    m.def("sincos_d3_d4_half_turn_fp16", &sincos_d3_d4_half_turn_fp16, "Paired half-turn polynomial sin D3/cos D4 with packed FP16 evaluation and output");
    m.def("sincos_d5_d4_half_turn_fp16", &sincos_d5_d4_half_turn_fp16, "Paired half-turn polynomial sin D5/cos D4 with packed FP16 evaluation and output");
    m.def("sincos_d5_d6_half_turn_fp16", &sincos_d5_d6_half_turn_fp16, "Paired half-turn polynomial sin D5/cos D6 with packed FP16 evaluation and output");
    m.def("sincos_d7_d6_half_turn_fp16", &sincos_d7_d6_half_turn_fp16, "Paired half-turn polynomial sin D7/cos D6 with packed FP16 evaluation and output");
    m.def("rope_sincos_native_fp16", &rope_sincos_native_fp16, "RoPE-specific native sin/cos table generation with FP16 output");
    m.def("rope_sincos_fixed_d3_d4_fp16", &rope_sincos_fixed_d3_d4_fp16, "RoPE-specific fixed-point phase reduction with packed FP16 sin D3/cos D4");
    m.def("rope_sincos_fixed_half_turn_d5_d6_fp16", &rope_sincos_fixed_half_turn_d5_d6_fp16, "RoPE-specific fixed-point half-turn reduction with FP16 sin D5/cos D6");
    m.def("rope_sincos_fixed_lut_fp16", &rope_sincos_fixed_lut_fp16, "RoPE-specific fixed-point phase lookup with packed FP16 linear correction");
    m.def("rope_sincos_native_fp16_compute", &rope_sincos_native_fp16_compute, "Compute-saturated RoPE-specific native sin/cos with FP16 output");
    m.def("rope_sincos_fixed_d3_d4_fp16_compute", &rope_sincos_fixed_d3_d4_fp16_compute, "Compute-saturated RoPE-specific fixed-point reduction with packed FP16 sin D3/cos D4");
    m.def("rope_sincos_fixed_half_turn_d5_d6_fp16_compute", &rope_sincos_fixed_half_turn_d5_d6_fp16_compute, "Compute-saturated RoPE-specific fixed-point half-turn reduction with FP16 sin D5/cos D6");
    m.def("rope_sincos_fixed_lut_fp16_compute", &rope_sincos_fixed_lut_fp16_compute, "Compute-saturated RoPE-specific fixed-point phase lookup with packed FP16 linear correction");
    m.def("rope_apply_cached_fp16", &rope_apply_cached_fp16, "Fused cached-table FP16 Q/K rotary embedding and transpose");
    m.def("rope_apply_native_fp16", &rope_apply_native_fp16, "Fused native-SFU FP16 Q/K rotary embedding and transpose");
    m.def("rope_apply_fixed_half_turn_d5_d6_fp16", &rope_apply_fixed_half_turn_d5_d6_fp16, "Fused polynomial FP16 Q/K rotary embedding and transpose");
    m.def("sincos_native_bf16_compute", &sincos_native_bf16_compute, "Compute-saturated native sin/cos with BF16 output");
    m.def("sincos_d3_d4_bf16_compute", &sincos_d3_d4_bf16_compute, "Compute-saturated packed-BF16 polynomial sin D3/cos D4");
    m.def("sincos_d3_d4_quarter_turn_bf16_compute", &sincos_d3_d4_quarter_turn_bf16_compute, "Compute-saturated quarter-turn polynomial sin D3/cos D4 with packed BF16 evaluation and output");
    m.def("sincos_d3_d4_half_turn_bf16_compute", &sincos_d3_d4_half_turn_bf16_compute, "Compute-saturated half-turn polynomial sin D3/cos D4 with packed BF16 evaluation and output");
    m.def("sincos_d5_d6_half_turn_bf16_compute", &sincos_d5_d6_half_turn_bf16_compute, "Compute-saturated half-turn polynomial sin D5/cos D6 with packed BF16 evaluation and output");
    m.def("sincos_d5_d4_half_turn_fp16_bf16_compute", &sincos_d5_d4_half_turn_fp16_bf16_compute, "Compute-saturated half-turn polynomial sin D5/cos D4 with packed FP16 evaluation and BF16 output");
    m.def("sincos_native_fp16_compute", &sincos_native_fp16_compute, "Compute-saturated native sin/cos with FP16 output");
    m.def("sincos_d3_d4_quarter_turn_fp16_compute", &sincos_d3_d4_quarter_turn_fp16_compute, "Compute-saturated quarter-turn polynomial sin D3/cos D4 with packed FP16 evaluation and output");
    m.def("sincos_d3_d4_half_turn_fp16_compute", &sincos_d3_d4_half_turn_fp16_compute, "Compute-saturated half-turn polynomial sin D3/cos D4 with packed FP16 evaluation and output");
    m.def("sincos_d5_d4_half_turn_fp16_compute", &sincos_d5_d4_half_turn_fp16_compute, "Compute-saturated half-turn polynomial sin D5/cos D4 with packed FP16 evaluation and output");
    m.def("sincos_d5_d6_half_turn_fp16_compute", &sincos_d5_d6_half_turn_fp16_compute, "Compute-saturated half-turn polynomial sin D5/cos D6 with packed FP16 evaluation and output");
    m.def("sincos_d7_d6_half_turn_fp16_compute", &sincos_d7_d6_half_turn_fp16_compute, "Compute-saturated half-turn polynomial sin D7/cos D6 with packed FP16 evaluation and output");

    // Sigmoid
    m.def("sigmoid_fwd",    &sigmoid_fwd,    "Sigmoid FWD (default D3)");
    m.def("sigmoid_fwd_d3", &sigmoid_fwd_d3, "Sigmoid FWD D3");
    m.def("sigmoid_fwd_d4", &sigmoid_fwd_d4, "Sigmoid FWD D4");
    m.def("sigmoid_fwd_d5", &sigmoid_fwd_d5, "Sigmoid FWD D5");
    m.def("sigmoid_fwd_d6", &sigmoid_fwd_d6, "Sigmoid FWD D6");
    m.def("sigmoid_fwd_variant", &sigmoid_fwd_variant, "Sigmoid FWD variant by degree/source");

    m.def("sigmoid_bwd",    &sigmoid_bwd,    "Sigmoid BWD (default D4)");
    m.def("sigmoid_bwd_d3", &sigmoid_bwd_d3, "Sigmoid BWD D3");
    m.def("sigmoid_bwd_d4", &sigmoid_bwd_d4, "Sigmoid BWD D4");
    m.def("sigmoid_bwd_d5", &sigmoid_bwd_d5, "Sigmoid BWD D5");
    m.def("sigmoid_bwd_d6", &sigmoid_bwd_d6, "Sigmoid BWD D6");
    m.def("sigmoid_bwd_variant", &sigmoid_bwd_variant, "Sigmoid BWD variant by degree/source");
    m.def("sigmoid_bwd_alg", &sigmoid_bwd_alg, "Sigmoid BWD algebraic (from y)");

    // Tanh
    m.def("tanh_fwd",    &tanh_fwd,    "Tanh FWD (default D3)");
    m.def("tanh_fwd_d3", &tanh_fwd_d3, "Tanh FWD D3");
    m.def("tanh_fwd_d4", &tanh_fwd_d4, "Tanh FWD D4");
    m.def("tanh_fwd_d5", &tanh_fwd_d5, "Tanh FWD D5");
    m.def("tanh_fwd_d6", &tanh_fwd_d6, "Tanh FWD D6");

    m.def("tanh_bwd",    &tanh_bwd,    "Tanh BWD (default D4)");
    m.def("tanh_bwd_d3", &tanh_bwd_d3, "Tanh BWD D3");
    m.def("tanh_bwd_d4", &tanh_bwd_d4, "Tanh BWD D4");
    m.def("tanh_bwd_d5", &tanh_bwd_d5, "Tanh BWD D5");
    m.def("tanh_bwd_d6", &tanh_bwd_d6, "Tanh BWD D6");
    m.def("tanh_bwd_alg", &tanh_bwd_alg, "Tanh BWD algebraic (from y)");

    // Swish
    m.def("swish_fwd",    &swish_fwd,    "Swish FWD (default D3)");
    m.def("swish_fwd_d3", &swish_fwd_d3, "Swish FWD D3");
    m.def("swish_fwd_d4", &swish_fwd_d4, "Swish FWD D4");
    m.def("swish_fwd_d5", &swish_fwd_d5, "Swish FWD D5");
    m.def("swish_fwd_d6", &swish_fwd_d6, "Swish FWD D6");

    m.def("swish_bwd",    &swish_bwd,    "Swish BWD (default D4)");
    m.def("swish_bwd_d3", &swish_bwd_d3, "Swish BWD D3");
    m.def("swish_bwd_d4", &swish_bwd_d4, "Swish BWD D4");
    m.def("swish_bwd_d5", &swish_bwd_d5, "Swish BWD D5");
    m.def("swish_bwd_d6", &swish_bwd_d6, "Swish BWD D6");
    m.def("swish_fwd_variant", &swish_fwd_variant, "Swish FWD variant by degree/source");
    m.def("swish_mul_fwd_variant", &swish_mul_fwd_variant, "Swish(gate) * up FWD variant by degree/source");
    m.def("swish_mul_packed_fwd_variant", &swish_mul_packed_fwd_variant, "Swish(packed gate) * packed up FWD variant by degree/source");
    m.def("swish_mul_packed_bwd_variant", &swish_mul_packed_bwd_variant, "Swish(packed gate) * packed up BWD variant by degree/source");
    m.def("swish_mul_bwd_variant", &swish_mul_bwd_variant, "Swish(gate) * up BWD variant by degree/source");
    m.def("swish_bwd_variant", &swish_bwd_variant, "Swish BWD variant by degree/source");
    m.def(
        "swish_ag_variant",
        &swish_fwd_variant_autograd,
        "Swish variant with C++ autograd by degree/source"
    );
    m.def(
        "swish_mul_ag_variant",
        &swish_mul_fwd_variant_autograd,
        "Swish(gate) * up variant with C++ autograd by degree/source"
    );

    // GeLU
    m.def("gelu_fwd",    &gelu_fwd,    "GeLU FWD (default D5)");
    m.def("gelu_fwd_d3", &gelu_fwd_d3, "GeLU FWD D3");
    m.def("gelu_fwd_d4", &gelu_fwd_d4, "GeLU FWD D4");
    m.def("gelu_fwd_d5", &gelu_fwd_d5, "GeLU FWD D5");
    m.def("gelu_fwd_d6", &gelu_fwd_d6, "GeLU FWD D6");

    m.def("gelu_bwd",    &gelu_bwd,    "GeLU BWD (default D5)");
    m.def("gelu_bwd_d3", &gelu_bwd_d3, "GeLU BWD D3");
    m.def("gelu_bwd_d4", &gelu_bwd_d4, "GeLU BWD D4");
    m.def("gelu_bwd_d5", &gelu_bwd_d5, "GeLU BWD D5");
    m.def("gelu_bwd_d6", &gelu_bwd_d6, "GeLU BWD D6");
    m.def("gelu_fwd_variant", &gelu_fwd_variant, "GeLU FWD variant by degree/source");
    m.def("gelu_bwd_variant", &gelu_bwd_variant, "GeLU BWD variant by degree/source");
    m.def(
        "gelu_ag_variant",
        &gelu_fwd_variant_autograd,
        "GeLU variant with C++ autograd by degree/source"
    );

    // C++ autograd path — fastest possible:
    // Python → pybind11 → C++ autograd::Function::apply → CUDA launcher
    // No Python autograd.Function, no TORCH_LIBRARY dispatcher
    m.def("sigmoid_ag", &sigmoid_fwd_autograd, "Sigmoid with C++ autograd");
    m.def("tanh_ag",    &tanh_fwd_autograd,    "Tanh with C++ autograd");
    m.def("swish_ag",   &swish_fwd_autograd,   "Swish with C++ autograd");

    // Hybrid FWD: 1 SFU + 3 Polynomial
    m.def("sigmoid_fwd_hybrid", &sigmoid_fwd_hybrid, "Sigmoid FWD hybrid 1SFU+3Poly");
    m.def("tanh_fwd_hybrid",    &tanh_fwd_hybrid,    "Tanh FWD hybrid 1SFU+3Poly");
    m.def("swish_fwd_hybrid",   &swish_fwd_hybrid,   "Swish FWD hybrid 1SFU+3Poly");

    // Hybrid BWD: 1 SFU + 3 Polynomial
    m.def("sigmoid_bwd_hybrid", &sigmoid_bwd_hybrid, "Sigmoid BWD hybrid 1SFU+3Poly");
    m.def("tanh_bwd_hybrid",    &tanh_bwd_hybrid,    "Tanh BWD hybrid 1SFU+3Poly");
    m.def("swish_bwd_hybrid",   &swish_bwd_hybrid,   "Swish BWD hybrid 1SFU+3Poly");

    // Fused FWD+BWD: single pass (needs go available)
    m.def("sigmoid_fused", &sigmoid_fused, "Sigmoid fused FWD+BWD");
    m.def("tanh_fused",    &tanh_fused,    "Tanh fused FWD+BWD");
    m.def("swish_fused",   &swish_fused,   "Swish fused FWD+BWD");

    // FWD with derivative: returns (y, f'(x)) — standard autograd pattern
    m.def("sigmoid_fwd_deriv_alg",  &sigmoid_fwd_deriv_alg,  "Sigmoid FWD+deriv (poly+algebraic)");
    m.def("sigmoid_fwd_deriv_poly", &sigmoid_fwd_deriv_poly, "Sigmoid FWD+deriv (poly+poly)");
    m.def("tanh_fwd_deriv_alg",     &tanh_fwd_deriv_alg,     "Tanh FWD+deriv (poly+algebraic)");
    m.def("tanh_fwd_deriv_poly",    &tanh_fwd_deriv_poly,    "Tanh FWD+deriv (poly+poly)");
    m.def("swish_fwd_deriv",        &swish_fwd_deriv,        "Swish FWD+deriv (shared sigmoid)");
    m.def("swish_fwd_deriv_poly",   &swish_fwd_deriv_poly,   "Swish FWD+deriv (poly+poly)");

    // Trivial multiply: gi = go * saved_dy
    m.def("multiply", &multiply, "Element-wise multiply (for backward)");
}

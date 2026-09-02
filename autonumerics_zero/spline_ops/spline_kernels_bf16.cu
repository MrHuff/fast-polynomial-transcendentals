// Copyright (c) 2026 Graphcore Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified in 2026 for the standalone fast-polynomial-transcendentals release.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "spline_structs_odd_bf16.cuh"
#include "spline_structs_sollya_bf16.cuh"

// =============================================================================
// Kernel constants
// =============================================================================
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
constexpr int PACKED_BLOCK_SIZE_BF16 = 128;
constexpr int PACKED_D3_BLOCK_SIZE_BF16 = 128;

static inline bool is_aligned_to_bf16(const void* ptr, std::uintptr_t alignment) {
    return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
}

static inline bool can_use_bfloat162(const void* a, const void* b, int size) {
    return (size % 2 == 0) && is_aligned_to_bf16(a, 4) && is_aligned_to_bf16(b, 4);
}

static inline bool can_use_bfloat162(const void* a, const void* b, const void* c, int size) {
    return can_use_bfloat162(a, b, size) && is_aligned_to_bf16(c, 4);
}

static inline bool can_use_bfloat162(
    const void* a, const void* b, const void* c, const void* d, const void* e, int size) {
    return can_use_bfloat162(a, b, c, size) && is_aligned_to_bf16(d, 4) && is_aligned_to_bf16(e, 4);
}

static inline bool can_use_int4_bf16(const void* a, const void* b, int size) {
    return (size % 8 == 0) && is_aligned_to_bf16(a, 16) && is_aligned_to_bf16(b, 16);
}

static inline bool can_use_int4_bf16(const void* a, const void* b, const void* c, int size) {
    return can_use_int4_bf16(a, b, size) && is_aligned_to_bf16(c, 16);
}

static inline bool can_use_int4_bf16(
    const void* a, const void* b, const void* c, const void* d, const void* e, int size) {
    return can_use_int4_bf16(a, b, c, size) && is_aligned_to_bf16(d, 16) && is_aligned_to_bf16(e, 16);
}

// =============================================================================
// Kernel templates — BF16 versions
// Using int4 vectorization (16 bytes = 4 x __nv_bfloat162)
// =============================================================================

// Unary FWD: out = F(in), vectorized
template <typename Func>
__global__ void __launch_bounds__(BLOCK_SIZE)
unary_vec_kernel_bf16(const int4* __restrict__ input,
                      int4* __restrict__ output,
                      int n_vec) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        int4 data = input[i];
        __nv_bfloat162* h2 = reinterpret_cast<__nv_bfloat162*>(&data);
        #pragma unroll
        for (int k = 0; k < 4; k++)
            h2[k] = Func::evaluate(h2[k]);
        output[i] = data;
    }
}

// Binary BWD: grad_in = grad_out * F'(in), vectorized
template <typename GradFunc>
__global__ void __launch_bounds__(BLOCK_SIZE)
binary_vec_kernel_bf16(const int4* __restrict__ grad_output,
                       const int4* __restrict__ input,
                       int4* __restrict__ grad_input,
                       int n_vec) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        int4 go_data = grad_output[i];
        int4 in_data = input[i];
        __nv_bfloat162* go = reinterpret_cast<__nv_bfloat162*>(&go_data);
        __nv_bfloat162* x  = reinterpret_cast<__nv_bfloat162*>(&in_data);
        int4 gi_data;
        __nv_bfloat162* gi = reinterpret_cast<__nv_bfloat162*>(&gi_data);
        #pragma unroll
        for (int k = 0; k < 4; k++)
            gi[k] = __hmul2(go[k], GradFunc::evaluate(x[k]));
        grad_input[i] = gi_data;
    }
}

// Scalar fallbacks
template <typename Func>
__global__ void unary_scalar_kernel_bf16(const __nv_bfloat162* __restrict__ input,
                                          __nv_bfloat162* __restrict__ output,
                                          int n_bf2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_bf2)
        output[idx] = Func::evaluate(input[idx]);
}

template <typename Func>
__global__ void unary_element_kernel_bf16(const __nv_bfloat16* __restrict__ input,
                                          __nv_bfloat16* __restrict__ output,
                                          int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        __nv_bfloat162 x = __halves2bfloat162(input[idx], input[idx]);
        output[idx] = __low2bfloat16(Func::evaluate(x));
    }
}

template <typename GradFunc>
__global__ void binary_scalar_kernel_bf16(const __nv_bfloat162* __restrict__ grad_output,
                                           const __nv_bfloat162* __restrict__ input,
                                           __nv_bfloat162* __restrict__ grad_input,
                                           int n_bf2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_bf2) {
        __nv_bfloat162 dy = GradFunc::evaluate(input[idx]);
        grad_input[idx] = __hmul2(grad_output[idx], dy);
    }
}

template <typename GradFunc>
__global__ void binary_element_kernel_bf16(const __nv_bfloat16* __restrict__ grad_output,
                                           const __nv_bfloat16* __restrict__ input,
                                           __nv_bfloat16* __restrict__ grad_input,
                                           int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        __nv_bfloat162 go = __halves2bfloat162(grad_output[idx], grad_output[idx]);
        __nv_bfloat162 x = __halves2bfloat162(input[idx], input[idx]);
        grad_input[idx] = __low2bfloat16(__hmul2(go, GradFunc::evaluate(x)));
    }
}

// Fused SwiGLU elementwise: out = swish(gate) * up, vectorized
template <typename Func>
__global__ void __launch_bounds__(BLOCK_SIZE)
swish_mul_vec_kernel_bf16(const int4* __restrict__ gate,
                          const int4* __restrict__ up,
                          int4* __restrict__ output,
                          int n_vec) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        int4 gate_data = gate[i];
        int4 up_data = up[i];
        int4 out_data;
        __nv_bfloat162* gate_h2 = reinterpret_cast<__nv_bfloat162*>(&gate_data);
        __nv_bfloat162* up_h2 = reinterpret_cast<__nv_bfloat162*>(&up_data);
        __nv_bfloat162* out_h2 = reinterpret_cast<__nv_bfloat162*>(&out_data);
        #pragma unroll
        for (int k = 0; k < 4; k++)
            out_h2[k] = __hmul2(Func::evaluate(gate_h2[k]), up_h2[k]);
        output[i] = out_data;
    }
}

template <typename Func>
__global__ void swish_mul_scalar_kernel_bf16(const __nv_bfloat162* __restrict__ gate,
                                             const __nv_bfloat162* __restrict__ up,
                                             __nv_bfloat162* __restrict__ output,
                                             int n_bf2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_bf2)
        output[idx] = __hmul2(Func::evaluate(gate[idx]), up[idx]);
}

template <typename Func>
__global__ void swish_mul_element_kernel_bf16(const __nv_bfloat16* __restrict__ gate,
                                              const __nv_bfloat16* __restrict__ up,
                                              __nv_bfloat16* __restrict__ output,
                                              int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        __nv_bfloat162 g = __halves2bfloat162(gate[idx], gate[idx]);
        __nv_bfloat162 u = __halves2bfloat162(up[idx], up[idx]);
        output[idx] = __low2bfloat16(__hmul2(Func::evaluate(g), u));
    }
}

template <typename FwdFunc, typename BwdFunc>
__global__ void __launch_bounds__(BLOCK_SIZE)
swish_mul_bwd_vec_kernel_bf16(const int4* __restrict__ grad_output,
                              const int4* __restrict__ gate,
                              const int4* __restrict__ up,
                              int4* __restrict__ grad_gate,
                              int4* __restrict__ grad_up,
                              int n_vec) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        int4 go_data = grad_output[i];
        int4 gate_data = gate[i];
        int4 up_data = up[i];
        int4 gg_data;
        int4 gu_data;
        __nv_bfloat162* go_h2 = reinterpret_cast<__nv_bfloat162*>(&go_data);
        __nv_bfloat162* gate_h2 = reinterpret_cast<__nv_bfloat162*>(&gate_data);
        __nv_bfloat162* up_h2 = reinterpret_cast<__nv_bfloat162*>(&up_data);
        __nv_bfloat162* gg_h2 = reinterpret_cast<__nv_bfloat162*>(&gg_data);
        __nv_bfloat162* gu_h2 = reinterpret_cast<__nv_bfloat162*>(&gu_data);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            gg_h2[k] = __hmul2(__hmul2(go_h2[k], up_h2[k]), BwdFunc::evaluate(gate_h2[k]));
            gu_h2[k] = __hmul2(go_h2[k], FwdFunc::evaluate(gate_h2[k]));
        }
        grad_gate[i] = gg_data;
        grad_up[i] = gu_data;
    }
}

template <typename FwdFunc, typename BwdFunc>
__global__ void swish_mul_bwd_scalar_kernel_bf16(
    const __nv_bfloat162* __restrict__ grad_output,
    const __nv_bfloat162* __restrict__ gate,
    const __nv_bfloat162* __restrict__ up,
    __nv_bfloat162* __restrict__ grad_gate,
    __nv_bfloat162* __restrict__ grad_up,
    int n_bf2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_bf2) {
        __nv_bfloat162 go = grad_output[idx];
        __nv_bfloat162 g = gate[idx];
        grad_gate[idx] = __hmul2(__hmul2(go, up[idx]), BwdFunc::evaluate(g));
        grad_up[idx] = __hmul2(go, FwdFunc::evaluate(g));
    }
}

template <typename FwdFunc, typename BwdFunc>
__global__ void swish_mul_bwd_element_kernel_bf16(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ grad_gate,
    __nv_bfloat16* __restrict__ grad_up,
    int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        __nv_bfloat162 go = __halves2bfloat162(grad_output[idx], grad_output[idx]);
        __nv_bfloat162 g = __halves2bfloat162(gate[idx], gate[idx]);
        __nv_bfloat162 u = __halves2bfloat162(up[idx], up[idx]);
        grad_gate[idx] = __low2bfloat16(__hmul2(__hmul2(go, u), BwdFunc::evaluate(g)));
        grad_up[idx] = __low2bfloat16(__hmul2(go, FwdFunc::evaluate(g)));
    }
}

// D3 backward needs both swish(x) and swish'(x). Deriving both from one
// sigmoid approximation shares the range reduction and Horner evaluation.
static __device__ __forceinline__ void swish_d3_fwd_bwd_alg_bf16(
    __nv_bfloat162 x,
    __nv_bfloat162& swish,
    __nv_bfloat162& derivative) {
    const __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
    const __nv_bfloat162 sigmoid = SIGMOID_FWD_D3_ODD_BF16::evaluate(x);
    swish = __hmul2(x, sigmoid);
    derivative = __hfma2(swish, __hsub2(one, sigmoid), sigmoid);
}

__global__ void __launch_bounds__(BLOCK_SIZE)
swish_mul_bwd_d3_alg_vec_kernel_bf16(
    const int4* __restrict__ grad_output,
    const int4* __restrict__ gate,
    const int4* __restrict__ up,
    int4* __restrict__ grad_gate,
    int4* __restrict__ grad_up,
    int n_vec) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        const int4 go_data = grad_output[i];
        const int4 gate_data = gate[i];
        const int4 up_data = up[i];
        int4 gg_data;
        int4 gu_data;
        const __nv_bfloat162* go_h2 =
            reinterpret_cast<const __nv_bfloat162*>(&go_data);
        const __nv_bfloat162* gate_h2 =
            reinterpret_cast<const __nv_bfloat162*>(&gate_data);
        const __nv_bfloat162* up_h2 =
            reinterpret_cast<const __nv_bfloat162*>(&up_data);
        __nv_bfloat162* gg_h2 = reinterpret_cast<__nv_bfloat162*>(&gg_data);
        __nv_bfloat162* gu_h2 = reinterpret_cast<__nv_bfloat162*>(&gu_data);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            __nv_bfloat162 swish;
            __nv_bfloat162 derivative;
            swish_d3_fwd_bwd_alg_bf16(gate_h2[k], swish, derivative);
            gg_h2[k] = __hmul2(__hmul2(go_h2[k], up_h2[k]), derivative);
            gu_h2[k] = __hmul2(go_h2[k], swish);
        }
        grad_gate[i] = gg_data;
        grad_up[i] = gu_data;
    }
}

__global__ void swish_mul_bwd_d3_alg_scalar_kernel_bf16(
    const __nv_bfloat162* __restrict__ grad_output,
    const __nv_bfloat162* __restrict__ gate,
    const __nv_bfloat162* __restrict__ up,
    __nv_bfloat162* __restrict__ grad_gate,
    __nv_bfloat162* __restrict__ grad_up,
    int n_bf2) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_bf2) {
        __nv_bfloat162 swish;
        __nv_bfloat162 derivative;
        swish_d3_fwd_bwd_alg_bf16(gate[idx], swish, derivative);
        grad_gate[idx] = __hmul2(
            __hmul2(grad_output[idx], up[idx]), derivative);
        grad_up[idx] = __hmul2(grad_output[idx], swish);
    }
}

__global__ void swish_mul_bwd_d3_alg_element_kernel_bf16(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ grad_gate,
    __nv_bfloat16* __restrict__ grad_up,
    int size) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        const __nv_bfloat162 go =
            __halves2bfloat162(grad_output[idx], grad_output[idx]);
        const __nv_bfloat162 g = __halves2bfloat162(gate[idx], gate[idx]);
        const __nv_bfloat162 u = __halves2bfloat162(up[idx], up[idx]);
        __nv_bfloat162 swish;
        __nv_bfloat162 derivative;
        swish_d3_fwd_bwd_alg_bf16(g, swish, derivative);
        grad_gate[idx] = __low2bfloat16(__hmul2(__hmul2(go, u), derivative));
        grad_up[idx] = __low2bfloat16(__hmul2(go, swish));
    }
}

template <typename FwdFunc, typename BwdFunc>
static __device__ __forceinline__ void swish_fwd_bwd_bf16(
    __nv_bfloat162 x,
    __nv_bfloat162& swish,
    __nv_bfloat162& derivative) {
    swish = FwdFunc::evaluate(x);
    derivative = BwdFunc::evaluate(x);
}

struct SWISH_FWD_NATIVE_BF16 {
    static __device__ __forceinline__ __nv_bfloat162 evaluate(
        __nv_bfloat162 x) {
        const float x0 = __bfloat162float(__low2bfloat16(x));
        const float x1 = __bfloat162float(__high2bfloat16(x));
        return __floats2bfloat162_rn(
            x0 / (1.0f + __expf(-x0)),
            x1 / (1.0f + __expf(-x1)));
    }
};
struct SWISH_BWD_NATIVE_BF16 {
    static __device__ __forceinline__ __nv_bfloat162 evaluate(
        __nv_bfloat162 x) {
        const float x0 = __bfloat162float(__low2bfloat16(x));
        const float x1 = __bfloat162float(__high2bfloat16(x));
        const float sigmoid0 = 1.0f / (1.0f + __expf(-x0));
        const float sigmoid1 = 1.0f / (1.0f + __expf(-x1));
        return __floats2bfloat162_rn(
            sigmoid0 * (1.0f + x0 * (1.0f - sigmoid0)),
            sigmoid1 * (1.0f + x1 * (1.0f - sigmoid1)));
    }
};

template <>
__device__ __forceinline__ void swish_fwd_bwd_bf16<
    SWISH_FWD_D3_ODD_BF16, SWISH_BWD_D3_ODD_BF16>(
    __nv_bfloat162 x,
    __nv_bfloat162& swish,
    __nv_bfloat162& derivative) {
    swish_d3_fwd_bwd_alg_bf16(x, swish, derivative);
}

template <typename FwdFunc, typename BwdFunc>
static __device__ __forceinline__ void swish_mul_packed_grads_bf16(
    __nv_bfloat162 grad_output,
    __nv_bfloat162 gate,
    __nv_bfloat162 up,
    __nv_bfloat162& grad_gate,
    __nv_bfloat162& grad_up) {
    __nv_bfloat162 swish;
    __nv_bfloat162 derivative;
    swish_fwd_bwd_bf16<FwdFunc, BwdFunc>(gate, swish, derivative);
    grad_gate = __hmul2(__hmul2(grad_output, up), derivative);
    grad_up = __hmul2(grad_output, swish);
}

template <>
__device__ __forceinline__ void swish_mul_packed_grads_bf16<
    SWISH_FWD_NATIVE_BF16, SWISH_BWD_NATIVE_BF16>(
    __nv_bfloat162 grad_output,
    __nv_bfloat162 gate,
    __nv_bfloat162 up,
    __nv_bfloat162& grad_gate,
    __nv_bfloat162& grad_up) {
    const __nv_bfloat162 dy = __hmul2(grad_output, up);
    const float dy0 = __bfloat162float(__low2bfloat16(dy));
    const float dy1 = __bfloat162float(__high2bfloat16(dy));
    const float x0 = __bfloat162float(__low2bfloat16(gate));
    const float x1 = __bfloat162float(__high2bfloat16(gate));
    const float sigmoid0 = 1.0f / (1.0f + __expf(-x0));
    const float sigmoid1 = 1.0f / (1.0f + __expf(-x1));
    grad_gate = __floats2bfloat162_rn(
        dy0 * sigmoid0 * (1.0f + x0 * (1.0f - sigmoid0)),
        dy1 * sigmoid1 * (1.0f + x1 * (1.0f - sigmoid1)));
    grad_up = __hmul2(
        grad_output,
        SWISH_FWD_NATIVE_BF16::evaluate(gate));
}

// Fused packed SwiGLU elementwise for tensors laid out as [..., gate, up].
template <typename Func>
__global__ void __launch_bounds__(PACKED_BLOCK_SIZE_BF16)
swish_mul_packed_vec_kernel_bf16(const __nv_bfloat16* __restrict__ packed,
                                 __nv_bfloat16* __restrict__ output,
                                 int rows,
                                 int hidden_size,
                                 int vecs_per_row) {
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        const int64_t row_offset = static_cast<int64_t>(row) * hidden_size;
        for (int vec_col = threadIdx.x + blockIdx.x * blockDim.x;
             vec_col < vecs_per_row;
             vec_col += blockDim.x * gridDim.x) {
        const int col = vec_col * 8;
        const __nv_bfloat16* gate_ptr = packed + row_offset * 2 + col;
        const __nv_bfloat16* up_ptr = gate_ptr + hidden_size;

        int4 gate_data = *reinterpret_cast<const int4*>(gate_ptr);
        int4 up_data = *reinterpret_cast<const int4*>(up_ptr);
        int4 out_data;
        __nv_bfloat162* gate_h2 = reinterpret_cast<__nv_bfloat162*>(&gate_data);
        __nv_bfloat162* up_h2 = reinterpret_cast<__nv_bfloat162*>(&up_data);
        __nv_bfloat162* out_h2 = reinterpret_cast<__nv_bfloat162*>(&out_data);
        #pragma unroll
        for (int k = 0; k < 4; k++)
            out_h2[k] = __hmul2(Func::evaluate(gate_h2[k]), up_h2[k]);
            reinterpret_cast<int4*>(output + row_offset)[vec_col] = out_data;
        }
    }
}

template <typename Func>
__global__ void __launch_bounds__(BLOCK_SIZE)
swish_mul_packed_h512_vec_kernel_bf16(const __nv_bfloat16* __restrict__ packed,
                                      __nv_bfloat16* __restrict__ output,
                                      int rows) {
    int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows)
        return;
    int vec_col = threadIdx.x & 63;
    const int64_t row_offset = static_cast<int64_t>(row) * 512;
    const __nv_bfloat16* gate_ptr = packed + row_offset * 2 + vec_col * 8;
    const __nv_bfloat16* up_ptr = gate_ptr + 512;

    int4 gate_data = *reinterpret_cast<const int4*>(gate_ptr);
    int4 up_data = *reinterpret_cast<const int4*>(up_ptr);
    int4 out_data;
    __nv_bfloat162* gate_h2 = reinterpret_cast<__nv_bfloat162*>(&gate_data);
    __nv_bfloat162* up_h2 = reinterpret_cast<__nv_bfloat162*>(&up_data);
    __nv_bfloat162* out_h2 = reinterpret_cast<__nv_bfloat162*>(&out_data);
    #pragma unroll
    for (int k = 0; k < 4; k++)
        out_h2[k] = __hmul2(Func::evaluate(gate_h2[k]), up_h2[k]);
    reinterpret_cast<int4*>(output + row_offset)[vec_col] = out_data;
}

__global__ void __launch_bounds__(PACKED_D3_BLOCK_SIZE_BF16)
swish_mul_packed_h512_d3_vec_kernel_bf16(const __nv_bfloat16* __restrict__ packed,
                                         __nv_bfloat16* __restrict__ output,
                                         int rows) {
    #pragma unroll
    for (int item = 0; item < 2; item++) {
        int flat = threadIdx.x + item * PACKED_D3_BLOCK_SIZE_BF16;
        int row = blockIdx.x * 4 + (flat >> 6);
        if (row >= rows)
            continue;
        int vec_col = flat & 63;
        const int64_t row_offset = static_cast<int64_t>(row) * 512;
        const __nv_bfloat16* gate_ptr = packed + row_offset * 2 + vec_col * 8;
        const __nv_bfloat16* up_ptr = gate_ptr + 512;

        int4 gate_data = *reinterpret_cast<const int4*>(gate_ptr);
        int4 up_data = *reinterpret_cast<const int4*>(up_ptr);
        int4 out_data;
        __nv_bfloat162* gate_h2 = reinterpret_cast<__nv_bfloat162*>(&gate_data);
        __nv_bfloat162* up_h2 = reinterpret_cast<__nv_bfloat162*>(&up_data);
        __nv_bfloat162* out_h2 = reinterpret_cast<__nv_bfloat162*>(&out_data);
        #pragma unroll
        for (int k = 0; k < 4; k++)
            out_h2[k] = __hmul2(SWISH_FWD_D3_ODD_BF16::evaluate(gate_h2[k]), up_h2[k]);
        reinterpret_cast<int4*>(output + row_offset)[vec_col] = out_data;
    }
}

template <typename Func>
__global__ void swish_mul_packed_scalar_kernel_bf16(const __nv_bfloat16* __restrict__ packed,
                                                    __nv_bfloat16* __restrict__ output,
                                                    int rows,
                                                    int hidden_size,
                                                    int pairs_per_row) {
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        const int64_t row_offset = static_cast<int64_t>(row) * hidden_size;
        for (int pair_col = threadIdx.x + blockIdx.x * blockDim.x;
             pair_col < pairs_per_row;
             pair_col += blockDim.x * gridDim.x) {
        const int col = pair_col * 2;
        const __nv_bfloat162* gate = reinterpret_cast<const __nv_bfloat162*>(
            packed + row_offset * 2 + col);
        const __nv_bfloat162* up = reinterpret_cast<const __nv_bfloat162*>(
            packed + row_offset * 2 + hidden_size + col);
            reinterpret_cast<__nv_bfloat162*>(output + row_offset)[pair_col] =
                __hmul2(Func::evaluate(*gate), *up);
        }
    }
}

template <typename FwdFunc, typename BwdFunc>
__global__ void __launch_bounds__(PACKED_BLOCK_SIZE_BF16)
swish_mul_packed_bwd_vec_kernel_bf16(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ grad_packed,
    int rows,
    int hidden_size,
    int vecs_per_row) {
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        const int64_t row_offset = static_cast<int64_t>(row) * hidden_size;
        for (int vec_col = threadIdx.x + blockIdx.x * blockDim.x;
             vec_col < vecs_per_row;
             vec_col += blockDim.x * gridDim.x) {
            const int col = vec_col * 8;
            const __nv_bfloat16* go_ptr = grad_output + row_offset + col;
            const __nv_bfloat16* gate_ptr = packed + row_offset * 2 + col;
            const __nv_bfloat16* up_ptr = gate_ptr + hidden_size;
            __nv_bfloat16* grad_gate_ptr = grad_packed + row_offset * 2 + col;
            __nv_bfloat16* grad_up_ptr = grad_gate_ptr + hidden_size;

            const int4 go_data = *reinterpret_cast<const int4*>(go_ptr);
            const int4 gate_data = *reinterpret_cast<const int4*>(gate_ptr);
            const int4 up_data = *reinterpret_cast<const int4*>(up_ptr);
            int4 grad_gate_data;
            int4 grad_up_data;
            const __nv_bfloat162* go_h2 =
                reinterpret_cast<const __nv_bfloat162*>(&go_data);
            const __nv_bfloat162* gate_h2 =
                reinterpret_cast<const __nv_bfloat162*>(&gate_data);
            const __nv_bfloat162* up_h2 =
                reinterpret_cast<const __nv_bfloat162*>(&up_data);
            __nv_bfloat162* grad_gate_h2 =
                reinterpret_cast<__nv_bfloat162*>(&grad_gate_data);
            __nv_bfloat162* grad_up_h2 =
                reinterpret_cast<__nv_bfloat162*>(&grad_up_data);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                swish_mul_packed_grads_bf16<FwdFunc, BwdFunc>(
                    go_h2[k],
                    gate_h2[k],
                    up_h2[k],
                    grad_gate_h2[k],
                    grad_up_h2[k]);
            }
            *reinterpret_cast<int4*>(grad_gate_ptr) = grad_gate_data;
            *reinterpret_cast<int4*>(grad_up_ptr) = grad_up_data;
        }
    }
}

template <typename FwdFunc, typename BwdFunc>
__global__ void swish_mul_packed_bwd_scalar_kernel_bf16(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ grad_packed,
    int rows,
    int hidden_size,
    int pairs_per_row) {
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        const int64_t row_offset = static_cast<int64_t>(row) * hidden_size;
        for (int pair_col = threadIdx.x + blockIdx.x * blockDim.x;
             pair_col < pairs_per_row;
             pair_col += blockDim.x * gridDim.x) {
            const int col = pair_col * 2;
            const __nv_bfloat162 go = *reinterpret_cast<const __nv_bfloat162*>(
                grad_output + row_offset + col);
            const __nv_bfloat162 gate = *reinterpret_cast<const __nv_bfloat162*>(
                packed + row_offset * 2 + col);
            const __nv_bfloat162 up = *reinterpret_cast<const __nv_bfloat162*>(
                packed + row_offset * 2 + hidden_size + col);
            __nv_bfloat162 grad_gate;
            __nv_bfloat162 grad_up;
            swish_mul_packed_grads_bf16<FwdFunc, BwdFunc>(
                go, gate, up, grad_gate, grad_up);
            *reinterpret_cast<__nv_bfloat162*>(
                grad_packed + row_offset * 2 + col) = grad_gate;
            *reinterpret_cast<__nv_bfloat162*>(
                grad_packed + row_offset * 2 + hidden_size + col) = grad_up;
        }
    }
}

// =============================================================================
// Grid sizing with occupancy API
// =============================================================================

template <auto Kernel>
static int compute_grid_bf16(int n_work_items) {
    int min_grid = (n_work_items + BLOCK_SIZE - 1) / BLOCK_SIZE;

    static int sm_count = 0;
    if (sm_count == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    }

    static const int blocks_per_sm = [] {
        int value = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &value, Kernel, BLOCK_SIZE, 0);
        return value;
    }();

    int max_grid = blocks_per_sm * sm_count;
    return min(min_grid, max_grid);
}

constexpr int VEC4_THRESHOLD_BF16 = 4096;

// =============================================================================
// Launch helpers
// =============================================================================

template <typename Func>
static void launch_unary_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t stream)
{
    if (size <= 0)
        return;
    int n_bf2 = size / 2;
    if (n_bf2 >= VEC4_THRESHOLD_BF16 && can_use_int4_bf16(in, out, size)) {
        int n_vec = n_bf2 / 4;
        auto kernel = unary_vec_kernel_bf16<Func>;
        int grid = compute_grid_bf16<unary_vec_kernel_bf16<Func>>(n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(in),
            reinterpret_cast<int4*>(out),
            n_vec);
    } else if (can_use_bfloat162(in, out, size)) {
        int grid = (n_bf2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unary_scalar_kernel_bf16<Func><<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat162*>(in),
            reinterpret_cast<__nv_bfloat162*>(out),
            n_bf2);
    } else {
        int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        unary_element_kernel_bf16<Func><<<grid, BLOCK_SIZE, 0, stream>>>(in, out, size);
    }
}

template <typename GradFunc>
static void launch_binary_bf16(
    __nv_bfloat16* grad_in, const __nv_bfloat16* grad_out, const __nv_bfloat16* in,
    int size, cudaStream_t stream)
{
    if (size <= 0)
        return;
    int n_bf2 = size / 2;
    if (n_bf2 >= VEC4_THRESHOLD_BF16 && can_use_int4_bf16(grad_out, in, grad_in, size)) {
        int n_vec = n_bf2 / 4;
        auto kernel = binary_vec_kernel_bf16<GradFunc>;
        int grid = compute_grid_bf16<binary_vec_kernel_bf16<GradFunc>>(n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(grad_out),
            reinterpret_cast<const int4*>(in),
            reinterpret_cast<int4*>(grad_in),
            n_vec);
    } else if (can_use_bfloat162(grad_out, in, grad_in, size)) {
        int grid = (n_bf2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        binary_scalar_kernel_bf16<GradFunc><<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat162*>(grad_out),
            reinterpret_cast<const __nv_bfloat162*>(in),
            reinterpret_cast<__nv_bfloat162*>(grad_in),
            n_bf2);
    } else {
        int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        binary_element_kernel_bf16<GradFunc><<<grid, BLOCK_SIZE, 0, stream>>>(
            grad_out,
            in,
            grad_in,
            size);
    }
}

template <typename Func>
static void launch_swish_mul_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int size, cudaStream_t stream)
{
    if (size <= 0)
        return;
    int n_bf2 = size / 2;
    if (n_bf2 >= VEC4_THRESHOLD_BF16 && can_use_int4_bf16(gate, up, out, size)) {
        int n_vec = n_bf2 / 4;
        auto kernel = swish_mul_vec_kernel_bf16<Func>;
        int grid = compute_grid_bf16<swish_mul_vec_kernel_bf16<Func>>(n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(gate),
            reinterpret_cast<const int4*>(up),
            reinterpret_cast<int4*>(out),
            n_vec);
    } else if (can_use_bfloat162(gate, up, out, size)) {
        int grid = (n_bf2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        swish_mul_scalar_kernel_bf16<Func><<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat162*>(gate),
            reinterpret_cast<const __nv_bfloat162*>(up),
            reinterpret_cast<__nv_bfloat162*>(out),
            n_bf2);
    } else {
        int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        swish_mul_element_kernel_bf16<Func><<<grid, BLOCK_SIZE, 0, stream>>>(
            gate,
            up,
            out,
            size);
    }
}

template <typename FwdFunc, typename BwdFunc>
static void launch_swish_mul_bwd_bf16(
    __nv_bfloat16* grad_gate,
    __nv_bfloat16* grad_up,
    const __nv_bfloat16* grad_out,
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    int size,
    cudaStream_t stream)
{
    if (size <= 0)
        return;
    int n_bf2 = size / 2;
    if (n_bf2 >= VEC4_THRESHOLD_BF16 && can_use_int4_bf16(grad_out, gate, up, grad_gate, grad_up, size)) {
        int n_vec = n_bf2 / 4;
        auto kernel = swish_mul_bwd_vec_kernel_bf16<FwdFunc, BwdFunc>;
        int grid = compute_grid_bf16<
            swish_mul_bwd_vec_kernel_bf16<FwdFunc, BwdFunc>>(n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(grad_out),
            reinterpret_cast<const int4*>(gate),
            reinterpret_cast<const int4*>(up),
            reinterpret_cast<int4*>(grad_gate),
            reinterpret_cast<int4*>(grad_up),
            n_vec);
    } else if (can_use_bfloat162(grad_out, gate, up, grad_gate, grad_up, size)) {
        int grid = (n_bf2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        swish_mul_bwd_scalar_kernel_bf16<FwdFunc, BwdFunc><<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat162*>(grad_out),
            reinterpret_cast<const __nv_bfloat162*>(gate),
            reinterpret_cast<const __nv_bfloat162*>(up),
            reinterpret_cast<__nv_bfloat162*>(grad_gate),
            reinterpret_cast<__nv_bfloat162*>(grad_up),
            n_bf2);
    } else {
        int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        swish_mul_bwd_element_kernel_bf16<FwdFunc, BwdFunc><<<grid, BLOCK_SIZE, 0, stream>>>(
            grad_out,
            gate,
            up,
            grad_gate,
            grad_up,
            size);
    }
}

static void launch_swish_mul_bwd_d3_alg_bf16(
    __nv_bfloat16* grad_gate,
    __nv_bfloat16* grad_up,
    const __nv_bfloat16* grad_out,
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    int size,
    cudaStream_t stream) {
    if (size <= 0)
        return;
    const int n_bf2 = size / 2;
    if (
        n_bf2 >= VEC4_THRESHOLD_BF16
        && can_use_int4_bf16(grad_out, gate, up, grad_gate, grad_up, size)
    ) {
        const int n_vec = n_bf2 / 4;
        auto kernel = swish_mul_bwd_d3_alg_vec_kernel_bf16;
        const int grid =
            compute_grid_bf16<swish_mul_bwd_d3_alg_vec_kernel_bf16>(n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(grad_out),
            reinterpret_cast<const int4*>(gate),
            reinterpret_cast<const int4*>(up),
            reinterpret_cast<int4*>(grad_gate),
            reinterpret_cast<int4*>(grad_up),
            n_vec);
    } else if (can_use_bfloat162(grad_out, gate, up, grad_gate, grad_up, size)) {
        int grid = (n_bf2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        swish_mul_bwd_d3_alg_scalar_kernel_bf16<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat162*>(grad_out),
            reinterpret_cast<const __nv_bfloat162*>(gate),
            reinterpret_cast<const __nv_bfloat162*>(up),
            reinterpret_cast<__nv_bfloat162*>(grad_gate),
            reinterpret_cast<__nv_bfloat162*>(grad_up),
            n_bf2);
    } else {
        int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        swish_mul_bwd_d3_alg_element_kernel_bf16<<<grid, BLOCK_SIZE, 0, stream>>>(
            grad_out,
            gate,
            up,
            grad_gate,
            grad_up,
            size);
    }
}

static void launch_swish_mul_packed_h512_d3_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* packed, int rows, cudaStream_t stream)
{
    int grid = (rows + 3) / 4;
    swish_mul_packed_h512_d3_vec_kernel_bf16<<<grid, PACKED_D3_BLOCK_SIZE_BF16, 0, stream>>>(
        packed,
        out,
        rows);
}

template <typename Func>
static void launch_swish_mul_packed_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* packed, int rows, int hidden_size,
    cudaStream_t stream)
{
    const int64_t size = static_cast<int64_t>(rows) * hidden_size;
    const int64_t n_bf2 = size / 2;
    if (n_bf2 >= VEC4_THRESHOLD_BF16 && hidden_size == 512) {
        int grid = (rows + 3) / 4;
        swish_mul_packed_h512_vec_kernel_bf16<Func><<<grid, BLOCK_SIZE, 0, stream>>>(
            packed,
            out,
            rows);
    } else if (n_bf2 >= VEC4_THRESHOLD_BF16 && (hidden_size % 8 == 0)) {
        int vecs_per_row = hidden_size / 8;
        int grid_x = (vecs_per_row + PACKED_BLOCK_SIZE_BF16 - 1) / PACKED_BLOCK_SIZE_BF16;
        int grid_y = min(rows, 65535);
        swish_mul_packed_vec_kernel_bf16<Func><<<dim3(grid_x, grid_y), PACKED_BLOCK_SIZE_BF16, 0, stream>>>(
            packed,
            out,
            rows,
            hidden_size,
            vecs_per_row);
    } else {
        int pairs_per_row = hidden_size / 2;
        int grid_x = (pairs_per_row + PACKED_BLOCK_SIZE_BF16 - 1) / PACKED_BLOCK_SIZE_BF16;
        if (grid_x == 0) grid_x = 1;
        int grid_y = min(rows, 65535);
        swish_mul_packed_scalar_kernel_bf16<Func><<<dim3(grid_x, grid_y), PACKED_BLOCK_SIZE_BF16, 0, stream>>>(
            packed,
            out,
            rows,
            hidden_size,
            pairs_per_row);
    }
}

template <typename FwdFunc, typename BwdFunc>
static void launch_swish_mul_packed_bwd_bf16(
    __nv_bfloat16* grad_packed,
    const __nv_bfloat16* grad_output,
    const __nv_bfloat16* packed,
    int rows,
    int hidden_size,
    cudaStream_t stream) {
    const int64_t size = static_cast<int64_t>(rows) * hidden_size;
    if (size <= 0)
        return;
    const int grid_y = min(rows, 65535);
    if (
        hidden_size % 8 == 0
        && is_aligned_to_bf16(grad_output, 16)
        && is_aligned_to_bf16(packed, 16)
        && is_aligned_to_bf16(grad_packed, 16)
    ) {
        const int vecs_per_row = hidden_size / 8;
        const int grid_x =
            (vecs_per_row + PACKED_BLOCK_SIZE_BF16 - 1) /
            PACKED_BLOCK_SIZE_BF16;
        swish_mul_packed_bwd_vec_kernel_bf16<FwdFunc, BwdFunc>
            <<<dim3(grid_x, grid_y), PACKED_BLOCK_SIZE_BF16, 0, stream>>>(
                grad_output, packed, grad_packed, rows, hidden_size, vecs_per_row);
    } else {
        const int pairs_per_row = hidden_size / 2;
        int grid_x =
            (pairs_per_row + PACKED_BLOCK_SIZE_BF16 - 1) /
            PACKED_BLOCK_SIZE_BF16;
        if (grid_x == 0) grid_x = 1;
        swish_mul_packed_bwd_scalar_kernel_bf16<FwdFunc, BwdFunc>
            <<<dim3(grid_x, grid_y), PACKED_BLOCK_SIZE_BF16, 0, stream>>>(
                grad_output, packed, grad_packed, rows, hidden_size, pairs_per_row);
    }
}

// =============================================================================
// Launchers — extern "C" for linkage from spline_ops.cpp
// =============================================================================

extern "C" {

// --- SIGMOID FWD D3-D6 (BF16) ---
void launch_sigmoid_fwd_d3_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SIGMOID_FWD_D3_ODD_BF16>(out, in, size, s);
}
void launch_sigmoid_fwd_d4_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SIGMOID_FWD_D4_ODD_BF16>(out, in, size, s);
}
void launch_sigmoid_fwd_d5_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SIGMOID_FWD_D5_ODD_BF16>(out, in, size, s);
}
void launch_sigmoid_fwd_d6_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SIGMOID_FWD_D6_ODD_BF16>(out, in, size, s);
}

// --- SIGMOID FWD D3-D6 (BF16, Sollya) ---
void launch_sigmoid_fwd_d3_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SIGMOID_FWD_D3_ODD_SOLLYA_BF16>(out, in, size, s);
}
void launch_sigmoid_fwd_d4_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SIGMOID_FWD_D4_ODD_SOLLYA_BF16>(out, in, size, s);
}
void launch_sigmoid_fwd_d5_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SIGMOID_FWD_D5_ODD_SOLLYA_BF16>(out, in, size, s);
}
void launch_sigmoid_fwd_d6_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SIGMOID_FWD_D6_ODD_SOLLYA_BF16>(out, in, size, s);
}

// --- SIGMOID BWD D3-D6 (BF16) ---
void launch_sigmoid_bwd_d3_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SIGMOID_BWD_D3_EVEN_BF16>(gi, go, in, size, s);
}
void launch_sigmoid_bwd_d4_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SIGMOID_BWD_D4_EVEN_BF16>(gi, go, in, size, s);
}
void launch_sigmoid_bwd_d5_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SIGMOID_BWD_D5_EVEN_BF16>(gi, go, in, size, s);
}
void launch_sigmoid_bwd_d6_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SIGMOID_BWD_D6_EVEN_BF16>(gi, go, in, size, s);
}

// --- SIGMOID BWD D3-D6 (BF16, Sollya) ---
void launch_sigmoid_bwd_d3_kernel_bf16_sollya(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SIGMOID_BWD_D3_EVEN_SOLLYA_BF16>(gi, go, in, size, s);
}
void launch_sigmoid_bwd_d4_kernel_bf16_sollya(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SIGMOID_BWD_D4_EVEN_SOLLYA_BF16>(gi, go, in, size, s);
}
void launch_sigmoid_bwd_d5_kernel_bf16_sollya(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SIGMOID_BWD_D5_EVEN_SOLLYA_BF16>(gi, go, in, size, s);
}
void launch_sigmoid_bwd_d6_kernel_bf16_sollya(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SIGMOID_BWD_D6_EVEN_SOLLYA_BF16>(gi, go, in, size, s);
}

// --- TANH FWD D3-D6 (BF16) ---
void launch_tanh_fwd_d3_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<TANH_FWD_D3_ODD_BF16>(out, in, size, s);
}
void launch_tanh_fwd_d4_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<TANH_FWD_D4_ODD_BF16>(out, in, size, s);
}
void launch_tanh_fwd_d5_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<TANH_FWD_D5_ODD_BF16>(out, in, size, s);
}
void launch_tanh_fwd_d6_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<TANH_FWD_D6_ODD_BF16>(out, in, size, s);
}

// --- TANH BWD D3-D6 (BF16) ---
void launch_tanh_bwd_d3_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<TANH_BWD_D3_EVEN_BF16>(gi, go, in, size, s);
}
void launch_tanh_bwd_d4_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<TANH_BWD_D4_EVEN_BF16>(gi, go, in, size, s);
}
void launch_tanh_bwd_d5_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<TANH_BWD_D5_EVEN_BF16>(gi, go, in, size, s);
}
void launch_tanh_bwd_d6_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<TANH_BWD_D6_EVEN_BF16>(gi, go, in, size, s);
}

// --- SWISH FWD D3-D6 (BF16) ---
void launch_swish_fwd_native_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SWISH_FWD_NATIVE_BF16>(out, in, size, s);
}
void launch_swish_fwd_d3_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SWISH_FWD_D3_ODD_BF16>(out, in, size, s);
}
void launch_swish_fwd_d4_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SWISH_FWD_D4_ODD_BF16>(out, in, size, s);
}
void launch_swish_fwd_d5_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SWISH_FWD_D5_ODD_BF16>(out, in, size, s);
}
void launch_swish_fwd_d6_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SWISH_FWD_D6_ODD_BF16>(out, in, size, s);
}

// --- SWISH MUL FWD native and D3-D6 (BF16): out = swish(gate) * up ---
void launch_swish_mul_fwd_native_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bf16<SWISH_FWD_NATIVE_BF16>(out, gate, up, size, s);
}
void launch_swish_mul_fwd_d3_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bf16<SWISH_FWD_D3_ODD_BF16>(out, gate, up, size, s);
}
void launch_swish_mul_fwd_d4_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bf16<SWISH_FWD_D4_ODD_BF16>(out, gate, up, size, s);
}
void launch_swish_mul_fwd_d5_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bf16<SWISH_FWD_D5_ODD_BF16>(out, gate, up, size, s);
}
void launch_swish_mul_fwd_d6_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bf16<SWISH_FWD_D6_ODD_BF16>(out, gate, up, size, s);
}

// --- SWISH MUL BWD native and D3-D6 (BF16): grad for out = swish(gate) * up ---
void launch_swish_mul_bwd_native_kernel_bf16(__nv_bfloat16* grad_gate, __nv_bfloat16* grad_up, const __nv_bfloat16* grad_out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bwd_bf16<SWISH_FWD_NATIVE_BF16, SWISH_BWD_NATIVE_BF16>(grad_gate, grad_up, grad_out, gate, up, size, s);
}
void launch_swish_mul_bwd_d3_kernel_bf16(__nv_bfloat16* grad_gate, __nv_bfloat16* grad_up, const __nv_bfloat16* grad_out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bwd_d3_alg_bf16(
        grad_gate, grad_up, grad_out, gate, up, size, s);
}
void launch_swish_mul_bwd_d4_kernel_bf16(__nv_bfloat16* grad_gate, __nv_bfloat16* grad_up, const __nv_bfloat16* grad_out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bwd_bf16<SWISH_FWD_D4_ODD_BF16, SWISH_BWD_D4_ODD_BF16>(grad_gate, grad_up, grad_out, gate, up, size, s);
}
void launch_swish_mul_bwd_d5_kernel_bf16(__nv_bfloat16* grad_gate, __nv_bfloat16* grad_up, const __nv_bfloat16* grad_out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bwd_bf16<SWISH_FWD_D5_ODD_BF16, SWISH_BWD_D5_ODD_BF16>(grad_gate, grad_up, grad_out, gate, up, size, s);
}
void launch_swish_mul_bwd_d6_kernel_bf16(__nv_bfloat16* grad_gate, __nv_bfloat16* grad_up, const __nv_bfloat16* grad_out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bwd_bf16<SWISH_FWD_D6_ODD_BF16, SWISH_BWD_D6_ODD_BF16>(grad_gate, grad_up, grad_out, gate, up, size, s);
}

// --- PACKED SWISH MUL FWD D3-D6 (BF16): out = swish(packed[..., :H]) * packed[..., H:] ---
void launch_swish_mul_packed_fwd_d3_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    if (hidden_size == 512) {
        launch_swish_mul_packed_h512_d3_bf16(out, packed, rows, s);
    } else {
        launch_swish_mul_packed_bf16<SWISH_FWD_D3_ODD_BF16>(out, packed, rows, hidden_size, s);
    }
}
void launch_swish_mul_packed_fwd_d4_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bf16<SWISH_FWD_D4_ODD_BF16>(out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_fwd_d5_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bf16<SWISH_FWD_D5_ODD_BF16>(out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_fwd_d6_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bf16<SWISH_FWD_D6_ODD_BF16>(out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_fwd_native_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bf16<SWISH_FWD_NATIVE_BF16>(out, packed, rows, hidden_size, s);
}

// --- PACKED SWISH MUL BWD D3-D6 (BF16): packed gradient for packed gate/up ---
void launch_swish_mul_packed_bwd_d3_kernel_bf16(__nv_bfloat16* grad_packed, const __nv_bfloat16* grad_out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bwd_bf16<SWISH_FWD_D3_ODD_BF16, SWISH_BWD_D3_ODD_BF16>(grad_packed, grad_out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_bwd_d4_kernel_bf16(__nv_bfloat16* grad_packed, const __nv_bfloat16* grad_out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bwd_bf16<SWISH_FWD_D4_ODD_BF16, SWISH_BWD_D4_ODD_BF16>(grad_packed, grad_out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_bwd_d5_kernel_bf16(__nv_bfloat16* grad_packed, const __nv_bfloat16* grad_out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bwd_bf16<SWISH_FWD_D5_ODD_BF16, SWISH_BWD_D5_ODD_BF16>(grad_packed, grad_out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_bwd_d6_kernel_bf16(__nv_bfloat16* grad_packed, const __nv_bfloat16* grad_out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bwd_bf16<SWISH_FWD_D6_ODD_BF16, SWISH_BWD_D6_ODD_BF16>(grad_packed, grad_out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_bwd_native_kernel_bf16(__nv_bfloat16* grad_packed, const __nv_bfloat16* grad_out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bwd_bf16<SWISH_FWD_NATIVE_BF16, SWISH_BWD_NATIVE_BF16>(grad_packed, grad_out, packed, rows, hidden_size, s);
}

// --- SWISH BWD D3-D6 (BF16) ---
void launch_swish_bwd_native_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SWISH_BWD_NATIVE_BF16>(gi, go, in, size, s);
}
void launch_swish_bwd_d3_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SWISH_BWD_D3_ODD_BF16>(gi, go, in, size, s);
}
void launch_swish_bwd_d4_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SWISH_BWD_D4_ODD_BF16>(gi, go, in, size, s);
}
void launch_swish_bwd_d5_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SWISH_BWD_D5_ODD_BF16>(gi, go, in, size, s);
}
void launch_swish_bwd_d6_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SWISH_BWD_D6_ODD_BF16>(gi, go, in, size, s);
}

// --- GELU FWD D3-D6 (BF16) ---
void launch_gelu_fwd_d3_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<GELU_FWD_D3_ODD_BF16>(out, in, size, s);
}
void launch_gelu_fwd_d4_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<GELU_FWD_D4_ODD_BF16>(out, in, size, s);
}
void launch_gelu_fwd_d5_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<GELU_FWD_D5_ODD_BF16>(out, in, size, s);
}
void launch_gelu_fwd_d6_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<GELU_FWD_D6_ODD_BF16>(out, in, size, s);
}

// --- GELU BWD D3-D6 (BF16) ---
void launch_gelu_bwd_d3_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<GELU_BWD_D3_ODD_BF16>(gi, go, in, size, s);
}
void launch_gelu_bwd_d4_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<GELU_BWD_D4_ODD_BF16>(gi, go, in, size, s);
}
void launch_gelu_bwd_d5_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<GELU_BWD_D5_ODD_BF16>(gi, go, in, size, s);
}
void launch_gelu_bwd_d6_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<GELU_BWD_D6_ODD_BF16>(gi, go, in, size, s);
}

// --- SWISH FWD D3-D6 (BF16, Sollya) ---
void launch_swish_fwd_d3_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SWISH_FWD_D3_ODD_SOLLYA_BF16>(out, in, size, s);
}
void launch_swish_fwd_d4_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SWISH_FWD_D4_ODD_SOLLYA_BF16>(out, in, size, s);
}
void launch_swish_fwd_d5_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SWISH_FWD_D5_ODD_SOLLYA_BF16>(out, in, size, s);
}
void launch_swish_fwd_d6_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SWISH_FWD_D6_ODD_SOLLYA_BF16>(out, in, size, s);
}

// --- SWISH BWD D3-D6 (BF16, Sollya) ---
void launch_swish_bwd_d3_kernel_bf16_sollya(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SWISH_BWD_D3_ODD_SOLLYA_BF16>(gi, go, in, size, s);
}
void launch_swish_bwd_d4_kernel_bf16_sollya(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SWISH_BWD_D4_ODD_SOLLYA_BF16>(gi, go, in, size, s);
}
void launch_swish_bwd_d5_kernel_bf16_sollya(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SWISH_BWD_D5_ODD_SOLLYA_BF16>(gi, go, in, size, s);
}
void launch_swish_bwd_d6_kernel_bf16_sollya(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SWISH_BWD_D6_ODD_SOLLYA_BF16>(gi, go, in, size, s);
}

// --- FUSED SWISH MUL FWD/BWD D3-D6 (BF16, Sollya) ---
void launch_swish_mul_fwd_d3_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bf16<SWISH_FWD_D3_ODD_SOLLYA_BF16>(out, gate, up, size, s);
}
void launch_swish_mul_fwd_d4_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bf16<SWISH_FWD_D4_ODD_SOLLYA_BF16>(out, gate, up, size, s);
}
void launch_swish_mul_fwd_d5_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bf16<SWISH_FWD_D5_ODD_SOLLYA_BF16>(out, gate, up, size, s);
}
void launch_swish_mul_fwd_d6_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bf16<SWISH_FWD_D6_ODD_SOLLYA_BF16>(out, gate, up, size, s);
}
void launch_swish_mul_bwd_d3_kernel_bf16_sollya(__nv_bfloat16* grad_gate, __nv_bfloat16* grad_up, const __nv_bfloat16* grad_out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bwd_bf16<SWISH_FWD_D3_ODD_SOLLYA_BF16, SWISH_BWD_D3_ODD_SOLLYA_BF16>(grad_gate, grad_up, grad_out, gate, up, size, s);
}
void launch_swish_mul_bwd_d4_kernel_bf16_sollya(__nv_bfloat16* grad_gate, __nv_bfloat16* grad_up, const __nv_bfloat16* grad_out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bwd_bf16<SWISH_FWD_D4_ODD_SOLLYA_BF16, SWISH_BWD_D4_ODD_SOLLYA_BF16>(grad_gate, grad_up, grad_out, gate, up, size, s);
}
void launch_swish_mul_bwd_d5_kernel_bf16_sollya(__nv_bfloat16* grad_gate, __nv_bfloat16* grad_up, const __nv_bfloat16* grad_out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bwd_bf16<SWISH_FWD_D5_ODD_SOLLYA_BF16, SWISH_BWD_D5_ODD_SOLLYA_BF16>(grad_gate, grad_up, grad_out, gate, up, size, s);
}
void launch_swish_mul_bwd_d6_kernel_bf16_sollya(__nv_bfloat16* grad_gate, __nv_bfloat16* grad_up, const __nv_bfloat16* grad_out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int size, cudaStream_t s) {
    launch_swish_mul_bwd_bf16<SWISH_FWD_D6_ODD_SOLLYA_BF16, SWISH_BWD_D6_ODD_SOLLYA_BF16>(grad_gate, grad_up, grad_out, gate, up, size, s);
}

// --- PACKED SWISH MUL FWD/BWD D3-D6 (BF16, Sollya) ---
void launch_swish_mul_packed_fwd_d3_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bf16<SWISH_FWD_D3_ODD_SOLLYA_BF16>(out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_fwd_d4_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bf16<SWISH_FWD_D4_ODD_SOLLYA_BF16>(out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_fwd_d5_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bf16<SWISH_FWD_D5_ODD_SOLLYA_BF16>(out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_fwd_d6_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bf16<SWISH_FWD_D6_ODD_SOLLYA_BF16>(out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_bwd_d3_kernel_bf16_sollya(__nv_bfloat16* grad_packed, const __nv_bfloat16* grad_out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bwd_bf16<SWISH_FWD_D3_ODD_SOLLYA_BF16, SWISH_BWD_D3_ODD_SOLLYA_BF16>(grad_packed, grad_out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_bwd_d4_kernel_bf16_sollya(__nv_bfloat16* grad_packed, const __nv_bfloat16* grad_out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bwd_bf16<SWISH_FWD_D4_ODD_SOLLYA_BF16, SWISH_BWD_D4_ODD_SOLLYA_BF16>(grad_packed, grad_out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_bwd_d5_kernel_bf16_sollya(__nv_bfloat16* grad_packed, const __nv_bfloat16* grad_out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bwd_bf16<SWISH_FWD_D5_ODD_SOLLYA_BF16, SWISH_BWD_D5_ODD_SOLLYA_BF16>(grad_packed, grad_out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_bwd_d6_kernel_bf16_sollya(__nv_bfloat16* grad_packed, const __nv_bfloat16* grad_out, const __nv_bfloat16* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bwd_bf16<SWISH_FWD_D6_ODD_SOLLYA_BF16, SWISH_BWD_D6_ODD_SOLLYA_BF16>(grad_packed, grad_out, packed, rows, hidden_size, s);
}

// --- GELU FWD D3-D6 (BF16, Sollya) ---
void launch_gelu_fwd_d3_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<GELU_FWD_D3_ODD_SOLLYA_BF16>(out, in, size, s);
}
void launch_gelu_fwd_d4_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<GELU_FWD_D4_ODD_SOLLYA_BF16>(out, in, size, s);
}
void launch_gelu_fwd_d5_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<GELU_FWD_D5_ODD_SOLLYA_BF16>(out, in, size, s);
}
void launch_gelu_fwd_d6_kernel_bf16_sollya(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<GELU_FWD_D6_ODD_SOLLYA_BF16>(out, in, size, s);
}

// --- GELU BWD D3-D6 (BF16, Sollya) ---
void launch_gelu_bwd_d3_kernel_bf16_sollya(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<GELU_BWD_D3_ODD_SOLLYA_BF16>(gi, go, in, size, s);
}
void launch_gelu_bwd_d4_kernel_bf16_sollya(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<GELU_BWD_D4_ODD_SOLLYA_BF16>(gi, go, in, size, s);
}
void launch_gelu_bwd_d5_kernel_bf16_sollya(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<GELU_BWD_D5_ODD_SOLLYA_BF16>(gi, go, in, size, s);
}
void launch_gelu_bwd_d6_kernel_bf16_sollya(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<GELU_BWD_D6_ODD_SOLLYA_BF16>(gi, go, in, size, s);
}

// --- BACKWARD COMPAT: old names (BF16) ---
void launch_sigmoid_fwd_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SIGMOID_FWD_D3_ODD_BF16>(out, in, size, s);
}
void launch_sigmoid_bwd_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SIGMOID_BWD_D4_EVEN_BF16>(gi, go, in, size, s);
}
void launch_tanh_fwd_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<TANH_FWD_D3_ODD_BF16>(out, in, size, s);
}
void launch_tanh_bwd_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<TANH_BWD_D4_EVEN_BF16>(gi, go, in, size, s);
}
void launch_swish_fwd_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_unary_bf16<SWISH_FWD_D3_ODD_BF16>(out, in, size, s);
}
void launch_swish_bwd_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* in, int size, cudaStream_t s) {
    launch_binary_bf16<SWISH_BWD_D4_ODD_BF16>(gi, go, in, size, s);
}

// --- ALGEBRAIC BACKWARD (BF16) ---
void launch_sigmoid_bwd_alg_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* y, int size, cudaStream_t s) {
    launch_binary_bf16<SIGMOID_BWD_ALGEBRAIC_BF16>(gi, go, y, size, s);
}
void launch_tanh_bwd_alg_kernel_bf16(__nv_bfloat16* gi, const __nv_bfloat16* go, const __nv_bfloat16* y, int size, cudaStream_t s) {
    launch_binary_bf16<TANH_BWD_ALGEBRAIC_BF16>(gi, go, y, size, s);
}

} // extern C

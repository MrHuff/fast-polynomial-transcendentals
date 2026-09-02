
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "spline_structs_odd.cuh"

// =============================================================================
// Vectorized kernels: 128-bit (int4) loads for optimal memory throughput
// Each thread processes 4 __half2 = 8 half values = 16 bytes per iteration
// Grid-stride loop for large tensors
// =============================================================================

constexpr int BLOCK_SIZE = 256;
constexpr int PACKED_BLOCK_SIZE = 128;
constexpr int PACKED_D3_BLOCK_SIZE = 128;

static inline bool is_aligned_to(const void* ptr, std::uintptr_t alignment) {
    return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
}

static inline bool can_use_half2(const void* a, const void* b, int size) {
    return (size % 2 == 0) && is_aligned_to(a, 4) && is_aligned_to(b, 4);
}

static inline bool can_use_half2(const void* a, const void* b, const void* c, int size) {
    return can_use_half2(a, b, size) && is_aligned_to(c, 4);
}

static inline bool can_use_half2(
    const void* a, const void* b, const void* c, const void* d, const void* e, int size) {
    return can_use_half2(a, b, c, size) && is_aligned_to(d, 4) && is_aligned_to(e, 4);
}

static inline bool can_use_int4(const void* a, const void* b, int size) {
    return (size % 8 == 0) && is_aligned_to(a, 16) && is_aligned_to(b, 16);
}

static inline bool can_use_int4(const void* a, const void* b, const void* c, int size) {
    return can_use_int4(a, b, size) && is_aligned_to(c, 16);
}

static inline bool can_use_int4(
    const void* a, const void* b, const void* c, const void* d, const void* e, int size) {
    return can_use_int4(a, b, c, size) && is_aligned_to(d, 16) && is_aligned_to(e, 16);
}

// Unary FWD: out = F(in), vectorized
template <typename Func>
__global__ void __launch_bounds__(BLOCK_SIZE)
unary_vec_kernel(const int4* __restrict__ input,
                 int4* __restrict__ output,
                 int n_vec) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        int4 data = input[i];
        __half2* h2 = reinterpret_cast<__half2*>(&data);
        #pragma unroll
        for (int k = 0; k < 4; k++)
            h2[k] = Func::evaluate(h2[k]);
        output[i] = data;
    }
}

// Binary BWD: grad_in = grad_out * F'(in), vectorized
template <typename GradFunc>
__global__ void __launch_bounds__(BLOCK_SIZE)
binary_vec_kernel(const int4* __restrict__ grad_output,
                  const int4* __restrict__ input,
                  int4* __restrict__ grad_input,
                  int n_vec) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        int4 go_data = grad_output[i];
        int4 in_data = input[i];
        __half2* go = reinterpret_cast<__half2*>(&go_data);
        __half2* x  = reinterpret_cast<__half2*>(&in_data);
        int4 out_data;
        __half2* gi = reinterpret_cast<__half2*>(&out_data);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            __half2 ds = GradFunc::evaluate(x[k]);
            gi[k] = __hmul2(go[k], ds);
        }
        grad_input[i] = out_data;
    }
}

// Scalar fallback for tail elements
template <typename Func>
__global__ void unary_scalar_kernel(const __half2* __restrict__ input,
                                     __half2* __restrict__ output,
                                     int n_h2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2)
        output[idx] = Func::evaluate(input[idx]);
}

template <typename Func>
__global__ void unary_element_kernel(const __half* __restrict__ input,
                                     __half* __restrict__ output,
                                     int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        __half2 x = __halves2half2(input[idx], input[idx]);
        output[idx] = __low2half(Func::evaluate(x));
    }
}

template <typename GradFunc>
__global__ void binary_scalar_kernel(const __half2* __restrict__ grad_output,
                                      const __half2* __restrict__ input,
                                      __half2* __restrict__ grad_input,
                                      int n_h2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __half2 ds = GradFunc::evaluate(input[idx]);
        grad_input[idx] = __hmul2(grad_output[idx], ds);
    }
}

template <typename GradFunc>
__global__ void binary_element_kernel(const __half* __restrict__ grad_output,
                                      const __half* __restrict__ input,
                                      __half* __restrict__ grad_input,
                                      int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        __half2 go = __halves2half2(grad_output[idx], grad_output[idx]);
        __half2 x = __halves2half2(input[idx], input[idx]);
        grad_input[idx] = __low2half(__hmul2(go, GradFunc::evaluate(x)));
    }
}

// Fused SwiGLU elementwise: out = swish(gate) * up, vectorized
template <typename Func>
__global__ void __launch_bounds__(BLOCK_SIZE)
swish_mul_vec_kernel(const int4* __restrict__ gate,
                     const int4* __restrict__ up,
                     int4* __restrict__ output,
                     int n_vec) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        int4 gate_data = gate[i];
        int4 up_data = up[i];
        int4 out_data;
        __half2* gate_h2 = reinterpret_cast<__half2*>(&gate_data);
        __half2* up_h2 = reinterpret_cast<__half2*>(&up_data);
        __half2* out_h2 = reinterpret_cast<__half2*>(&out_data);
        #pragma unroll
        for (int k = 0; k < 4; k++)
            out_h2[k] = __hmul2(Func::evaluate(gate_h2[k]), up_h2[k]);
        output[i] = out_data;
    }
}

template <typename Func>
__global__ void swish_mul_scalar_kernel(const __half2* __restrict__ gate,
                                        const __half2* __restrict__ up,
                                        __half2* __restrict__ output,
                                        int n_h2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2)
        output[idx] = __hmul2(Func::evaluate(gate[idx]), up[idx]);
}

template <typename Func>
__global__ void swish_mul_element_kernel(const __half* __restrict__ gate,
                                         const __half* __restrict__ up,
                                         __half* __restrict__ output,
                                         int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        __half2 g = __halves2half2(gate[idx], gate[idx]);
        __half2 u = __halves2half2(up[idx], up[idx]);
        output[idx] = __low2half(__hmul2(Func::evaluate(g), u));
    }
}

template <typename FwdFunc, typename BwdFunc>
__global__ void __launch_bounds__(BLOCK_SIZE)
swish_mul_bwd_vec_kernel(const int4* __restrict__ grad_output,
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
        __half2* go_h2 = reinterpret_cast<__half2*>(&go_data);
        __half2* gate_h2 = reinterpret_cast<__half2*>(&gate_data);
        __half2* up_h2 = reinterpret_cast<__half2*>(&up_data);
        __half2* gg_h2 = reinterpret_cast<__half2*>(&gg_data);
        __half2* gu_h2 = reinterpret_cast<__half2*>(&gu_data);
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
__global__ void swish_mul_bwd_scalar_kernel(const __half2* __restrict__ grad_output,
                                            const __half2* __restrict__ gate,
                                            const __half2* __restrict__ up,
                                            __half2* __restrict__ grad_gate,
                                            __half2* __restrict__ grad_up,
                                            int n_h2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __half2 go = grad_output[idx];
        __half2 g = gate[idx];
        grad_gate[idx] = __hmul2(__hmul2(go, up[idx]), BwdFunc::evaluate(g));
        grad_up[idx] = __hmul2(go, FwdFunc::evaluate(g));
    }
}

template <typename FwdFunc, typename BwdFunc>
__global__ void swish_mul_bwd_element_kernel(const __half* __restrict__ grad_output,
                                             const __half* __restrict__ gate,
                                             const __half* __restrict__ up,
                                             __half* __restrict__ grad_gate,
                                             __half* __restrict__ grad_up,
                                             int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        __half2 go = __halves2half2(grad_output[idx], grad_output[idx]);
        __half2 g = __halves2half2(gate[idx], gate[idx]);
        __half2 u = __halves2half2(up[idx], up[idx]);
        grad_gate[idx] = __low2half(__hmul2(__hmul2(go, u), BwdFunc::evaluate(g)));
        grad_up[idx] = __low2half(__hmul2(go, FwdFunc::evaluate(g)));
    }
}

// Fused packed SwiGLU elementwise for tensors laid out as [..., gate, up].
template <typename Func>
__global__ void __launch_bounds__(PACKED_BLOCK_SIZE)
swish_mul_packed_vec_kernel(const __half* __restrict__ packed,
                            __half* __restrict__ output,
                            int rows,
                            int hidden_size,
                            int vecs_per_row) {
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        const int64_t row_offset = static_cast<int64_t>(row) * hidden_size;
        for (int vec_col = threadIdx.x + blockIdx.x * blockDim.x;
             vec_col < vecs_per_row;
             vec_col += blockDim.x * gridDim.x) {
        const int col = vec_col * 8;
        const __half* gate_ptr = packed + row_offset * 2 + col;
        const __half* up_ptr = gate_ptr + hidden_size;

        int4 gate_data = *reinterpret_cast<const int4*>(gate_ptr);
        int4 up_data = *reinterpret_cast<const int4*>(up_ptr);
        int4 out_data;
        __half2* gate_h2 = reinterpret_cast<__half2*>(&gate_data);
        __half2* up_h2 = reinterpret_cast<__half2*>(&up_data);
        __half2* out_h2 = reinterpret_cast<__half2*>(&out_data);
        #pragma unroll
        for (int k = 0; k < 4; k++)
            out_h2[k] = __hmul2(Func::evaluate(gate_h2[k]), up_h2[k]);
            reinterpret_cast<int4*>(output + row_offset)[vec_col] = out_data;
        }
    }
}

template <typename Func>
__global__ void __launch_bounds__(BLOCK_SIZE)
swish_mul_packed_h512_vec_kernel(const __half* __restrict__ packed,
                                 __half* __restrict__ output,
                                 int rows) {
    int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows)
        return;
    int vec_col = threadIdx.x & 63;
    const int64_t row_offset = static_cast<int64_t>(row) * 512;
    const __half* gate_ptr = packed + row_offset * 2 + vec_col * 8;
    const __half* up_ptr = gate_ptr + 512;

    int4 gate_data = *reinterpret_cast<const int4*>(gate_ptr);
    int4 up_data = *reinterpret_cast<const int4*>(up_ptr);
    int4 out_data;
    __half2* gate_h2 = reinterpret_cast<__half2*>(&gate_data);
    __half2* up_h2 = reinterpret_cast<__half2*>(&up_data);
    __half2* out_h2 = reinterpret_cast<__half2*>(&out_data);
    #pragma unroll
    for (int k = 0; k < 4; k++)
        out_h2[k] = __hmul2(Func::evaluate(gate_h2[k]), up_h2[k]);
    reinterpret_cast<int4*>(output + row_offset)[vec_col] = out_data;
}

__global__ void __launch_bounds__(PACKED_D3_BLOCK_SIZE)
swish_mul_packed_h512_d3_vec_kernel(const __half* __restrict__ packed,
                                    __half* __restrict__ output,
                                    int rows) {
    #pragma unroll
    for (int item = 0; item < 2; item++) {
        int flat = threadIdx.x + item * PACKED_D3_BLOCK_SIZE;
        int row = blockIdx.x * 4 + (flat >> 6);
        if (row >= rows)
            continue;
        int vec_col = flat & 63;
        const int64_t row_offset = static_cast<int64_t>(row) * 512;
        const __half* gate_ptr = packed + row_offset * 2 + vec_col * 8;
        const __half* up_ptr = gate_ptr + 512;

        int4 gate_data = *reinterpret_cast<const int4*>(gate_ptr);
        int4 up_data = *reinterpret_cast<const int4*>(up_ptr);
        int4 out_data;
        __half2* gate_h2 = reinterpret_cast<__half2*>(&gate_data);
        __half2* up_h2 = reinterpret_cast<__half2*>(&up_data);
        __half2* out_h2 = reinterpret_cast<__half2*>(&out_data);
        #pragma unroll
        for (int k = 0; k < 4; k++)
            out_h2[k] = __hmul2(SWISH_FWD_D3_ODD::evaluate(gate_h2[k]), up_h2[k]);
        reinterpret_cast<int4*>(output + row_offset)[vec_col] = out_data;
    }
}

template <typename Func>
__global__ void swish_mul_packed_scalar_kernel(const __half* __restrict__ packed,
                                               __half* __restrict__ output,
                                               int rows,
                                               int hidden_size,
                                               int pairs_per_row) {
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        const int64_t row_offset = static_cast<int64_t>(row) * hidden_size;
        for (int pair_col = threadIdx.x + blockIdx.x * blockDim.x;
             pair_col < pairs_per_row;
             pair_col += blockDim.x * gridDim.x) {
        const int col = pair_col * 2;
        const __half2* gate = reinterpret_cast<const __half2*>(
            packed + row_offset * 2 + col);
        const __half2* up = reinterpret_cast<const __half2*>(
            packed + row_offset * 2 + hidden_size + col);
            reinterpret_cast<__half2*>(output + row_offset)[pair_col] =
                __hmul2(Func::evaluate(*gate), *up);
        }
    }
}

template <typename FwdFunc, typename BwdFunc>
static __device__ __forceinline__ void swish_fwd_bwd(
    __half2 x,
    __half2& swish,
    __half2& derivative) {
    swish = FwdFunc::evaluate(x);
    derivative = BwdFunc::evaluate(x);
}

struct SWISH_FWD_NATIVE {
    static __device__ __forceinline__ __half2 evaluate(__half2 x) {
        const float x0 = __half2float(__low2half(x));
        const float x1 = __half2float(__high2half(x));
        return __floats2half2_rn(
            x0 / (1.0f + __expf(-x0)),
            x1 / (1.0f + __expf(-x1)));
    }
};
struct SWISH_BWD_NATIVE {
    static __device__ __forceinline__ __half2 evaluate(__half2 x) {
        const float x0 = __half2float(__low2half(x));
        const float x1 = __half2float(__high2half(x));
        const float sigmoid0 = 1.0f / (1.0f + __expf(-x0));
        const float sigmoid1 = 1.0f / (1.0f + __expf(-x1));
        return __floats2half2_rn(
            sigmoid0 * (1.0f + x0 * (1.0f - sigmoid0)),
            sigmoid1 * (1.0f + x1 * (1.0f - sigmoid1)));
    }
};

template <typename FwdFunc, typename BwdFunc>
static __device__ __forceinline__ void swish_mul_packed_grads(
    __half2 grad_output,
    __half2 gate,
    __half2 up,
    __half2& grad_gate,
    __half2& grad_up) {
    __half2 swish;
    __half2 derivative;
    swish_fwd_bwd<FwdFunc, BwdFunc>(gate, swish, derivative);
    grad_gate = __hmul2(__hmul2(grad_output, up), derivative);
    grad_up = __hmul2(grad_output, swish);
}

template <>
__device__ __forceinline__ void swish_mul_packed_grads<
    SWISH_FWD_NATIVE, SWISH_BWD_NATIVE>(
    __half2 grad_output,
    __half2 gate,
    __half2 up,
    __half2& grad_gate,
    __half2& grad_up) {
    const __half2 dy = __hmul2(grad_output, up);
    const float dy0 = __half2float(__low2half(dy));
    const float dy1 = __half2float(__high2half(dy));
    const float x0 = __half2float(__low2half(gate));
    const float x1 = __half2float(__high2half(gate));
    const float sigmoid0 = 1.0f / (1.0f + __expf(-x0));
    const float sigmoid1 = 1.0f / (1.0f + __expf(-x1));
    grad_gate = __floats2half2_rn(
        dy0 * sigmoid0 * (1.0f + x0 * (1.0f - sigmoid0)),
        dy1 * sigmoid1 * (1.0f + x1 * (1.0f - sigmoid1)));
    grad_up = __hmul2(grad_output, SWISH_FWD_NATIVE::evaluate(gate));
}

template <typename FwdFunc, typename BwdFunc>
__global__ void __launch_bounds__(PACKED_BLOCK_SIZE)
swish_mul_packed_bwd_vec_kernel(
    const __half* __restrict__ grad_output,
    const __half* __restrict__ packed,
    __half* __restrict__ grad_packed,
    int rows,
    int hidden_size,
    int vecs_per_row) {
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        const int64_t row_offset = static_cast<int64_t>(row) * hidden_size;
        for (int vec_col = threadIdx.x + blockIdx.x * blockDim.x;
             vec_col < vecs_per_row;
             vec_col += blockDim.x * gridDim.x) {
            const int col = vec_col * 8;
            const __half* go_ptr = grad_output + row_offset + col;
            const __half* gate_ptr = packed + row_offset * 2 + col;
            const __half* up_ptr = gate_ptr + hidden_size;
            __half* grad_gate_ptr = grad_packed + row_offset * 2 + col;
            __half* grad_up_ptr = grad_gate_ptr + hidden_size;

            const int4 go_data = *reinterpret_cast<const int4*>(go_ptr);
            const int4 gate_data = *reinterpret_cast<const int4*>(gate_ptr);
            const int4 up_data = *reinterpret_cast<const int4*>(up_ptr);
            int4 grad_gate_data;
            int4 grad_up_data;
            const __half2* go_h2 = reinterpret_cast<const __half2*>(&go_data);
            const __half2* gate_h2 = reinterpret_cast<const __half2*>(&gate_data);
            const __half2* up_h2 = reinterpret_cast<const __half2*>(&up_data);
            __half2* grad_gate_h2 = reinterpret_cast<__half2*>(&grad_gate_data);
            __half2* grad_up_h2 = reinterpret_cast<__half2*>(&grad_up_data);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                swish_mul_packed_grads<FwdFunc, BwdFunc>(
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
__global__ void swish_mul_packed_bwd_scalar_kernel(
    const __half* __restrict__ grad_output,
    const __half* __restrict__ packed,
    __half* __restrict__ grad_packed,
    int rows,
    int hidden_size,
    int pairs_per_row) {
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        const int64_t row_offset = static_cast<int64_t>(row) * hidden_size;
        for (int pair_col = threadIdx.x + blockIdx.x * blockDim.x;
             pair_col < pairs_per_row;
             pair_col += blockDim.x * gridDim.x) {
            const int col = pair_col * 2;
            const __half2 go = *reinterpret_cast<const __half2*>(
                grad_output + row_offset + col);
            const __half2 gate = *reinterpret_cast<const __half2*>(
                packed + row_offset * 2 + col);
            const __half2 up = *reinterpret_cast<const __half2*>(
                packed + row_offset * 2 + hidden_size + col);
            __half2 grad_gate;
            __half2 grad_up;
            swish_mul_packed_grads<FwdFunc, BwdFunc>(
                go, gate, up, grad_gate, grad_up);
            *reinterpret_cast<__half2*>(
                grad_packed + row_offset * 2 + col) = grad_gate;
            *reinterpret_cast<__half2*>(
                grad_packed + row_offset * 2 + hidden_size + col) = grad_up;
        }
    }
}

// =============================================================================
// Grid sizing with occupancy API
// =============================================================================

template <auto Kernel>
static int compute_grid(int n_work_items) {
    int min_grid = (n_work_items + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Query device L2 size once
    static int l2_bytes = 0;
    static int sm_count = 0;
    if (l2_bytes == 0) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device);
    }

    // Each work item = 16 bytes (int4). Estimate total footprint as
    // 2-3 tensors * n_work_items * 16B. Use 2x as conservative threshold.
    long long footprint = (long long)n_work_items * 16LL * 2LL;

    if (footprint > (long long)l2_bytes) {
        // HBM-bound: uncapped grid for best memory interleaving.
        // Many blocks each doing 1 pass >> few blocks doing 100s of waves.
        return min_grid;
    }

    // L2-resident: cap at occupancy * SMs.
    // Grid-stride loop gives each thread multiple iterations, providing
    // instruction-level parallelism that hides polynomial evaluation latency.
    static const int blocks_per_sm = [] {
        int value = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &value, Kernel, BLOCK_SIZE, 0);
        return value;
    }();

    int max_grid = blocks_per_sm * sm_count;
    return min(min_grid, max_grid);
}

constexpr int VEC4_THRESHOLD = 4096;

// =============================================================================
// Launch helpers — dispatch vectorized or scalar based on size/alignment
// =============================================================================

template <typename Func>
static void launch_unary(
    __half* out, const __half* in, int size, cudaStream_t stream)
{
    if (size <= 0)
        return;
    int n_h2 = size / 2;
    if (n_h2 >= VEC4_THRESHOLD && can_use_int4(in, out, size)) {
        int n_vec = n_h2 / 4;
        auto kernel = unary_vec_kernel<Func>;
        int grid = compute_grid<unary_vec_kernel<Func>>(n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(in),
            reinterpret_cast<int4*>(out),
            n_vec);
    } else if (can_use_half2(in, out, size)) {
        int grid = (n_h2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        unary_scalar_kernel<Func><<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half2*>(in),
            reinterpret_cast<__half2*>(out),
            n_h2);
    } else {
        int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        unary_element_kernel<Func><<<grid, BLOCK_SIZE, 0, stream>>>(in, out, size);
    }
}

template <typename GradFunc>
static void launch_binary(
    __half* grad_in, const __half* grad_out, const __half* in,
    int size, cudaStream_t stream)
{
    if (size <= 0)
        return;
    int n_h2 = size / 2;
    if (n_h2 >= VEC4_THRESHOLD && can_use_int4(grad_out, in, grad_in, size)) {
        int n_vec = n_h2 / 4;
        auto kernel = binary_vec_kernel<GradFunc>;
        int grid = compute_grid<binary_vec_kernel<GradFunc>>(n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(grad_out),
            reinterpret_cast<const int4*>(in),
            reinterpret_cast<int4*>(grad_in),
            n_vec);
    } else if (can_use_half2(grad_out, in, grad_in, size)) {
        int grid = (n_h2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        binary_scalar_kernel<GradFunc><<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half2*>(grad_out),
            reinterpret_cast<const __half2*>(in),
            reinterpret_cast<__half2*>(grad_in),
            n_h2);
    } else {
        int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        binary_element_kernel<GradFunc><<<grid, BLOCK_SIZE, 0, stream>>>(
            grad_out,
            in,
            grad_in,
            size);
    }
}

template <typename Func>
static void launch_swish_mul(
    __half* out, const __half* gate, const __half* up, int size, cudaStream_t stream)
{
    if (size <= 0)
        return;
    int n_h2 = size / 2;
    if (n_h2 >= VEC4_THRESHOLD && can_use_int4(gate, up, out, size)) {
        int n_vec = n_h2 / 4;
        auto kernel = swish_mul_vec_kernel<Func>;
        int grid = compute_grid<swish_mul_vec_kernel<Func>>(n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(gate),
            reinterpret_cast<const int4*>(up),
            reinterpret_cast<int4*>(out),
            n_vec);
    } else if (can_use_half2(gate, up, out, size)) {
        int grid = (n_h2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        swish_mul_scalar_kernel<Func><<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half2*>(gate),
            reinterpret_cast<const __half2*>(up),
            reinterpret_cast<__half2*>(out),
            n_h2);
    } else {
        int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        swish_mul_element_kernel<Func><<<grid, BLOCK_SIZE, 0, stream>>>(
            gate,
            up,
            out,
            size);
    }
}

template <typename FwdFunc, typename BwdFunc>
static void launch_swish_mul_bwd(
    __half* grad_gate,
    __half* grad_up,
    const __half* grad_out,
    const __half* gate,
    const __half* up,
    int size,
    cudaStream_t stream)
{
    if (size <= 0)
        return;
    int n_h2 = size / 2;
    if (n_h2 >= VEC4_THRESHOLD && can_use_int4(grad_out, gate, up, grad_gate, grad_up, size)) {
        int n_vec = n_h2 / 4;
        auto kernel = swish_mul_bwd_vec_kernel<FwdFunc, BwdFunc>;
        int grid = compute_grid<swish_mul_bwd_vec_kernel<FwdFunc, BwdFunc>>(n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(grad_out),
            reinterpret_cast<const int4*>(gate),
            reinterpret_cast<const int4*>(up),
            reinterpret_cast<int4*>(grad_gate),
            reinterpret_cast<int4*>(grad_up),
            n_vec);
    } else if (can_use_half2(grad_out, gate, up, grad_gate, grad_up, size)) {
        int grid = (n_h2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        swish_mul_bwd_scalar_kernel<FwdFunc, BwdFunc><<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half2*>(grad_out),
            reinterpret_cast<const __half2*>(gate),
            reinterpret_cast<const __half2*>(up),
            reinterpret_cast<__half2*>(grad_gate),
            reinterpret_cast<__half2*>(grad_up),
            n_h2);
    } else {
        int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        swish_mul_bwd_element_kernel<FwdFunc, BwdFunc><<<grid, BLOCK_SIZE, 0, stream>>>(
            grad_out,
            gate,
            up,
            grad_gate,
            grad_up,
            size);
    }
}

static void launch_swish_mul_packed_h512_d3(
    __half* out, const __half* packed, int rows, cudaStream_t stream)
{
    int grid = (rows + 3) / 4;
    swish_mul_packed_h512_d3_vec_kernel<<<grid, PACKED_D3_BLOCK_SIZE, 0, stream>>>(
        packed,
        out,
        rows);
}

template <typename Func>
static void launch_swish_mul_packed(
    __half* out, const __half* packed, int rows, int hidden_size, cudaStream_t stream)
{
    const int64_t size = static_cast<int64_t>(rows) * hidden_size;
    const int64_t n_h2 = size / 2;
    if (n_h2 >= VEC4_THRESHOLD && hidden_size == 512) {
        int grid = (rows + 3) / 4;
        swish_mul_packed_h512_vec_kernel<Func><<<grid, BLOCK_SIZE, 0, stream>>>(
            packed,
            out,
            rows);
    } else if (n_h2 >= VEC4_THRESHOLD && (hidden_size % 8 == 0)) {
        int vecs_per_row = hidden_size / 8;
        int grid_x = (vecs_per_row + PACKED_BLOCK_SIZE - 1) / PACKED_BLOCK_SIZE;
        int grid_y = min(rows, 65535);
        swish_mul_packed_vec_kernel<Func><<<dim3(grid_x, grid_y), PACKED_BLOCK_SIZE, 0, stream>>>(
            packed,
            out,
            rows,
            hidden_size,
            vecs_per_row);
    } else {
        int pairs_per_row = hidden_size / 2;
        int grid_x = (pairs_per_row + PACKED_BLOCK_SIZE - 1) / PACKED_BLOCK_SIZE;
        if (grid_x == 0) grid_x = 1;
        int grid_y = min(rows, 65535);
        swish_mul_packed_scalar_kernel<Func><<<dim3(grid_x, grid_y), PACKED_BLOCK_SIZE, 0, stream>>>(
            packed,
            out,
            rows,
            hidden_size,
            pairs_per_row);
    }
}

template <typename FwdFunc, typename BwdFunc>
static void launch_swish_mul_packed_bwd(
    __half* grad_packed,
    const __half* grad_output,
    const __half* packed,
    int rows,
    int hidden_size,
    cudaStream_t stream) {
    const int64_t size = static_cast<int64_t>(rows) * hidden_size;
    if (size <= 0)
        return;
    const int grid_y = min(rows, 65535);
    if (
        hidden_size % 8 == 0
        && is_aligned_to(grad_output, 16)
        && is_aligned_to(packed, 16)
        && is_aligned_to(grad_packed, 16)
    ) {
        const int vecs_per_row = hidden_size / 8;
        const int grid_x =
            (vecs_per_row + PACKED_BLOCK_SIZE - 1) / PACKED_BLOCK_SIZE;
        swish_mul_packed_bwd_vec_kernel<FwdFunc, BwdFunc>
            <<<dim3(grid_x, grid_y), PACKED_BLOCK_SIZE, 0, stream>>>(
                grad_output, packed, grad_packed, rows, hidden_size, vecs_per_row);
    } else {
        const int pairs_per_row = hidden_size / 2;
        int grid_x =
            (pairs_per_row + PACKED_BLOCK_SIZE - 1) / PACKED_BLOCK_SIZE;
        if (grid_x == 0) grid_x = 1;
        swish_mul_packed_bwd_scalar_kernel<FwdFunc, BwdFunc>
            <<<dim3(grid_x, grid_y), PACKED_BLOCK_SIZE, 0, stream>>>(
                grad_output, packed, grad_packed, rows, hidden_size, pairs_per_row);
    }
}

// =============================================================================
// SFU helper functions (for hybrid kernels)
// =============================================================================

__device__ __forceinline__ __half2 sigmoid_sfu(__half2 val) {
    __half2 one = __float2half2_rn(1.0f);
    return h2rcp(__hadd2(one, h2exp(__hneg2(val))));
}

__device__ __forceinline__ __half2 tanh_sfu(__half2 val) {
    return h2tanh_approx(val);
}

__device__ __forceinline__ __half2 swish_sfu(__half2 val) {
    __half2 one = __float2half2_rn(1.0f);
    return __hmul2(val, h2rcp(__hadd2(one, h2exp(__hneg2(val)))));
}

// SFU-based backward helpers
__device__ __forceinline__ __half2 sigmoid_bwd_sfu(__half2 x) {
    __half2 one = __float2half2_rn(1.0f);
    __half2 y = h2rcp(__hadd2(one, h2exp(__hneg2(x))));
    return __hmul2(y, __hsub2(one, y));  // y*(1-y)
}

__device__ __forceinline__ __half2 tanh_bwd_sfu(__half2 x) {
    __half2 one = __float2half2_rn(1.0f);
    __half2 y = h2tanh_approx(x);
    return __hsub2(one, __hmul2(y, y));  // 1-y²
}

__device__ __forceinline__ __half2 swish_bwd_sfu(__half2 x) {
    __half2 one = __float2half2_rn(1.0f);
    __half2 sig = h2rcp(__hadd2(one, h2exp(__hneg2(x))));
    return __hmul2(sig, __hadd2(one, __hmul2(x, __hsub2(one, sig))));
    // sig * (1 + x * (1 - sig))
}

// =============================================================================
// Hybrid SFU+FMA Forward Kernel — 4-wide, route SFU_N through SFU, rest poly
// =============================================================================

template <int SFU_N, typename SplineFunc, __half2 (*SfuFunc)(__half2)>
__global__ void __launch_bounds__(BLOCK_SIZE)
hybrid_vec_kernel(const int4* __restrict__ input,
                  int4* __restrict__ output,
                  int n_vec) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        int4 data = input[i];
        __half2* h2 = reinterpret_cast<__half2*>(&data);
        // Route first SFU_N through SFU pipe, rest through FMA (polynomial)
        if constexpr (SFU_N >= 1) h2[0] = SfuFunc(h2[0]);
        else                      h2[0] = SplineFunc::evaluate(h2[0]);
        if constexpr (SFU_N >= 2) h2[1] = SfuFunc(h2[1]);
        else                      h2[1] = SplineFunc::evaluate(h2[1]);
        if constexpr (SFU_N >= 3) h2[2] = SfuFunc(h2[2]);
        else                      h2[2] = SplineFunc::evaluate(h2[2]);
        if constexpr (SFU_N >= 4) h2[3] = SfuFunc(h2[3]);
        else                      h2[3] = SplineFunc::evaluate(h2[3]);
        output[i] = data;
    }
}

// Hybrid backward variant — same pattern for BWD
template <int SFU_N, typename GradFunc, __half2 (*SfuBwdFunc)(__half2)>
__global__ void __launch_bounds__(BLOCK_SIZE)
hybrid_bwd_vec_kernel(const int4* __restrict__ grad_output,
                       const int4* __restrict__ input,
                       int4* __restrict__ grad_input,
                       int n_vec) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        int4 go_data = grad_output[i];
        int4 in_data = input[i];
        __half2* go = reinterpret_cast<__half2*>(&go_data);
        __half2* x  = reinterpret_cast<__half2*>(&in_data);
        int4 out_data;
        __half2* gi = reinterpret_cast<__half2*>(&out_data);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            __half2 ds;
            if (k < SFU_N) ds = SfuBwdFunc(x[k]);
            else           ds = GradFunc::evaluate(x[k]);
            gi[k] = __hmul2(go[k], ds);
        }
        grad_input[i] = out_data;
    }
}

// =============================================================================
// Fused FWD+BWD Kernel — computes y=f(x) and gi=go*f'(y) in one pass
// Reads x once instead of twice. Saves 20% HBM traffic.
// =============================================================================

template <typename FwdFunc, typename AlgBwdFunc>
__global__ void __launch_bounds__(BLOCK_SIZE)
fused_fwd_bwd_kernel(const int4* __restrict__ input,
                      const int4* __restrict__ grad_output,
                      int4* __restrict__ fwd_out,
                      int4* __restrict__ grad_input,
                      int n_vec) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        int4 x_data  = input[i];
        int4 go_data = grad_output[i];
        __half2* x  = reinterpret_cast<__half2*>(&x_data);
        __half2* go = reinterpret_cast<__half2*>(&go_data);
        int4 y_data, gi_data;
        __half2* y  = reinterpret_cast<__half2*>(&y_data);
        __half2* gi = reinterpret_cast<__half2*>(&gi_data);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            y[k] = FwdFunc::evaluate(x[k]);         // forward
            __half2 dy = AlgBwdFunc::evaluate(y[k]); // derivative from y
            gi[k] = __hmul2(go[k], dy);              // grad_input
        }
        fwd_out[i]    = y_data;
        grad_input[i] = gi_data;
    }
}

// Fused launch helper
template <typename FwdFunc, typename AlgBwdFunc>
static void launch_fused(
    __half* y_out, __half* gi_out,
    const __half* x_in, const __half* go_in,
    int size, cudaStream_t stream)
{
    int n_h2 = size / 2;
    if (n_h2 >= VEC4_THRESHOLD && (n_h2 % 4 == 0)) {
        int n_vec = n_h2 / 4;
        auto kernel = fused_fwd_bwd_kernel<FwdFunc, AlgBwdFunc>;
        int grid = compute_grid<fused_fwd_bwd_kernel<FwdFunc, AlgBwdFunc>>(n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(x_in),
            reinterpret_cast<const int4*>(go_in),
            reinterpret_cast<int4*>(y_out),
            reinterpret_cast<int4*>(gi_out),
            n_vec);
    } else {
        // Fallback: separate FWD + BWD
        launch_unary<FwdFunc>(y_out, x_in, size, stream);
        launch_binary<AlgBwdFunc>(gi_out, go_in, y_out, size, stream);
    }
}

// =============================================================================
// FWD-WITH-DERIVATIVE KERNELS (for standard autograd)
// Forward computes y=f(x) AND dy=f'(x), saves dy for backward.
// Backward is just gi = go * saved_dy (trivial multiply).
// =============================================================================

// Variant A: Poly FWD + algebraic derivative from y
template <typename FwdFunc, typename AlgBwdFunc>
__global__ void __launch_bounds__(BLOCK_SIZE)
fwd_deriv_alg_kernel(const int4* __restrict__ input,
                      int4* __restrict__ fwd_out,
                      int4* __restrict__ deriv_out,
                      int n_vec) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        int4 x_data = input[i];
        __half2* x = reinterpret_cast<__half2*>(&x_data);
        int4 y_data, dy_data;
        __half2* y  = reinterpret_cast<__half2*>(&y_data);
        __half2* dy = reinterpret_cast<__half2*>(&dy_data);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            y[k]  = FwdFunc::evaluate(x[k]);         // polynomial forward
            dy[k] = AlgBwdFunc::evaluate(y[k]);      // algebraic derivative from y
        }
        fwd_out[i]   = y_data;
        deriv_out[i] = dy_data;
    }
}

// Variant B: Poly FWD + poly BWD (both from x, sharing interval logic in x)
template <typename FwdFunc, typename BwdFunc>
__global__ void __launch_bounds__(BLOCK_SIZE)
fwd_deriv_poly_kernel(const int4* __restrict__ input,
                       int4* __restrict__ fwd_out,
                       int4* __restrict__ deriv_out,
                       int n_vec) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        int4 x_data = input[i];
        __half2* x = reinterpret_cast<__half2*>(&x_data);
        int4 y_data, dy_data;
        __half2* y  = reinterpret_cast<__half2*>(&y_data);
        __half2* dy = reinterpret_cast<__half2*>(&dy_data);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            y[k]  = FwdFunc::evaluate(x[k]);    // polynomial forward
            dy[k] = BwdFunc::evaluate(x[k]);    // polynomial derivative from x
        }
        fwd_out[i]   = y_data;
        deriv_out[i] = dy_data;
    }
}

// Variant C: Swish-specific with shared sigmoid computation
// y = x * sigmoid(x), f'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
// Both share the sigmoid polynomial evaluation!
template <typename SigFwdFunc>
__global__ void __launch_bounds__(BLOCK_SIZE)
fwd_deriv_swish_kernel(const int4* __restrict__ input,
                        int4* __restrict__ fwd_out,
                        int4* __restrict__ deriv_out,
                        int n_vec) {
    __half2 one = __float2half2_rn(1.0f);
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        int4 x_data = input[i];
        __half2* x = reinterpret_cast<__half2*>(&x_data);
        int4 y_data, dy_data;
        __half2* y  = reinterpret_cast<__half2*>(&y_data);
        __half2* dy = reinterpret_cast<__half2*>(&dy_data);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            __half2 sig = SigFwdFunc::evaluate(x[k]);   // sigmoid(x) — computed once!
            y[k]  = __hmul2(x[k], sig);                  // swish = x * sigmoid(x)
            // f'(x) = sig * (1 + x * (1 - sig))
            dy[k] = __hmul2(sig, __hfma2(x[k], __hsub2(one, sig), one));
        }
        fwd_out[i]   = y_data;
        deriv_out[i] = dy_data;
    }
}

// Trivial multiply kernel for backward: gi = go * saved_dy
__global__ void __launch_bounds__(BLOCK_SIZE)
multiply_vec_kernel(const int4* __restrict__ a,
                     const int4* __restrict__ b,
                     int4* __restrict__ out,
                     int n_vec) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_vec;
         i += blockDim.x * gridDim.x) {
        int4 a_data = a[i];
        int4 b_data = b[i];
        __half2* av = reinterpret_cast<__half2*>(&a_data);
        __half2* bv = reinterpret_cast<__half2*>(&b_data);
        int4 c_data;
        __half2* cv = reinterpret_cast<__half2*>(&c_data);
        #pragma unroll
        for (int k = 0; k < 4; k++)
            cv[k] = __hmul2(av[k], bv[k]);
        out[i] = c_data;
    }
}

// Launch helpers for fwd_with_deriv variants
template <typename FwdFunc, typename AlgBwdFunc>
static void launch_fwd_deriv_alg(
    __half* y_out, __half* dy_out, const __half* x_in,
    int size, cudaStream_t stream)
{
    int n_h2 = size / 2;
    if (n_h2 >= VEC4_THRESHOLD && (n_h2 % 4 == 0)) {
        int n_vec = n_h2 / 4;
        auto kernel = fwd_deriv_alg_kernel<FwdFunc, AlgBwdFunc>;
        int grid = compute_grid<fwd_deriv_alg_kernel<FwdFunc, AlgBwdFunc>>(n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(x_in),
            reinterpret_cast<int4*>(y_out),
            reinterpret_cast<int4*>(dy_out),
            n_vec);
    } else {
        launch_unary<FwdFunc>(y_out, x_in, size, stream);
        launch_binary<AlgBwdFunc>(dy_out, y_out, y_out, size, stream);  // hack: go=y, x=y — alg just uses "x"
    }
}

template <typename FwdFunc, typename BwdFunc>
static void launch_fwd_deriv_poly(
    __half* y_out, __half* dy_out, const __half* x_in,
    int size, cudaStream_t stream)
{
    int n_h2 = size / 2;
    if (n_h2 >= VEC4_THRESHOLD && (n_h2 % 4 == 0)) {
        int n_vec = n_h2 / 4;
        auto kernel = fwd_deriv_poly_kernel<FwdFunc, BwdFunc>;
        int grid = compute_grid<fwd_deriv_poly_kernel<FwdFunc, BwdFunc>>(n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(x_in),
            reinterpret_cast<int4*>(y_out),
            reinterpret_cast<int4*>(dy_out),
            n_vec);
    } else {
        launch_unary<FwdFunc>(y_out, x_in, size, stream);
        // For scalar fallback, call BWD poly with dummy go=ones
        launch_unary<BwdFunc>(dy_out, x_in, size, stream);
    }
}

template <typename SigFwdFunc>
static void launch_fwd_deriv_swish(
    __half* y_out, __half* dy_out, const __half* x_in,
    int size, cudaStream_t stream)
{
    int n_h2 = size / 2;
    if (n_h2 >= VEC4_THRESHOLD && (n_h2 % 4 == 0)) {
        int n_vec = n_h2 / 4;
        auto kernel = fwd_deriv_swish_kernel<SigFwdFunc>;
        int grid = compute_grid<fwd_deriv_swish_kernel<SigFwdFunc>>(n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(x_in),
            reinterpret_cast<int4*>(y_out),
            reinterpret_cast<int4*>(dy_out),
            n_vec);
    } else {
        // Fallback: separate evaluations
        launch_unary<SWISH_FWD_D3_ODD>(y_out, x_in, size, stream);
        launch_unary<SWISH_BWD_D4_ODD>(dy_out, x_in, size, stream);
    }
}

// Multiply launcher
static void launch_multiply(
    __half* out, const __half* a, const __half* b,
    int size, cudaStream_t stream)
{
    int n_h2 = size / 2;
    if (n_h2 >= VEC4_THRESHOLD && (n_h2 % 4 == 0)) {
        int n_vec = n_h2 / 4;
        auto kernel = multiply_vec_kernel;
        int grid = compute_grid<multiply_vec_kernel>(n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(a),
            reinterpret_cast<const int4*>(b),
            reinterpret_cast<int4*>(out),
            n_vec);
    } else {
        // scalar fallback
        int grid = (n_h2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid == 0) grid = 1;
        // Reuse binary_scalar_kernel with a dummy "identity" struct? No — just use at::mul
        // For now, just call vec version with adjusted size
        multiply_vec_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(a),
            reinterpret_cast<const int4*>(b),
            reinterpret_cast<int4*>(out),
            n_h2 / 4);
    }
}



// Hybrid launch helper
template <int SFU_N, typename SplineFunc, __half2 (*SfuFunc)(__half2)>
static void launch_hybrid_unary(
    __half* out, const __half* in, int size, cudaStream_t stream)
{
    int n_h2 = size / 2;
    if (n_h2 >= VEC4_THRESHOLD && (n_h2 % 4 == 0)) {
        int n_vec = n_h2 / 4;
        auto kernel = hybrid_vec_kernel<SFU_N, SplineFunc, SfuFunc>;
        int grid = compute_grid<hybrid_vec_kernel<SFU_N, SplineFunc, SfuFunc>>(
            n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(in),
            reinterpret_cast<int4*>(out),
            n_vec);
    } else {
        // Fallback to pure polynomial for small sizes
        launch_unary<SplineFunc>(out, in, size, stream);
    }
}

// Hybrid BWD launch helper
template <int SFU_N, typename GradFunc, __half2 (*SfuBwdFunc)(__half2)>
static void launch_hybrid_binary(
    __half* gi, const __half* go, const __half* in,
    int size, cudaStream_t stream)
{
    int n_h2 = size / 2;
    if (n_h2 >= VEC4_THRESHOLD && (n_h2 % 4 == 0)) {
        int n_vec = n_h2 / 4;
        auto kernel = hybrid_bwd_vec_kernel<SFU_N, GradFunc, SfuBwdFunc>;
        int grid = compute_grid<hybrid_bwd_vec_kernel<SFU_N, GradFunc, SfuBwdFunc>>(
            n_vec);
        kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const int4*>(go),
            reinterpret_cast<const int4*>(in),
            reinterpret_cast<int4*>(gi),
            n_vec);
    } else {
        launch_binary<GradFunc>(gi, go, in, size, stream);
    }
}

// =============================================================================
// Launchers — extern "C" for linkage from spline_ops.cpp
// =============================================================================

extern "C" {

// --- SIGMOID FWD D3-D6 ---
void launch_sigmoid_fwd_d3_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<SIGMOID_FWD_D3_ODD>(out, in, size, s);
}
void launch_sigmoid_fwd_d4_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<SIGMOID_FWD_D4_ODD>(out, in, size, s);
}
void launch_sigmoid_fwd_d5_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<SIGMOID_FWD_D5_ODD>(out, in, size, s);
}
void launch_sigmoid_fwd_d6_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<SIGMOID_FWD_D6_ODD>(out, in, size, s);
}

// --- SIGMOID BWD D3-D6 ---
void launch_sigmoid_bwd_d3_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<SIGMOID_BWD_D3_EVEN>(gi, go, in, size, s);
}
void launch_sigmoid_bwd_d4_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<SIGMOID_BWD_D4_EVEN>(gi, go, in, size, s);
}
void launch_sigmoid_bwd_d5_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<SIGMOID_BWD_D5_EVEN>(gi, go, in, size, s);
}
void launch_sigmoid_bwd_d6_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<SIGMOID_BWD_D6_EVEN>(gi, go, in, size, s);
}

// --- TANH FWD D3-D6 ---
void launch_tanh_fwd_d3_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<TANH_FWD_D3_ODD>(out, in, size, s);
}
void launch_tanh_fwd_d4_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<TANH_FWD_D4_ODD>(out, in, size, s);
}
void launch_tanh_fwd_d5_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<TANH_FWD_D5_ODD>(out, in, size, s);
}
void launch_tanh_fwd_d6_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<TANH_FWD_D6_ODD>(out, in, size, s);
}

// --- TANH BWD D3-D6 ---
void launch_tanh_bwd_d3_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<TANH_BWD_D3_EVEN>(gi, go, in, size, s);
}
void launch_tanh_bwd_d4_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<TANH_BWD_D4_EVEN>(gi, go, in, size, s);
}
void launch_tanh_bwd_d5_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<TANH_BWD_D5_EVEN>(gi, go, in, size, s);
}
void launch_tanh_bwd_d6_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<TANH_BWD_D6_EVEN>(gi, go, in, size, s);
}

// --- SWISH FWD D3-D6 ---
void launch_swish_fwd_native_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<SWISH_FWD_NATIVE>(out, in, size, s);
}
void launch_swish_fwd_d3_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<SWISH_FWD_D3_ODD>(out, in, size, s);
}
void launch_swish_fwd_d4_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<SWISH_FWD_D4_ODD>(out, in, size, s);
}
void launch_swish_fwd_d5_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<SWISH_FWD_D5_ODD>(out, in, size, s);
}
void launch_swish_fwd_d6_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<SWISH_FWD_D6_ODD>(out, in, size, s);
}

// --- SWISH MUL FWD native and D3-D6: out = swish(gate) * up ---
void launch_swish_mul_fwd_native_kernel(__half* out, const __half* gate, const __half* up, int size, cudaStream_t s) {
    launch_swish_mul<SWISH_FWD_NATIVE>(out, gate, up, size, s);
}
void launch_swish_mul_fwd_d3_kernel(__half* out, const __half* gate, const __half* up, int size, cudaStream_t s) {
    launch_swish_mul<SWISH_FWD_D3_ODD>(out, gate, up, size, s);
}
void launch_swish_mul_fwd_d4_kernel(__half* out, const __half* gate, const __half* up, int size, cudaStream_t s) {
    launch_swish_mul<SWISH_FWD_D4_ODD>(out, gate, up, size, s);
}
void launch_swish_mul_fwd_d5_kernel(__half* out, const __half* gate, const __half* up, int size, cudaStream_t s) {
    launch_swish_mul<SWISH_FWD_D5_ODD>(out, gate, up, size, s);
}
void launch_swish_mul_fwd_d6_kernel(__half* out, const __half* gate, const __half* up, int size, cudaStream_t s) {
    launch_swish_mul<SWISH_FWD_D6_ODD>(out, gate, up, size, s);
}

// --- SWISH MUL BWD native and D3-D6: grad for out = swish(gate) * up ---
void launch_swish_mul_bwd_native_kernel(__half* grad_gate, __half* grad_up, const __half* grad_out, const __half* gate, const __half* up, int size, cudaStream_t s) {
    launch_swish_mul_bwd<SWISH_FWD_NATIVE, SWISH_BWD_NATIVE>(grad_gate, grad_up, grad_out, gate, up, size, s);
}
void launch_swish_mul_bwd_d3_kernel(__half* grad_gate, __half* grad_up, const __half* grad_out, const __half* gate, const __half* up, int size, cudaStream_t s) {
    launch_swish_mul_bwd<SWISH_FWD_D3_ODD, SWISH_BWD_D3_ODD>(grad_gate, grad_up, grad_out, gate, up, size, s);
}
void launch_swish_mul_bwd_d4_kernel(__half* grad_gate, __half* grad_up, const __half* grad_out, const __half* gate, const __half* up, int size, cudaStream_t s) {
    launch_swish_mul_bwd<SWISH_FWD_D4_ODD, SWISH_BWD_D4_ODD>(grad_gate, grad_up, grad_out, gate, up, size, s);
}
void launch_swish_mul_bwd_d5_kernel(__half* grad_gate, __half* grad_up, const __half* grad_out, const __half* gate, const __half* up, int size, cudaStream_t s) {
    launch_swish_mul_bwd<SWISH_FWD_D5_ODD, SWISH_BWD_D5_ODD>(grad_gate, grad_up, grad_out, gate, up, size, s);
}
void launch_swish_mul_bwd_d6_kernel(__half* grad_gate, __half* grad_up, const __half* grad_out, const __half* gate, const __half* up, int size, cudaStream_t s) {
    launch_swish_mul_bwd<SWISH_FWD_D6_ODD, SWISH_BWD_D6_ODD>(grad_gate, grad_up, grad_out, gate, up, size, s);
}

// --- PACKED SWISH MUL FWD D3-D6: out = swish(packed[..., :H]) * packed[..., H:] ---
void launch_swish_mul_packed_fwd_d3_kernel(__half* out, const __half* packed, int rows, int hidden_size, cudaStream_t s) {
    if (hidden_size == 512) {
        launch_swish_mul_packed_h512_d3(out, packed, rows, s);
    } else {
        launch_swish_mul_packed<SWISH_FWD_D3_ODD>(out, packed, rows, hidden_size, s);
    }
}
void launch_swish_mul_packed_fwd_d4_kernel(__half* out, const __half* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed<SWISH_FWD_D4_ODD>(out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_fwd_d5_kernel(__half* out, const __half* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed<SWISH_FWD_D5_ODD>(out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_fwd_d6_kernel(__half* out, const __half* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed<SWISH_FWD_D6_ODD>(out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_fwd_native_kernel(__half* out, const __half* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed<SWISH_FWD_NATIVE>(out, packed, rows, hidden_size, s);
}

// --- PACKED SWISH MUL BWD D3-D6: packed gradient for packed gate/up ---
void launch_swish_mul_packed_bwd_d3_kernel(__half* grad_packed, const __half* grad_out, const __half* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bwd<SWISH_FWD_D3_ODD, SWISH_BWD_D3_ODD>(grad_packed, grad_out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_bwd_d4_kernel(__half* grad_packed, const __half* grad_out, const __half* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bwd<SWISH_FWD_D4_ODD, SWISH_BWD_D4_ODD>(grad_packed, grad_out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_bwd_d5_kernel(__half* grad_packed, const __half* grad_out, const __half* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bwd<SWISH_FWD_D5_ODD, SWISH_BWD_D5_ODD>(grad_packed, grad_out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_bwd_d6_kernel(__half* grad_packed, const __half* grad_out, const __half* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bwd<SWISH_FWD_D6_ODD, SWISH_BWD_D6_ODD>(grad_packed, grad_out, packed, rows, hidden_size, s);
}
void launch_swish_mul_packed_bwd_native_kernel(__half* grad_packed, const __half* grad_out, const __half* packed, int rows, int hidden_size, cudaStream_t s) {
    launch_swish_mul_packed_bwd<SWISH_FWD_NATIVE, SWISH_BWD_NATIVE>(grad_packed, grad_out, packed, rows, hidden_size, s);
}

// --- SWISH BWD D3-D6 ---
void launch_swish_bwd_native_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<SWISH_BWD_NATIVE>(gi, go, in, size, s);
}
void launch_swish_bwd_d3_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<SWISH_BWD_D3_ODD>(gi, go, in, size, s);
}
void launch_swish_bwd_d4_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<SWISH_BWD_D4_ODD>(gi, go, in, size, s);
}
void launch_swish_bwd_d5_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<SWISH_BWD_D5_ODD>(gi, go, in, size, s);
}
void launch_swish_bwd_d6_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<SWISH_BWD_D6_ODD>(gi, go, in, size, s);
}

// --- GELU FWD D3-D6 ---
void launch_gelu_fwd_d3_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<GELU_FWD_D3_ODD>(out, in, size, s);
}
void launch_gelu_fwd_d4_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<GELU_FWD_D4_ODD>(out, in, size, s);
}
void launch_gelu_fwd_d5_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<GELU_FWD_D5_ODD>(out, in, size, s);
}
void launch_gelu_fwd_d6_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<GELU_FWD_D6_ODD>(out, in, size, s);
}

// --- GELU BWD D3-D6 ---
void launch_gelu_bwd_d3_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<GELU_BWD_D3_ODD>(gi, go, in, size, s);
}
void launch_gelu_bwd_d4_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<GELU_BWD_D4_ODD>(gi, go, in, size, s);
}
void launch_gelu_bwd_d5_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<GELU_BWD_D5_ODD>(gi, go, in, size, s);
}
void launch_gelu_bwd_d6_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<GELU_BWD_D6_ODD>(gi, go, in, size, s);
}

// --- BACKWARD COMPAT: old names ---
void launch_sigmoid_fwd_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<SIGMOID_FWD_D3_ODD>(out, in, size, s);
}
void launch_sigmoid_bwd_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<SIGMOID_BWD_D4_EVEN>(gi, go, in, size, s);
}
void launch_tanh_fwd_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<TANH_FWD_D3_ODD>(out, in, size, s);
}
void launch_tanh_bwd_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<TANH_BWD_D4_EVEN>(gi, go, in, size, s);
}
void launch_swish_fwd_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_unary<SWISH_FWD_D3_ODD>(out, in, size, s);
}
void launch_swish_bwd_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_binary<SWISH_BWD_D4_ODD>(gi, go, in, size, s);
}

// --- ALGEBRAIC BACKWARD (uses cached forward output y, not raw input x) ---
void launch_sigmoid_bwd_alg_kernel(__half* gi, const __half* go, const __half* y, int size, cudaStream_t s) {
    launch_binary<SIGMOID_BWD_ALGEBRAIC>(gi, go, y, size, s);
}
void launch_tanh_bwd_alg_kernel(__half* gi, const __half* go, const __half* y, int size, cudaStream_t s) {
    launch_binary<TANH_BWD_ALGEBRAIC>(gi, go, y, size, s);
}

// --- HYBRID FWD: 1 SFU + 3 Polynomial ---
void launch_sigmoid_fwd_hybrid_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_hybrid_unary<1, SIGMOID_FWD_D3_ODD, sigmoid_sfu>(out, in, size, s);
}
void launch_tanh_fwd_hybrid_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_hybrid_unary<1, TANH_FWD_D3_ODD, tanh_sfu>(out, in, size, s);
}
void launch_swish_fwd_hybrid_kernel(__half* out, const __half* in, int size, cudaStream_t s) {
    launch_hybrid_unary<1, SWISH_FWD_D3_ODD, swish_sfu>(out, in, size, s);
}

// --- HYBRID BWD: 1 SFU + 3 Polynomial ---
void launch_sigmoid_bwd_hybrid_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_hybrid_binary<1, SIGMOID_BWD_D4_EVEN, sigmoid_bwd_sfu>(gi, go, in, size, s);
}
void launch_tanh_bwd_hybrid_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_hybrid_binary<1, TANH_BWD_D4_EVEN, tanh_bwd_sfu>(gi, go, in, size, s);
}
void launch_swish_bwd_hybrid_kernel(__half* gi, const __half* go, const __half* in, int size, cudaStream_t s) {
    launch_hybrid_binary<1, SWISH_BWD_D4_ODD, swish_bwd_sfu>(gi, go, in, size, s);
}

// --- FUSED FWD+BWD: single pass, reads x once (needs go available) ---
void launch_sigmoid_fused_kernel(
    __half* y_out, __half* gi_out,
    const __half* x_in, const __half* go_in, int size, cudaStream_t s) {
    launch_fused<SIGMOID_FWD_D3_ODD, SIGMOID_BWD_ALGEBRAIC>(y_out, gi_out, x_in, go_in, size, s);
}
void launch_tanh_fused_kernel(
    __half* y_out, __half* gi_out,
    const __half* x_in, const __half* go_in, int size, cudaStream_t s) {
    launch_fused<TANH_FWD_D3_ODD, TANH_BWD_ALGEBRAIC>(y_out, gi_out, x_in, go_in, size, s);
}
void launch_swish_fused_kernel(
    __half* y_out, __half* gi_out,
    const __half* x_in, const __half* go_in, int size, cudaStream_t s) {
    // Swish fused: compute sigmoid poly → derive both y and gi
    // Uses fwd_deriv_swish internally + multiply
    // TODO: implement a true fused kernel for swish
    launch_fwd_deriv_swish<SIGMOID_FWD_D3_ODD>(y_out, gi_out, x_in, size, s);
    // gi_out now holds f'(x), need to multiply by go — BUT this approach
    // doesn't work for the "fused with go" pattern. Use separate approach:
    // Actually, let's just do: compute sigmoid, y = x*sig, dy = sig*(1+x*(1-sig)), gi = go*dy
    // We need a 3-input fused kernel for swish. For now, fall back to separate.
    launch_unary<SWISH_FWD_D3_ODD>(y_out, x_in, size, s);
    launch_binary<SWISH_BWD_D4_ODD>(gi_out, go_in, x_in, size, s);
}

// --- FWD WITH DERIVATIVE: standard autograd pattern ---
// Forward writes (y, dy), backward just does gi = go * dy

// Sigmoid: poly FWD + algebraic derivative from y
void launch_sigmoid_fwd_deriv_alg_kernel(
    __half* y_out, __half* dy_out, const __half* x_in, int size, cudaStream_t s) {
    launch_fwd_deriv_alg<SIGMOID_FWD_D3_ODD, SIGMOID_BWD_ALGEBRAIC>(y_out, dy_out, x_in, size, s);
}
// Sigmoid: poly FWD + poly BWD (both from x)
void launch_sigmoid_fwd_deriv_poly_kernel(
    __half* y_out, __half* dy_out, const __half* x_in, int size, cudaStream_t s) {
    launch_fwd_deriv_poly<SIGMOID_FWD_D3_ODD, SIGMOID_BWD_D4_EVEN>(y_out, dy_out, x_in, size, s);
}
// Tanh: poly FWD + algebraic derivative from y
void launch_tanh_fwd_deriv_alg_kernel(
    __half* y_out, __half* dy_out, const __half* x_in, int size, cudaStream_t s) {
    launch_fwd_deriv_alg<TANH_FWD_D3_ODD, TANH_BWD_ALGEBRAIC>(y_out, dy_out, x_in, size, s);
}
// Tanh: poly FWD + poly BWD (both from x)
void launch_tanh_fwd_deriv_poly_kernel(
    __half* y_out, __half* dy_out, const __half* x_in, int size, cudaStream_t s) {
    launch_fwd_deriv_poly<TANH_FWD_D3_ODD, TANH_BWD_D4_EVEN>(y_out, dy_out, x_in, size, s);
}
// Swish: shared sigmoid computation
void launch_swish_fwd_deriv_kernel(
    __half* y_out, __half* dy_out, const __half* x_in, int size, cudaStream_t s) {
    launch_fwd_deriv_swish<SIGMOID_FWD_D3_ODD>(y_out, dy_out, x_in, size, s);
}
// Swish: poly FWD + poly BWD (both from x)
void launch_swish_fwd_deriv_poly_kernel(
    __half* y_out, __half* dy_out, const __half* x_in, int size, cudaStream_t s) {
    launch_fwd_deriv_poly<SWISH_FWD_D3_ODD, SWISH_BWD_D4_ODD>(y_out, dy_out, x_in, size, s);
}

// Trivial multiply: gi = go * saved_dy
void launch_multiply_kernel(
    __half* out, const __half* a, const __half* b, int size, cudaStream_t s) {
    launch_multiply(out, a, b, size, s);
}

} // extern C

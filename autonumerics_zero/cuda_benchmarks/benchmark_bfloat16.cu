// =============================================================================
// SOTA BFloat16 Activation Benchmark — Spline ODD Kernels D3-D6 (BFloat16)
// =============================================================================
// BFloat16 version. Benchmarks spline-based activation functions vs native baselines.
// Covers: Sigmoid, Tanh, Swish, ERF, GELU (FWD D3-D6 and BWD D3-D6).
//
// Build: nvcc -O3 -arch=sm_100 --use_fast_math \
//        -I../spline_ops -o benchmark_bfloat16 benchmark_bfloat16.cu
// =============================================================================

#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

#define N_ELEMENTS (1024 * 1024 * 16) // 16M elements
#define BLOCK_SIZE 256
#define N_REPEATS 100
#define INNER_REPEATS 10

// Include struct definitions
#include "spline_structs_odd_bf16.cuh"

// =============================================================================
// UTILITY KERNELS
// =============================================================================

inline void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

__global__ void init_kernel(__nv_bfloat162* data, int n_h2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        float f = (float)(idx % 20000) * 0.001f - 10.0f;
        data[idx] = __float2bfloat162_rn(f);
    }
}

__global__ void verify_kernel(const __nv_bfloat162* ref, const __nv_bfloat162* test, int n_h2, float* max_err) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        float2 r = __bfloat1622float2(ref[idx]);
        float2 t = __bfloat1622float2(test[idx]);
        float err = fmaxf(fabsf(r.x - t.x), fabsf(r.y - t.y));
        int* err_int = (int*)max_err;
        atomicMax(err_int, __float_as_int(err));
    }
}

void verify_func(const char* name, void(*ref_kernel)(__nv_bfloat162*,int,int),
                 void(*test_kernel)(__nv_bfloat162*,int,int), int n_h2, float max_tol) {
    printf("Verifying %-45s", name);
    size_t bytes = n_h2 * sizeof(__nv_bfloat162);
    __nv_bfloat162 *d_ref, *d_test;
    float *d_max_err;
    check(cudaMalloc(&d_ref, bytes), "malloc ref");
    check(cudaMalloc(&d_test, bytes), "malloc test");
    check(cudaMalloc(&d_max_err, sizeof(float)), "malloc err");
    init_kernel<<<(n_h2+255)/256, 256>>>(d_ref, n_h2);
    init_kernel<<<(n_h2+255)/256, 256>>>(d_test, n_h2);
    ref_kernel<<<(n_h2+255)/256, 256>>>(d_ref, n_h2, 1);
    test_kernel<<<(n_h2+255)/256, 256>>>(d_test, n_h2, 1);
    check(cudaMemset(d_max_err, 0, sizeof(float)), "zero err");
    verify_kernel<<<(n_h2+255)/256, 256>>>(d_ref, d_test, n_h2, d_max_err);
    float max_err = 0.0f;
    check(cudaMemcpy(&max_err, d_max_err, sizeof(float), cudaMemcpyDeviceToHost), "memcpy err");
    if (max_err > max_tol)
        printf("Max Err: %f  *** HIGH ERROR *** (Tol: %f)\n", max_err, max_tol);
    else
        printf("Max Err: %f  OK\n", max_err);
    cudaFree(d_ref); cudaFree(d_test); cudaFree(d_max_err);
}

void run_bench(const char* name, void(*kernel)(__nv_bfloat162*,int,int), __nv_bfloat162* d_data, int n_h2) {
    printf("%-40s ... ", name); fflush(stdout);
    int block = BLOCK_SIZE;
    int grid = (n_h2 + block - 1) / block;
    init_kernel<<<grid, block>>>(d_data, n_h2);
    check(cudaDeviceSynchronize(), "init sync");
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create");
    check(cudaEventCreate(&stop), "event create");
    kernel<<<grid, block>>>(d_data, n_h2, INNER_REPEATS);  // warmup
    check(cudaDeviceSynchronize(), "warmup sync");
    check(cudaEventRecord(start), "record start");
    for (int i = 0; i < N_REPEATS; ++i)
        kernel<<<grid, block>>>(d_data, n_h2, INNER_REPEATS);
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "event sync");
    float ms_total = 0.0f;
    check(cudaEventElapsedTime(&ms_total, start, stop), "elapsed time");
    printf("%8.4f ms\n", ms_total / N_REPEATS); fflush(stdout);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// =============================================================================
// MACRO: Generate spline kernel wrapper from struct
// =============================================================================
#define SPLINE_KERNEL(name, StructType)                                    \
__global__ void name(__nv_bfloat162* data, int n_h2, int inner_repeats) {         \
    int idx = threadIdx.x + blockIdx.x * blockDim.x;                       \
    if (idx < n_h2) {                                                      \
        __nv_bfloat162 val = data[idx];                                           \
        _Pragma("unroll")                                                  \
        for (int i = 0; i < inner_repeats; ++i)                            \
            val = StructType::evaluate(val);                               \
        data[idx] = val;                                                   \
    }                                                                      \
}

// =============================================================================
// NATIVE BASELINE KERNELS
// =============================================================================

__global__ void fwd_sigmoid_native(__nv_bfloat162* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __nv_bfloat162 val = data[idx];
        __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
        #pragma unroll
        for (int i = 0; i < inner_repeats; ++i)
            val = h2rcp(__hadd2(one, h2exp(__hneg2(val))));
        data[idx] = val;
    }
}

__global__ void fwd_tanh_native(__nv_bfloat162* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __nv_bfloat162 val = data[idx];
        #pragma unroll
        for (int i = 0; i < inner_repeats; ++i)
            val = __hsub2(__hmul2(__float2bfloat162_rn(2.0f), h2rcp(__hadd2(__float2bfloat162_rn(1.0f), h2exp(__hneg2(__hmul2(__float2bfloat162_rn(2.0f), val)))))), __float2bfloat162_rn(1.0f));
        data[idx] = val;
    }
}

__global__ void fwd_swish_native(__nv_bfloat162* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __nv_bfloat162 val = data[idx];
        __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
        #pragma unroll
        for (int i = 0; i < inner_repeats; ++i) {
            __nv_bfloat162 sig = h2rcp(__hadd2(one, h2exp(__hneg2(val))));
            val = __hmul2(val, sig);
        }
        data[idx] = val;
    }
}

__global__ void grad_sigmoid_native(__nv_bfloat162* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __nv_bfloat162 val = data[idx];
        __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
        #pragma unroll
        for (int i = 0; i < inner_repeats; ++i) {
            __nv_bfloat162 sig = h2rcp(__hadd2(one, h2exp(__hneg2(val))));
            val = __hmul2(sig, __hsub2(one, sig));
        }
        data[idx] = val;
    }
}

__global__ void grad_tanh_native(__nv_bfloat162* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __nv_bfloat162 val = data[idx];
        __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
        #pragma unroll
        for (int i = 0; i < inner_repeats; ++i) {
            __nv_bfloat162 t = __hsub2(__hmul2(__float2bfloat162_rn(2.0f), h2rcp(__hadd2(__float2bfloat162_rn(1.0f), h2exp(__hneg2(__hmul2(__float2bfloat162_rn(2.0f), val)))))), __float2bfloat162_rn(1.0f));
            val = __hsub2(one, __hmul2(t, t));
        }
        data[idx] = val;
    }
}

__global__ void grad_swish_native(__nv_bfloat162* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __nv_bfloat162 val = data[idx];
        __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
        #pragma unroll
        for (int i = 0; i < inner_repeats; ++i) {
            __nv_bfloat162 sig = h2rcp(__hadd2(one, h2exp(__hneg2(val))));
            val = __hmul2(sig, __hadd2(one, __hmul2(val, __hsub2(one, sig))));
        }
        data[idx] = val;
    }
}

// --- Native GELU: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))) ---
__global__ void fwd_gelu_native(__nv_bfloat162* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __nv_bfloat162 val = data[idx];
        float2 f = __bfloat1622float2(val);
        #pragma unroll
        for (int i = 0; i < inner_repeats; ++i) {
            f.x = f.x * 0.5f * (1.0f + erff(f.x * 0.7071067811865475f));
            f.y = f.y * 0.5f * (1.0f + erff(f.y * 0.7071067811865475f));
        }
        data[idx] = __float22bfloat162_rn(f);
    }
}

// --- Native GELU backward: GELU'(x) = 0.5*(1+erf(x/sqrt(2))) + x*exp(-x²/2)/sqrt(2*pi) ---
__global__ void grad_gelu_native(__nv_bfloat162* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __nv_bfloat162 val = data[idx];
        float2 f = __bfloat1622float2(val);
        #pragma unroll
        for (int i = 0; i < inner_repeats; ++i) {
            float inv_sqrt2 = 0.7071067811865475f;
            float inv_sqrt2pi = 0.3989422804014327f;
            float phi_x = 0.5f * (1.0f + erff(f.x * inv_sqrt2));
            float phi_y = 0.5f * (1.0f + erff(f.y * inv_sqrt2));
            f.x = phi_x + f.x * expf(-0.5f * f.x * f.x) * inv_sqrt2pi;
            f.y = phi_y + f.y * expf(-0.5f * f.y * f.y) * inv_sqrt2pi;
        }
        data[idx] = __float22bfloat162_rn(f);
    }
}

// --- Native ERF: erf(x) via SFU (expf-based) ---
__global__ void fwd_erf_native(__nv_bfloat162* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __nv_bfloat162 val = data[idx];
        float2 f = __bfloat1622float2(val);
        #pragma unroll
        for (int i = 0; i < inner_repeats; ++i) {
            f.x = erff(f.x);
            f.y = erff(f.y);
        }
        data[idx] = __float22bfloat162_rn(f);
    }
}


// =============================================================================
// SPLINE KERNELS — All D3-D6 via macro
// =============================================================================

// --- Sigmoid FWD D3-D6 ---
SPLINE_KERNEL(fwd_sigmoid_d3, SIGMOID_FWD_D3_ODD_BF16)
SPLINE_KERNEL(fwd_sigmoid_d4, SIGMOID_FWD_D4_ODD_BF16)
SPLINE_KERNEL(fwd_sigmoid_d5, SIGMOID_FWD_D5_ODD_BF16)
SPLINE_KERNEL(fwd_sigmoid_d6, SIGMOID_FWD_D6_ODD_BF16)

// --- Tanh FWD D3-D6 ---
SPLINE_KERNEL(fwd_tanh_d3, TANH_FWD_D3_ODD_BF16)
SPLINE_KERNEL(fwd_tanh_d4, TANH_FWD_D4_ODD_BF16)
SPLINE_KERNEL(fwd_tanh_d5, TANH_FWD_D5_ODD_BF16)
SPLINE_KERNEL(fwd_tanh_d6, TANH_FWD_D6_ODD_BF16)

// --- Swish FWD D3-D6 ---
SPLINE_KERNEL(fwd_swish_d3, SWISH_FWD_D3_ODD_BF16)
SPLINE_KERNEL(fwd_swish_d4, SWISH_FWD_D4_ODD_BF16)
SPLINE_KERNEL(fwd_swish_d5, SWISH_FWD_D5_ODD_BF16)
SPLINE_KERNEL(fwd_swish_d6, SWISH_FWD_D6_ODD_BF16)

// --- Sigmoid BWD D3-D6 ---
SPLINE_KERNEL(grad_sigmoid_d3, SIGMOID_BWD_D3_EVEN_BF16)
SPLINE_KERNEL(grad_sigmoid_d4, SIGMOID_BWD_D4_EVEN_BF16)
SPLINE_KERNEL(grad_sigmoid_d5, SIGMOID_BWD_D5_EVEN_BF16)
SPLINE_KERNEL(grad_sigmoid_d6, SIGMOID_BWD_D6_EVEN_BF16)

// --- Tanh BWD D3-D6 ---
SPLINE_KERNEL(grad_tanh_d3, TANH_BWD_D3_EVEN_BF16)
SPLINE_KERNEL(grad_tanh_d4, TANH_BWD_D4_EVEN_BF16)
SPLINE_KERNEL(grad_tanh_d5, TANH_BWD_D5_EVEN_BF16)
SPLINE_KERNEL(grad_tanh_d6, TANH_BWD_D6_EVEN_BF16)

// --- Swish BWD D3-D6 ---
SPLINE_KERNEL(grad_swish_d3, SWISH_BWD_D3_ODD_BF16)
SPLINE_KERNEL(grad_swish_d4, SWISH_BWD_D4_ODD_BF16)
SPLINE_KERNEL(grad_swish_d5, SWISH_BWD_D5_ODD_BF16)
SPLINE_KERNEL(grad_swish_d6, SWISH_BWD_D6_ODD_BF16)

// --- ERF FWD D3-D6 ---
SPLINE_KERNEL(fwd_erf_d3, ERF_FWD_D3_ODD_BF16)
SPLINE_KERNEL(fwd_erf_d4, ERF_FWD_D4_ODD_BF16)
SPLINE_KERNEL(fwd_erf_d5, ERF_FWD_D5_ODD_BF16)
SPLINE_KERNEL(fwd_erf_d6, ERF_FWD_D6_ODD_BF16)

// --- GELU FWD D3-D6 ---
SPLINE_KERNEL(fwd_gelu_d3, GELU_FWD_D3_ODD_BF16)
SPLINE_KERNEL(fwd_gelu_d4, GELU_FWD_D4_ODD_BF16)
SPLINE_KERNEL(fwd_gelu_d5, GELU_FWD_D5_ODD_BF16)
SPLINE_KERNEL(fwd_gelu_d6, GELU_FWD_D6_ODD_BF16)

// --- GELU BWD D3-D6 ---
SPLINE_KERNEL(grad_gelu_d3, GELU_BWD_D3_ODD_BF16)
SPLINE_KERNEL(grad_gelu_d4, GELU_BWD_D4_ODD_BF16)
SPLINE_KERNEL(grad_gelu_d5, GELU_BWD_D5_ODD_BF16)
SPLINE_KERNEL(grad_gelu_d6, GELU_BWD_D6_ODD_BF16)

// =============================================================================
// SFU HELPER FUNCTIONS
// =============================================================================

__device__ __forceinline__ __nv_bfloat162 sigmoid_sfu(__nv_bfloat162 val) {
    __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
    return h2rcp(__hadd2(one, h2exp(__hneg2(val))));
}

__device__ __forceinline__ __nv_bfloat162 tanh_sfu(__nv_bfloat162 val) {
    return __hsub2(__hmul2(__float2bfloat162_rn(2.0f), h2rcp(__hadd2(__float2bfloat162_rn(1.0f), h2exp(__hneg2(__hmul2(__float2bfloat162_rn(2.0f), val)))))), __float2bfloat162_rn(1.0f));
}

__device__ __forceinline__ __nv_bfloat162 swish_sfu(__nv_bfloat162 val) {
    __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
    return __hmul2(val, h2rcp(__hadd2(one, h2exp(__hneg2(val)))));
}

// =============================================================================
// HYBRID SFU+POLY KERNELS
// Each thread loads 2 × int4 = 8 × __nv_bfloat162 = 16 half values.
// Routes SFU_N through SFU and (8-SFU_N) through polynomial (FMA cores).
// SFU and FMA pipelines execute in parallel on different hardware units.
// =============================================================================

template <int SFU_N, typename SplineFunc,
          __nv_bfloat162 (*SfuFunc)(__nv_bfloat162)>
__global__ void hybrid_kernel(__nv_bfloat162* data, int n_h2, int inner_repeats) {
    int base = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int stride = blockDim.x * gridDim.x * 2;
    for (int idx = base; idx < n_h2; idx += stride) {
        if (idx + 1 >= n_h2) break;
        // Load 2 half2 values (could be part of a larger vectorized load)
        __nv_bfloat162 v0 = data[idx];
        __nv_bfloat162 v1 = data[idx + 1];
        #pragma unroll
        for (int i = 0; i < inner_repeats; ++i) {
            // Interleave SFU and FMA instructions for pipeline parallelism
            if constexpr (SFU_N >= 1) v0 = SfuFunc(v0);
            else                      v0 = SplineFunc::evaluate(v0);
            if constexpr (SFU_N >= 2) v1 = SfuFunc(v1);
            else                      v1 = SplineFunc::evaluate(v1);
        }
        data[idx] = v0;
        data[idx + 1] = v1;
    }
}

// Wider version: 4 half2 per thread for more ILP
template <int SFU_N, typename SplineFunc,
          __nv_bfloat162 (*SfuFunc)(__nv_bfloat162)>
__global__ void hybrid_kernel_4wide(__nv_bfloat162* data, int n_h2, int inner_repeats) {
    int base = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    int stride = blockDim.x * gridDim.x * 4;
    for (int idx = base; idx < n_h2; idx += stride) {
        if (idx + 3 >= n_h2) break;
        __nv_bfloat162 v0 = data[idx];
        __nv_bfloat162 v1 = data[idx + 1];
        __nv_bfloat162 v2 = data[idx + 2];
        __nv_bfloat162 v3 = data[idx + 3];
        #pragma unroll
        for (int i = 0; i < inner_repeats; ++i) {
            // Route first SFU_N values through SFU, rest through polynomial
            if constexpr (SFU_N >= 1) v0 = SfuFunc(v0);
            else                      v0 = SplineFunc::evaluate(v0);
            if constexpr (SFU_N >= 2) v1 = SfuFunc(v1);
            else                      v1 = SplineFunc::evaluate(v1);
            if constexpr (SFU_N >= 3) v2 = SfuFunc(v2);
            else                      v2 = SplineFunc::evaluate(v2);
            if constexpr (SFU_N >= 4) v3 = SfuFunc(v3);
            else                      v3 = SplineFunc::evaluate(v3);
        }
        data[idx] = v0;
        data[idx + 1] = v1;
        data[idx + 2] = v2;
        data[idx + 3] = v3;
    }
}

// 8-wide version: maximum ILP, matches FA's approach
template <int SFU_N, typename SplineFunc,
          __nv_bfloat162 (*SfuFunc)(__nv_bfloat162)>
__global__ void hybrid_kernel_8wide(__nv_bfloat162* data, int n_h2, int inner_repeats) {
    int base = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
    int stride = blockDim.x * gridDim.x * 8;
    for (int idx = base; idx < n_h2; idx += stride) {
        if (idx + 7 >= n_h2) break;
        __nv_bfloat162 v[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) v[k] = data[idx + k];
        #pragma unroll
        for (int i = 0; i < inner_repeats; ++i) {
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                if (k < SFU_N)
                    v[k] = SfuFunc(v[k]);
                else
                    v[k] = SplineFunc::evaluate(v[k]);
            }
        }
        #pragma unroll
        for (int k = 0; k < 8; k++) data[idx + k] = v[k];
    }
}

// Convenience macro-based kernel wrappers for benchmarking:

#define HYBRID_4W(name, SFU_N, Struct, SfuFn)                                  \
__global__ void name(__nv_bfloat162* data, int n_h2, int inner_repeats) {              \
    int base = (threadIdx.x + blockIdx.x * blockDim.x) * 4;                    \
    int stride = blockDim.x * gridDim.x * 4;                                   \
    for (int idx = base; idx < n_h2; idx += stride) {                          \
        if (idx + 3 >= n_h2) break;                                            \
        __nv_bfloat162 v0 = data[idx];                                                \
        __nv_bfloat162 v1 = data[idx + 1];                                            \
        __nv_bfloat162 v2 = data[idx + 2];                                            \
        __nv_bfloat162 v3 = data[idx + 3];                                            \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < inner_repeats; ++i) {                              \
            if constexpr (SFU_N >= 1) v0 = SfuFn(v0);                          \
            else                      v0 = Struct::evaluate(v0);               \
            if constexpr (SFU_N >= 2) v1 = SfuFn(v1);                          \
            else                      v1 = Struct::evaluate(v1);               \
            if constexpr (SFU_N >= 3) v2 = SfuFn(v2);                          \
            else                      v2 = Struct::evaluate(v2);               \
            if constexpr (SFU_N >= 4) v3 = SfuFn(v3);                          \
            else                      v3 = Struct::evaluate(v3);               \
        }                                                                      \
        data[idx] = v0;                                                        \
        data[idx + 1] = v1;                                                    \
        data[idx + 2] = v2;                                                    \
        data[idx + 3] = v3;                                                    \
    }                                                                          \
}

// Pure polynomial baseline (4-wide for fair comparison)
HYBRID_4W(fwd_sigmoid_poly4,  0, SIGMOID_FWD_D3_ODD_BF16, sigmoid_sfu)
// Hybrid splits: 1 SFU + 3 poly, 2+2, 3+1
HYBRID_4W(fwd_sigmoid_sfu1p3, 1, SIGMOID_FWD_D3_ODD_BF16, sigmoid_sfu)
HYBRID_4W(fwd_sigmoid_sfu2p2, 2, SIGMOID_FWD_D3_ODD_BF16, sigmoid_sfu)
HYBRID_4W(fwd_sigmoid_sfu3p1, 3, SIGMOID_FWD_D3_ODD_BF16, sigmoid_sfu)
// Pure SFU baseline (4-wide)
HYBRID_4W(fwd_sigmoid_sfu4,   4, SIGMOID_FWD_D3_ODD_BF16, sigmoid_sfu)

// Tanh
HYBRID_4W(fwd_tanh_poly4,  0, TANH_FWD_D3_ODD_BF16, tanh_sfu)
HYBRID_4W(fwd_tanh_sfu1p3, 1, TANH_FWD_D3_ODD_BF16, tanh_sfu)
HYBRID_4W(fwd_tanh_sfu2p2, 2, TANH_FWD_D3_ODD_BF16, tanh_sfu)
HYBRID_4W(fwd_tanh_sfu3p1, 3, TANH_FWD_D3_ODD_BF16, tanh_sfu)
HYBRID_4W(fwd_tanh_sfu4,   4, TANH_FWD_D3_ODD_BF16, tanh_sfu)

// Swish
HYBRID_4W(fwd_swish_poly4,  0, SWISH_FWD_D3_ODD_BF16, swish_sfu)
HYBRID_4W(fwd_swish_sfu1p3, 1, SWISH_FWD_D3_ODD_BF16, swish_sfu)
HYBRID_4W(fwd_swish_sfu2p2, 2, SWISH_FWD_D3_ODD_BF16, swish_sfu)
HYBRID_4W(fwd_swish_sfu3p1, 3, SWISH_FWD_D3_ODD_BF16, swish_sfu)
HYBRID_4W(fwd_swish_sfu4,   4, SWISH_FWD_D3_ODD_BF16, swish_sfu)

// BWD variants
HYBRID_4W(grad_sigmoid_poly4,  0, SIGMOID_BWD_D3_EVEN_BF16, sigmoid_sfu)
HYBRID_4W(grad_sigmoid_sfu2p2, 2, SIGMOID_BWD_D3_EVEN_BF16, sigmoid_sfu)

HYBRID_4W(grad_tanh_poly4,  0, TANH_BWD_D3_EVEN_BF16, tanh_sfu)
HYBRID_4W(grad_tanh_sfu2p2, 2, TANH_BWD_D3_EVEN_BF16, tanh_sfu)

// =============================================================================
// MAIN
// =============================================================================

int main() {
    int n_elements = N_ELEMENTS;
    int n_h2 = n_elements / 2;
    size_t bytes = n_h2 * sizeof(__nv_bfloat162);

    __nv_bfloat162* d_data;
    check(cudaMalloc(&d_data, bytes), "malloc data");

    printf("=========================================================\n");
    printf("SOTA BFloat16 Activation Benchmark — %dM elements, %d inner reps\n",
           n_elements / (1024*1024), INNER_REPEATS);
    printf("=========================================================\n\n");

    // ========== VERIFICATION ==========
    printf("--- VERIFICATION ---\n");
    // Sigmoid FWD
    verify_func("Sigmoid FWD D3", fwd_sigmoid_native, fwd_sigmoid_d3, n_h2, 0.01f);
    verify_func("Sigmoid FWD D4", fwd_sigmoid_native, fwd_sigmoid_d4, n_h2, 0.01f);
    verify_func("Sigmoid FWD D5", fwd_sigmoid_native, fwd_sigmoid_d5, n_h2, 0.01f);
    verify_func("Sigmoid FWD D6", fwd_sigmoid_native, fwd_sigmoid_d6, n_h2, 0.01f);
    // Tanh FWD
    verify_func("Tanh FWD D3",    fwd_tanh_native, fwd_tanh_d3, n_h2, 0.02f);
    verify_func("Tanh FWD D4",    fwd_tanh_native, fwd_tanh_d4, n_h2, 0.02f);
    verify_func("Tanh FWD D5",    fwd_tanh_native, fwd_tanh_d5, n_h2, 0.01f);
    verify_func("Tanh FWD D6",    fwd_tanh_native, fwd_tanh_d6, n_h2, 0.015f);
    // Swish FWD
    verify_func("Swish FWD D3",   fwd_swish_native, fwd_swish_d3, n_h2, 0.05f);
    verify_func("Swish FWD D4",   fwd_swish_native, fwd_swish_d4, n_h2, 0.04f);
    verify_func("Swish FWD D5",   fwd_swish_native, fwd_swish_d5, n_h2, 0.04f);
    verify_func("Swish FWD D6",   fwd_swish_native, fwd_swish_d6, n_h2, 0.05f);
    // Sigmoid BWD
    verify_func("Sigmoid BWD D3", grad_sigmoid_native, grad_sigmoid_d3, n_h2, 0.02f);
    verify_func("Sigmoid BWD D4", grad_sigmoid_native, grad_sigmoid_d4, n_h2, 0.01f);
    verify_func("Sigmoid BWD D5", grad_sigmoid_native, grad_sigmoid_d5, n_h2, 0.01f);
    verify_func("Sigmoid BWD D6", grad_sigmoid_native, grad_sigmoid_d6, n_h2, 0.01f);
    // Tanh BWD
    verify_func("Tanh BWD D3",    grad_tanh_native, grad_tanh_d3, n_h2, 0.05f);
    verify_func("Tanh BWD D4",    grad_tanh_native, grad_tanh_d4, n_h2, 0.03f);
    verify_func("Tanh BWD D5",    grad_tanh_native, grad_tanh_d5, n_h2, 0.025f);
    verify_func("Tanh BWD D6",    grad_tanh_native, grad_tanh_d6, n_h2, 0.05f);
    // Swish BWD
    verify_func("Swish BWD D3",   grad_swish_native, grad_swish_d3, n_h2, 0.04f);
    verify_func("Swish BWD D4",   grad_swish_native, grad_swish_d4, n_h2, 0.02f);
    verify_func("Swish BWD D5",   grad_swish_native, grad_swish_d5, n_h2, 0.02f);
    verify_func("Swish BWD D6",   grad_swish_native, grad_swish_d6, n_h2, 0.02f);
    // ERF FWD
    verify_func("ERF FWD D3",     fwd_erf_native, fwd_erf_d3, n_h2, 0.02f);
    verify_func("ERF FWD D4",     fwd_erf_native, fwd_erf_d4, n_h2, 0.01f);
    verify_func("ERF FWD D5",     fwd_erf_native, fwd_erf_d5, n_h2, 0.01f);
    verify_func("ERF FWD D6",     fwd_erf_native, fwd_erf_d6, n_h2, 0.01f);
    // GELU FWD
    verify_func("GELU FWD D3",    fwd_gelu_native, fwd_gelu_d3, n_h2, 0.05f);
    verify_func("GELU FWD D4",    fwd_gelu_native, fwd_gelu_d4, n_h2, 0.015f);
    verify_func("GELU FWD D5",    fwd_gelu_native, fwd_gelu_d5, n_h2, 0.015f);
    verify_func("GELU FWD D6",    fwd_gelu_native, fwd_gelu_d6, n_h2, 0.01f);
    // GELU BWD
    verify_func("GELU BWD D3",    grad_gelu_native, grad_gelu_d3, n_h2, 0.03f);
    verify_func("GELU BWD D4",    grad_gelu_native, grad_gelu_d4, n_h2, 0.02f);
    verify_func("GELU BWD D5",    grad_gelu_native, grad_gelu_d5, n_h2, 0.02f);
    verify_func("GELU BWD D6",    grad_gelu_native, grad_gelu_d6, n_h2, 0.02f);

    // ========== FORWARD BENCHMARKS ==========
    printf("\n--- FORWARD PASS: Degree Comparison ---\n");
    printf("%-40s %10s\n", "Kernel", "Time");
    printf("----------------------------------------------------\n");

    // Sigmoid FWD
    run_bench("Sigmoid FWD Native",   fwd_sigmoid_native, d_data, n_h2);
    run_bench("Sigmoid FWD D3",       fwd_sigmoid_d3,     d_data, n_h2);
    run_bench("Sigmoid FWD D4",       fwd_sigmoid_d4,     d_data, n_h2);
    run_bench("Sigmoid FWD D5",       fwd_sigmoid_d5,     d_data, n_h2);
    run_bench("Sigmoid FWD D6",       fwd_sigmoid_d6,     d_data, n_h2);
    printf("\n");

    // Tanh FWD
    run_bench("Tanh FWD Native",      fwd_tanh_native,    d_data, n_h2);
    run_bench("Tanh FWD D3",          fwd_tanh_d3,        d_data, n_h2);
    run_bench("Tanh FWD D4",          fwd_tanh_d4,        d_data, n_h2);
    run_bench("Tanh FWD D5",          fwd_tanh_d5,        d_data, n_h2);
    run_bench("Tanh FWD D6",          fwd_tanh_d6,        d_data, n_h2);
    printf("\n");

    // Swish FWD
    run_bench("Swish FWD Native",     fwd_swish_native,   d_data, n_h2);
    run_bench("Swish FWD D3",         fwd_swish_d3,       d_data, n_h2);
    run_bench("Swish FWD D4",         fwd_swish_d4,       d_data, n_h2);
    run_bench("Swish FWD D5",         fwd_swish_d5,       d_data, n_h2);
    run_bench("Swish FWD D6",         fwd_swish_d6,       d_data, n_h2);
    printf("\n");

    // ERF FWD
    run_bench("ERF FWD Native",       fwd_erf_native,     d_data, n_h2);
    run_bench("ERF FWD D3",           fwd_erf_d3,         d_data, n_h2);
    run_bench("ERF FWD D4",           fwd_erf_d4,         d_data, n_h2);
    run_bench("ERF FWD D5",           fwd_erf_d5,         d_data, n_h2);
    run_bench("ERF FWD D6",           fwd_erf_d6,         d_data, n_h2);
    printf("\n");

    // GELU FWD
    run_bench("GELU FWD Native",      fwd_gelu_native,    d_data, n_h2);
    run_bench("GELU FWD D3",          fwd_gelu_d3,        d_data, n_h2);
    run_bench("GELU FWD D4",          fwd_gelu_d4,        d_data, n_h2);
    run_bench("GELU FWD D5",          fwd_gelu_d5,        d_data, n_h2);
    run_bench("GELU FWD D6",          fwd_gelu_d6,        d_data, n_h2);

    // ========== BACKWARD BENCHMARKS ==========
    printf("\n--- BACKWARD PASS: Degree Comparison ---\n");
    printf("%-40s %10s\n", "Kernel", "Time");
    printf("----------------------------------------------------\n");

    // Sigmoid BWD
    run_bench("Sigmoid BWD Native",   grad_sigmoid_native, d_data, n_h2);
    run_bench("Sigmoid BWD D3",       grad_sigmoid_d3,     d_data, n_h2);
    run_bench("Sigmoid BWD D4",       grad_sigmoid_d4,     d_data, n_h2);
    run_bench("Sigmoid BWD D5",       grad_sigmoid_d5,     d_data, n_h2);
    run_bench("Sigmoid BWD D6",       grad_sigmoid_d6,     d_data, n_h2);
    printf("\n");

    // Tanh BWD
    run_bench("Tanh BWD Native",      grad_tanh_native,    d_data, n_h2);
    run_bench("Tanh BWD D3",          grad_tanh_d3,        d_data, n_h2);
    run_bench("Tanh BWD D4",          grad_tanh_d4,        d_data, n_h2);
    run_bench("Tanh BWD D5",          grad_tanh_d5,        d_data, n_h2);
    run_bench("Tanh BWD D6",          grad_tanh_d6,        d_data, n_h2);
    printf("\n");

    // Swish BWD
    run_bench("Swish BWD Native",     grad_swish_native,   d_data, n_h2);
    run_bench("Swish BWD D3",         grad_swish_d3,       d_data, n_h2);
    run_bench("Swish BWD D4",         grad_swish_d4,       d_data, n_h2);
    run_bench("Swish BWD D5",         grad_swish_d5,       d_data, n_h2);
    run_bench("Swish BWD D6",         grad_swish_d6,       d_data, n_h2);
    printf("\n");

    // GELU BWD
    run_bench("GELU BWD Native",      grad_gelu_native,    d_data, n_h2);
    run_bench("GELU BWD D3",          grad_gelu_d3,        d_data, n_h2);
    run_bench("GELU BWD D4",          grad_gelu_d4,        d_data, n_h2);
    run_bench("GELU BWD D5",          grad_gelu_d5,        d_data, n_h2);
    run_bench("GELU BWD D6",          grad_gelu_d6,        d_data, n_h2);

    // ========== HYBRID SFU+POLY BENCHMARKS ==========
    printf("\n--- HYBRID: SFU + Polynomial Split (4-wide) ---\n");
    printf("%-40s %10s\n", "Kernel", "Time");
    printf("----------------------------------------------------\n");

    printf("\n  Sigmoid FWD:\n");
    run_bench("  Poly 4/4",           fwd_sigmoid_poly4,   d_data, n_h2);
    run_bench("  SFU 1 + Poly 3",     fwd_sigmoid_sfu1p3,  d_data, n_h2);
    run_bench("  SFU 2 + Poly 2",     fwd_sigmoid_sfu2p2,  d_data, n_h2);
    run_bench("  SFU 3 + Poly 1",     fwd_sigmoid_sfu3p1,  d_data, n_h2);
    run_bench("  SFU 4/4",            fwd_sigmoid_sfu4,    d_data, n_h2);

    printf("\n  Tanh FWD:\n");
    run_bench("  Poly 4/4",           fwd_tanh_poly4,      d_data, n_h2);
    run_bench("  SFU 1 + Poly 3",     fwd_tanh_sfu1p3,     d_data, n_h2);
    run_bench("  SFU 2 + Poly 2",     fwd_tanh_sfu2p2,     d_data, n_h2);
    run_bench("  SFU 3 + Poly 1",     fwd_tanh_sfu3p1,     d_data, n_h2);
    run_bench("  SFU 4/4",            fwd_tanh_sfu4,       d_data, n_h2);

    printf("\n  Swish FWD:\n");
    run_bench("  Poly 4/4",           fwd_swish_poly4,     d_data, n_h2);
    run_bench("  SFU 1 + Poly 3",     fwd_swish_sfu1p3,    d_data, n_h2);
    run_bench("  SFU 2 + Poly 2",     fwd_swish_sfu2p2,    d_data, n_h2);
    run_bench("  SFU 3 + Poly 1",     fwd_swish_sfu3p1,    d_data, n_h2);
    run_bench("  SFU 4/4",            fwd_swish_sfu4,      d_data, n_h2);

    printf("\n  Sigmoid BWD:\n");
    run_bench("  Poly 4/4",           grad_sigmoid_poly4,  d_data, n_h2);
    run_bench("  SFU 2 + Poly 2",     grad_sigmoid_sfu2p2, d_data, n_h2);

    printf("\n  Tanh BWD:\n");
    run_bench("  Poly 4/4",           grad_tanh_poly4,     d_data, n_h2);
    run_bench("  SFU 2 + Poly 2",     grad_tanh_sfu2p2,    d_data, n_h2);


    printf("\n=========================================================\n");
    cudaFree(d_data);
    return 0;
}
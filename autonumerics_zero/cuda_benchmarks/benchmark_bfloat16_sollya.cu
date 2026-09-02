// =============================================================================
// BF16 Sollya vs Current Device-Side Activation Benchmark
// =============================================================================
// Compares the shipped BF16 spline structs against Sollya-generated BF16
// structs under the exact same runtime evaluation shape and clamp.
//
// Build:
//   nvcc -O3 -arch=sm_100 --use_fast_math \
//        -I../spline_ops -o benchmark_bfloat16_sollya benchmark_bfloat16_sollya.cu
// =============================================================================

#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

#define N_ELEMENTS (1024 * 1024 * 16)
#define BLOCK_SIZE 256
#define N_REPEATS 50
#define INNER_REPEATS 10

#include "spline_structs_odd_bf16.cuh"
#include "spline_structs_sollya_bf16.cuh"

inline void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s: %s\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

__global__ void init_kernel(__nv_bfloat162* data, int n_h2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        float f = (float)(idx % 20000) * 0.001f - 10.0f;
        data[idx] = __float2bfloat162_rn(f);
    }
}

void reset_input(__nv_bfloat162* d_data, int n_h2) {
    int grid = (n_h2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_kernel<<<grid, BLOCK_SIZE>>>(d_data, n_h2);
    check(cudaDeviceSynchronize(), "reset_input sync");
}

float run_bench(void(*kernel)(__nv_bfloat162*, int, int), __nv_bfloat162* d_data, int n_h2) {
    int grid = (n_h2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reset_input(d_data, n_h2);

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");

    kernel<<<grid, BLOCK_SIZE>>>(d_data, n_h2, INNER_REPEATS);
    check(cudaDeviceSynchronize(), "warmup sync");

    check(cudaEventRecord(start), "record start");
    for (int i = 0; i < N_REPEATS; ++i) {
        kernel<<<grid, BLOCK_SIZE>>>(d_data, n_h2, INNER_REPEATS);
    }
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "event sync");

    float ms_total = 0.0f;
    check(cudaEventElapsedTime(&ms_total, start, stop), "elapsed time");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms_total / N_REPEATS;
}

#define SPLINE_KERNEL(name, StructType)                                            \
__global__ void name(__nv_bfloat162* data, int n_h2, int inner_repeats) {         \
    int idx = threadIdx.x + blockIdx.x * blockDim.x;                               \
    if (idx < n_h2) {                                                              \
        __nv_bfloat162 val = data[idx];                                            \
        _Pragma("unroll")                                                          \
        for (int i = 0; i < inner_repeats; ++i) {                                  \
            val = StructType::evaluate(val);                                       \
        }                                                                          \
        data[idx] = val;                                                           \
    }                                                                              \
}

#define DECLARE_KERNEL_PAIR(prefix, OursType, SollyaType) \
    SPLINE_KERNEL(prefix##_ours, OursType)                \
    SPLINE_KERNEL(prefix##_sollya, SollyaType)

DECLARE_KERNEL_PAIR(sigmoid_fwd_d3, SIGMOID_FWD_D3_ODD_BF16, SIGMOID_FWD_D3_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(sigmoid_fwd_d4, SIGMOID_FWD_D4_ODD_BF16, SIGMOID_FWD_D4_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(sigmoid_fwd_d5, SIGMOID_FWD_D5_ODD_BF16, SIGMOID_FWD_D5_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(sigmoid_fwd_d6, SIGMOID_FWD_D6_ODD_BF16, SIGMOID_FWD_D6_ODD_SOLLYA_BF16)

DECLARE_KERNEL_PAIR(tanh_fwd_d3, TANH_FWD_D3_ODD_BF16, TANH_FWD_D3_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(tanh_fwd_d4, TANH_FWD_D4_ODD_BF16, TANH_FWD_D4_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(tanh_fwd_d5, TANH_FWD_D5_ODD_BF16, TANH_FWD_D5_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(tanh_fwd_d6, TANH_FWD_D6_ODD_BF16, TANH_FWD_D6_ODD_SOLLYA_BF16)

DECLARE_KERNEL_PAIR(swish_fwd_d3, SWISH_FWD_D3_ODD_BF16, SWISH_FWD_D3_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(swish_fwd_d4, SWISH_FWD_D4_ODD_BF16, SWISH_FWD_D4_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(swish_fwd_d5, SWISH_FWD_D5_ODD_BF16, SWISH_FWD_D5_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(swish_fwd_d6, SWISH_FWD_D6_ODD_BF16, SWISH_FWD_D6_ODD_SOLLYA_BF16)

DECLARE_KERNEL_PAIR(gelu_fwd_d3, GELU_FWD_D3_ODD_BF16, GELU_FWD_D3_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(gelu_fwd_d4, GELU_FWD_D4_ODD_BF16, GELU_FWD_D4_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(gelu_fwd_d5, GELU_FWD_D5_ODD_BF16, GELU_FWD_D5_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(gelu_fwd_d6, GELU_FWD_D6_ODD_BF16, GELU_FWD_D6_ODD_SOLLYA_BF16)

DECLARE_KERNEL_PAIR(sigmoid_bwd_d3, SIGMOID_BWD_D3_EVEN_BF16, SIGMOID_BWD_D3_EVEN_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(sigmoid_bwd_d4, SIGMOID_BWD_D4_EVEN_BF16, SIGMOID_BWD_D4_EVEN_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(sigmoid_bwd_d5, SIGMOID_BWD_D5_EVEN_BF16, SIGMOID_BWD_D5_EVEN_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(sigmoid_bwd_d6, SIGMOID_BWD_D6_EVEN_BF16, SIGMOID_BWD_D6_EVEN_SOLLYA_BF16)

DECLARE_KERNEL_PAIR(tanh_bwd_d3, TANH_BWD_D3_EVEN_BF16, TANH_BWD_D3_EVEN_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(tanh_bwd_d4, TANH_BWD_D4_EVEN_BF16, TANH_BWD_D4_EVEN_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(tanh_bwd_d5, TANH_BWD_D5_EVEN_BF16, TANH_BWD_D5_EVEN_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(tanh_bwd_d6, TANH_BWD_D6_EVEN_BF16, TANH_BWD_D6_EVEN_SOLLYA_BF16)

DECLARE_KERNEL_PAIR(swish_bwd_d3, SWISH_BWD_D3_ODD_BF16, SWISH_BWD_D3_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(swish_bwd_d4, SWISH_BWD_D4_ODD_BF16, SWISH_BWD_D4_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(swish_bwd_d5, SWISH_BWD_D5_ODD_BF16, SWISH_BWD_D5_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(swish_bwd_d6, SWISH_BWD_D6_ODD_BF16, SWISH_BWD_D6_ODD_SOLLYA_BF16)

DECLARE_KERNEL_PAIR(gelu_bwd_d3, GELU_BWD_D3_ODD_BF16, GELU_BWD_D3_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(gelu_bwd_d4, GELU_BWD_D4_ODD_BF16, GELU_BWD_D4_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(gelu_bwd_d5, GELU_BWD_D5_ODD_BF16, GELU_BWD_D5_ODD_SOLLYA_BF16)
DECLARE_KERNEL_PAIR(gelu_bwd_d6, GELU_BWD_D6_ODD_BF16, GELU_BWD_D6_ODD_SOLLYA_BF16)

struct BenchRow {
    const char* family;
    int degree;
    void(*ours)(__nv_bfloat162*, int, int);
    void(*sollya)(__nv_bfloat162*, int, int);
};

int main() {
    int n_h2 = N_ELEMENTS / 2;
    size_t bytes = n_h2 * sizeof(__nv_bfloat162);
    __nv_bfloat162* d_data = nullptr;
    check(cudaMalloc(&d_data, bytes), "malloc data");

    BenchRow rows[] = {
        {"sigmoid_fwd", 3, sigmoid_fwd_d3_ours, sigmoid_fwd_d3_sollya},
        {"sigmoid_fwd", 4, sigmoid_fwd_d4_ours, sigmoid_fwd_d4_sollya},
        {"sigmoid_fwd", 5, sigmoid_fwd_d5_ours, sigmoid_fwd_d5_sollya},
        {"sigmoid_fwd", 6, sigmoid_fwd_d6_ours, sigmoid_fwd_d6_sollya},
        {"tanh_fwd", 3, tanh_fwd_d3_ours, tanh_fwd_d3_sollya},
        {"tanh_fwd", 4, tanh_fwd_d4_ours, tanh_fwd_d4_sollya},
        {"tanh_fwd", 5, tanh_fwd_d5_ours, tanh_fwd_d5_sollya},
        {"tanh_fwd", 6, tanh_fwd_d6_ours, tanh_fwd_d6_sollya},
        {"swish_fwd", 3, swish_fwd_d3_ours, swish_fwd_d3_sollya},
        {"swish_fwd", 4, swish_fwd_d4_ours, swish_fwd_d4_sollya},
        {"swish_fwd", 5, swish_fwd_d5_ours, swish_fwd_d5_sollya},
        {"swish_fwd", 6, swish_fwd_d6_ours, swish_fwd_d6_sollya},
        {"gelu_fwd", 3, gelu_fwd_d3_ours, gelu_fwd_d3_sollya},
        {"gelu_fwd", 4, gelu_fwd_d4_ours, gelu_fwd_d4_sollya},
        {"gelu_fwd", 5, gelu_fwd_d5_ours, gelu_fwd_d5_sollya},
        {"gelu_fwd", 6, gelu_fwd_d6_ours, gelu_fwd_d6_sollya},
        {"sigmoid_bwd", 3, sigmoid_bwd_d3_ours, sigmoid_bwd_d3_sollya},
        {"sigmoid_bwd", 4, sigmoid_bwd_d4_ours, sigmoid_bwd_d4_sollya},
        {"sigmoid_bwd", 5, sigmoid_bwd_d5_ours, sigmoid_bwd_d5_sollya},
        {"sigmoid_bwd", 6, sigmoid_bwd_d6_ours, sigmoid_bwd_d6_sollya},
        {"tanh_bwd", 3, tanh_bwd_d3_ours, tanh_bwd_d3_sollya},
        {"tanh_bwd", 4, tanh_bwd_d4_ours, tanh_bwd_d4_sollya},
        {"tanh_bwd", 5, tanh_bwd_d5_ours, tanh_bwd_d5_sollya},
        {"tanh_bwd", 6, tanh_bwd_d6_ours, tanh_bwd_d6_sollya},
        {"swish_bwd", 3, swish_bwd_d3_ours, swish_bwd_d3_sollya},
        {"swish_bwd", 4, swish_bwd_d4_ours, swish_bwd_d4_sollya},
        {"swish_bwd", 5, swish_bwd_d5_ours, swish_bwd_d5_sollya},
        {"swish_bwd", 6, swish_bwd_d6_ours, swish_bwd_d6_sollya},
        {"gelu_bwd", 3, gelu_bwd_d3_ours, gelu_bwd_d3_sollya},
        {"gelu_bwd", 4, gelu_bwd_d4_ours, gelu_bwd_d4_sollya},
        {"gelu_bwd", 5, gelu_bwd_d5_ours, gelu_bwd_d5_sollya},
        {"gelu_bwd", 6, gelu_bwd_d6_ours, gelu_bwd_d6_sollya},
    };

    printf("RESULT_HEADER family degree ours_ms sollya_ms sollya_over_ours\n");
    for (const auto& row : rows) {
        float ours_ms = run_bench(row.ours, d_data, n_h2);
        float sollya_ms = run_bench(row.sollya, d_data, n_h2);
        printf(
            "RESULT %s D%d ours_ms=%.6f sollya_ms=%.6f sollya_over_ours=%.6f\n",
            row.family,
            row.degree,
            ours_ms,
            sollya_ms,
            sollya_ms / ours_ms
        );
        fflush(stdout);
    }

    cudaFree(d_data);
    return 0;
}

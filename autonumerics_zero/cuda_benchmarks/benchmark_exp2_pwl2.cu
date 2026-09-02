// Focused GB200 benchmark for the historical two-piece linear exp2 fit.
//
// Build:
//   /usr/local/cuda-13.0/bin/nvcc -O3 -std=c++17 -arch=sm_100 \
//     -lineinfo -o benchmark_exp2_pwl2 benchmark_exp2_pwl2.cu
//
// The fractional benchmarks isolate 2^x on x in [0, 1), where the old
// packed-FP16 fit operates. The full-range benchmarks include the FP32 range
// reduction and exponent reconstruction required by FA4.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kComputeThreads = 1 << 16;
constexpr int kComputeIterations = 256;
constexpr int kComputeChains = 8;
constexpr int kTimingSamples = 9;
constexpr int kWarmupLaunches = 5;
constexpr int kTimedLaunches = 20;
constexpr int kErrorPairs = 1 << 20;
constexpr int kL2PairsH2 = 1 << 20;   // 4 MiB input + 4 MiB output.
constexpr int kHbmPairsH2 = 1 << 26;  // 256 MiB input + 256 MiB output.
constexpr int kL2PairsF32 = 1 << 19;  // 4 MiB input + 4 MiB output.
constexpr int kHbmPairsF32 = 1 << 25; // 256 MiB input + 256 MiB output.

inline void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

__device__ __forceinline__ __half2 half2_from_bits(uint32_t bits) {
    return *reinterpret_cast<__half2*>(&bits);
}

__device__ __forceinline__ uint32_t half2_to_bits(__half2 value) {
    return *reinterpret_cast<uint32_t*>(&value);
}

struct FractionalNative {
    __device__ __forceinline__ static __half2 eval(__half2 x) {
        return h2exp2(x);
    }
};

struct FractionalPwl2Historical {
    // Historical N=2, D=1 coefficients:
    //   [0, 0.5): 0x3a91 * x + 0x3bea
    //   [0.5, 1): 0x3ca7 * x + 0x3a8c
    __device__ __forceinline__ static __half2 eval(__half2 x) {
        constexpr uint32_t kSlopePacked = 0x3ca73a91;
        constexpr uint32_t kOffsetPacked = 0x3a8c3bea;
        const __half2 zero = half2_from_bits(0x00000000);
        // 0x3bff + 1 rounds to 2.0 in FP16, which clears the mantissa index
        // bit. Clamp one additional ULP down so the historical bit trick is
        // well-defined at the upper endpoint.
        const __half2 largest_below_one = half2_from_bits(0x3bfe3bfe);
        const __half2 one = half2_from_bits(0x3c003c00);
        const __half2 clamped = __hmin2(__hmax2(x, zero), largest_below_one);
        const uint32_t normalized = half2_to_bits(__hadd2(clamped, one));
        const int shift_lo = static_cast<int>((normalized >> 9) & 1u) << 4;
        const int shift_hi = static_cast<int>((normalized >> 25) & 1u) << 4;
        const uint32_t slope = ((kSlopePacked >> shift_lo) & 0xffffu) |
                               (((kSlopePacked >> shift_hi) & 0xffffu) << 16);
        const uint32_t offset = ((kOffsetPacked >> shift_lo) & 0xffffu) |
                                (((kOffsetPacked >> shift_hi) & 0xffffu) << 16);
        return __hfma2(clamped, half2_from_bits(slope), half2_from_bits(offset));
    }
};

struct FractionalPwl2Hinge {
    // Continuous least-squares form of the same two-segment fit:
    //   p(x) = 0.9892578125 + 0.82080078125*x
    //          + 0.342529296875*max(x - 0.5, 0).
    // All constants are exact FP16 values. This removes coefficient selection.
    __device__ __forceinline__ static __half2 eval(__half2 x) {
        const __half2 zero = half2_from_bits(0x00000000);
        const __half2 half = half2_from_bits(0x38003800);
        const __half2 slope = half2_from_bits(0x3a913a91);
        const __half2 offset = half2_from_bits(0x3bea3bea);
        const __half2 slope_delta = half2_from_bits(0x357b357b);
        const __half2 hinge = __hmax2(__hsub2(x, half), zero);
        const __half2 base = __hfma2(x, slope, offset);
        return __hfma2(hinge, slope_delta, base);
    }
};

struct FullNative {
    __device__ __forceinline__ static float2 eval(float2 x) {
        return make_float2(exp2f(x.x), exp2f(x.y));
    }
};

struct FullDegree3 {
    __device__ __forceinline__ static float2 eval(float2 value) {
        int out_x;
        int out_y;
        asm volatile(
            "{\n\t"
            ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
            ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
            ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
            "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
            "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
            "mov.b64 l1, {f1, f2};\n\t"
            "mov.f32 f3, 0f4B400000;\n\t"
            "mov.b64 l2, {f3, f3};\n\t"
            "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
            "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
            "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
            "mov.f32 f7, 0f3D9DF09D;\n\t"
            "mov.b64 l6, {f7, f7};\n\t"
            "mov.f32 f6, 0f3E6906A4;\n\t"
            "mov.b64 l5, {f6, f6};\n\t"
            "mov.f32 f5, 0f3F31F519;\n\t"
            "mov.b64 l4, {f5, f5};\n\t"
            "mov.f32 f4, 0f3F800000;\n\t"
            "mov.b64 l3, {f4, f4};\n\t"
            "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
            "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
            "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
            "mov.b64 {r1, r2}, l7;\n\t"
            "mov.b64 {r3, r4}, l10;\n\t"
            "shl.b32 r5, r1, 23;\n\t"
            "add.s32 r7, r5, r3;\n\t"
            "shl.b32 r6, r2, 23;\n\t"
            "add.s32 r8, r6, r4;\n\t"
            "mov.b32 %0, r7;\n\t"
            "mov.b32 %1, r8;\n\t"
            "}"
            : "=r"(out_x), "=r"(out_y)
            : "f"(value.x), "f"(value.y));
        return make_float2(__int_as_float(out_x), __int_as_float(out_y));
    }
};

struct FullPwl2Hinge {
    // Full range reduction plus a continuous two-piece FP32 least-squares fit.
    __device__ __forceinline__ static float2 eval(float2 value) {
        int out_x;
        int out_y;
        asm volatile(
            "{\n\t"
            ".reg .f32 f1, f2, f3, f4, f5, f6, f7, f8;\n\t"
            ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11;\n\t"
            ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
            "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
            "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
            "mov.b64 l1, {f1, f2};\n\t"
            "mov.f32 f3, 0f4B400000;\n\t"
            "mov.b64 l2, {f3, f3};\n\t"
            "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
            "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
            "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
            "mov.f32 f4, 0f3F000000;\n\t"
            "mov.b64 l3, {f4, f4};\n\t"
            "sub.rn.ftz.f32x2 l10, l9, l3;\n\t"
            "mov.f32 f5, 0f00000000;\n\t"
            "mov.b64 l4, {f5, f5};\n\t"
            "mov.b64 {f1, f2}, l10;\n\t"
            "max.ftz.f32 f1, f1, 0f00000000;\n\t"
            "max.ftz.f32 f2, f2, 0f00000000;\n\t"
            "mov.b64 l10, {f1, f2};\n\t"
            "mov.f32 f6, 0f3F522364;\n\t"
            "mov.b64 l5, {f6, f6};\n\t"
            "mov.f32 f7, 0f3F7D4D54;\n\t"
            "mov.b64 l6, {f7, f7};\n\t"
            "fma.rn.ftz.f32x2 l11, l9, l5, l6;\n\t"
            "mov.f32 f8, 0f3EAF5701;\n\t"
            "mov.b64 l5, {f8, f8};\n\t"
            "fma.rn.ftz.f32x2 l11, l10, l5, l11;\n\t"
            "mov.b64 {r1, r2}, l7;\n\t"
            "mov.b64 {r3, r4}, l11;\n\t"
            "shl.b32 r5, r1, 23;\n\t"
            "add.s32 r7, r5, r3;\n\t"
            "shl.b32 r6, r2, 23;\n\t"
            "add.s32 r8, r6, r4;\n\t"
            "mov.b32 %0, r7;\n\t"
            "mov.b32 %1, r8;\n\t"
            "}"
            : "=r"(out_x), "=r"(out_y)
            : "f"(value.x), "f"(value.y));
        return make_float2(__int_as_float(out_x), __int_as_float(out_y));
    }
};

template <typename Evaluator>
__global__ void compute_h2_kernel(const __half2* input, __half2* output, int iterations) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kComputeThreads) {
        return;
    }
    __half2 x[kComputeChains];
#pragma unroll
    for (int chain = 0; chain < kComputeChains; ++chain) {
        x[chain] = input[index + chain * kComputeThreads];
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
        for (int chain = 0; chain < kComputeChains; ++chain) {
            const __half2 value = Evaluator::eval(x[chain]);
            if (chain < kComputeChains / 2) {
                x[chain] = __hfma2(
                    value, half2_from_bits(0x38003800), half2_from_bits(0xb800b800));
            } else {
                x[chain] = __hfma2(
                    value, half2_from_bits(0xb800b800), half2_from_bits(0x3e003e00));
            }
        }
    }
    output[index] = x[index & (kComputeChains - 1)];
}

template <typename Evaluator>
__global__ void compute_f32_kernel(const float2* input, float2* output, int iterations) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kComputeThreads) {
        return;
    }
    float2 x[kComputeChains];
#pragma unroll
    for (int chain = 0; chain < kComputeChains; ++chain) {
        x[chain] = input[index + chain * kComputeThreads];
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
        for (int chain = 0; chain < kComputeChains; ++chain) {
            const float2 value = Evaluator::eval(x[chain]);
            if (chain < kComputeChains / 2) {
                x[chain] = make_float2(
                    fmaf(value.x, 8.0f, -16.0f), fmaf(value.y, 8.0f, -16.0f));
            } else {
                x[chain] = make_float2(-8.0f * value.x, -8.0f * value.y);
            }
        }
    }
    output[index] = x[index & (kComputeChains - 1)];
}

template <typename Evaluator>
__global__ void memory_h2_kernel(const __half2* input, __half2* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = Evaluator::eval(input[index]);
    }
}

template <typename Evaluator>
__global__ void memory_f32_kernel(const float2* input, float2* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = Evaluator::eval(input[index]);
    }
}

template <typename Kernel, typename... Args>
float time_kernel(Kernel kernel, dim3 grid, dim3 block, Args... args) {
    for (int i = 0; i < kWarmupLaunches; ++i) {
        kernel<<<grid, block>>>(args...);
    }
    check_cuda(cudaDeviceSynchronize(), "benchmark warmup");

    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    std::vector<float> samples;
    samples.reserve(kTimingSamples);
    for (int sample = 0; sample < kTimingSamples; ++sample) {
        check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
        for (int launch = 0; launch < kTimedLaunches; ++launch) {
            kernel<<<grid, block>>>(args...);
        }
        check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
        samples.push_back(elapsed_ms / static_cast<float>(kTimedLaunches));
    }
    check_cuda(cudaGetLastError(), "timed kernel launch");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

struct ErrorMetrics {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double sum_sq = 0.0;
    size_t count = 0;

    void add(double actual, double reference) {
        const double absolute = std::abs(actual - reference);
        max_abs = std::max(max_abs, absolute);
        max_rel = std::max(max_rel, absolute / std::abs(reference));
        sum_sq += absolute * absolute;
        ++count;
    }

    double rmse() const {
        return std::sqrt(sum_sq / static_cast<double>(count));
    }
};

std::vector<__half2> make_fractional_h2(size_t count) {
    std::vector<__half2> values(count);
    for (size_t i = 0; i < count; ++i) {
        const float x = static_cast<float>((2 * i) % 65521) / 65521.0f;
        const float y = static_cast<float>((2 * i + 1) % 65521) / 65521.0f;
        values[i] = __floats2half2_rn(x, y);
    }
    return values;
}

std::vector<float2> make_full_f32(size_t count) {
    std::vector<float2> values(count);
    for (size_t i = 0; i < count; ++i) {
        const float u = static_cast<float>((2 * i) % 1048573) / 1048573.0f;
        const float v = static_cast<float>((2 * i + 1) % 1048573) / 1048573.0f;
        values[i] = make_float2(-16.0f * u, -16.0f * v);
    }
    return values;
}

template <typename Evaluator>
ErrorMetrics error_fractional_h2(const std::vector<__half2>& input) {
    const size_t bytes = input.size() * sizeof(__half2);
    __half2* device_input = nullptr;
    __half2* device_output = nullptr;
    check_cuda(cudaMalloc(&device_input, bytes), "cudaMalloc(error h2 input)");
    check_cuda(cudaMalloc(&device_output, bytes), "cudaMalloc(error h2 output)");
    check_cuda(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(error h2 input)");
    const dim3 block(kBlockSize);
    const dim3 grid((input.size() + kBlockSize - 1) / kBlockSize);
    memory_h2_kernel<Evaluator><<<grid, block>>>(device_input, device_output, input.size());
    std::vector<__half2> output(input.size());
    check_cuda(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(error h2 output)");
    ErrorMetrics metrics;
    for (size_t i = 0; i < input.size(); ++i) {
        const float2 x = __half22float2(input[i]);
        const float2 y = __half22float2(output[i]);
        metrics.add(y.x, std::exp2(static_cast<double>(x.x)));
        metrics.add(y.y, std::exp2(static_cast<double>(x.y)));
    }
    cudaFree(device_input);
    cudaFree(device_output);
    return metrics;
}

template <typename Evaluator>
ErrorMetrics error_full_f32(const std::vector<float2>& input) {
    const size_t bytes = input.size() * sizeof(float2);
    float2* device_input = nullptr;
    float2* device_output = nullptr;
    check_cuda(cudaMalloc(&device_input, bytes), "cudaMalloc(error f32 input)");
    check_cuda(cudaMalloc(&device_output, bytes), "cudaMalloc(error f32 output)");
    check_cuda(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(error f32 input)");
    const dim3 block(kBlockSize);
    const dim3 grid((input.size() + kBlockSize - 1) / kBlockSize);
    memory_f32_kernel<Evaluator><<<grid, block>>>(device_input, device_output, input.size());
    std::vector<float2> output(input.size());
    check_cuda(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(error f32 output)");
    ErrorMetrics metrics;
    for (size_t i = 0; i < input.size(); ++i) {
        metrics.add(output[i].x, std::exp2(static_cast<double>(input[i].x)));
        metrics.add(output[i].y, std::exp2(static_cast<double>(input[i].y)));
    }
    cudaFree(device_input);
    cudaFree(device_output);
    return metrics;
}

template <typename Evaluator>
float benchmark_compute_h2(const std::vector<__half2>& host_input) {
    const size_t input_count = kComputeThreads * kComputeChains;
    const size_t input_bytes = input_count * sizeof(__half2);
    __half2* device_input = nullptr;
    __half2* device_output = nullptr;
    check_cuda(cudaMalloc(&device_input, input_bytes), "cudaMalloc(compute h2 input)");
    check_cuda(cudaMalloc(&device_output, kComputeThreads * sizeof(__half2)),
               "cudaMalloc(compute h2 output)");
    check_cuda(cudaMemcpy(device_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(compute h2 input)");
    const float ms = time_kernel(
        compute_h2_kernel<Evaluator>, dim3(kComputeThreads / kBlockSize), dim3(kBlockSize),
        device_input, device_output, kComputeIterations);
    cudaFree(device_input);
    cudaFree(device_output);
    return ms;
}

template <typename Evaluator>
float benchmark_compute_f32(const std::vector<float2>& host_input) {
    const size_t input_count = kComputeThreads * kComputeChains;
    const size_t input_bytes = input_count * sizeof(float2);
    float2* device_input = nullptr;
    float2* device_output = nullptr;
    check_cuda(cudaMalloc(&device_input, input_bytes), "cudaMalloc(compute f32 input)");
    check_cuda(cudaMalloc(&device_output, kComputeThreads * sizeof(float2)),
               "cudaMalloc(compute f32 output)");
    check_cuda(cudaMemcpy(device_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(compute f32 input)");
    const float ms = time_kernel(
        compute_f32_kernel<Evaluator>, dim3(kComputeThreads / kBlockSize), dim3(kBlockSize),
        device_input, device_output, kComputeIterations);
    cudaFree(device_input);
    cudaFree(device_output);
    return ms;
}

template <typename Evaluator>
float benchmark_memory_h2(const std::vector<__half2>& host_seed, int count) {
    const size_t bytes = static_cast<size_t>(count) * sizeof(__half2);
    __half2* device_input = nullptr;
    __half2* device_output = nullptr;
    check_cuda(cudaMalloc(&device_input, bytes), "cudaMalloc(memory h2 input)");
    check_cuda(cudaMalloc(&device_output, bytes), "cudaMalloc(memory h2 output)");
    check_cuda(cudaMemcpy(device_input, host_seed.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(memory h2 input)");
    const float ms = time_kernel(
        memory_h2_kernel<Evaluator>, dim3((count + kBlockSize - 1) / kBlockSize),
        dim3(kBlockSize), device_input, device_output, count);
    cudaFree(device_input);
    cudaFree(device_output);
    return ms;
}

template <typename Evaluator>
float benchmark_memory_f32(const std::vector<float2>& host_seed, int count) {
    const size_t bytes = static_cast<size_t>(count) * sizeof(float2);
    float2* device_input = nullptr;
    float2* device_output = nullptr;
    check_cuda(cudaMalloc(&device_input, bytes), "cudaMalloc(memory f32 input)");
    check_cuda(cudaMalloc(&device_output, bytes), "cudaMalloc(memory f32 output)");
    check_cuda(cudaMemcpy(device_input, host_seed.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(memory f32 input)");
    const float ms = time_kernel(
        memory_f32_kernel<Evaluator>, dim3((count + kBlockSize - 1) / kBlockSize),
        dim3(kBlockSize), device_input, device_output, count);
    cudaFree(device_input);
    cudaFree(device_output);
    return ms;
}

void print_error(const char* domain, const char* variant, const ErrorMetrics& metrics) {
    std::printf(
        "ERROR domain=%s variant=%s max_abs=%.9g max_rel=%.9g rmse=%.9g samples=%zu\n",
        domain, variant, metrics.max_abs, metrics.max_rel, metrics.rmse(), metrics.count);
}

void print_compute(
    const char* datatype, const char* variant, float ms, float native_ms, double scalar_evals) {
    const double ns_per_value = static_cast<double>(ms) * 1.0e6 / scalar_evals;
    std::printf(
        "RESULT regime=compute datatype=%s variant=%s ms=%.6f ns_per_value=%.9g "
        "speedup=%.6f\n",
        datatype, variant, ms, ns_per_value, native_ms / ms);
}

void print_memory(
    const char* regime,
    const char* datatype,
    const char* variant,
    float ms,
    float native_ms,
    size_t bytes) {
    const double bandwidth_gbs = static_cast<double>(bytes) / (static_cast<double>(ms) * 1.0e6);
    std::printf(
        "RESULT regime=%s datatype=%s variant=%s ms=%.6f bandwidth_gbs=%.6f "
        "speedup=%.6f\n",
        regime, datatype, variant, ms, bandwidth_gbs, native_ms / ms);
}

} // namespace

int main() {
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    std::printf("GPU name=%s cc=%d.%d\n", properties.name, properties.major, properties.minor);

    const auto fractional_error_input = make_fractional_h2(kErrorPairs);
    const auto full_error_input = make_full_f32(kErrorPairs);
    print_error("fractional_h2", "native", error_fractional_h2<FractionalNative>(fractional_error_input));
    print_error(
        "fractional_h2", "pwl2_historical",
        error_fractional_h2<FractionalPwl2Historical>(fractional_error_input));
    print_error(
        "fractional_h2", "pwl2_hinge",
        error_fractional_h2<FractionalPwl2Hinge>(fractional_error_input));
    print_error("full_f32", "native", error_full_f32<FullNative>(full_error_input));
    print_error("full_f32", "degree3", error_full_f32<FullDegree3>(full_error_input));
    print_error("full_f32", "pwl2_hinge", error_full_f32<FullPwl2Hinge>(full_error_input));

    const auto compute_h2_input = make_fractional_h2(kComputeThreads * kComputeChains);
    const auto compute_f32_input = make_full_f32(kComputeThreads * kComputeChains);
    const double scalar_compute_evals = static_cast<double>(kComputeThreads) *
                                        kComputeChains * kComputeIterations * 2.0;
    const float compute_h2_native = benchmark_compute_h2<FractionalNative>(compute_h2_input);
    print_compute("fp16x2_fractional", "native", compute_h2_native, compute_h2_native,
                  scalar_compute_evals);
    print_compute(
        "fp16x2_fractional", "pwl2_historical",
        benchmark_compute_h2<FractionalPwl2Historical>(compute_h2_input), compute_h2_native,
        scalar_compute_evals);
    print_compute(
        "fp16x2_fractional", "pwl2_hinge",
        benchmark_compute_h2<FractionalPwl2Hinge>(compute_h2_input), compute_h2_native,
        scalar_compute_evals);

    const float compute_f32_native = benchmark_compute_f32<FullNative>(compute_f32_input);
    print_compute("f32x2_full", "native", compute_f32_native, compute_f32_native,
                  scalar_compute_evals);
    print_compute(
        "f32x2_full", "degree3", benchmark_compute_f32<FullDegree3>(compute_f32_input),
        compute_f32_native, scalar_compute_evals);
    print_compute(
        "f32x2_full", "pwl2_hinge", benchmark_compute_f32<FullPwl2Hinge>(compute_f32_input),
        compute_f32_native, scalar_compute_evals);

    const auto l2_h2_input = make_fractional_h2(kL2PairsH2);
    const float l2_h2_native = benchmark_memory_h2<FractionalNative>(l2_h2_input, kL2PairsH2);
    const size_t l2_h2_bytes = 2ull * kL2PairsH2 * sizeof(__half2);
    print_memory("l2", "fp16x2_fractional", "native", l2_h2_native, l2_h2_native, l2_h2_bytes);
    print_memory(
        "l2", "fp16x2_fractional", "pwl2_historical",
        benchmark_memory_h2<FractionalPwl2Historical>(l2_h2_input, kL2PairsH2), l2_h2_native,
        l2_h2_bytes);
    print_memory(
        "l2", "fp16x2_fractional", "pwl2_hinge",
        benchmark_memory_h2<FractionalPwl2Hinge>(l2_h2_input, kL2PairsH2), l2_h2_native,
        l2_h2_bytes);

    const auto l2_f32_input = make_full_f32(kL2PairsF32);
    const float l2_f32_native = benchmark_memory_f32<FullNative>(l2_f32_input, kL2PairsF32);
    const size_t l2_f32_bytes = 2ull * kL2PairsF32 * sizeof(float2);
    print_memory("l2", "f32x2_full", "native", l2_f32_native, l2_f32_native, l2_f32_bytes);
    print_memory(
        "l2", "f32x2_full", "degree3",
        benchmark_memory_f32<FullDegree3>(l2_f32_input, kL2PairsF32), l2_f32_native,
        l2_f32_bytes);
    print_memory(
        "l2", "f32x2_full", "pwl2_hinge",
        benchmark_memory_f32<FullPwl2Hinge>(l2_f32_input, kL2PairsF32), l2_f32_native,
        l2_f32_bytes);

    // Allocate HBM seeds only after the smaller measurements to keep host memory use bounded.
    const auto hbm_h2_input = make_fractional_h2(kHbmPairsH2);
    const float hbm_h2_native = benchmark_memory_h2<FractionalNative>(hbm_h2_input, kHbmPairsH2);
    const size_t hbm_h2_bytes = 2ull * kHbmPairsH2 * sizeof(__half2);
    print_memory("hbm", "fp16x2_fractional", "native", hbm_h2_native, hbm_h2_native,
                 hbm_h2_bytes);
    print_memory(
        "hbm", "fp16x2_fractional", "pwl2_historical",
        benchmark_memory_h2<FractionalPwl2Historical>(hbm_h2_input, kHbmPairsH2), hbm_h2_native,
        hbm_h2_bytes);
    print_memory(
        "hbm", "fp16x2_fractional", "pwl2_hinge",
        benchmark_memory_h2<FractionalPwl2Hinge>(hbm_h2_input, kHbmPairsH2), hbm_h2_native,
        hbm_h2_bytes);

    const auto hbm_f32_input = make_full_f32(kHbmPairsF32);
    const float hbm_f32_native = benchmark_memory_f32<FullNative>(hbm_f32_input, kHbmPairsF32);
    const size_t hbm_f32_bytes = 2ull * kHbmPairsF32 * sizeof(float2);
    print_memory("hbm", "f32x2_full", "native", hbm_f32_native, hbm_f32_native,
                 hbm_f32_bytes);
    print_memory(
        "hbm", "f32x2_full", "degree3",
        benchmark_memory_f32<FullDegree3>(hbm_f32_input, kHbmPairsF32), hbm_f32_native,
        hbm_f32_bytes);
    print_memory(
        "hbm", "f32x2_full", "pwl2_hinge",
        benchmark_memory_f32<FullPwl2Hinge>(hbm_f32_input, kHbmPairsF32), hbm_f32_native,
        hbm_f32_bytes);

    return 0;
}

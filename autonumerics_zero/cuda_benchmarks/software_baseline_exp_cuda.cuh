#define SOFTWARE_BASELINE_EXP_CUDA_CUH
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath> // For nearbyintf

/*
 * This file provides fast, "software-emulated" implementations of exp
 * for F32, F16, and F16x2. These serve as a fair "apples-to-apples"
 * baseline to compare against your evolved direct-to-sigmoid polynomial.
 *
 * All implementations use the same FMA-based algorithm:
 * 1. Range Reduction: e^x = 2^i * 2^f, where f is in [-0.5, 0.5]
 * 2. Polynomial: Approximate 2^f with a 5th-order FMA polynomial.
 * 3. Reconstruction: Use fast integer bit-twiddling (a "software ldexp")
 * to compute the final result.
 */

// -----------------------------------------------------------------
// --- F32 Implementation (Unchanged) ---
// -----------------------------------------------------------------
struct software_baseline_exp_CUDA_F32 {

    // --- Constants ---
    static constexpr float LOG2_E = 1.44269504089f;
    static constexpr float MAX_ARG = 88.0f;
    static constexpr float MIN_ARG = -88.0f;
    static constexpr float P5_C1 = 0.693147182f;
    static constexpr float P5_C2 = 0.240226469f;
    static constexpr float P5_C3 = 0.055504112f;
    static constexpr float P5_C4 = 0.009618127f;
    static constexpr float P5_C5 = 0.001333148f;

    static __forceinline__ __device__ float evaluate(float x) {
        // 1. Clamp input
        x = fminf(fmaxf(x, MIN_ARG), MAX_ARG);

        // 2. Range Reduction
        float t = x * LOG2_E;
        float i = nearbyintf(t); // round to nearest integer
        float f = t - i;         // fractional part in [-0.5, 0.5]

        // 3. Compute 2^f using 5th-order polynomial (Horner's method for FMA)
        float poly = fmaf(f, P5_C5, P5_C4);
        poly = fmaf(f, poly, P5_C3);
        poly = fmaf(f, poly, P5_C2);
        poly = fmaf(f, poly, P5_C1);
        poly = fmaf(f, poly, 1.0f); // poly is now approx 2^f

        // 4. Reconstruct: compute 2^i * 2^f
        // This is the software "ldexpf(poly, i)"
        int exp_int = (int)i;
        int poly_bits = __float_as_int(poly);
        int result_bits = poly_bits + (exp_int << 23);
        return __int_as_float(result_bits);
    }
};

// -----------------------------------------------------------------
// --- NATIVE H (FP16) Implementation (Unchanged) ---
// -----------------------------------------------------------------
struct software_baseline_exp_CUDA_H {

    // --- Constants (as inlined functions to be safe with __half) ---
    static __forceinline__ __device__ __half H_LOG2_E() { return __float2half_rn(1.442695f); }
    static __forceinline__ __device__ __half H_MAX_ARG() { return __float2half_rn(11.0f); } // log(65504) ~ 11.09
    static __forceinline__ __device__ __half H_MIN_ARG() { return __float2half_rn(-10.0f); }
    static __forceinline__ __device__ __half H_ONE() { return __float2half_rn(1.0f); }
    static __forceinline__ __device__ __half H_C1() { return __float2half_rn(0.693147f); }
    static __forceinline__ __device__ __half H_C2() { return __float2half_rn(0.240226f); }
    static __forceinline__ __device__ __half H_C3() { return __float2half_rn(0.055504f); }
    static __forceinline__ __device__ __half H_C4() { return __float2half_rn(0.009618f); }
    static __forceinline__ __device__ __half H_C5() { return __float2half_rn(0.001333f); }

    static __forceinline__ __device__ __half evaluate(__half x) {
        // 1. Clamp input
        x = __hmax(x, H_MIN_ARG());
        x = __hmin(x, H_MAX_ARG());

        // 2. Range Reduction
        __half t = __hmul(x, H_LOG2_E());
        __half i = hrint(t); // round to nearest floating-point integer
        __half f = __hsub(t, i); // fractional part in [-0.5, 0.5]

        // 3. Compute 2^f using polynomial and native __hfma
        __half poly = __hfma(f, H_C5(), H_C4());
        poly = __hfma(f, poly, H_C3());
        poly = __hfma(f, poly, H_C2());
        poly = __hfma(f, poly, H_C1());
        poly = __hfma(f, poly, H_ONE()); // poly is now approx 2^f

        // 4. Reconstruct: "software ldexp(poly, i)"
        // Convert floating-point integer (e.g., 3.0h) to C int (e.g., 3)
        short i_int = __half2short_rd(i);

        // Get the 16-bit representation
        unsigned short poly_bits = __half_as_ushort(poly);

        // Add the integer exponent (i << 10) to the FP16 exponent
        unsigned short result_bits = poly_bits + (i_int << 10);

        return __ushort_as_half(result_bits);
    }
};


// -----------------------------------------------------------------
// --- NATIVE H2 (FP16x2) Implementation [FIXED] ---
// -----------------------------------------------------------------
struct software_baseline_exp_CUDA_H2 {

    // --- Constants (as inlined functions for __half2) ---
    static __forceinline__ __device__ __half2 H2_LOG2_E() { return __float2half2_rn(1.442695f); }
    static __forceinline__ __device__ __half2 H2_MAX_ARG() { return __float2half2_rn(11.0f); }
    static __forceinline__ __device__ __half2 H2_MIN_ARG() { return __float2half2_rn(-10.0f); }
    static __forceinline__ __device__ __half2 H2_ONE() { return __float2half2_rn(1.0f); }
    static __forceinline__ __device__ __half2 H2_C1() { return __float2half2_rn(0.693147f); }
    static __forceinline__ __device__ __half2 H2_C2() { return __float2half2_rn(0.240226f); }
    static __forceinline__ __device__ __half2 H2_C3() { return __float2half2_rn(0.055504f); }
    static __forceinline__ __device__ __half2 H2_C4() { return __float2half2_rn(0.009618f); }
    static __forceinline__ __device__ __half2 H2_C5() { return __float2half2_rn(0.001333f); }

    static __forceinline__ __device__ __half2 evaluate(__half2 x) {
        // 1. Clamp input
        x = __hmax2(x, H2_MIN_ARG());
        x = __hmin2(x, H2_MAX_ARG());

        // 2. Range Reduction
        __half2 t = __hmul2(x, H2_LOG2_E());
        __half2 i = h2rint(t); // round to nearest floating-point integer
        __half2 f = __hsub2(t, i); // fractional part in [-0.5, 0.5]

        // 3. Compute 2^f using polynomial and native __hfma2
        __half2 poly = __hfma2(f, H2_C5(), H2_C4());
        poly = __hfma2(f, poly, H2_C3());
        poly = __hfma2(f, poly, H2_C2());
        poly = __hfma2(f, poly, H2_C1());
        poly = __hfma2(f, poly, H2_ONE()); // poly is now approx 2^f

        // 4. Reconstruct: "software ldexp(poly, i)" for H2
        // Convert floating-point integers (e.g., 3.0h) to C integers (e.g., 3)

        // Portable __half2 ldexp reconstruction
        __half lo_i = __low2half(i);
        __half hi_i = __high2half(i);
        int i_lo = __half2int_rd(lo_i);
        int i_hi = __half2int_rd(hi_i);

        // reinterpret __half2 -> uint32_t
        unsigned int poly_bits = *reinterpret_cast<unsigned int*>(&poly);

        unsigned short poly_lo = poly_bits & 0xFFFF;
        unsigned short poly_hi = poly_bits >> 16;

        unsigned short res_lo = poly_lo + (i_lo << 10);
        unsigned short res_hi = poly_hi + (i_hi << 10);

        unsigned int result_bits = (res_hi << 16) | res_lo;

        // reinterpret uint -> __half2
        return *reinterpret_cast<__half2*>(&result_bits);

    }
};

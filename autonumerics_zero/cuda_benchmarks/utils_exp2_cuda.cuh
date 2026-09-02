#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Exact implementation of 'e2e_asm2' from flash_attn/cute/utils.py
// Adapted for CUDA C++
struct utils_exp2_cuda {

    static __forceinline__ __device__ float2 evaluate_f32x2(float2 val) {
        float x = val.x;
        float y = val.y;

        // constants
        const float MAGIC = 12582912.0f; // 0x4B400000
        const float C3 = 0.077119089663028717041015625f; // 0x3D9DF09D
        const float C2 = 0.227564394474029541015625f;    // 0x3E6906A4
        const float C1 = 0.695146143436431884765625f;    // 0x3F31F519
        const float C0 = 1.0f;                           // 0x3F800000
        const float NEG_127 = -127.0f;

        // max.ftz.f32 (Clamp)
        x = fmaxf(x, NEG_127);
        y = fmaxf(y, NEG_127);

        // add.rm.ftz.f32 (Round Minus Infinity / Down)
        // x_rounded = x + MAGIC
        float x_rounded = __fadd_rd(x, MAGIC);
        float y_rounded = __fadd_rd(y, MAGIC);

        // sub.rn.ftz.f32 (Round Nearest)
        // x_rounded_back = x_rounded - MAGIC
        float x_rounded_back = __fsub_rn(x_rounded, MAGIC);
        float y_rounded_back = __fsub_rn(y_rounded, MAGIC);

        // x_frac = x - x_rounded_back
        float x_frac = __fsub_rn(x, x_rounded_back);
        float y_frac = __fsub_rn(y, y_rounded_back);

        // Polynomial Evaluation (Degree 3) - FMA pipeline
        // p = x_frac * C3 + C2
        float p_x = __fmaf_rn(x_frac, C3, C2);
        float p_y = __fmaf_rn(y_frac, C3, C2);

        // p = p * x_frac + C1
        p_x = __fmaf_rn(p_x, x_frac, C1);
        p_y = __fmaf_rn(p_y, y_frac, C1);

        // p = p * x_frac + C0
        p_x = __fmaf_rn(p_x, x_frac, C0);
        p_y = __fmaf_rn(p_y, y_frac, C0);

        // Reconstruction
        // Extract integer bits from x_rounded (which is float)
        int i_x = __float_as_int(x_rounded);
        int i_y = __float_as_int(y_rounded);

        // Extract bits from poly result
        int p_bits_x = __float_as_int(p_x);
        int p_bits_y = __float_as_int(p_y);

        // Shift integer part (exponent) into position
        int e_x = i_x << 23;
        int e_y = i_y << 23;

        // Add to mantissa/poly
        int res_x = e_x + p_bits_x;
        int res_y = e_y + p_bits_y;

        return make_float2(__int_as_float(res_x), __int_as_float(res_y));
    }
};

struct utils_exp2_cuda_CUDA_H2 {
    static __forceinline__ __device__ __half2 evaluate(__half2 val) {
        float2 val_f2 = __half22float2(val);
        float2 res_f2 = utils_exp2_cuda::evaluate_f32x2(val_f2);
        return __float22half2_rn(res_f2);
    }
};

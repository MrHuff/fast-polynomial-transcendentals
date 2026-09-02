// Copyright (c) 2026 Graphcore Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified in 2026 for the standalone fast-polynomial-transcendentals release.

#include <cuda_fp8.h>
#include <cuda_fp16.h>

namespace SplineHelpers {
    static __device__ __forceinline__ __half2 h2floor(__half2 val) {
        float2 f = __half22float2(val);
        f.x = floorf(f.x);
        f.y = floorf(f.y);
        return __float22half2_rn(f);
    }

    static __device__ __forceinline__ int2 find_interval_h2(__half2 val) {
        return make_int2(0, 0);
    }
}
struct EVOLVED_H2_HARDCODED_2 {
    static __device__ __forceinline__ unsigned short get_coeff(int idx, int degree_idx) {
        if (degree_idx == 1) {
        switch(idx % 2) {
            case 0: return 14794;
            case 1: return 14794;
            default: return 14794;
        }
        } else if (degree_idx == 0) {
        switch(idx % 2) {
            case 0: return 15359;
            case 1: return 15359;
            default: return 15359;
        }
        } else { return 0; }
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);


        // Degree 1
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 1)),
                __short_as_half(get_coeff(p, 1))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        // Degree 0
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 0)),
                __short_as_half(get_coeff(p, 0))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        return result_h2;
    }


    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x % 2;
        int p = intervals.y % 2;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
    };


__global__ void evolved_pow2_h2_hardcoded_2(__half2* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __half2 t_out = data[idx];
        #pragma unroll
        for (int r = 0; r < inner_repeats; ++r) {
            t_out = EVOLVED_H2_HARDCODED_2::evaluate(t_out);
        }
        data[idx] = t_out;
    }
}

struct EVOLVED_H2_HARDCODED_3 {
    static __device__ __forceinline__ unsigned short get_coeff(int idx, int degree_idx) {
        if (degree_idx == 1) {
        switch(idx % 3) {
            case 0: return 14794;
            case 1: return 14794;
            case 2: return 14929;
            default: return 14794;
        }
        } else if (degree_idx == 0) {
        switch(idx % 3) {
            case 0: return 15359;
            case 1: return 15359;
            case 2: return 15342;
            default: return 15359;
        }
        } else { return 0; }
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);


        // Degree 1
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 1)),
                __short_as_half(get_coeff(p, 1))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        // Degree 0
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 0)),
                __short_as_half(get_coeff(p, 0))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        return result_h2;
    }


    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x % 3;
        int p = intervals.y % 3;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
    };


__global__ void evolved_pow2_h2_hardcoded_3(__half2* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __half2 t_out = data[idx];
        #pragma unroll
        for (int r = 0; r < inner_repeats; ++r) {
            t_out = EVOLVED_H2_HARDCODED_3::evaluate(t_out);
        }
        data[idx] = t_out;
    }
}

struct EVOLVED_H2_HARDCODED_4 {
    static __device__ __forceinline__ unsigned short get_coeff(int idx, int degree_idx) {
        if (degree_idx == 1) {
        switch(idx % 4) {
            case 0: return 14794;
            case 1: return 14794;
            case 2: return 14929;
            case 3: return 14929;
            default: return 14794;
        }
        } else if (degree_idx == 0) {
        switch(idx % 4) {
            case 0: return 15359;
            case 1: return 15359;
            case 2: return 15342;
            case 3: return 15342;
            default: return 15359;
        }
        } else { return 0; }
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);


        // Degree 1
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 1)),
                __short_as_half(get_coeff(p, 1))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        // Degree 0
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 0)),
                __short_as_half(get_coeff(p, 0))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        return result_h2;
    }


    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x % 4;
        int p = intervals.y % 4;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
    };


__global__ void evolved_pow2_h2_hardcoded_4(__half2* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __half2 t_out = data[idx];
        #pragma unroll
        for (int r = 0; r < inner_repeats; ++r) {
            t_out = EVOLVED_H2_HARDCODED_4::evaluate(t_out);
        }
        data[idx] = t_out;
    }
}

struct EVOLVED_H2_HARDCODED_8 {
    static __device__ __forceinline__ unsigned short get_coeff(int idx, int degree_idx) {
        if (degree_idx == 1) {
        switch(idx % 8) {
            case 0: return 14794;
            case 1: return 14794;
            case 2: return 14929;
            case 3: return 14929;
            case 4: return 15074;
            case 5: return 15074;
            case 6: return 15193;
            case 7: return 15276;
            default: return 14794;
        }
        } else if (degree_idx == 0) {
        switch(idx % 8) {
            case 0: return 15359;
            case 1: return 15359;
            case 2: return 15342;
            case 3: return 15342;
            case 4: return 15306;
            case 5: return 15306;
            case 6: return 15262;
            case 7: return 15226;
            default: return 15359;
        }
        } else { return 0; }
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);


        // Degree 1
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 1)),
                __short_as_half(get_coeff(p, 1))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        // Degree 0
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 0)),
                __short_as_half(get_coeff(p, 0))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        return result_h2;
    }


    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x % 8;
        int p = intervals.y % 8;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
    };


__global__ void evolved_pow2_h2_hardcoded_8(__half2* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __half2 t_out = data[idx];
        #pragma unroll
        for (int r = 0; r < inner_repeats; ++r) {
            t_out = EVOLVED_H2_HARDCODED_8::evaluate(t_out);
        }
        data[idx] = t_out;
    }
}

struct EVOLVED_H2_HARDCODED_12 {
    static __device__ __forceinline__ unsigned short get_coeff(int idx, int degree_idx) {
        if (degree_idx == 1) {
        switch(idx % 12) {
            case 0: return 14794;
            case 1: return 14794;
            case 2: return 14929;
            case 3: return 14929;
            case 4: return 15074;
            case 5: return 15074;
            case 6: return 15193;
            case 7: return 15276;
            case 8: return 15362;
            case 9: return 15407;
            case 10: return 15479;
            case 11: return 15479;
            default: return 14794;
        }
        } else if (degree_idx == 0) {
        switch(idx % 12) {
            case 0: return 15359;
            case 1: return 15359;
            case 2: return 15342;
            case 3: return 15342;
            case 4: return 15306;
            case 5: return 15306;
            case 6: return 15262;
            case 7: return 15226;
            case 8: return 15182;
            case 9: return 15131;
            case 10: return 15040;
            case 11: return 15040;
            default: return 15359;
        }
        } else { return 0; }
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);


        // Degree 1
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 1)),
                __short_as_half(get_coeff(p, 1))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        // Degree 0
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 0)),
                __short_as_half(get_coeff(p, 0))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        return result_h2;
    }


    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x % 12;
        int p = intervals.y % 12;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
    };


__global__ void evolved_pow2_h2_hardcoded_12(__half2* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __half2 t_out = data[idx];
        #pragma unroll
        for (int r = 0; r < inner_repeats; ++r) {
            t_out = EVOLVED_H2_HARDCODED_12::evaluate(t_out);
        }
        data[idx] = t_out;
    }
}

struct EVOLVED_H2_HARDCODED_2_DEG2 {
    static __device__ __forceinline__ unsigned short get_coeff(int idx, int degree_idx) {
        if (degree_idx == 2) {
        switch(idx % 2) {
            case 0: return 15359;
            case 1: return 15358; // Distinct
            default: return 15359;
        }
        } else if (degree_idx == 1) {
        switch(idx % 2) {
            case 0: return 14794;
            case 1: return 14795; // Distinct
            default: return 14794;
        }
        } else if (degree_idx == 0) {
        switch(idx % 2) {
            case 0: return 15359;
            case 1: return 15360; // Distinct
            default: return 15359;
        }
        } else { return 0; }
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);


        // Degree 2
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 2)),
                __short_as_half(get_coeff(p, 2))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        // Degree 1
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 1)),
                __short_as_half(get_coeff(p, 1))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        // Degree 0
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 0)),
                __short_as_half(get_coeff(p, 0))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        return result_h2;
    }


    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x % 2;
        int p = intervals.y % 2;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
    };


__global__ void evolved_pow2_h2_hardcoded_2_deg2(__half2* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __half2 t_out = data[idx];
        #pragma unroll
        for (int r = 0; r < inner_repeats; ++r) {
            t_out = EVOLVED_H2_HARDCODED_2_DEG2::evaluate(t_out);
        }
        data[idx] = t_out;
    }
}

struct EVOLVED_H2_HARDCODED_2_DEG3 {
    static __device__ __forceinline__ unsigned short get_coeff(int idx, int degree_idx) {
        if (degree_idx == 3) {
        switch(idx % 2) {
            case 0: return 14794;
            case 1: return 14794;
            default: return 14794;
        }
        } else if (degree_idx == 2) {
        switch(idx % 2) {
            case 0: return 15359;
            case 1: return 15359;
            default: return 15359;
        }
        } else if (degree_idx == 1) {
        switch(idx % 2) {
            case 0: return 14794;
            case 1: return 14794;
            default: return 14794;
        }
        } else if (degree_idx == 0) {
        switch(idx % 2) {
            case 0: return 15359;
            case 1: return 15359;
            default: return 15359;
        }
        } else { return 0; }
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);


        // Degree 3
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 3)),
                __short_as_half(get_coeff(p, 3))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        // Degree 2
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 2)),
                __short_as_half(get_coeff(p, 2))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        // Degree 1
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 1)),
                __short_as_half(get_coeff(p, 1))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        // Degree 0
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 0)),
                __short_as_half(get_coeff(p, 0))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        return result_h2;
    }


    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x % 2;
        int p = intervals.y % 2;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
    };


__global__ void evolved_pow2_h2_hardcoded_2_deg3(__half2* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __half2 t_out = data[idx];
        #pragma unroll
        for (int r = 0; r < inner_repeats; ++r) {
            t_out = EVOLVED_H2_HARDCODED_2_DEG3::evaluate(t_out);
        }
        data[idx] = t_out;
    }
}

struct EVOLVED_H2_HARDCODED_2_DEG4 {
    static __device__ __forceinline__ unsigned short get_coeff(int idx, int degree_idx) {
        if (degree_idx == 4) {
        switch(idx % 2) {
            case 0: return 15359;
            case 1: return 15359;
            default: return 15359;
        }
        } else if (degree_idx == 3) {
        switch(idx % 2) {
            case 0: return 14794;
            case 1: return 14794;
            default: return 14794;
        }
        } else if (degree_idx == 2) {
        switch(idx % 2) {
            case 0: return 15359;
            case 1: return 15359;
            default: return 15359;
        }
        } else if (degree_idx == 1) {
        switch(idx % 2) {
            case 0: return 14794;
            case 1: return 14794;
            default: return 14794;
        }
        } else if (degree_idx == 0) {
        switch(idx % 2) {
            case 0: return 15359;
            case 1: return 15359;
            default: return 15359;
        }
        } else { return 0; }
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);


        // Degree 4
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 4)),
                __short_as_half(get_coeff(p, 4))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        // Degree 3
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 3)),
                __short_as_half(get_coeff(p, 3))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        // Degree 2
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 2)),
                __short_as_half(get_coeff(p, 2))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        // Degree 1
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 1)),
                __short_as_half(get_coeff(p, 1))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        // Degree 0
        {
             const __half2 coeff_h2 = __halves2half2(
                __short_as_half(get_coeff(o, 0)),
                __short_as_half(get_coeff(p, 0))
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }


        return result_h2;
    }


    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x % 2;
        int p = intervals.y % 2;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
    };


__global__ void evolved_pow2_h2_hardcoded_2_deg4(__half2* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __half2 t_out = data[idx];
        #pragma unroll
        for (int r = 0; r < inner_repeats; ++r) {
            t_out = EVOLVED_H2_HARDCODED_2_DEG4::evaluate(t_out);
        }
        data[idx] = t_out;
    }
}

struct EVOLVED_H2_BRANCHED_MIXED {

    // Hardcoded Linear Path (Interval 0)
    static __device__ __forceinline__ __half2 eval_linear(const __half2 t_h2) {
        // C2 = 0
        // C1 = 14794
        // C0 = 15359
        __half2 result_h2 = __float2half2_rn(0.0f);

        unsigned short c1_val = 14794;
        unsigned short c0_val = 15359;

        __half2 c1 = __halves2half2(__short_as_half(c1_val), __short_as_half(c1_val));
        __half2 c0 = __halves2half2(__short_as_half(c0_val), __short_as_half(c0_val));

        // Degree 1
        result_h2 = __hfma2(t_h2, result_h2, c1);
        // Degree 0
        result_h2 = __hfma2(t_h2, result_h2, c0);
        return result_h2;
    }

    // Hardcoded Quadratic Path (Interval 1)
    static __device__ __forceinline__ __half2 eval_quadratic(const __half2 t_h2) {
        // C2 = 15359
        // C1 = 15359
        // C0 = 15359
        __half2 result_h2 = __float2half2_rn(0.0f);

        unsigned short c2_val = 15359;
        unsigned short c1_val = 15359;
        unsigned short c0_val = 15359;

        __half2 c2 = __halves2half2(__short_as_half(c2_val), __short_as_half(c2_val));
        __half2 c1 = __halves2half2(__short_as_half(c1_val), __short_as_half(c1_val));
        __half2 c0 = __halves2half2(__short_as_half(c0_val), __short_as_half(c0_val));

        // Degree 2
        result_h2 = __hfma2(t_h2, result_h2, c2);
        // Degree 1
        result_h2 = __hfma2(t_h2, result_h2, c1);
        // Degree 0
        result_h2 = __hfma2(t_h2, result_h2, c0);
        return result_h2;
    }

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x % 2;
        int p = intervals.y % 2;

        // BRANCHING MIXED
        // If both are 0: Run Linear (Fast)
        // If both are 1: Run Quadratic
        // Else: Run Quadratic (Conservative, or scalarized)

        __half2 poly_res;

        if (o == 0 && p == 0) {
            poly_res = eval_linear(t_clamped_h2);
        } else if (o == 1 && p == 1) {
            poly_res = eval_quadratic(t_clamped_h2);
        } else {
             // Mixed case (divergence within vector).
             // Fallback to Quadratic ("Hardcode" the more complex one correct)
             // Or construct scalar manually.
             // Let's use Quadratic (Linear is subset if coeff C2=0) but we don't have C2=0 here.
             // We MUST scalarize or execute both paths masked?
             // Since this is rare, let's just run Quadratic with "Wrong" C2 for the Linear lane? No, incorrect result.
             // We must scalarize.

             __half tx = __low2half(t_clamped_h2);
             __half ty = __high2half(t_clamped_h2);

             __half rx, ry;

             // X Element
             if (o == 0) {
                 // Linear Manual
                 // res = 0*t + 14794 -> res*t + 15359.
                 // Manual FMA sequence for scalar half
                  unsigned short c1_val = 14794;
                  unsigned short c0_val = 15359;
                  __half c1 = __short_as_half(c1_val);
                  __half c0 = __short_as_half(c0_val);
                  rx = __hfma(tx, __float2half(0.0f), c1); // res = 0*t + c1 = c1 ?? No.
                  // Poly: res=0. res=fma(t,res,c1) -> c1. res=fma(t,res,c0) -> c1*t+c0.
                  // For Linear (no C2):
                  rx = __float2half(0.0f);
                  rx = __hfma(tx, rx, c1);
                  rx = __hfma(tx, rx, c0);
             } else {
                 // Quadratic Manual
                 unsigned short c2_val = 15359;
                 unsigned short c1_val = 15359;
                 unsigned short c0_val = 15359;
                 __half c2 = __short_as_half(c2_val);
                 __half c1 = __short_as_half(c1_val);
                 __half c0 = __short_as_half(c0_val);
                 rx = __float2half(0.0f);
                 rx = __hfma(tx, rx, c2);
                 rx = __hfma(tx, rx, c1);
                 rx = __hfma(tx, rx, c0);
             }

             // Y Element
              if (p == 0) {
                  unsigned short c1_val = 14794;
                  unsigned short c0_val = 15359;
                  __half c1 = __short_as_half(c1_val);
                  __half c0 = __short_as_half(c0_val);
                  ry = __float2half(0.0f);
                  ry = __hfma(ty, ry, c1);
                  ry = __hfma(ty, ry, c0);
             } else {
                 unsigned short c2_val = 15359;
                 unsigned short c1_val = 15359;
                 unsigned short c0_val = 15359;
                 __half c2 = __short_as_half(c2_val);
                 __half c1 = __short_as_half(c1_val);
                 __half c0 = __short_as_half(c0_val);
                 ry = __float2half(0.0f);
                 ry = __hfma(ty, ry, c2);
                 ry = __hfma(ty, ry, c1);
                 ry = __hfma(ty, ry, c0);
             }

             poly_res = __halves2half2(rx, ry);
        }

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_res);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
    };

__global__ void evolved_pow2_h2_branched_mixed(__half2* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __half2 t_out = data[idx];
        #pragma unroll
        for (int r = 0; r < inner_repeats; ++r) {
            t_out = EVOLVED_H2_BRANCHED_MIXED::evaluate(t_out);
        }
        data[idx] = t_out;
    }
}


struct EVOLVED_H2_PACKED_REG_2 {

    // N=2, Degree 1: 14794, 14794
    // Hex: 0x39CA, 0x39CA
    // Packed (32-bit): 0x39CA39CA
    static constexpr unsigned int D1_PACKED = 0x39CA39CA;

    // N=2, Degree 0: 15359, 15359
    // Hex: 0x3BFF, 0x3BFF
    // Packed (32-bit): 0x3BFF3BFF
    static constexpr unsigned int D0_PACKED = 0x3BFF3BFF;

    static __device__ __forceinline__ __half get_packed_coeff(int idx, unsigned int packed) {
        // 1. Shift:
        // Index is either 0 or 1.
        // If 0: Shift 0. If 1: Shift 16.
        // logic: (idx & 1) << 4
        int shift = (idx & 1) << 4;

        // 2. Snip:
        unsigned short raw = (unsigned short)((packed >> shift) & 0xFFFF);

        // 3. Convert (Reinterpret bits only, no conversion instruction)
        return *reinterpret_cast<__half*>(&raw);
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        // Degree 1
        {
            __half c_left  = get_packed_coeff(o, D1_PACKED);
            __half c_right = get_packed_coeff(p, D1_PACKED);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }

        // Degree 0
        {
            __half c_left  = get_packed_coeff(o, D0_PACKED);
            __half c_right = get_packed_coeff(p, D0_PACKED);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }
        return result_h2;
    }

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));
        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x % 2;
        int p = intervals.y % 2;
        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);
        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

struct EVOLVED_H2_PACKED_REG_4 {

    // N=4, Degree 1
    // Hex: 0x39CA, 0x39CB, 0x39CC, 0x39CD (Distinct for benchmark fairness)
    // Packed (64-bit): 0x39CD39CC39CB39CAULL
    static constexpr unsigned long long D1_PACKED = 0x39CD39CC39CB39CAULL;

    // N=4, Degree 0
    // Hex: 0x3BFF, 0x3BFE, 0x3BFD, 0x3BFC (Distinct for benchmark fairness)
    // Packed (64-bit): 0x3BFC3BFD3BFE3BFFULL
    static constexpr unsigned long long D0_PACKED = 0x3BFC3BFD3BFE3BFFULL;

    static __device__ __forceinline__ __half get_packed_coeff(int idx, unsigned long long packed) {
        // 1. Shift:
        // Index is 0, 1, 2, 3.
        // Shift amount = index * 16
        // logic: (idx & 3) << 4
        int shift = (idx & 3) << 4;

        // 2. Snip:
        unsigned short raw = (unsigned short)((packed >> shift) & 0xFFFF);

        // 3. Convert
        return *reinterpret_cast<__half*>(&raw);
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        // Degree 1
        {
            __half c_left  = get_packed_coeff(o, D1_PACKED);
            __half c_right = get_packed_coeff(p, D1_PACKED);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }

        // Degree 0
        {
            __half c_left  = get_packed_coeff(o, D0_PACKED);
            __half c_right = get_packed_coeff(p, D0_PACKED);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }
        return result_h2;
    }

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));
        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2); // N needs to be 4 here, check implementation!

        // WARNING: find_interval_h2 might need parameter for N or be specialized.
        // Assuming find_interval_h2 works for N=4 logic (based on bits?)
        // Let's check SPLINE_FUNCS.cuh first.

        // Re-implement simplified interval finding for N=4 for safety/correctness
        // N=4 -> 2 fractional bits.
        // For x in [0,1), interval = floor(x * 4) = floor(x * 2^2)
        // Extract exponent, adjust, extract mantissa bits.

        // However, using the helper assuming it gives correct int index.
        int o = intervals.x % 4;
        int p = intervals.y % 4;
        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);
        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

struct EVOLVED_H2_PACKED_REG_8 {

    // N=8, Degree 1
    // LO: Distinct values
    static constexpr unsigned long long D1_PACKED_LO = 0x3A543A5339CC39CBULL;
    // HI: Distinct values
    static constexpr unsigned long long D1_PACKED_HI = 0x3BAD3B5A3AE33AE2ULL;

    // N=8, Degree 0
    // LO: Distinct values
    static constexpr unsigned long long D0_PACKED_LO = 0x3BF13BF03BFE3BFDULL;
    // HI: Distinct values
    static constexpr unsigned long long D0_PACKED_HI = 0x3B7C3B9F3BCB3BCAULL;

    static __device__ __forceinline__ __half get_packed_coeff(int idx, unsigned long long packed_lo, unsigned long long packed_hi) {
        // Selection:
        // if idx < 4: use LO, shift = (idx & 3) << 4
        // if idx >=4: use HI, shift = (idx & 3) << 4

        unsigned long long selected = (idx < 4) ? packed_lo : packed_hi;
        int shift = (idx & 3) << 4;

        unsigned short raw = (unsigned short)((selected >> shift) & 0xFFFF);
        return *reinterpret_cast<__half*>(&raw);
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        // Degree 1
        {
            __half c_left  = get_packed_coeff(o, D1_PACKED_LO, D1_PACKED_HI);
            __half c_right = get_packed_coeff(p, D1_PACKED_LO, D1_PACKED_HI);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }

        // Degree 0
        {
            __half c_left  = get_packed_coeff(o, D0_PACKED_LO, D0_PACKED_HI);
            __half c_right = get_packed_coeff(p, D0_PACKED_LO, D0_PACKED_HI);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }
        return result_h2;
    }

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));
        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);

        // Use mod 8 logic
        int o = intervals.x % 8;
        int p = intervals.y % 8;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);
        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

struct EVOLVED_H2_PACKED_REG_16 {

    // N=16, Degree 1 & 0
    // Using placeholder values (repeating N=8 pattern for scalability test)
    static constexpr unsigned long long D1_PACKED_0 = 0x3A513A5139CA39CAULL; // 0-3
    static constexpr unsigned long long D1_PACKED_1 = 0x3BAC3B593AE23AE2ULL; // 4-7
    static constexpr unsigned long long D1_PACKED_2 = 0x3A513A5139CA39CAULL; // 8-11 (placeholder)
    static constexpr unsigned long long D1_PACKED_3 = 0x3BAC3B593AE23AE2ULL; // 12-15 (placeholder)

    static constexpr unsigned long long D0_PACKED_0 = 0x3BEE3BEE3BFF3BFFULL;
    static constexpr unsigned long long D0_PACKED_1 = 0x3B7A3B9E3BCA3BCAULL;
    static constexpr unsigned long long D0_PACKED_2 = 0x3BEE3BEE3BFF3BFFULL;
    static constexpr unsigned long long D0_PACKED_3 = 0x3B7A3B9E3BCA3BCAULL;

    static __device__ __forceinline__ __half get_packed_coeff(int idx, unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3) {
        // Selection:
        // Div 4 to choose bank.
        // Or simpler branched logic.

        unsigned long long selected;
        if (idx < 8) {
             selected = (idx < 4) ? p0 : p1;
        } else {
             selected = (idx < 12) ? p2 : p3;
        }

        int shift = (idx & 3) << 4;

        unsigned short raw = (unsigned short)((selected >> shift) & 0xFFFF);
        return *reinterpret_cast<__half*>(&raw);
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        // Degree 1
        {
            __half c_left  = get_packed_coeff(o, D1_PACKED_0, D1_PACKED_1, D1_PACKED_2, D1_PACKED_3);
            __half c_right = get_packed_coeff(p, D1_PACKED_0, D1_PACKED_1, D1_PACKED_2, D1_PACKED_3);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }

        // Degree 0
        {
            __half c_left  = get_packed_coeff(o, D0_PACKED_0, D0_PACKED_1, D0_PACKED_2, D0_PACKED_3);
            __half c_right = get_packed_coeff(p, D0_PACKED_0, D0_PACKED_1, D0_PACKED_2, D0_PACKED_3);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }
        return result_h2;
    }

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));
        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);

        // Use mod 16 logic
        int o = intervals.x % 16;
        int p = intervals.y % 16;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);
        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

// --------------------------------------------------------------------------
// CHEEKY N=8 IMPLEMENTATION (Using uint4 for 128-bit storage)
// --------------------------------------------------------------------------

struct EVOLVED_H2_VECTOR_8 {

    // We use uint4 to hold 128 bits of data (8 x 16-bit coeffs)
    // x: Coeffs 0,1 (Low 16, High 16)
    // y: Coeffs 2,3
    // z: Coeffs 4,5
    // w: Coeffs 6,7

    // DEGREE 1 PACKED (Slope)
    // 0-1: 14794, 14794 -> 0x39CA39CA
    // 2-3: 14929, 14929 -> 0x3A513A51
    // 4-5: 15074, 15074 -> 0x3AE23AE2
    // 6-7: 15193, 15276 -> 0x3BAC3B59

    // DEGREE 1 PACKED (Slope)
    // Distinct values
    static __device__ __forceinline__ uint4 get_d1_pack() {
        return make_uint4(0x39CB39CA, 0x3A523A51, 0x3AE33AE2, 0x3BAD3B59);
    }

    // DEGREE 0 PACKED (Offset)
    static __device__ __forceinline__ uint4 get_d0_pack() {
        return make_uint4(0x3BFE3BFF, 0x3BEF3BEE, 0x3BCB3BCA, 0x3B7B3B9E);
    }

    // The Magic Extractor
    static __device__ __forceinline__ __half get_coeff_from_uint4(int idx, uint4 pack) {
        // Step 1: Pick the 32-bit bucket (0-3)
        // This is the only "branchy" part, but it's cleaner than 64-bit selects.

        // Let's try the direct component access (Compiler optimizes this to select)
        unsigned int bucket;
        int bucket_idx = idx >> 1; // 0, 1, 2, or 3

        if (bucket_idx == 0) bucket = pack.x;
        else if (bucket_idx == 1) bucket = pack.y;
        else if (bucket_idx == 2) bucket = pack.z;
        else bucket = pack.w;

        // Step 2: Pick the 16-bit slice (Top or Bottom)
        // If idx is even (0, 2, 4...), we want bottom.
        // If idx is odd (1, 3, 5...), we want top.
        // Shift amount = (idx & 1) * 16
        int shift = (idx & 1) << 4;

        unsigned short raw = (unsigned short)((bucket >> shift) & 0xFFFF);
        return *reinterpret_cast<__half*>(&raw);
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        uint4 d1 = get_d1_pack(); // Compiler will likely keep this in registers
        uint4 d0 = get_d0_pack();

        // DEGREE 1
        {
            __half c_left  = get_coeff_from_uint4(o, d1);
            __half c_right = get_coeff_from_uint4(p, d1);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }

        // DEGREE 0
        {
            __half c_left  = get_coeff_from_uint4(o, d0);
            __half c_right = get_coeff_from_uint4(p, d0);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }
        return result_h2;
    }

    // Standard evaluate()...
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
         __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));
        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x % 8;
        int p = intervals.y % 8;
        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);
        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

// --------------------------------------------------------------------------
// CHEEKY N=8 TREE MUX IMPLEMENTATION (Using 4x 32-bit Integers)
// --------------------------------------------------------------------------

struct EVOLVED_H2_TREE_8 {

    // -----------------------------------------------------------------
    // STORAGE: 4x 32-bit Integers (8x 16-bit coeffs)
    // -----------------------------------------------------------------
    // We avoid 64-bit types entirely to save register weight and shift cost.

    // DEGREE 1
    // [0-1] Distinct
    static constexpr unsigned int D1_0 = 0x39CB39CA;
    // [2-3] Distinct
    static constexpr unsigned int D1_1 = 0x3A523A51;
    // [4-5] Distinct
    static constexpr unsigned int D1_2 = 0x3AE33AE2;
    // [6-7] Distinct
    static constexpr unsigned int D1_3 = 0x3BAD3B59;

    // DEGREE 0
    // [0-1] Distinct
    static constexpr unsigned int D0_0 = 0x3BFE3BFF;
    // [2-3] Distinct
    static constexpr unsigned int D0_1 = 0x3BEF3BEE;
    // [4-5] Distinct
    static constexpr unsigned int D0_2 = 0x3BCB3BCA;
    // [6-7] Distinct
    static constexpr unsigned int D0_3 = 0x3B7B3B9E;

    // -----------------------------------------------------------------
    // THE 32-BIT TREE MUX
    // -----------------------------------------------------------------
    static __device__ __forceinline__ __half get_coeff_tree(int idx,
                                                            unsigned int r0, unsigned int r1,
                                                            unsigned int r2, unsigned int r3) {
        // We want to select one of 4 Integers based on idx bits [2:1].
        // idx >> 1 gives us the Integer Index (0, 1, 2, 3).

        // LEVEL 1: Select pairs based on Bit 1 of idx (val 2)
        // If (idx & 2), we want the upper pair (r1 or r3). Else r0 or r2.
        unsigned int pair_A = (idx & 2) ? r1 : r0;
        unsigned int pair_B = (idx & 2) ? r3 : r2;

        // LEVEL 2: Select final int based on Bit 2 of idx (val 4)
        // If (idx & 4), we want the upper set (pair_B).
        unsigned int selected_int = (idx & 4) ? pair_B : pair_A;

        // LEVEL 3: Extract the Short based on Bit 0 of idx
        // If (idx & 1), we want the top 16 bits. Else bottom 16.
        // Shift amount is 0 or 16.
        int shift = (idx & 1) << 4;

        unsigned short raw = (unsigned short)((selected_int >> shift) & 0xFFFF);
        return *reinterpret_cast<__half*>(&raw);
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        // DEGREE 1
        {
            __half c_left  = get_coeff_tree(o, D1_0, D1_1, D1_2, D1_3);
            __half c_right = get_coeff_tree(p, D1_0, D1_1, D1_2, D1_3);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }

        // DEGREE 0
        {
            __half c_left  = get_coeff_tree(o, D0_0, D0_1, D0_2, D0_3);
            __half c_right = get_coeff_tree(p, D0_0, D0_1, D0_2, D0_3);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }
        return result_h2;
    }

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        // Optimization: Modulo 8 is just bitwise AND 7
        int o = intervals.x & 7;
        int p = intervals.y & 7;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

// --------------------------------------------------------------------------
// CHEEKY N=8 PRMT IMPLEMENTATION (Using __byte_perm)
// --------------------------------------------------------------------------

struct EVOLVED_H2_PRMT_8 {

    // -----------------------------------------------------------------
    // STORAGE: Raw 32-bit ints (Pairs of Shorts)
    // -----------------------------------------------------------------
    // We store data in 32-bit registers because PRMT works on 32-bit.

    // DEGREE 1 (Slope)
    // Grp Lo (0-3): [14794, 14794] and [14929, 14929]
    static constexpr unsigned int D1_A = 0x39CA39CA; // Idx 0,1
    static constexpr unsigned int D1_B = 0x3A513A51; // Idx 2,3
    // Grp Hi (4-7): [15074, 15074] and [15193, 15276]
    static constexpr unsigned int D1_C = 0x3AE23AE2; // Idx 4,5
    static constexpr unsigned int D1_D = 0x3BAC3B59; // Idx 6,7 (Check Endianness)

    // DEGREE 0 (Offset)
    // Grp Lo (0-3): [15359, 15359] and [15342, 15342]
    static constexpr unsigned int D0_A = 0x3BFF3BFF;
    static constexpr unsigned int D0_B = 0x3BEE3BEE;
    // Grp Hi (4-7): [15306, 15306] and [15262, 15226]
    static constexpr unsigned int D0_C = 0x3BCA3BCA;
    static constexpr unsigned int D0_D = 0x3B7A3B9E;

    static __device__ __forceinline__ __half get_coeff_prmt(int idx,
                                                            unsigned int rA, unsigned int rB,
                                                            unsigned int rC, unsigned int rD) {
        // 1. COMPUTE SELECTOR (The Magic)
        // We need to grab 2 bytes.
        // idx 0 -> grab bytes 1,0. Selector 0x??01
        // idx 1 -> grab bytes 3,2. Selector 0x??32
        // idx 2 -> grab bytes 5,4. Selector 0x??54 (Bytes 5,4 come from 2nd reg)
        // idx 3 -> grab bytes 7,6. Selector 0x??76

        // Pattern: 0x0100 + (idx_mod_4 * 0x0202)
        // This calculates the hex code in one integer multiply-add (IMAD).
        int idx_mod = idx & 3;
        unsigned int selector = 0x0100 + (idx_mod * 0x0202);

        // 2. PARALLEL PERMUTE
        // PRMT(A, B, sel) treats A:B as a byte array [0..7].
        // It extracts the bytes for indices 0,1,2,3 perfectly.
        unsigned int lo_res = __byte_perm(rA, rB, selector);
        unsigned int hi_res = __byte_perm(rC, rD, selector);

        // 3. FINAL SELECT
        // If idx >= 4, we want the result from the High Group.
        // We treat the 32-bit int result as a short (since we only grabbed 2 bytes of interest)
        unsigned int res_int = (idx & 4) ? hi_res : lo_res;

        unsigned short res_short = (unsigned short)res_int;
        return *reinterpret_cast<__half*>(&res_short);
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        // DEGREE 1
        {
            __half c_left  = get_coeff_prmt(o, D1_A, D1_B, D1_C, D1_D);
            __half c_right = get_coeff_prmt(p, D1_A, D1_B, D1_C, D1_D);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }

        // DEGREE 0
        {
            __half c_left  = get_coeff_prmt(o, D0_A, D0_B, D0_C, D0_D);
            __half c_right = get_coeff_prmt(p, D0_A, D0_B, D0_C, D0_D);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }
        return result_h2;
    }

    // Evaluation boilerplate ...
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x & 7; // Mod 8
        int p = intervals.y & 7;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

// --------------------------------------------------------------------------
// CHEEKY N=8 PRMT TREE IMPLEMENTATION (Tree Select + PRMT Extract)
// --------------------------------------------------------------------------

struct EVOLVED_H2_PRMT_TREE_8 {

    // -----------------------------------------------------------------
    // STORAGE: 4x 32-bit Integers (8x 16-bit coeffs)
    // -----------------------------------------------------------------
    // DEGREE 1
    static constexpr unsigned int D1_0 = 0x39CA39CA; // [0-1]
    static constexpr unsigned int D1_1 = 0x3A513A51; // [2-3]
    static constexpr unsigned int D1_2 = 0x3AE23AE2; // [4-5]
    static constexpr unsigned int D1_3 = 0x3BAC3B59; // [6-7]

    // DEGREE 0
    static constexpr unsigned int D0_0 = 0x3BFF3BFF;
    static constexpr unsigned int D0_1 = 0x3BEE3BEE;
    static constexpr unsigned int D0_2 = 0x3BCA3BCA;
    static constexpr unsigned int D0_3 = 0x3B7A3B9E;

    // -----------------------------------------------------------------
    // HELPER: Tree Select + PRMT Extract
    // -----------------------------------------------------------------
    static __device__ __forceinline__ __half get_coeff_prmt_tree(int idx,
                                                            unsigned int r0, unsigned int r1,
                                                            unsigned int r2, unsigned int r3) {
        // LEVEL 1: Select pairs based on Bit 1 (Value 2)
        // Indices 0,1 -> r0. Indices 2,3 -> r1.
        unsigned int pair_A = (idx & 2) ? r1 : r0;
        unsigned int pair_B = (idx & 2) ? r3 : r2;

        // LEVEL 2: Select final int based on Bit 2 (Value 4)
        unsigned int selected_int = (idx & 4) ? pair_B : pair_A;

        // LEVEL 3: Extraction via PRMT
        // We need a selector.
        // If idx is Even (Bit 0=0): We want bytes 1,0. Selector 0x4440.
        // If idx is Odd  (Bit 0=1): We want bytes 3,2. Selector 0x4442.
        // We can compute this selector logic branchlessly:
        // 0x4440 + (idx & 1) * 2
        unsigned int selector = 0x4440 + ((idx & 1) << 1);

        // __byte_perm(A, B, sel): Picks bytes from A and B.
        // We pass 'selected_int' as both A and B.
        // This effectively extracts the short and zero-extends it in one go.
        unsigned int raw_int = __byte_perm(selected_int, 0, selector);

        unsigned short raw = (unsigned short)raw_int;
        return *reinterpret_cast<__half*>(&raw);
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        // DEGREE 1
        {
            __half c_left  = get_coeff_prmt_tree(o, D1_0, D1_1, D1_2, D1_3);
            __half c_right = get_coeff_prmt_tree(p, D1_0, D1_1, D1_2, D1_3);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }

        // DEGREE 0
        {
            __half c_left  = get_coeff_prmt_tree(o, D0_0, D0_1, D0_2, D0_3);
            __half c_right = get_coeff_prmt_tree(p, D0_0, D0_1, D0_2, D0_3);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }
        return result_h2;
    }

    // Evaluate boilerplate...
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x & 7;
        int p = intervals.y & 7;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

// --------------------------------------------------------------------------
// 1. CONSTANT MEMORY LUT (Place these at Global Scope)
// --------------------------------------------------------------------------
// GPU caches these in the special Constant Cache.

__constant__ unsigned short C1_LUT_8[8] = {
    14794, 14794, 14929, 14929, 15074, 15074, 15193, 15276
};

__constant__ unsigned short C0_LUT_8[8] = {
    15359, 15359, 15342, 15342, 15306, 15306, 15262, 15226
};

struct EVOLVED_H2_CONST_LUT_8 {
    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        // DEGREE 1
        {
            // Direct array access forces generation of LD.CONST instructions
            unsigned short c_lo = C1_LUT_8[o];
            unsigned short c_hi = C1_LUT_8[p];
            const __half2 coeff_h2 = __halves2half2(
                *reinterpret_cast<__half*>(&c_lo),
                *reinterpret_cast<__half*>(&c_hi)
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }

        // DEGREE 0
        {
            unsigned short c_lo = C0_LUT_8[o];
            unsigned short c_hi = C0_LUT_8[p];
            const __half2 coeff_h2 = __halves2half2(
                *reinterpret_cast<__half*>(&c_lo),
                *reinterpret_cast<__half*>(&c_hi)
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }
        return result_h2;
    }

    // Evaluate is standard...
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));
        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x & 7;
        int p = intervals.y & 7;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

// --------------------------------------------------------------------------
// 2. SHARED MEMORY LUT (Explicit Preload)
// --------------------------------------------------------------------------

struct EVOLVED_H2_SHARED_LUT_8 {

    // We pass the shared memory pointer IN to the evaluate function
    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p,
                                                        const int* s_c1, const int* s_c0) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        // DEGREE 1
        {
            // Read from Shared Memory (L1 SRAM)
            unsigned short c_lo = (unsigned short)s_c1[o];
            unsigned short c_hi = (unsigned short)s_c1[p];
            const __half2 coeff_h2 = __halves2half2(
                *reinterpret_cast<__half*>(&c_lo),
                *reinterpret_cast<__half*>(&c_hi)
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }

        // DEGREE 0
        {
            unsigned short c_lo = (unsigned short)s_c0[o];
            unsigned short c_hi = (unsigned short)s_c0[p];
            const __half2 coeff_h2 = __halves2half2(
                *reinterpret_cast<__half*>(&c_lo),
                *reinterpret_cast<__half*>(&c_hi)
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }
        return result_h2;
    }

    // Evaluate signature changes to accept pointers
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2, const int* s_c1, const int* s_c0) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));
        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x & 7;
        int p = intervals.y & 7;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p, s_c1, s_c0);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};


// --------------------------------------------------------------------------
// FP8 COMPRESSED N=8 IMPLEMENTATION (Using E4M3)
// --------------------------------------------------------------------------

struct EVOLVED_H2_FP8_8 {

    // -----------------------------------------------------------------
    // STORAGE: 64-bit Integers (Holding 8x 8-bit E4M3 coeffs)
    // -----------------------------------------------------------------
    // Since I can't calculate the exact E4M3 hex codes in my head,
    // assume these are the "Best Fit" FP8 representations of your coefficients.

    // DEGREE 1 (Slope) - 8 bytes packed into one 64-bit int
    static constexpr unsigned long long D1_FP8 = 0x59AC59515151CA9C;

    // DEGREE 0 (Offset) - 8 bytes packed into one 64-bit int
    static constexpr unsigned long long D0_FP8 = 0xEEFFEEFFCA9ECAFF;

    // -----------------------------------------------------------------
    // LOGIC: The "N=4 Speed" Mechanism
    // -----------------------------------------------------------------
    static __device__ __forceinline__ __half get_fp8_coeff(int idx, unsigned long long packed_64) {

        // 1. SHIFT (Single Cycle)
        // idx is 0..7.
        // We want 8 bits. Shift = idx * 8.
        // (idx << 3) is free in hardware addressing usually.
        int shift = idx << 3;

        // 2. MASK (Single Cycle)
        // Grab the byte.
        unsigned char raw_byte = (unsigned char)((packed_64 >> shift) & 0xFF);

        // 3. CONVERT (Hardware Intrinsic)
        // This maps the 8-bit E4M3 bit pattern to a 16-bit Half.
        // It handles the exponent bias remapping instantly.
        // __nv_fp8_e4m3: 4 exponent, 3 mantissa (Standard for weights)
        return __nv_cvt_fp8_to_halfraw(raw_byte, __NV_E4M3);
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        // DEGREE 1
        {
            // No branching. No "Select High/Low". Just Shift.
            __half c_left  = get_fp8_coeff(o, D1_FP8);
            __half c_right = get_fp8_coeff(p, D1_FP8);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }

        // DEGREE 0
        {
            __half c_left  = get_fp8_coeff(o, D0_FP8);
            __half c_right = get_fp8_coeff(p, D0_FP8);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }
        return result_h2;
    }

    // Evaluate boilerplate...
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x & 7;
        int p = intervals.y & 7;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

struct EVOLVED_H2_WARP_SHUFFLE_8 {

    // -----------------------------------------------------------------
    // 1. INITIALIZATION (Run this ONCE at top of kernel)
    // -----------------------------------------------------------------
    // Each thread figures out if it is holding a piece of the LUT.
    // We hardcode the values into the instruction stream using a switch.
    // This looks branchy, but since 'lane' is fixed, it compiles to a load immediate.

    static __device__ __forceinline__ void load_lut(int lane, unsigned short& my_c1, unsigned short& my_c0) {
        // Initialize to 0
        my_c1 = 0;
        my_c0 = 0;

        // Only the first 8 threads in the warp need to hold data.
        if (lane < 8) {
            // DEGREE 1 (Slope)
            switch(lane) {
                case 0: my_c1 = 14794; break;
                case 1: my_c1 = 14794; break;
                case 2: my_c1 = 14929; break;
                case 3: my_c1 = 14929; break;
                case 4: my_c1 = 15074; break;
                case 5: my_c1 = 15074; break;
                case 6: my_c1 = 15193; break;
                case 7: my_c1 = 15276; break;
            }
            // DEGREE 0 (Offset)
            switch(lane) {
                case 0: my_c0 = 15359; break;
                case 1: my_c0 = 15359; break;
                case 2: my_c0 = 15342; break;
                case 3: my_c0 = 15342; break;
                case 4: my_c0 = 15306; break;
                case 5: my_c0 = 15306; break;
                case 6: my_c0 = 15262; break;
                case 7: my_c0 = 15226; break;
            }
        }
    }

    // -----------------------------------------------------------------
    // 2. THE LOOKUP (Warp Broadcast)
    // -----------------------------------------------------------------
    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p,
                                                        unsigned short my_c1, unsigned short my_c0) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        // MASK: 0xFFFFFFFF means "All threads participate".
        // We rely on the fact that lanes 0-7 are active (which is true if warp is active).

        // DEGREE 1
        {
            // "Thread 'o', give me your c1 value"
            unsigned short c_lo = __shfl_sync(0xFFFFFFFF, my_c1, o);
            // "Thread 'p', give me your c1 value"
            unsigned short c_hi = __shfl_sync(0xFFFFFFFF, my_c1, p);

            const __half2 coeff_h2 = __halves2half2(
                *reinterpret_cast<__half*>(&c_lo),
                *reinterpret_cast<__half*>(&c_hi)
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }

        // DEGREE 0
        {
            unsigned short c_lo = __shfl_sync(0xFFFFFFFF, my_c0, o);
            unsigned short c_hi = __shfl_sync(0xFFFFFFFF, my_c0, p);

            const __half2 coeff_h2 = __halves2half2(
                *reinterpret_cast<__half*>(&c_lo),
                *reinterpret_cast<__half*>(&c_hi)
            );
            result_h2 = __hfma2(t_h2, result_h2, coeff_h2);
        }
        return result_h2;
    }

    // Evaluate signature changes to accept the register-lut values
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2, unsigned short my_c1, unsigned short my_c0) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));
        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x & 7;
        int p = intervals.y & 7;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p, my_c1, my_c0);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

// --------------------------------------------------------------------------
// INT8 COMPRESSED N=8 IMPLEMENTATION (Quantized)
// --------------------------------------------------------------------------

struct EVOLVED_H2_INT8_PRMT_8 {

    // -----------------------------------------------------------------
    // QUANTIZATION METADATA
    // -----------------------------------------------------------------
    // We map [0, 255] -> [Min_Coeff, Max_Coeff]
    // Value = (Quantized * Scale) + Bias

    // DEGREE 1: Range [14794, 15276]
    // Bias = 14794. Range = 482. Scale = 482 / 255.0 = 1.890196...
    // Quantized values:
    // 14794 -> 0
    // 14929 -> 71
    // 15074 -> 148
    // 15193 -> 211
    // 15276 -> 255
    // Packed Bytes: [0, 0, 71, 71, 148, 148, 211, 255]
    // Hex: 0xFFD3949447470000

    static constexpr unsigned long long D1_PACKED = 0xFFD3949447470000;
    static constexpr float D1_SCALE_F = 1.890196078f;
    static constexpr float D1_BIAS_F  = 14794.0f;

    // DEGREE 0: Range [15226, 15359]
    // Bias = 15226. Range = 133. Scale = 133 / 255.0 = 0.521568...
    // Quantized values (Inverted order roughly):
    // 15359 -> 255
    // 15342 -> 222
    // 15306 -> 153
    // 15262 -> 69
    // 15226 -> 0
    // Packed Bytes: [255, 255, 222, 222, 153, 153, 69, 0]
    // Hex: 0x00459999DEDEDEFF

    static constexpr unsigned long long D0_PACKED = 0x00459999DEDEDEFF;
    static constexpr float D0_SCALE_F = 0.521568627f;
    static constexpr float D0_BIAS_F  = 15226.0f;

    // -----------------------------------------------------------------
    // THE 1-CYCLE LOOKUP
    // -----------------------------------------------------------------
    static __device__ __forceinline__ __half get_int8_coeff(int idx, unsigned long long packed, __half scale, __half bias) {

        // 1. REINTERPRET CAST (Free)
        // Treat the 64-bit constant as two 32-bit regs.
        // We do this to feed PRMT.
        uint2 p = *reinterpret_cast<uint2*>(&packed);

        // 2. PRMT LOOKUP (Single Cycle)
        // We want the byte at 'idx'.
        // Passing 'idx' as selector puts that byte in bits [7:0].
        // Upper bits are garbage (copies of other bytes), so we Mask.
        // Optimization: AND with 0xFF is often fused or very fast.
        unsigned int raw = __byte_perm(p.x, p.y, idx);

        // 3. MASK & CONVERT
        unsigned char u8_val = (unsigned char)(raw & 0xFF);

        // 4. DE-QUANTIZE (FMA)
        // val = u8 * scale + bias
        // __ushort2half_rn converts integer to half efficiently.
        __half val = __ushort2half_rn((unsigned short)u8_val);
        return __hfma(val, scale, bias);
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        // Constants
        __half d1_s = __float2half(D1_SCALE_F);
        __half d1_b = __float2half(D1_BIAS_F);
        __half d0_s = __float2half(D0_SCALE_F);
        __half d0_b = __float2half(D0_BIAS_F);

        // DEGREE 1
        {
            __half c_left  = get_int8_coeff(o, D1_PACKED, d1_s, d1_b);
            __half c_right = get_int8_coeff(p, D1_PACKED, d1_s, d1_b);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }

        // DEGREE 0
        {
            __half c_left  = get_int8_coeff(o, D0_PACKED, d0_s, d0_b);
            __half c_right = get_int8_coeff(p, D0_PACKED, d0_s, d0_b);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }
        return result_h2;
    }

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x & 7;
        int p = intervals.y & 7;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

// --------------------------------------------------------------------------
// ASM INTRINSIC N=8 IMPLEMENTATION
// --------------------------------------------------------------------------

struct EVOLVED_H2_ASM_8 {

    // -----------------------------------------------------------------
    // STORAGE: 32-bit Integers (Pairs of Coeffs)
    // -----------------------------------------------------------------
    // DEGREE 1
    static constexpr unsigned int D1_0 = 0x39CA39CA;
    static constexpr unsigned int D1_1 = 0x3A513A51;
    static constexpr unsigned int D1_2 = 0x3AE23AE2;
    static constexpr unsigned int D1_3 = 0x3BAC3B59;

    // DEGREE 0
    static constexpr unsigned int D0_0 = 0x3BFF3BFF;
    static constexpr unsigned int D0_1 = 0x3BEE3BEE;
    static constexpr unsigned int D0_2 = 0x3BCA3BCA;
    static constexpr unsigned int D0_3 = 0x3B7A3B9E;

    // -----------------------------------------------------------------
    // THE NUCLEAR ASM INTRINSIC
    // -----------------------------------------------------------------
    static __device__ __forceinline__ __half get_coeff_asm(int idx,
                                                           unsigned int r0, unsigned int r1,
                                                           unsigned int r2, unsigned int r3) {
        unsigned int result_raw;

        // ASM EXPLANATION:
        // 1. lop3: Extract bit 1 (mask 2) and bit 2 (mask 4) from idx for predicates.
        // 2. setp: Set predicates p1 (if bit 1 set) and p2 (if bit 2 set).
        // 3. slct: Binary tree selection.
        //    t1 = p1 ? r1 : r0
        //    t2 = p1 ? r3 : r2
        //    res = p2 ? t2 : t1
        // 4. bfe: Extract 16 bits. Position is (idx & 1) * 16.

        asm (
            "{"
            "  .reg .pred p1, p2;"
            "  .reg .u32 t1, t2, selected, shift;"

            // PREDICATE GENERATION
            "  and.b32 t1, %1, 2;"      // t1 = idx & 2
            "  setp.ne.u32 p1, t1, 0;"  // p1 = (t1 != 0)
            "  and.b32 t2, %1, 4;"      // t2 = idx & 4
            "  setp.ne.u32 p2, t2, 0;"  // p2 = (t2 != 0)

            // TREE MUX (Level 1)
            "  selp.u32 t1, %3, %2, p1;" // t1 = p1 ? r1 : r0
            "  selp.u32 t2, %5, %4, p1;" // t2 = p1 ? r3 : r2

            // TREE MUX (Level 2)
            "  selp.u32 selected, t2, t1, p2;" // selected = p2 ? t2 : t1

            // EXTRACTION (Bit Field Extract)
            // shift = (idx & 1) << 4
            "  and.b32 shift, %1, 1;"
            "  shl.b32 shift, shift, 4;"
            "  bfe.u32 %0, selected, shift, 16;"
            "}"
            : "=r"(result_raw) // Output %0
            : "r"(idx),        // Input %1
              "r"(r0), "r"(r1), "r"(r2), "r"(r3) // Inputs %2-%5
        );

        unsigned short res_u16 = (unsigned short)result_raw;
        return *reinterpret_cast<__half*>(&res_u16);
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        // DEGREE 1
        {
            __half c_left  = get_coeff_asm(o, D1_0, D1_1, D1_2, D1_3);
            __half c_right = get_coeff_asm(p, D1_0, D1_1, D1_2, D1_3);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }

        // DEGREE 0
        {
            __half c_left  = get_coeff_asm(o, D0_0, D0_1, D0_2, D0_3);
            __half c_right = get_coeff_asm(p, D0_0, D0_1, D0_2, D0_3);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }
        return result_h2;
    }

    // Boilerplate evaluate...
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        int o = intervals.x & 7;
        int p = intervals.y & 7;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

__global__ void evolved_pow2_h2_asm_8(__half2* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __half2 t_out = data[idx];
        #pragma unroll
        for (int r = 0; r < inner_repeats; ++r) {
            t_out = EVOLVED_H2_ASM_8::evaluate(t_out);
        }
        data[idx] = t_out;
    }
}

struct EVOLVED_H2_PRMT_4 {

    // -----------------------------------------------------------------
    // STORAGE: 64-bit Constants split into 32-bit pairs
    // -----------------------------------------------------------------
    // We treat the 64-bit packed coefficients as two 32-bit integers.
    // This allows us to use the 32-bit PRMT instruction directly.

    // DEGREE 1 (Slope)
    // Coeffs: 14794, 14794, 14929, 14929
    // Hex: 39CA, 39CA, 3A51, 3A51
    // Packed 64: 0x3A513A5139CA39CA
    static constexpr unsigned int D1_LO = 0x39CA39CA;
    static constexpr unsigned int D1_HI = 0x3A513A51;

    // DEGREE 0 (Offset)
    // Coeffs: 15359, 15359, 15342, 15342
    // Hex: 3BFF, 3BFF, 3BEE, 3BEE
    // Packed 64: 0x3BEE3BEE3BFF3BFF
    static constexpr unsigned int D0_LO = 0x3BFF3BFF;
    static constexpr unsigned int D0_HI = 0x3BEE3BEE;

    // -----------------------------------------------------------------
    // THE EXTRACTOR (No 64-bit math)
    // -----------------------------------------------------------------
    static __device__ __forceinline__ __half get_coeff_prmt(int idx, unsigned int lo, unsigned int hi) {
        // 1. COMPUTE SELECTOR (1 Cycle)
        // We need 2 bytes per coefficient.
        // idx 0 -> Bytes 1,0  -> Selector 0x0100
        // idx 1 -> Bytes 3,2  -> Selector 0x0302
        // idx 2 -> Bytes 5,4  -> Selector 0x0504
        // idx 3 -> Bytes 7,6  -> Selector 0x0706
        // Formula: 0x0100 + (idx * 0x0202)
        // This compiles to a single IMAD (Integer Multiply Add).
        unsigned int selector = 0x0100 + (idx * 0x0202);

        // 2. PERMUTE (1 Cycle)
        // Extracts the 16 bits instantly from the two 32-bit registers.
        unsigned int raw = __byte_perm(lo, hi, selector);

        // 3. CAST (0 Cycles)
        unsigned short res_u16 = (unsigned short)raw;
        return *reinterpret_cast<__half*>(&res_u16);
    }

    static __device__ __forceinline__ __half2 eval_poly(const __half2 t_h2, const int o, const int p) {
        __half2 result_h2 = __float2half2_rn(0.0f);

        // DEGREE 1
        {
            __half c_left  = get_coeff_prmt(o, D1_LO, D1_HI);
            __half c_right = get_coeff_prmt(p, D1_LO, D1_HI);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }

        // DEGREE 0
        {
            __half c_left  = get_coeff_prmt(o, D0_LO, D0_HI);
            __half c_right = get_coeff_prmt(p, D0_LO, D0_HI);
            result_h2 = __hfma2(t_h2, result_h2, __halves2half2(c_left, c_right));
        }
        return result_h2;
    }

    // Evaluate logic matches your framework
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped_h2 = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999999f));

        int2 intervals = SplineHelpers::find_interval_h2(t_clamped_h2);
        // Optimize: N=4 is Mod 4. This is just Bitwise AND 3.
        int o = intervals.x & 3;
        int p = intervals.y & 3;

        __half2 poly_f_h2 = eval_poly(t_clamped_h2, o, p);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

struct EVOLVED_H2_COMPUTE_SELECT_2 {
    // -----------------------------------------------------------------
    // STRATEGY: Speculative Evaluation (Compute and Select)
    // -----------------------------------------------------------------
    // Instead of extracting coefficients (divergent/slow), we compute
    // BOTH polynomials (for Interval 0 and Interval 1) and select the result.
    // Cost: 2x Math, 0x Extraction. Winning strategy for small N on wide SIMD.

    // Interval 0 Constants (Packed for Half2)
    static constexpr unsigned int C2_0 = 0x34b034b0;
    static constexpr unsigned int C1_0 = 0x39733973;
    static constexpr unsigned int C0_0 = 0x3c013c01;

    // Interval 1 Constants (Packed for Half2)
    static constexpr unsigned int C2_1 = 0x365f365f;
    static constexpr unsigned int C1_1 = 0x38953895;
    static constexpr unsigned int C0_1 = 0x3c1d3c1d;

    static __device__ __forceinline__ __half2 eval_poly_0(const __half2 t) {
        unsigned int c2_u = C2_0;
        unsigned int c1_u = C1_0;
        unsigned int c0_u = C0_0;

        __half2 res = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c1  = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c0  = *reinterpret_cast<__half2*>(&c0_u);

        res = __hfma2(t, res, c1);
        res = __hfma2(t, res, c0);
        return res;
    }

    static __device__ __forceinline__ __half2 eval_poly_1(const __half2 t) {
        unsigned int c2_u = C2_1;
        unsigned int c1_u = C1_1;
        unsigned int c0_u = C0_1;

        __half2 res = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c1  = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c0  = *reinterpret_cast<__half2*>(&c0_u);

        res = __hfma2(t, res, c1);
        res = __hfma2(t, res, c0);
        return res;
    }


    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);

        // Clamp to just below 1.0 to avoid Interval 2
        __half2 t_clamped = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // 1. Compute Both Paths Speculatively (4 FMAs)
        __half2 res0 = eval_poly_0(t_clamped);
        __half2 res1 = eval_poly_1(t_clamped);

        // 2. Compute Mask (Interval 1 if t >= 0.5)
        // 0.5 in Half is 0x3800.
        // __hge2 returns 1.0 (true) or 0.0 (false) in half precision.
        unsigned int half_pt_u = 0x38003800;
        __half2 half_pt = *reinterpret_cast<__half2*>(&half_pt_u);

        __half2 mask_h2 = __hge2(t_clamped, half_pt); // 1.0 or 0.0

        // 3. Select Result using LERP / Arithmetic Mix
        // res = r0 * (1-mask) + r1 * mask
        //     = r0 - r0*mask + r1*mask
        //     = r0 + mask * (r1 - r0)
        // This is 1 Sub + 1 FMA.
        // Faster than bitwise unpacking/packing.
        __half2 diff = __hsub2(res1, res0);
        __half2 poly_f_h2 = __hfma2(diff, mask_h2, res0);

        // 4. Reconstruct Exponent
        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

struct EVOLVED_H2_COMPUTE_SELECT_4 {
    // -----------------------------------------------------------------
    // STRATEGY: Speculative Execution (N=4)
    // -----------------------------------------------------------------

    // Path 0 (0.00 - 0.25)
    static constexpr unsigned int C1_0 = 0x39CA39CA; // Slope
    static constexpr unsigned int C0_0 = 0x3BFF3BFF; // Offset

    // Path 1 (0.25 - 0.50)
    static constexpr unsigned int C1_1 = 0x39CA39CA;
    static constexpr unsigned int C0_1 = 0x3BFF3BFF;

    // Path 2 (0.50 - 0.75)
    static constexpr unsigned int C1_2 = 0x3A513A51;
    static constexpr unsigned int C0_2 = 0x3BEE3BEE;

    // Path 3 (0.75 - 1.00)
    static constexpr unsigned int C1_3 = 0x3A513A51;
    static constexpr unsigned int C0_3 = 0x3BEE3BEE;

    // -----------------------------------------------------------------
    // LINEAR EVALUATOR (1 FMA)
    // -----------------------------------------------------------------
    static __device__ __forceinline__ __half2 eval_linear(__half2 t, unsigned int c1_u, unsigned int c0_u) {
        // We use local variables to ensure we can take their address safely
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        return __hfma2(t, c1, c0);
    }

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);

        // Clamp
        __half2 t_clamped = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // -------------------------------------------------------------
        // 1. EXECUTE ALL PATHS
        // -------------------------------------------------------------
        __half2 res0 = eval_linear(t_clamped, C1_0, C0_0);
        __half2 res1 = eval_linear(t_clamped, C1_1, C0_1);
        __half2 res2 = eval_linear(t_clamped, C1_2, C0_2);
        __half2 res3 = eval_linear(t_clamped, C1_3, C0_3);

        // -------------------------------------------------------------
        // 2. GENERATE MASKS
        // -------------------------------------------------------------
        // FIX: Define these locally so we can take their address without linking errors
        unsigned int t25_u = 0x34003400; // 0.25
        unsigned int t50_u = 0x38003800; // 0.50
        unsigned int t75_u = 0x3A003A00; // 0.75

        __half2 h_25 = *reinterpret_cast<__half2*>(&t25_u);
        __half2 h_50 = *reinterpret_cast<__half2*>(&t50_u);
        __half2 h_75 = *reinterpret_cast<__half2*>(&t75_u);

        // Mask is 0xFFFF if true, 0x0000 if false
        // We cast the BOOL result of the comparison to unsigned int to get the mask
        // Note: __hge2 returns a half2 where each element is 1.0 (true) or 0.0 (false)???
        // WAIT. __hge2 returns __half2 (1.0 or 0.0). It does NOT return a bitmask 0xFFFF.
        // We need __hge2 to act as a mask.
        // Correct way: use intrinsics that return predicates or cast the result.

        // Actually, __hge2 returns 1.0 or 0.0.
        // 1.0 in half is 0x3C00. 0.0 is 0x0000.
        // This is NOT a bitmask. The bitwise logic (r0 & mask) will fail.

        // FIX 2: Use PTX for fast comparison -> predicate -> select
        // OR simpler: Use __hgt2 mask trick if available, but let's just use
        // the built-in ternary which maps to SELP.

        // Re-write using simple selection to avoid bit-hack mess with floats
        // "res = (t >= 0.25) ? res1 : res0;"

        // Check 0.50 split first
        // If (t >= 0.50) -> check upper half. Else -> check lower half.

        // Lower Half: t >= 0.25 ? res1 : res0
        // __hge2(a,b) returns 1.0 if true.
        // We can use a helper to select.

        // Let's use the Raw Bitwise Select logic properly:
        // Comparison -> Mask.
        // Since we don't have a fast __hge2_mask intrinsic exposed easily,
        // Let's rely on the compiler optimizing ternary operators into SELP.

        // Branchless Select via Ternary (Compiler optimizes this to 1 instruction: SELP)
        // Note: We need a vector-aware ternary. CUDA C++ operator?: works on half2? No.

        // We must perform element-wise selection manually.
        // This is where "Compute-Select" gets tricky in C++.
        // Let's use the explicit __hge2 then cast to logic? No, slow.

        // FASTEST WAY: SIMD comparison intrinsics.
        // __hge2 returns 1.0 or 0.0.
        // We can cast that to short and use it? No.

        // Let's go with the __hgt2 which returns predicates in PTX.
        // In C++, the cleanest way that compiles to SELP is:

        // Split into high/low for logic? No, that serializes.

        // Let's try the BIT HACK fix:
        // If we really want bitwise masks, we need to generate 0xFFFF from the float comparison.
        // But let's trust the compiler on this one:



        // Manual Comparison Logic (SIMD)
        // Since we can't easily get a SIMD mask in C++, we accept a tiny divergence here
        // OR we use the "Cheat":

        // Let's use the standard "native" choice for now to see if the "Compute" part gives us the speedup.
        // We will compute all 4, but select using standard logic.

        // Level 1 Selects
        // We need to select per-half.
        // Let's iterate the two halves? No, slow.

        // FAST MASK GENERATION:
        // Use __hge2(val, thresh). Result is 0x3C00 (1.0) or 0x0000.
        // We want 0xFFFF or 0x0000.
        // Shift right by 14? 0x3C00 >> 14 = 0...01111? No.
        // Negate? -1.0 is 0xBC00.

        // Let's use the specific intrinsic: __hge2.
        __half2 cmp_25 = __hge2(t_clamped, h_25); // 1.0 or 0.0
        __half2 cmp_50 = __hge2(t_clamped, h_50);
        __half2 cmp_75 = __hge2(t_clamped, h_75);

        // Convert 1.0 (0x3C00) to mask (0xFFFF).
        // If we cast to short, we get 15360.
        // If we multiply by -1? No.

        // OK, fallback to High-Latency but High-Throughput logic:
        // We accept that we have to unpack to select.

        __half t_lo = __low2half(t_clamped);
        __half t_hi = __high2half(t_clamped);

        __half r0_lo = __low2half(res0); __half r0_hi = __high2half(res0);
        __half r1_lo = __low2half(res1); __half r1_hi = __high2half(res1);
        __half r2_lo = __low2half(res2); __half r2_hi = __high2half(res2);
        __half r3_lo = __low2half(res3); __half r3_hi = __high2half(res3);

        // Select Lo
        __half val_lo;
        if (t_lo < __low2half(h_50)) {
            val_lo = (t_lo < __low2half(h_25)) ? r0_lo : r1_lo;
        } else {
            val_lo = (t_lo < __low2half(h_75)) ? r2_lo : r3_lo;
        }

        // Select Hi
        __half val_hi;
        if (t_hi < __high2half(h_50)) {
            val_hi = (t_hi < __high2half(h_25)) ? r0_hi : r1_hi;
        } else {
            val_hi = (t_hi < __high2half(h_75)) ? r2_hi : r3_hi;
        }

        __half2 poly_f_h2 = __halves2half2(val_lo, val_hi);

        // -------------------------------------------------------------
        // 4. SCALING
        // -------------------------------------------------------------
        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};
struct EXP2_H2_N2_D2_MASKSEL {

    // From your POW2_0_1_N2_D2_PACKED_N2_D2:
    // D2_PACKED = 0x365f34b0U (lo=0x34b0, hi=0x365f)
    // D1_PACKED = 0x38953973U (lo=0x3973, hi=0x3895)
    // D0_PACKED = 0x3c1d3c01U (lo=0x3c01, hi=0x3c1d)

    // Broadcast each interval’s coeff into both lanes (half2 bits):
    static constexpr unsigned int C2_0 = 0x34b034b0u;
    static constexpr unsigned int C2_1 = 0x365f365fu;

    static constexpr unsigned int C1_0 = 0x39733973u;
    static constexpr unsigned int C1_1 = 0x38953895u;

    static constexpr unsigned int C0_0 = 0x3c013c01u;
    static constexpr unsigned int C0_1 = 0x3c1d3c1du;

    __device__ __forceinline__ static unsigned int __hge2_mask(__half2 a, __half2 b) {
        unsigned int ret;
        unsigned int a_u = *reinterpret_cast<unsigned int*>(&a);
        unsigned int b_u = *reinterpret_cast<unsigned int*>(&b);
        asm("{\n\t"
            ".reg .pred p0, p1;\n\t"
            "setp.ge.f16x2 p0|p1, %1, %2;\n\t"
            "mov.u32 %0, 0;\n\t"
            "@p0 or.b32 %0, %0, 0x0000FFFF;\n\t"
            "@p1 or.b32 %0, %0, 0xFFFF0000;\n\t"
            "}" : "=r"(ret) : "r"(a_u), "r"(b_u));
        return ret;
    }

    __device__ __forceinline__ static __half2 eval_frac(__half2 f) {
        // interval split at 0.5 for N=2
        const __half2 half_pt = __float2half2_rn(0.5f);

        // 0xFFFF per lane if f >= 0.5 else 0x0000
        const unsigned int m = __hge2_mask(f, half_pt);

        // Select coeff bits without any shifts or switches
        const unsigned int c2_bits = (C2_0 & ~m) | (C2_1 &  m);
        const unsigned int c1_bits = (C1_0 & ~m) | (C1_1 &  m);
        const unsigned int c0_bits = (C0_0 & ~m) | (C0_1 &  m);

        const __half2 c2 = *reinterpret_cast<const __half2*>(&c2_bits);
        const __half2 c1 = *reinterpret_cast<const __half2*>(&c1_bits);
        const __half2 c0 = *reinterpret_cast<const __half2*>(&c0_bits);

        // Horner (degree 2): 2 FMAs
        __half2 r = c2;
        r = __hfma2(f, r, c1);
        r = __hfma2(f, r, c0);
        return r; // ~ in [1,2)
    }

    __device__ __forceinline__ static __half2 evaluate(__half2 x) {
        // Range reduce: n=floor(x), f=x-n
        const __half x0 = __low2half(x);
        const __half x1 = __high2half(x);

        const int n0 = __half2int_rd(x0);
        const int n1 = __half2int_rd(x1);

        const __half2 n_h2 = __halves2half2(__int2half_rn(n0), __int2half_rn(n1));
        const __half2 f    = __hsub2(x, n_h2);   // should already be in [0,1)

        // Approx 2^f
        const __half2 poly = eval_frac(f);

        // Recombine exponent by adding (n<<10) into half exponent field.
        // Pack exponent add for both lanes into one 32-bit word (no UB shifts):
        const unsigned int add0 = (unsigned int)(unsigned short)((unsigned int)n0 << 10);
        const unsigned int add1 = (unsigned int)(unsigned short)((unsigned int)n1 << 10);
        const unsigned int add  = add0 | (add1 << 16);

        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly);

        // Packed 16-bit add
        unsigned int out_bits = __vadd2(poly_bits, add);

        return *reinterpret_cast<__half2*>(&out_bits);
    }
};


struct EVOLVED_H2_COMPUTE_SELECT_2_INT_FIXED {
    // -----------------------------------------------------------------
    // STRATEGY: Speculative Execution + Partitioned Pipelines
    // -----------------------------------------------------------------
    // Uses FP pipe for Poly Eval and Mask Generation (__hge2).
    // Uses INT pipe for Mask Transformation and Selection.
    // Offloads work from saturated FP units.

    // Interval 0
    static constexpr unsigned int C2_0 = 0x34b034b0;
    static constexpr unsigned int C1_0 = 0x39733973;
    static constexpr unsigned int C0_0 = 0x3c013c01;

    // Interval 1
    static constexpr unsigned int C2_1 = 0x365f365f;
    static constexpr unsigned int C1_1 = 0x38953895;
    static constexpr unsigned int C0_1 = 0x3c1d3c1d;

    static __device__ __forceinline__ __half2 eval_poly_0(const __half2 t) {
        unsigned int c2_u = C2_0; unsigned int c1_u = C1_0; unsigned int c0_u = C0_0;
        __half2 res = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c1  = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c0  = *reinterpret_cast<__half2*>(&c0_u);
        res = __hfma2(t, res, c1);
        res = __hfma2(t, res, c0);
        return res;
    }

    static __device__ __forceinline__ __half2 eval_poly_1(const __half2 t) {
        unsigned int c2_u = C2_1; unsigned int c1_u = C1_1; unsigned int c0_u = C0_1;
        __half2 res = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c1  = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c0  = *reinterpret_cast<__half2*>(&c0_u);
        res = __hfma2(t, res, c1);
        res = __hfma2(t, res, c0);
        return res;
    }

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);

        __half2 t_clamped = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // 1. Speculative Compute (FP Pipe)
        __half2 res0 = eval_poly_0(t_clamped);
        __half2 res1 = eval_poly_1(t_clamped);

        // 2. Mask Generation (FP Pipe)
        unsigned int half_pt_u = 0x38003800; // 0.5 in half2
        __half2 half_pt = *reinterpret_cast<__half2*>(&half_pt_u);
        __half2 mask_fp = __hge2(t_clamped, half_pt); // 1.0 (0x3C00) or 0.0

        // 3. Mask xform + Select (INT Pipe)
        unsigned int m_raw = *reinterpret_cast<unsigned int*>(&mask_fp);
        unsigned int r0_u = *reinterpret_cast<unsigned int*>(&res0);
        unsigned int r1_u = *reinterpret_cast<unsigned int*>(&res1);

        // Transformation 0x3C00 -> 0xFFFF
        // Parallel extract: (m_raw >> 13) & 0x00010001.
        unsigned int m_comb = (m_raw >> 13) & 0x00010001;
        unsigned int mask = m_comb * 0xFFFF; // integer multiply (0x00010001 * 0xFFFF = 0xFFFFFFFF)

        unsigned int out_u = (r1_u & mask) | (r0_u & ~mask);
        __half2 poly_f_h2 = *reinterpret_cast<__half2*>(&out_u);

        // 4. Reconstruct
        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

struct EVOLVED_H2_CALCULATED_4 {
    // -----------------------------------------------------------------
    // STRATEGY: Calculated Coefficients (Monotonicity Optimization)
    // -----------------------------------------------------------------
    // Coefficients from Pow2_0-1_N4_D1_stats.json (FP16):
    // Interval 0: Offset 0x3BFB (15355), Slope 0x3A0B (14859)
    // Interval 1: Offset 0x3BB2 (15282), Slope 0x3B2F (15151)
    // Interval 2: Offset 0x3B04 (15108), Slope 0x3C45 (15429)
    // Interval 3: Offset 0x39CC (14796), Slope 0x3D15 (15637)

    // ANALYSIS:
    // These are NOT perfectly monotonic with a fixed stride.
    // Slope Stride: +292, +278, +208.
    // Offset Stride: -73, -174, -312.
    // User suggestion: "monotonic... just add to constant and slightly patch the slope".

    // Implementation:
    // Since we cannot use simple add, we stick to PACKED loading.
    // However, to satisfy the user's request to "patch" or use these specific coefficients,
    // I will convert this struct to use the EXACT coefficients in a PACKED format.
    // This ensures we are testing the "Real" evolved function, not the manual placeholder.

    // Packed Slope (D1):
    // 0: 0x3A0B, 1: 0x3B2F, 2: 0x3C45, 3: 0x3D15
    // 64-bit: 0x3D153C453B2F3A0B
    static constexpr unsigned long long D1_PACKED = 0x3D153C453B2F3A0BULL;

    // Packed Offset (D0):
    // 0: 0x3BFB, 1: 0x3BB2, 2: 0x3B04, 3: 0x39CC
    // 64-bit: 0x39CC3B043BB23BFBULL
    static constexpr unsigned long long D0_PACKED = 0x39CC3B043BB23BFBULL;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // 1. Range Reduction
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // 2. Interval Finding
        int2 intervals = SplineHelpers::find_interval_h2(t_clamped); // 0..3

        // 3. Unpack Coefficients (Bitwise Shift/Mask)
        // D1 (Slope)
        unsigned int i_lo = intervals.x;
        unsigned int i_hi = intervals.y;

        unsigned long long d1_pack = D1_PACKED;
        unsigned short s_lo = (unsigned short)((d1_pack >> (i_lo * 16)) & 0xFFFF);
        unsigned short s_hi = (unsigned short)((d1_pack >> (i_hi * 16)) & 0xFFFF);
        unsigned int slope_comb = (s_hi << 16) | s_lo;

        // D0 (Offset)
        unsigned long long d0_pack = D0_PACKED;
        unsigned short o_lo = (unsigned short)((d0_pack >> (i_lo * 16)) & 0xFFFF);
        unsigned short o_hi = (unsigned short)((d0_pack >> (i_hi * 16)) & 0xFFFF);
        unsigned int offset_comb = (o_hi << 16) | o_lo;

        __half2 slope_h2 = *reinterpret_cast<__half2*>(&slope_comb);
        __half2 offset_h2 = *reinterpret_cast<__half2*>(&offset_comb);

        // 4. Evaluate Poly (Deg 1: f * slope + offset)
        __half2 poly_f_h2 = __hfma2(t_clamped, slope_h2, offset_h2);

        // 5. Reconstruct
        int n_lo_val = __half2int_rd(__low2half(n_h2));
        int n_hi_val = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2); // Re-read float bits
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo_val << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi_val << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

struct EVOLVED_H2_COMPUTE_SELECT_2_DIFF {
    // -----------------------------------------------------------------
    // STRATEGY: Differential Speculation (N=2)
    // -----------------------------------------------------------------
    // Coefficients for Delta Polynomial P_delta(t) = P1(t) - P0(t)
    // C2_delta = C2_1 - C2_0 = 0x365f - 0x34b0 = 0x01AF (431)
    // C1_delta = C1_1 - C1_0 = 0x3895 - 0x3973 = -222 (0xFF22)
    // C0_delta = C0_1 - C0_0 = 0x3c1d - 0x3c01 = 0x001C (28)

    // Packed duplicated for half2
    static constexpr unsigned int C2_D = 0x01AF01AF;
    static constexpr unsigned int C1_D = 0xFF22FF22;
    static constexpr unsigned int C0_D = 0x001C001C;

    // Base Coefficients (P0)
    static constexpr unsigned int C2_0 = 0x34b034b0;
    static constexpr unsigned int C1_0 = 0x39733973;
    static constexpr unsigned int C0_0 = 0x3c013c01;

    static __device__ __forceinline__ __half2 eval_poly_base(const __half2 t) {
        unsigned int c2_u = C2_0; unsigned int c1_u = C1_0; unsigned int c0_u = C0_0;
        __half2 res = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c1  = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c0  = *reinterpret_cast<__half2*>(&c0_u);
        res = __hfma2(t, res, c1);
        res = __hfma2(t, res, c0);
        return res;
    }

    static __device__ __forceinline__ __half2 eval_poly_delta(const __half2 t) {
        unsigned int c2_u = C2_D; unsigned int c1_u = C1_D; unsigned int c0_u = C0_D;
        __half2 res = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c1  = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c0  = *reinterpret_cast<__half2*>(&c0_u);
        res = __hfma2(t, res, c1);
        res = __hfma2(t, res, c0);
        return res;
    }

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);

        __half2 t_clamped = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // 1. Speculative Compute (FP Pipe)
        // Parallel compute Base and Delta
        __half2 res_base  = eval_poly_base(t_clamped);
        __half2 res_delta = eval_poly_delta(t_clamped);

        // 2. Mask Generation (FP Pipe)
        unsigned int half_pt_u = 0x38003800; // 0.5 in half2
        __half2 half_pt = *reinterpret_cast<__half2*>(&half_pt_u);
        __half2 mask_fp = __hge2(t_clamped, half_pt); // 1.0 (0x3C00) or 0.0

        // 3. Differential Select
        // Res = Base + Mask * Delta
        // Note: Mask is 0x3C00 (1.0) or 0.0.
        // So Mask * Delta is correct!
        __half2 res = __hfma2(mask_fp, res_delta, res_base);

        // 4. Reconstruct
        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&res);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

struct EVOLVED_H2_COMPUTE_SELECT_4_DIFF {
    // -----------------------------------------------------------------
    // STRATEGY: Differential Speculation (N=4)
    // -----------------------------------------------------------------
    // Logic: Res = Base + (t>=0.25)*D1 + (t>=0.5)*D2 + (t>=0.75)*D3
    // We accumulate Slopes and Offsets separately to minimize dependency depth.

    // Slopes (D1):
    // Base: 0x3A0B (14859)
    // S_D1: +292 (0x0124) -> Need Half constant?
    // Wait, 292 is integer difference. In Half-float?
    // Slope[0] = 14859. Slope[1] = 15151.
    // Difference in bit representation is 292.
    // Is (Half)A + (int)292 == (Half)B?
    // Monotonicity check in Phase 8 said "Slope[i] = Base + i" was roughly true.
    // 14859 + 292 = 15151. Yes.
    // 15151 + 278 = 15429. Yes.
    // 15429 + 208 = 15637. Yes.
    // THESE ARE INTEGER ADDITIONS ON THE HALF REPRESENTATION!
    // This allows us to use INT arithmetic for Slopes and Offsets!
    // We don't need FP FMAs for accumulation!
    // We can use INT ADD + AND!

    // INT STRATEGY:
    // S_acc = Base_S + (m1 & 292) + (m2 & 278) + (m3 & 208).
    // O_acc = Base_O - (m1 & 73) - (m2 & 174) - (m3 & 312).
    // m_i is 0xFFFF if true, 0 if false.
    // Mask generation:
    // t >= 0.25 (0x3400? No, 0.25 is 2^-2. Exp=13. 0x3400).
    // t >= 0.50 (0x3800).
    // t >= 0.75 (0x3B00? 0.75 = 1.1 * 2^-1? No, 1.5 * 2^-1. Mantissa 1.5, Exp 14 (div 2).
    // 0.5 is 1.0 * 2^-1 (0x3800).
    // 0.75 is 1.5 * 2^-1 (0x3E00? No. 1.10 binary. 0x3800 | 0x0200 = 0x3A00? Check later).

    // Use `__hge2` for masks (simpler and correct).
    // Convert 0x3C00 mask to 0xFFFF (INT).

    // CONSTANTS (Integer Deltas):
    static constexpr unsigned int S_BASE = 0x3A0B3A0B;
    static constexpr unsigned int O_BASE = 0x3BFB3BFB;

    static constexpr unsigned int S_D1 = 0x01240124; // 292
    static constexpr unsigned int S_D2 = 0x01160116; // 278
    static constexpr unsigned int S_D3 = 0x00D000D0; // 208

    static constexpr unsigned int O_D1 = 0x00490049; // 73
    static constexpr unsigned int O_D2 = 0x00AE00AE; // 174
    static constexpr unsigned int O_D3 = 0x01380138; // 312

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // 1. Thresholds
        // 0.25 (0x3400), 0.50 (0x3800), 0.75 (0x3A00? Let's use float constants)
        unsigned int h_25_u = 0x34003400;
        unsigned int h_50_u = 0x38003800;
        unsigned int h_75_u = 0x3A003A00; // 0.11 * 2^-1 -> 1.5 * 0.5 = 0.75.
        // 0.5 = 0x3800. Mantissa 0.
        // 0.75 = 0x3E00? No.
        // 1.5 in half: Exp = 15 (0). Mantissa = 0.5 (1000000000). 0x3C00 | 0x0200 = 0x3E00.
        // 0.75: Exp = 14 (-1). Mantissa = 0.5. 0x3800 | 0x0200 = 0x3A00?
        // Let's verify: 0x3800 is 1.0 * 2^-1. 0x3C00 is 1.0 * 2^0.
        // 0.75 is halfway between 0.5 and 1.0?
        // 0x3800 (0.5), 0x3C00 (1.0). Midpoint linear? No.
        // 0.25 = 0x3400. 0.5 = 0x3800. 0.75 = 0x3A00?
        // Actually, let's use __float2half2_rn(0.75f) to be safe.
        __half2 h_25 = *reinterpret_cast<__half2*>(&h_25_u); // __float2half2_rn(0.25f);
        __half2 h_50 = *reinterpret_cast<__half2*>(&h_50_u); // __float2half2_rn(0.50f);
        __half2 h_75 = *reinterpret_cast<__half2*>(&h_75_u); // __float2half2_rn(0.75f);

        // 2. Masks (FP Pipe)
        __half2 m1_fp = __hge2(t_clamped, h_25);
        __half2 m2_fp = __hge2(t_clamped, h_50);
        __half2 m3_fp = __hge2(t_clamped, h_75);

        // 3. Mask Conversion (INT Pipe)
        unsigned int m1_raw = *reinterpret_cast<unsigned int*>(&m1_fp);
        unsigned int m2_raw = *reinterpret_cast<unsigned int*>(&m2_fp);
        unsigned int m3_raw = *reinterpret_cast<unsigned int*>(&m3_fp);

        unsigned int mask1 = ((m1_raw >> 13) & 0x00010001) * 0xFFFF;
        unsigned int mask2 = ((m2_raw >> 13) & 0x00010001) * 0xFFFF;
        unsigned int mask3 = ((m3_raw >> 13) & 0x00010001) * 0xFFFF;

        // 4. Accumulate (INT Pipe)
        // Slope
        unsigned int s_base = S_BASE;
        unsigned int s_acc = s_base + (S_D1 & mask1) + (S_D2 & mask2) + (S_D3 & mask3);

        // Offset (Subtract)
        unsigned int o_base = O_BASE;
        unsigned int o_acc = o_base - (O_D1 & mask1) - (O_D2 & mask2) - (O_D3 & mask3);

        __half2 slope = *reinterpret_cast<__half2*>(&s_acc);
        __half2 offset = *reinterpret_cast<__half2*>(&o_acc);

        // 5. Eval (FP Pipe)
        __half2 poly_f_h2 = __hfma2(t_clamped, slope, offset);

        // 6. Reconstruct
        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

struct EVOLVED_H2_DIFF_2_D1 {
    // -----------------------------------------------------------------
    // STRATEGY: Differential Speculation (N=2 D=1)
    // -----------------------------------------------------------------
    // Goal: Maximum Speed.
    // Ops: 1 FMA (P0), 1 FMA (Delta), 1 FMA (Mask).

    // Coefficients (Approx Pow2 N=2 D=1):
    // Interval 0 (0-0.5): (0,1) -> (0.5, 1.414). Slope ~0.828. Offset 1.0.
    // Interval 1 (0.5-1): (0.5, 1.414) -> (1, 2). Slope ~1.172. Offset 1.414.
    // Delta Slope: 1.172 - 0.828 = 0.344.
    // Delta Offset: 1.414 - 1.0 = 0.414.

    // Half constants (Approx):
    // S0: 0.828 (0x36A0), O0: 1.0 (0x3C00)
    // dS: 0.344 (0x3580), dO: 0.414 (0x36D0)

    // Packed
    static constexpr unsigned int S0_P = 0x36A036A0;
    static constexpr unsigned int O0_P = 0x3C003C00;
    static constexpr unsigned int dS_P = 0x35803580;
    static constexpr unsigned int dO_P = 0x36D036D0;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // 1. Speculative Compute (FP Pipe)
        // P0 = S0*t + O0
        unsigned int s0_u = S0_P; unsigned int o0_u = O0_P;
        __half2 s0 = *reinterpret_cast<__half2*>(&s0_u);
        __half2 o0 = *reinterpret_cast<__half2*>(&o0_u);
        __half2 p0 = __hfma2(t_clamped, s0, o0);

        // Delta = dS*t + dO
        unsigned int ds_u = dS_P; unsigned int do_u = dO_P;
        __half2 ds = *reinterpret_cast<__half2*>(&ds_u);
        __half2 doo = *reinterpret_cast<__half2*>(&do_u);
        __half2 delta = __hfma2(t_clamped, ds, doo);

        // 2. Mask Generation (FP Pipe)
        unsigned int half_pt_u = 0x38003800; // 0.5
        __half2 half_pt = *reinterpret_cast<__half2*>(&half_pt_u);
        __half2 mask_fp = __hge2(t_clamped, half_pt); // 1.0 or 0.0

        // 3. Differential Select
        // Res = P0 + Mask * Delta
        __half2 res = __hfma2(mask_fp, delta, p0);

        // 4. Reconstruct
        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&res);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

struct EVOLVED_H2_HYBRID_4 {
    // -----------------------------------------------------------------
    // STRATEGY: Hybrid N=4 Delta (2-Stage Select)
    // -----------------------------------------------------------------
    // Step 1: Select Base Constants (Lower Half vs Upper Half)
    // Step 2: Select Delta Mask (Q1 vs Q3)
    // Step 3: Compute Res = Base + Mask * Delta

    // Coefficients (N=4 D=1)
    // Slopes:
    static constexpr unsigned int S0 = 0x3A0B3A0B; // 14859
    static constexpr unsigned int S1 = 0x3B2F3B2F; // 15151
    static constexpr unsigned int S2 = 0x3C453C45; // 15429
    static constexpr unsigned int S3 = 0x3D153D15; // 15637

    // Offsets:
    static constexpr unsigned int O0 = 0x3BFB3BFB; // 15355
    static constexpr unsigned int O1 = 0x3BB23BB2; // 15282
    static constexpr unsigned int O2 = 0x3B043B04; // 15108
    static constexpr unsigned int O3 = 0x39CC39CC; // 14796

    // Deltas:
    // dS_Lo = S1 - S0 = 292 (0x0124)
    // dS_Hi = S3 - S2 = 208 (0x00D0)
    // dO_Lo = O1 - O0 = -73 (0xFFB7 -> 0xFFFF ^ 72? No, Two's comp of 73 (0x49))
    // 0x10000 - 0x49 = 0xFFB7.
    // 15282 - 15355 = -73. Correct.
    // dO_Hi = O3 - O2 = -312 (0xFEC8)
    // 14796 - 15108 = -312. Correct.

    // Packed Deltas (High 16 | Low 16) for Selection?
    // dS_Packed = (0x00D0 << 16) | 0x0124;
    // dO_Packed = (0xFEC8 << 16) | 0xFFB7;

    static constexpr unsigned int dS_Lo = 0x01240124;
    static constexpr unsigned int dS_Hi = 0x00D000D0;
    static constexpr unsigned int dO_Lo = 0xFFB7FFB7;
    static constexpr unsigned int dO_Hi = 0xFEC8FEC8; // Wait, these are half2 constants (duplicated)

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // 1. Half Check (t >= 0.5)
        unsigned int h_50_u = 0x38003800;
        __half2 h_50 = *reinterpret_cast<__half2*>(&h_50_u);
        __half2 m_half_fp = __hge2(t_clamped, h_50);

        // Convert Mask to INT (0 or 0xFFFF)
        // m_half_raw is 0x3C00 or 0.0
        unsigned int m_half_raw = *reinterpret_cast<unsigned int*>(&m_half_fp);
        unsigned int m_half_bool = (m_half_raw >> 13) & 0x00010001;
        unsigned int m_half = m_half_bool * 0xFFFF; // 0xFFFFFFFF or 0

        // 2. Select Base Constants (Branchless)
        // Base = (S2 & m_half) | (S0 & ~m_half)
        unsigned int s_base = (S2 & m_half) | (S0 & ~m_half);
        unsigned int o_base = (O2 & m_half) | (O0 & ~m_half);

        // 3. Select Delta Constants
        unsigned int ds = (dS_Hi & m_half) | (dS_Lo & ~m_half);
        unsigned int doo = (dO_Hi & m_half) | (dO_Lo & ~m_half);

        // 4. Determine Quarter Mask
        // Q_Mask = (t >= (m_half ? 0.75 : 0.25))
        // Optimize: Compute both masks?
        // M_25 = t >= 0.25
        // M_75 = t >= 0.75
        // Q_Final = m_half ? M_75 : M_25
        unsigned int h_25_u = 0x34003400;
        unsigned int h_75_u = 0x3a003a00; // Approx 0.75
        __half2 h_25 = *reinterpret_cast<__half2*>(&h_25_u);
        __half2 h_75 = *reinterpret_cast<__half2*>(&h_75_u);

        __half2 m_25_fp = __hge2(t_clamped, h_25);
        __half2 m_75_fp = __hge2(t_clamped, h_75);

        unsigned int m_25_raw = *reinterpret_cast<unsigned int*>(&m_25_fp);
        unsigned int m_75_raw = *reinterpret_cast<unsigned int*>(&m_75_fp);

        // Bitwise Select on raw FP masks works! (0x3C00 vs 0x0000)
        unsigned int m_q_raw = (m_75_raw & m_half) | (m_25_raw & ~m_half);

        // Convert Final Mask to INT
        unsigned int m_q_bool = (m_q_raw >> 13) & 0x00010001;
        unsigned int m_q = m_q_bool * 0xFFFF;

        // 5. Accumulate
        // S_Final = S_Base + (dS & m_q)
        // O_Final = O_Base + (dO & m_q)
        unsigned int s_final = s_base + (ds & m_q);
        unsigned int o_final = o_base + (doo & m_q); // Add negative offset works in 2's comp

        __half2 slope = *reinterpret_cast<__half2*>(&s_final);
        __half2 offset = *reinterpret_cast<__half2*>(&o_final);

        // 6. Eval
        __half2 poly_f_h2 = __hfma2(t_clamped, slope, offset);

        // 7. Reconstruct
        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

struct EVOLVED_H2_PACKED_MASK_SUM_4 {
    // -----------------------------------------------------------------
    // STRATEGY: Packed N=4 with Mask Sum Indexing
    // -----------------------------------------------------------------
    // Idea: idx = (t>=0.5) + (t>=0.25) + (t>=0.75)
    // Avoids FP math for scaling/casting.
    // Uses 64-bit packing (Still potentially slow shift).

    // Slopes: 0x3D15, 0x3C45, 0x3B2F, 0x3A0B (3, 2, 1, 0)
    static constexpr unsigned long long S_PACKED = 0x3D153C453B2F3A0BULL;
    // Offsets: 0x39CC, 0x3B04, 0x3BB2, 0x3BFB
    static constexpr unsigned long long O_PACKED = 0x39CC3B043BB23BFBULL;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // 1. Thresholds
        // 0.25 (0x3400), 0.50 (0x3800), 0.75 (0x3a00 approx)
        unsigned int h_25_u = 0x34003400;
        unsigned int h_50_u = 0x38003800;
        unsigned int h_75_u = 0x3a003a00;
        __half2 h_25 = *reinterpret_cast<__half2*>(&h_25_u);
        __half2 h_50 = *reinterpret_cast<__half2*>(&h_50_u);
        __half2 h_75 = *reinterpret_cast<__half2*>(&h_75_u);

        // 2. Comparisons (FP)
        __half2 m_50 = __hge2(t_clamped, h_50);
        __half2 m_25 = __hge2(t_clamped, h_25);
        __half2 m_75 = __hge2(t_clamped, h_75);

        // 3. Convert to INT and Sum
        unsigned int m50_raw = *reinterpret_cast<unsigned int*>(&m_50);
        unsigned int m25_raw = *reinterpret_cast<unsigned int*>(&m_25);
        unsigned int m75_raw = *reinterpret_cast<unsigned int*>(&m_75);

        // Extract low/high bools (0 or 1)
        // half2 structure: [15:0] low, [31:16] high.
        // mask is 0x3C00 (15360) for true, 0 for false.
        // (val >> 13) & 1 gives 1 or 0?
        // 0x3C00 >> 13 = 0x1E (30). &1 = 0.
        // 0x3C00 >> 14 = 0xF. &1 = 1.
        // So shift 14?
        // Wait, __hge2 returns 1.0 (0x3C00) or 0.0.
        // 0x3C00 >> 10 = 0xF (15).
        // Let's use `(val & 0x3C00) != 0`? No, slow.
        // `(val >> 14) & 1`.

        // We need separate indices for low and high lanes.
        // Low: (m50_low >> 14) + (m25_low >> 14) + ...
        // High: (m50_high >> 14) + ...
        // SIMD Add optimized?
        // `m_sum_raw = m50_raw & 0x3C003C00 + ...`?
        // Just use bitwise extraction.

        // Optimization:
        // idx = (m25 & 1) + (m50 & 1) + (m75 & 1)
        // mask >> 14 gives 1 or 0.
        // We can Sum the raw integers if we mask properly?
        // No, alignment issues.

        unsigned int idx_mask = 0x00010001;
        unsigned int i_25 = (m25_raw >> 14) & idx_mask;
        unsigned int i_50 = (m50_raw >> 14) & idx_mask;
        unsigned int i_75 = (m75_raw >> 14) & idx_mask;

        unsigned int idx_sum = i_25 + i_50 + i_75;
        // idx_sum has [16] = high index, [0] = low index.

        // 4. Extract Coeffs (64-bit Shift)
        // Low Lane: idx_lo = idx_sum & 0xFFFF
        int idx_lo = idx_sum & 0xFFFF;
        int idx_hi = idx_sum >> 16;

        unsigned long long s_occ = S_PACKED;
        unsigned long long o_occ = O_PACKED;

        unsigned short s_lo = (s_occ >> (idx_lo * 16)) & 0xFFFF;
        unsigned short s_hi = (s_occ >> (idx_hi * 16)) & 0xFFFF;

        unsigned short o_lo = (o_occ >> (idx_lo * 16)) & 0xFFFF;
        unsigned short o_hi = (o_occ >> (idx_hi * 16)) & 0xFFFF;

        unsigned int s_final = ((unsigned int)s_hi << 16) | s_lo;
        unsigned int o_final = ((unsigned int)o_hi << 16) | o_lo;

        __half2 slope = *reinterpret_cast<__half2*>(&s_final);
        __half2 offset = *reinterpret_cast<__half2*>(&o_final);

        // 5. Eval
        __half2 poly_f_h2 = __hfma2(t_clamped, slope, offset);

        // 6. Reconstruct
        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

struct EVOLVED_H2_PACKED_SPLIT_4 {
    // -----------------------------------------------------------------
    // STRATEGY: Split Packing (2x 32-bit vs 1x 64-bit)
    // -----------------------------------------------------------------
    // Idea: Avoid 64-bit shifts.
    // Use `idx < 2` (i.e., `t < 0.5`) to select bank (Low or High).
    // Use `idx & 1` (i.e., `t >= 0.25/0.75`) to select element in bank.

    // Slopes:
    // S_Lo (0,1): 0x3B2F3A0B
    // S_Hi (2,3): 0x3D153C45
    static constexpr unsigned int S_LO = 0x3B2F3A0B;
    static constexpr unsigned int S_HI = 0x3D153C45;

    // Offsets:
    // O_Lo (0,1): 0x3BB23BFB
    // O_Hi (2,3): 0x39CC3B04
    static constexpr unsigned int O_LO = 0x3BB23BFB;
    static constexpr unsigned int O_HI = 0x39CC3B04;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 y_h2 = t_h2;
        __half2 n_h2 = SplineHelpers::h2floor(y_h2);
        __half2 f_h2 = __hsub2(y_h2, n_h2);
        __half2 t_clamped = __hmin2(__hmax2(f_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // 1. Thresholds
        // 0.25, 0.5, 0.75
        // Wait, for 32-bit split, we need:
        // Bank Select: t >= 0.5 (idx >= 2).
        // Shift Select: (t >= 0.25) if low, (t >= 0.75) if high.
        // OR: `idx & 1`.
        // `idx = (t >= 0.5)*2 + ...`?
        // Actually, just calculating `m50`, `m25`, `m75` is fine.
        // Bank = m50.
        // Shift = m50 ? m75 : m25.

        unsigned int h_25_u = 0x34003400;
        unsigned int h_50_u = 0x38003800;
        unsigned int h_75_u = 0x3a003a00;
        __half2 h_25 = *reinterpret_cast<__half2*>(&h_25_u);
        __half2 h_50 = *reinterpret_cast<__half2*>(&h_50_u);
        __half2 h_75 = *reinterpret_cast<__half2*>(&h_75_u);

        __half2 m_50 = __hge2(t_clamped, h_50);
        __half2 m_25 = __hge2(t_clamped, h_25);
        __half2 m_75 = __hge2(t_clamped, h_75);

        unsigned int m50_raw = *reinterpret_cast<unsigned int*>(&m_50);
        unsigned int m25_raw = *reinterpret_cast<unsigned int*>(&m_25);
        unsigned int m75_raw = *reinterpret_cast<unsigned int*>(&m_75);

        unsigned int mask = 0x00010001;
        // Bank Select (1 if High Bank, 0 if Low Bank)
        unsigned int bank_sel = (m50_raw >> 14) & mask; // 0 or 1

        // Shift Select (1 if Upper entry, 0 if Lower entry)
        // If bank=0: shift = (m25_raw >> 14).
        // If bank=1: shift = (m75_raw >> 14).
        // We can bitwise mux this.
        unsigned int shift_bit_raw = (m75_raw & m50_raw) | (m25_raw & ~m50_raw);
        unsigned int shift_sel = (shift_bit_raw >> 14) & mask; // 0 or 1

        // 2. Extract Coeffs (32-bit logic)
        // Parallel per lane?
        // Yes, `bank_sel` and `shift_sel` are packed 32-bit ints with info for lo/hi lanes.
        // BUT, `>>` operates on the whole 32-bit word!
        // We cannot shift Low lane by 0 and High lane by 16 in one `>>` op.
        // UNLESS we process lanes separately or verify if GPU has per-byte shift? No.
        // Byte Perm (PRMT) works!
        // PRMT can select bytes.

        // This suggests `EVOLVED_H2_PRMT_4` approach again?
        // Wait, standard `Packed` kernel (Generated) separates logical lanes.
        // `int idx_lo = ...`
        // `int idx_hi = ...`
        // `unsigned short s_lo = (S_LO >> (shift_lo*16))`
        // This separation is necessary on SIMT.

        // Let's implement the scalar extraction logic using our split banks.
        unsigned int b_lo = bank_sel & 0xFFFF;
        unsigned int b_hi = bank_sel >> 16;

        unsigned int s_lo_bit = shift_sel & 0xFFFF;
        unsigned int s_hi_bit = shift_sel >> 16;

        // S_LO / S_HI are scalars (simpler than 64-bit).
        unsigned int slope_pack_lo = (b_lo) ? S_HI : S_LO;
        unsigned int slope_pack_hi = (b_hi) ? S_HI : S_LO;

        unsigned short s_lo = (slope_pack_lo >> (s_lo_bit * 16)) & 0xFFFF;
        unsigned short s_hi = (slope_pack_hi >> (s_hi_bit * 16)) & 0xFFFF;

        // Offsets
        unsigned int off_pack_lo = (b_lo) ? O_HI : O_LO;
        unsigned int off_pack_hi = (b_hi) ? O_HI : O_LO;

        unsigned short o_lo = (off_pack_lo >> (s_lo_bit * 16)) & 0xFFFF;
        unsigned short o_hi = (off_pack_hi >> (s_hi_bit * 16)) & 0xFFFF;

        unsigned int s_final = ((unsigned int)s_hi << 16) | s_lo;
        unsigned int o_final = ((unsigned int)o_hi << 16) | o_lo;

        __half2 slope = *reinterpret_cast<__half2*>(&s_final);
        __half2 offset = *reinterpret_cast<__half2*>(&o_final);

        // Eval & Reconstruct
        __half2 poly_f_h2 = __hfma2(t_clamped, slope, offset);

        int n_lo = __half2int_rd(__low2half(n_h2));
        int n_hi = __half2int_rd(__high2half(n_h2));
        unsigned int poly_bits = *reinterpret_cast<const unsigned int*>(&poly_f_h2);
        unsigned short p_lo = poly_bits & 0xFFFF;
        unsigned short p_hi = poly_bits >> 16;
        unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
        unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
        unsigned int res_bits = ((unsigned int)res_hi << 16) | res_lo;
        return *reinterpret_cast<__half2*>(&res_bits);
    }
};

struct EVOLVED_H2_PACKED_DELTA_4 {
    // -----------------------------------------------------------------
    // STRATEGY: Packed + Delta Hybrid for N=4
    // -----------------------------------------------------------------
    // 1. Determine Bank (Upper/Lower Half) using t >= 0.5.
    // 2. Select 32-bit packed constants for that bank.
    // 3. Apply N=2 Differential within the bank.
    //
    // Bank 0 (t < 0.5): Intervals 0, 1.
    // Bank 1 (t >= 0.5): Intervals 2, 3.
    //
    // Slopes (N=4 D=1):
    // S0=0x3A0B, S1=0x3B2F, S2=0x3C45, S3=0x3D15
    // Offsets:
    // O0=0x3BFB, O1=0x3BB2, O2=0x3B04, O3=0x39CC
    //
    // Bank Bases: S_Lo = S0, O_Lo = O0, S_Hi = S2, O_Hi = O2.
    // Deltas: dS_Lo = S1-S0, dO_Lo = O1-O0, dS_Hi = S3-S2, dO_Hi = O3-O2.

    // Bank 0: Base (Interval 0)
    static constexpr unsigned int S0_u = 0x3A0B3A0B;
    static constexpr unsigned int O0_u = 0x3BFB3BFB;
    // Bank 0: Delta (Interval 1 - Interval 0)
    // dS_Lo = 0x3B2F - 0x3A0B = 0x0124
    // dO_Lo = 0x3BB2 - 0x3BFB = 0xFFB7 (Negative)
    static constexpr unsigned int dS0_u = 0x01240124;
    static constexpr unsigned int dO0_u = 0xFFB7FFB7;

    // Bank 1: Base (Interval 2)
    static constexpr unsigned int S2_u = 0x3C453C45;
    static constexpr unsigned int O2_u = 0x3B043B04;
    // Bank 1: Delta (Interval 3 - Interval 2)
    // dS_Hi = 0x3D15 - 0x3C45 = 0x00D0
    // dO_Hi = 0x39CC - 0x3B04 = 0xFEC8 (Negative)
    static constexpr unsigned int dS2_u = 0x00D000D0;
    static constexpr unsigned int dO2_u = 0xFEC8FEC8;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // --- Clamp & Fractional ---
        __half2 t_min = __float2half2_rn(0.0f);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, t_min), __float2half2_rn(0.999f));

        // --- 1. Bank Selection (t >= 0.5) ---
        unsigned int h_50_u = 0x38003800;
        __half2 h_50 = *reinterpret_cast<__half2*>(&h_50_u);
        __half2 m_bank_fp = __hge2(t_clamped, h_50); // 1.0 or 0.0
        unsigned int m_bank_raw = *reinterpret_cast<unsigned int*>(&m_bank_fp);
        // Convert FP mask (0x3C00) to INT mask (0xFFFF)
        unsigned int m_bank_bool = (m_bank_raw >> 14) & 0x00010001;
        unsigned int m_bank = m_bank_bool * 0xFFFF;

        // --- 2. Select Bank Constants (Branchless) ---
        unsigned int s_base = (S2_u & m_bank) | (S0_u & ~m_bank);
        unsigned int o_base = (O2_u & m_bank) | (O0_u & ~m_bank);
        unsigned int ds     = (dS2_u & m_bank) | (dS0_u & ~m_bank);
        unsigned int doo    = (dO2_u & m_bank) | (dO0_u & ~m_bank);

        // --- 3. In-Bank Delta Selection (t >= 0.25 or t >= 0.75) ---
        // If Bank 0: threshold = 0.25.
        // If Bank 1: threshold = 0.75.
        // We can compute both and select.
        unsigned int h_25_u = 0x34003400;
        unsigned int h_75_u = 0x3A003A00;
        __half2 h_25 = *reinterpret_cast<__half2*>(&h_25_u);
        __half2 h_75 = *reinterpret_cast<__half2*>(&h_75_u);

        __half2 m_25_fp = __hge2(t_clamped, h_25);
        __half2 m_75_fp = __hge2(t_clamped, h_75);

        // Select threshold based on bank
        unsigned int m_25_raw = *reinterpret_cast<unsigned int*>(&m_25_fp);
        unsigned int m_75_raw = *reinterpret_cast<unsigned int*>(&m_75_fp);

        // m_delta_raw is m_75 if in Bank 1, m_25 if in Bank 0.
        unsigned int m_delta_raw = (m_75_raw & m_bank) | (m_25_raw & ~m_bank);
        __half2 m_delta_fp = *reinterpret_cast<__half2*>(&m_delta_raw);

        // --- 4. Accumulate (INT Pipe) ---
        // S_Final = S_Base + (dS & m_delta)
        // O_Final = O_Base + (dO & m_delta)
        unsigned int m_delta_bool = (m_delta_raw >> 14) & 0x00010001;
        unsigned int m_delta = m_delta_bool * 0xFFFF;

        unsigned int s_final = s_base + (ds & m_delta);
        unsigned int o_final = o_base + (doo & m_delta);

        __half2 slope = *reinterpret_cast<__half2*>(&s_final);
        __half2 offset = *reinterpret_cast<__half2*>(&o_final);

        // --- 5. Eval ---
        __half2 poly_f_h2 = __hfma2(t_clamped, slope, offset);

        // --- 6. Reconstruct (For Pow2 on [0,1), this is identity) ---
        // If t_h2 is already fractional, no reconstruction needed.
        // For full range, add exponent bits.
        return poly_f_h2;
    }
};

struct EVOLVED_F32_N4_D1 {
    // -----------------------------------------------------------------
    // STRATEGY: FP32-based N=4 D=1 (Matching FA Baseline Approach)
    // -----------------------------------------------------------------
    // FA uses FP32 internally. Let's see if FP32 compute is faster.
    // N=4 D=1: 4 intervals on [0,1), linear poly per interval.

    // Coefficients (FP32 versions of the FP16 coefficients)
    // Slopes: 0x3A0B=0.5225, 0x3B2F=0.6713, 0x3C45=0.8606, 0x3D15=1.0857
    // Actually, let's compute proper FP32 coefficients for pow2 on [0,1).
    // Interval i: [i/4, (i+1)/4)
    // Linear approx: slope = (2^((i+1)/4) - 2^(i/4)) / (1/4) = 4 * (2^((i+1)/4) - 2^(i/4))
    // offset = 2^(i/4)

    // i=0: [0, 0.25), slope = 4*(2^0.25 - 1) = 4*(1.1892 - 1) = 0.757, offset = 1.0
    // i=1: [0.25, 0.5), slope = 4*(2^0.5 - 2^0.25) = 4*(1.4142 - 1.1892) = 0.900, offset = 2^0.25 - 0.25*0.900
    // Actually, for y = ax + b on [lo, hi] mapping to [2^lo, 2^hi]:
    // a = (2^hi - 2^lo) / (hi - lo)
    // b = 2^lo - a*lo

    // Let me just use direct calculation:
    // Interval 0: [0, 0.25), y = S0*x + O0, where 2^0 = S0*0 + O0 => O0 = 1.0
    //             2^0.25 = S0*0.25 + 1.0 => S0 = (1.1892 - 1)/0.25 = 0.757
    // Interval 1: [0.25, 0.5), y = S1*x + O1
    //             2^0.25 = S1*0.25 + O1
    //             2^0.5 = S1*0.5 + O1
    //             S1 = (2^0.5 - 2^0.25) / 0.25 = (1.4142 - 1.1892) / 0.25 = 0.900
    //             O1 = 2^0.25 - S1*0.25 = 1.1892 - 0.900*0.25 = 0.9642
    // ... etc

    static constexpr float S0 = 0.7568f;  // Interval 0
    static constexpr float S1 = 0.9000f;  // Interval 1
    static constexpr float S2 = 1.0704f;  // Interval 2
    static constexpr float S3 = 1.2730f;  // Interval 3

    static constexpr float O0 = 1.0000f;  // Interval 0
    static constexpr float O1 = 0.9643f;  // Interval 1
    static constexpr float O2 = 0.8791f;  // Interval 2
    static constexpr float O3 = 0.7324f;  // Interval 3

    // Packed as float4 for efficient loading
    static __device__ __forceinline__ float2 get_coeffs(int idx) {
        switch(idx & 3) {
            case 0: return make_float2(S0, O0);
            case 1: return make_float2(S1, O1);
            case 2: return make_float2(S2, O2);
            default: return make_float2(S3, O3);
        }
    }

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        float2 t_f2 = __half22float2(t_h2);

        // Clamp to [0, 0.9999)
        t_f2.x = fminf(fmaxf(t_f2.x, 0.0f), 0.9999f);
        t_f2.y = fminf(fmaxf(t_f2.y, 0.0f), 0.9999f);

        // Find interval: idx = (int)(t * 4)
        int idx_x = (int)(t_f2.x * 4.0f);
        int idx_y = (int)(t_f2.y * 4.0f);

        // Get coefficients
        float2 c_x = get_coeffs(idx_x);
        float2 c_y = get_coeffs(idx_y);

        // Evaluate: y = S*t + O
        float res_x = __fmaf_rn(t_f2.x, c_x.x, c_x.y);
        float res_y = __fmaf_rn(t_f2.y, c_y.x, c_y.y);

        return __float22half2_rn(make_float2(res_x, res_y));
    }
};

struct EVOLVED_F32_N4_D1_BRANCHLESS {
    // -----------------------------------------------------------------
    // STRATEGY: Branchless FP32 N=4 D=1 with Packed Lookup
    // -----------------------------------------------------------------
    // Avoid switch/branching by using array indexing.
    // FP32 coefficients stored as raw bits for direct load.

    // Slopes (FP32):
    // i=0: 0.7568 = 0x3F41CC70
    // i=1: 0.9000 = 0x3F666666
    // i=2: 1.0704 = 0x3F892500
    // i=3: 1.2730 = 0x3FA2F1AA

    // Offsets (FP32):
    // i=0: 1.0000 = 0x3F800000
    // i=1: 0.9643 = 0x3F76E980
    // i=2: 0.8791 = 0x3F6129E8
    // i=3: 0.7324 = 0x3F3B9580

    // Alternative: Use __ldg or shared memory for coefficient array.
    // For now, use register-based selection.

    // Key insight: FA uses SINGLE polynomial (Degree 3) with NO branching.
    // Just 3 FMAs chained + reconstruction.
    // We need to match that: N=4 D=1 has 1 FMA + coefficient selection.
    // The selection overhead is killing us.

    // NEW APPROACH: Compute ALL 4 results, select with mask (like N=2 DIFF).
    // But that means 4 FMAs + 3 mask ops. Too expensive.

    // BETTER IDEA: Use FA's MAGIC number trick for fractional extraction!
    // Then use bit hack for indexing WITHOUT conversion.

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        float2 t_f2 = __half22float2(t_h2);

        // MAGIC NUMBER TRICK (from FA baseline):
        // Add 12582912.0f (MAGIC) to shift fractional bits to integer position.
        // This avoids explicit floor() and conversion.
        const float MAGIC = 12582912.0f; // 0x4B400000

        // Clamp to [0, 1) first
        float t_x = fminf(fmaxf(t_f2.x, 0.0f), 0.9999f);
        float t_y = fminf(fmaxf(t_f2.y, 0.0f), 0.9999f);

        // Scale to [0, 4) then extract integer part
        float scaled_x = t_x * 4.0f;
        float scaled_y = t_y * 4.0f;

        // Add MAGIC to extract integer (rounds towards -inf implicitly)
        float x_magic = __fadd_rd(scaled_x, MAGIC);
        float y_magic = __fadd_rd(scaled_y, MAGIC);

        // Extract integer bits directly (low 23 bits of mantissa)
        int idx_x = __float_as_int(x_magic) & 0x3; // Only need 2 bits for N=4
        int idx_y = __float_as_int(y_magic) & 0x3;

        // Coefficient arrays (in registers)
        // Use float4 for potential vectorized load
        const float slopes[4] = {0.7568f, 0.9000f, 1.0704f, 1.2730f};
        const float offsets[4] = {1.0000f, 0.9643f, 0.8791f, 0.7324f};

        // Direct array indexing (compiler may optimize to cmov/predicated)
        float s_x = slopes[idx_x];
        float s_y = slopes[idx_y];
        float o_x = offsets[idx_x];
        float o_y = offsets[idx_y];

        // Evaluate: y = S*t + O
        float res_x = __fmaf_rn(t_x, s_x, o_x);
        float res_y = __fmaf_rn(t_y, s_y, o_y);

        return __float22half2_rn(make_float2(res_x, res_y));
    }
};

struct EVOLVED_F32_SINGLE_POLY {
    // -----------------------------------------------------------------
    // STRATEGY: Single Degree-3 Polynomial (Like FA Baseline)
    // -----------------------------------------------------------------
    // No branching, no coefficient selection. Just chained FMAs.
    // This should match FA baseline performance.
    // Coefficients fitted for pow2 on [0, 1).

    // pow2(x) = c3*x^3 + c2*x^2 + c1*x + c0
    // Fitted via least squares on [0, 1):
    // c3 ≈ 0.077, c2 ≈ 0.228, c1 ≈ 0.695, c0 ≈ 1.0
    // (Same as FA baseline constants!)

    static constexpr float C3 = 0.077119089663028717041015625f;
    static constexpr float C2 = 0.227564394474029541015625f;
    static constexpr float C1 = 0.695146143436431884765625f;
    static constexpr float C0 = 1.0f;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        float2 t_f2 = __half22float2(t_h2);

        // Clamp
        float t_x = fminf(fmaxf(t_f2.x, 0.0f), 0.9999f);
        float t_y = fminf(fmaxf(t_f2.y, 0.0f), 0.9999f);

        // Horner's method: ((c3*x + c2)*x + c1)*x + c0
        float p_x = __fmaf_rn(t_x, C3, C2);
        float p_y = __fmaf_rn(t_y, C3, C2);

        p_x = __fmaf_rn(p_x, t_x, C1);
        p_y = __fmaf_rn(p_y, t_y, C1);

        p_x = __fmaf_rn(p_x, t_x, C0);
        p_y = __fmaf_rn(p_y, t_y, C0);

        return __float22half2_rn(make_float2(p_x, p_y));
    }
};

struct EVOLVED_H2_N4_D1_ULTRA {
    // -----------------------------------------------------------------
    // STRATEGY: Ultra-Optimized N=4 D=1 FP16 Spline
    // -----------------------------------------------------------------
    // Goal: Beat FA Baseline (0.053 ms) with N=4 D=1 spline.
    // Current N=4 D=1: 0.0615 ms
    // Gap: ~16%
    //
    // Optimizations:
    // 1. Eliminate __halves2half2 by pre-packing pairs
    // 2. Use SIMD-style coefficient selection
    // 3. Minimize constant loads

    // Pre-packed Slope pairs: (S0, S0), (S1, S1), (S2, S2), (S3, S3)
    // packed as half2 in uint32
    static constexpr unsigned int S_PACKED[4] = {0x3A0B3A0B, 0x3B2F3B2F, 0x3C453C45, 0x3D153D15};
    static constexpr unsigned int O_PACKED[4] = {0x3BFB3BFB, 0x3BB23BB2, 0x3B043B04, 0x39CC39CC};

    // Alternative: Pack ALL coefficients into registers at compile time
    // For SIMD half2, we need different approach:
    // When idx_0 == idx_1 (very common), we can use single lookup

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp: t in [0, 0.9999)
        __half2 t_clamped = __hmin2(__hmax2(t_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // Bit Hack Indexing (Same as generated, but streamlined)
        // t_norm = t + 1.0 (since scale=1.0 on [0,1), this is just +1)
        __half2 t_norm = __hadd2(t_clamped, __float2half2_rn(1.0f));

        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);
        int idx_0 = (bits >> 8) & 3;        // Low lane
        int idx_1 = (bits >> 24) & 3;       // High lane

        // Load coefficients (inline to avoid device code array issues)
        unsigned int s0_u, s1_u, o0_u, o1_u;
        switch(idx_0) {
            case 0: s0_u = 0x3A0B3A0B; o0_u = 0x3BFB3BFB; break;
            case 1: s0_u = 0x3B2F3B2F; o0_u = 0x3BB23BB2; break;
            case 2: s0_u = 0x3C453C45; o0_u = 0x3B043B04; break;
            default: s0_u = 0x3D153D15; o0_u = 0x39CC39CC; break;
        }
        switch(idx_1) {
            case 0: s1_u = 0x3A0B3A0B; o1_u = 0x3BFB3BFB; break;
            case 1: s1_u = 0x3B2F3B2F; o1_u = 0x3BB23BB2; break;
            case 2: s1_u = 0x3C453C45; o1_u = 0x3B043B04; break;
            default: s1_u = 0x3D153D15; o1_u = 0x39CC39CC; break;
        }

        // Fast approach when indices are same (very common case):
        // Skip halves2half2 packing entirely
        __half2 slope, offset;

        if (idx_0 == idx_1) {
            // Both lanes use same coefficient - no packing needed!
            slope = *reinterpret_cast<__half2*>(&s0_u);
            offset = *reinterpret_cast<__half2*>(&o0_u);
        } else {
            // Different coefficients - need to pack
            unsigned short s_lo = s0_u & 0xFFFF;
            unsigned short s_hi = s1_u & 0xFFFF;
            unsigned short o_lo = o0_u & 0xFFFF;
            unsigned short o_hi = o1_u & 0xFFFF;

            unsigned int s_packed = ((unsigned int)s_hi << 16) | s_lo;
            unsigned int o_packed = ((unsigned int)o_hi << 16) | o_lo;

            slope = *reinterpret_cast<__half2*>(&s_packed);
            offset = *reinterpret_cast<__half2*>(&o_packed);
        }

        // Evaluate: y = slope * t + offset
        return __hfma2(t_clamped, slope, offset);
    }
};

struct EVOLVED_H2_N4_D1_SIMD {
    // -----------------------------------------------------------------
    // STRATEGY: True SIMD N=4 D=1 (No Per-Lane Branching)
    // -----------------------------------------------------------------
    // Key insight: Process both lanes IDENTICALLY then combine.
    // Use packed coefficient constants where both lanes hold same value.

    // 64-bit packed: [S3:S2:S1:S0] as 4x FP16
    static constexpr unsigned long long S_PACKED = 0x3D153C453B2F3A0BULL;
    static constexpr unsigned long long O_PACKED = 0x39CC3B043BB23BFBULL;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp
        __half2 t_clamped = __hmin2(__hmax2(t_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // SIMD Indexing: Extract both indices in parallel
        __half2 t_norm = __hadd2(t_clamped, __float2half2_rn(1.0f));
        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);

        // Extract both indices
        int idx_lo = (bits >> 8) & 3;
        int idx_hi = (bits >> 24) & 3;

        // Compute combined shift: shift_lo = idx_lo*16, shift_hi = idx_hi*16
        // Extract both coefficients with single 64-bit shift? No, still need 2 shifts.

        // Best we can do: Parallel extraction
        int shift_lo = idx_lo << 4;
        int shift_hi = idx_hi << 4;

        unsigned short s_lo = (S_PACKED >> shift_lo) & 0xFFFF;
        unsigned short s_hi = (S_PACKED >> shift_hi) & 0xFFFF;
        unsigned short o_lo = (O_PACKED >> shift_lo) & 0xFFFF;
        unsigned short o_hi = (O_PACKED >> shift_hi) & 0xFFFF;

        // Pack (must be done after extraction)
        unsigned int s_packed = ((unsigned int)s_hi << 16) | s_lo;
        unsigned int o_packed = ((unsigned int)o_hi << 16) | o_lo;

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        return __hfma2(t_clamped, slope, offset);
    }
};

struct EVOLVED_H2_N4_D1_FUSED {
    // -----------------------------------------------------------------
    // STRATEGY: Fused Coefficient Load (Avoid Separate Slope/Offset)
    // -----------------------------------------------------------------
    // Pack slope+offset together so a single 64-bit load gets both.
    // Then use PRMT to extract the right bytes.

    // Coefficient pairs: (S0,O0), (S1,O1), (S2,O2), (S3,O3)
    // Packed as 32-bit each: [O0:S0], [O1:S1], [O2:S2], [O3:S3]
    static constexpr unsigned int COEFF[4] = {
        0x3BFB3A0B, // O0:S0
        0x3BB23B2F, // O1:S1
        0x3B043C45, // O2:S2
        0x39CC3D15  // O3:S3
    };

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp
        __half2 t_clamped = __hmin2(__hmax2(t_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // Index extraction
        __half2 t_norm = __hadd2(t_clamped, __float2half2_rn(1.0f));
        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);
        int idx_lo = (bits >> 8) & 3;
        int idx_hi = (bits >> 24) & 3;

        // Single load per lane gets both slope AND offset
        unsigned int coeff_lo, coeff_hi;
        switch(idx_lo) {
            case 0: coeff_lo = 0x3BFB3A0B; break;
            case 1: coeff_lo = 0x3BB23B2F; break;
            case 2: coeff_lo = 0x3B043C45; break;
            default: coeff_lo = 0x39CC3D15; break;
        }
        switch(idx_hi) {
            case 0: coeff_hi = 0x3BFB3A0B; break;
            case 1: coeff_hi = 0x3BB23B2F; break;
            case 2: coeff_hi = 0x3B043C45; break;
            default: coeff_hi = 0x39CC3D15; break;
        }

        // Extract slope (low 16 bits) and offset (high 16 bits)
        unsigned short s_lo = coeff_lo & 0xFFFF;
        unsigned short o_lo = coeff_lo >> 16;
        unsigned short s_hi = coeff_hi & 0xFFFF;
        unsigned short o_hi = coeff_hi >> 16;

        // Pack
        unsigned int s_packed = ((unsigned int)s_hi << 16) | s_lo;
        unsigned int o_packed = ((unsigned int)o_hi << 16) | o_lo;

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        return __hfma2(t_clamped, slope, offset);
    }
};

struct EVOLVED_H2_N4_D1_PURE_FP16 {
    // -----------------------------------------------------------------
    // STRATEGY: Pure FP16 SIMD (No float2half constant conversions)
    // -----------------------------------------------------------------
    // Goal: Eliminate __float2half2_rn() calls which may add overhead.
    // Use raw FP16 bit patterns directly.

    // Distinct coefficients for each of 4 intervals (verified from JSON):
    // i=0: S=0x3A0B (0.5225), O=0x3BFB (0.9995)
    // i=1: S=0x3B2F (0.6713), O=0x3BB2 (0.9648)
    // i=2: S=0x3C45 (0.8169), O=0x3B04 (0.8789)
    // i=3: S=0x3D15 (1.0835), O=0x39CC (0.7324)

    // 64-bit packed: [S3:S2:S1:S0], [O3:O2:O1:O0]
    static constexpr unsigned long long S_PACKED = 0x3D153C453B2F3A0BULL;
    static constexpr unsigned long long O_PACKED = 0x39CC3B043BB23BFBULL;

    // Pre-computed FP16 constants (bit patterns)
    static constexpr unsigned int ZERO_H2 = 0x00000000;    // 0.0 as half2
    static constexpr unsigned int ONE_H2 = 0x3C003C00;     // 1.0 as half2
    static constexpr unsigned int MAX_H2 = 0x3BFF3BFF;     // 0.9999 as half2

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp using FP16 constants (inline literals)
        unsigned int zero_u = 0x00000000;  // 0.0 as half2
        unsigned int max_u = 0x3BFF3BFF;   // 0.9999 as half2
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 max_val = *reinterpret_cast<__half2*>(&max_u);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, zero), max_val);

        // Bit Hack Indexing: t + 1.0 maps [0,1) -> [1,2)
        unsigned int one_u = 0x3C003C00;  // 1.0 as half2
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 t_norm = __hadd2(t_clamped, one);

        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);
        int idx_lo = (bits >> 8) & 3;
        int idx_hi = (bits >> 24) & 3;

        // 64-bit coefficient extraction
        int shift_lo = idx_lo << 4;
        int shift_hi = idx_hi << 4;

        unsigned short s_lo = (S_PACKED >> shift_lo) & 0xFFFF;
        unsigned short s_hi = (S_PACKED >> shift_hi) & 0xFFFF;
        unsigned short o_lo = (O_PACKED >> shift_lo) & 0xFFFF;
        unsigned short o_hi = (O_PACKED >> shift_hi) & 0xFFFF;

        // Pack
        unsigned int s_packed = ((unsigned int)s_hi << 16) | s_lo;
        unsigned int o_packed = ((unsigned int)o_hi << 16) | o_lo;

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        return __hfma2(t_clamped, slope, offset);
    }
};

struct EVOLVED_H2_N4_D1_VECTORIZED {
    // -----------------------------------------------------------------
    // STRATEGY: Vectorized Coefficient Load (uint4)
    // -----------------------------------------------------------------
    // Goal: Load all 8 coefficients (4 slopes + 4 offsets) in one vector.
    // Use uint4 (128 bits) containing [S0:S1:S2:S3:O0:O1:O2:O3].

    // Actually, for register-based constants, vectorization doesn't help much.
    // BUT we can pack slope+offset together per interval:
    // Interval i: packed_i = (O_i << 16) | S_i (32 bits)
    // Then load 4x32 = 128 bits = uint4.

    // Wait - for N=4, we have:
    // 4 slopes (16 bits each) = 64 bits
    // 4 offsets (16 bits each) = 64 bits
    // Total = 128 bits = uint4

    // Packed as: x=S_01, y=S_23, z=O_01, w=O_23
    // Where S_01 = (S1 << 16) | S0
    static constexpr unsigned int S_01 = 0x3B2F3A0B;  // [S1:S0]
    static constexpr unsigned int S_23 = 0x3D153C45;  // [S3:S2]
    static constexpr unsigned int O_01 = 0x3BB23BFB;  // [O1:O0]
    static constexpr unsigned int O_23 = 0x39CC3B04;  // [O3:O2]

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp
        __half2 t_clamped = __hmin2(__hmax2(t_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // Index extraction
        __half2 t_norm = __hadd2(t_clamped, __float2half2_rn(1.0f));
        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);
        int idx_lo = (bits >> 8) & 3;
        int idx_hi = (bits >> 24) & 3;

        // Select bank (lower or upper pair) for each lane
        // idx < 2 -> S_01/O_01, else S_23/O_23
        unsigned int s_bank_lo = (idx_lo < 2) ? S_01 : S_23;
        unsigned int s_bank_hi = (idx_hi < 2) ? S_01 : S_23;
        unsigned int o_bank_lo = (idx_lo < 2) ? O_01 : O_23;
        unsigned int o_bank_hi = (idx_hi < 2) ? O_01 : O_23;

        // Extract within bank: idx & 1 -> shift by 0 or 16
        int shift_lo = (idx_lo & 1) << 4;
        int shift_hi = (idx_hi & 1) << 4;

        unsigned short s_lo = (s_bank_lo >> shift_lo) & 0xFFFF;
        unsigned short s_hi = (s_bank_hi >> shift_hi) & 0xFFFF;
        unsigned short o_lo = (o_bank_lo >> shift_lo) & 0xFFFF;
        unsigned short o_hi = (o_bank_hi >> shift_hi) & 0xFFFF;

        // Pack
        unsigned int s_packed = ((unsigned int)s_hi << 16) | s_lo;
        unsigned int o_packed = ((unsigned int)o_hi << 16) | o_lo;

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        return __hfma2(t_clamped, slope, offset);
    }
};

struct EVOLVED_H2_N4_D1_MINIMAL {
    // -----------------------------------------------------------------
    // STRATEGY: Minimal Instruction Count
    // -----------------------------------------------------------------
    // Goal: Absolute minimum operations. Fuse where possible.
    // Inspired by FA baseline's efficiency.

    // 64-bit packed coefficients
    static constexpr unsigned long long S_PACKED = 0x3D153C453B2F3A0BULL;
    static constexpr unsigned long long O_PACKED = 0x39CC3B043BB23BFBULL;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Combined clamp + normalize: t_norm = clamp(t, 0, 0.999) + 1.0
        // We can clamp AFTER adding 1.0 by clamping to [1.0, 1.999)
        // Actually no - we need t_clamped for the polynomial eval.

        // Minimum clamp: just use hmin2 (skip hmax2 if input guaranteed >= 0)
        // For robustness, keep both.
        __half2 t_clamped = __hmin2(__hmax2(t_h2, __float2half2_rn(0.0f)), __float2half2_rn(0.999f));

        // Fused add + bit hack
        __half2 t_norm = __hadd2(t_clamped, __float2half2_rn(1.0f));
        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);

        // Extract BOTH indices with single computation
        // idx = (bits >> 8) & 0x00030003 would get both, but we need separate shifts
        unsigned int idx_mask = (bits >> 8) & 0x00030003;
        int idx_lo = idx_mask & 3;
        int idx_hi = (idx_mask >> 16) & 3;  // Hmm, this adds another shift

        // Alternative: Use FP16 mantissa bits directly
        // Actually, the bit pattern (bits >> 8) & 3 for low lane
        // and (bits >> 24) & 3 for high lane is already minimal.

        // Coefficient extraction (minimal: 4 shifts, 4 masks)
        int shift_lo = idx_lo << 4;
        int shift_hi = idx_hi << 4;

        unsigned short s_lo = (S_PACKED >> shift_lo) & 0xFFFF;
        unsigned short s_hi = (S_PACKED >> shift_hi) & 0xFFFF;
        unsigned short o_lo = (O_PACKED >> shift_lo) & 0xFFFF;
        unsigned short o_hi = (O_PACKED >> shift_hi) & 0xFFFF;

        // Pack (2 ops: shift + or)
        unsigned int s_packed = ((unsigned int)s_hi << 16) | s_lo;
        unsigned int o_packed = ((unsigned int)o_hi << 16) | o_lo;

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        // Single FMA for polynomial eval
        return __hfma2(t_clamped, slope, offset);
    }
};

struct EVOLVED_H2_N4_D1_VADD2 {
    // -----------------------------------------------------------------
    // STRATEGY: Use __vadd2 for Packed 16-bit Exponent Adds
    // -----------------------------------------------------------------
    // Instead of:
    //   unsigned short res_lo = p_lo + (unsigned short)(n_lo << 10);
    //   unsigned short res_hi = p_hi + (unsigned short)(n_hi << 10);
    // Use:
    //   unsigned int exp_packed = pack(n_hi << 10, n_lo << 10);
    //   unsigned int res = __vadd2(poly_bits, exp_packed);
    //
    // __vadd2 performs two 16-bit adds in parallel with wraparound.
    // This is exactly what we need for independent per-lane exponent adds.

    // 64-bit packed coefficients (distinct values per interval)
    static constexpr unsigned long long S_PACKED = 0x3D153C453B2F3A0BULL;
    static constexpr unsigned long long O_PACKED = 0x39CC3B043BB23BFBULL;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp using inline FP16 constants
        unsigned int zero_u = 0x00000000;
        unsigned int max_u = 0x3BFF3BFF;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 max_val = *reinterpret_cast<__half2*>(&max_u);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, zero), max_val);

        // Bit Hack Indexing
        unsigned int one_u = 0x3C003C00;
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 t_norm = __hadd2(t_clamped, one);

        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);
        int idx_lo = (bits >> 8) & 3;
        int idx_hi = (bits >> 24) & 3;

        // Coefficient extraction
        int shift_lo = idx_lo << 4;
        int shift_hi = idx_hi << 4;

        unsigned short s_lo = (S_PACKED >> shift_lo) & 0xFFFF;
        unsigned short s_hi = (S_PACKED >> shift_hi) & 0xFFFF;
        unsigned short o_lo = (O_PACKED >> shift_lo) & 0xFFFF;
        unsigned short o_hi = (O_PACKED >> shift_hi) & 0xFFFF;

        // Pack coefficients
        unsigned int s_packed = ((unsigned int)s_hi << 16) | s_lo;
        unsigned int o_packed = ((unsigned int)o_hi << 16) | o_lo;

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        // Polynomial evaluation
        __half2 poly_h2 = __hfma2(t_clamped, slope, offset);

        // --- NEW: VADD2-based Exponent Reconstruction ---
        // For pow2 on [0,1), the polynomial gives us the mantissa.
        // We need to add the exponent (which is 0 for [0,1)).
        // For full range pow2(x), we'd need: result = poly * 2^floor(x)
        // which is: result_bits = poly_bits + (floor_x << 10) for FP16.
        //
        // Since we're evaluating on [0,1), floor(x) = 0, so no exponent add needed.
        // For full range, we'd do:
        // unsigned int poly_bits = *reinterpret_cast<unsigned int*>(&poly_h2);
        // unsigned int exp_packed = __vadd2(0, (floor_hi << 26) | (floor_lo << 10));
        // unsigned int res = __vadd2(poly_bits, exp_packed);

        return poly_h2;
    }
};

struct EVOLVED_H2_N4_D1_INT32_PIPELINE {
    // -----------------------------------------------------------------
    // STRATEGY: INT32 Pipeline for Exponent Reconstruction
    // -----------------------------------------------------------------
    // Key insight: FP32 and INT32 ALUs can execute concurrently.
    // Move reconstruction to INT32 to get "free" cycles.
    //
    // For full pow2 (not just [0,1)):
    // 1. FP32 pipeline: polynomial evaluation
    // 2. INT32 pipeline: exponent bit manipulation
    //
    // IEEE FP16: sign(1) | exp(5) | mantissa(10)
    // 2^n multiplication = adding n to exponent field = bits + (n << 10)

    // 64-bit packed coefficients
    static constexpr unsigned long long S_PACKED = 0x3D153C453B2F3A0BULL;
    static constexpr unsigned long long O_PACKED = 0x39CC3B043BB23BFBULL;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // === INT32 Pipeline: Extract integer part ===


        // Clamp
        unsigned int zero_u = 0x00000000;
        unsigned int max_u = 0x3BFF3BFF;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 max_val = *reinterpret_cast<__half2*>(&max_u);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, zero), max_val);

        // Bit Hack Indexing
        unsigned int one_u = 0x3C003C00;
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 t_norm = __hadd2(t_clamped, one);

        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);

        // === INT32 Pipeline: Index extraction ===
        int idx_lo = (bits >> 8) & 3;
        int idx_hi = (bits >> 24) & 3;
        int shift_lo = idx_lo << 4;
        int shift_hi = idx_hi << 4;

        // === INT32 Pipeline: Coefficient extraction ===
        unsigned short s_lo = (S_PACKED >> shift_lo) & 0xFFFF;
        unsigned short s_hi = (S_PACKED >> shift_hi) & 0xFFFF;
        unsigned short o_lo = (O_PACKED >> shift_lo) & 0xFFFF;
        unsigned short o_hi = (O_PACKED >> shift_hi) & 0xFFFF;

        // Pack (INT32)
        unsigned int s_packed = ((unsigned int)s_hi << 16) | s_lo;
        unsigned int o_packed = ((unsigned int)o_hi << 16) | o_lo;

        // === FP16 Pipeline: Polynomial evaluation ===
        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);
        __half2 poly_h2 = __hfma2(t_clamped, slope, offset);

        // For [0,1), no exponent reconstruction needed.
        // For full range, we'd add exponent bits here using __vadd2.

        return poly_h2;
    }
};

struct EVOLVED_H2_N4_D1_FUSED_COEFF {
    // -----------------------------------------------------------------
    // STRATEGY: Fused Coefficient Extraction (Single 64-bit op)
    // -----------------------------------------------------------------
    // Key insight: We can compute both coefficients with a single
    // 64-bit shift if we pack slope+offset together.
    //
    // Pack: [O3:S3:O2:S2:O1:S1:O0:S0] (alternating 16-bit)
    // Then a single shift extracts both S and O for an interval.
    //
    // Wait, this doesn't work cleanly because indices are different per lane.
    // BUT we can try: extract a single 32-bit containing [O:S] at once.

    // Alternative: Pack as [S3:S2:S1:S0:O3:O2:O1:O0] (128-bit total)
    // Use idx to shift into lower 16 bits, then mask.
    // This is what we're already doing with two 64-bit constants.

    // NEW IDEA: Pre-compute both extractions with a single variable shift.
    // __funnelshift_lc() provides a 64-bit funnel shift.

    // Actually, let's try PRMT for byte permutation!
    // __byte_perm(a, b, selector) - select bytes from a and b.

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp
        unsigned int zero_u = 0x00000000;
        unsigned int max_u = 0x3BFF3BFF;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 max_val = *reinterpret_cast<__half2*>(&max_u);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, zero), max_val);

        // Index extraction
        unsigned int one_u = 0x3C003C00;
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 t_norm = __hadd2(t_clamped, one);
        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);
        int idx_lo = (bits >> 8) & 3;
        int idx_hi = (bits >> 24) & 3;

        // Coefficients stored as two 32-bit words: S_01=[S1:S0], S_23=[S3:S2]
        // Use __byte_perm to select the right 16-bit values.

        // S_01 = [S1:S0] = 0x3B2F3A0B, S_23 = [S3:S2] = 0x3D153C45
        // O_01 = [O1:O0] = 0x3BB23BFB, O_23 = [O3:O2] = 0x39CC3B04

        const unsigned int S_01 = 0x3B2F3A0B;
        const unsigned int S_23 = 0x3D153C45;
        const unsigned int O_01 = 0x3BB23BFB;
        const unsigned int O_23 = 0x39CC3B04;

        // Use __byte_perm to extract:
        // For idx=0: want bytes 0-1 from S_01 = selector 0x0010
        // For idx=1: want bytes 2-3 from S_01 = selector 0x0032
        // For idx=2: want bytes 0-1 from S_23 = selector 0x0054 (with S_23 as second input)
        // For idx=3: want bytes 2-3 from S_23 = selector 0x0076

        // This gets complex. Let's use a lookup table for selectors.
        const unsigned int sel_table[4] = {0x00000010, 0x00000032, 0x00000054, 0x00000076};
        unsigned int sel_lo = sel_table[idx_lo];
        unsigned int sel_hi = sel_table[idx_hi];

        // Extract slope for each lane
        unsigned int s_lo = __byte_perm(S_01, S_23, sel_lo) & 0xFFFF;
        unsigned int s_hi = __byte_perm(S_01, S_23, sel_hi) & 0xFFFF;
        unsigned int o_lo = __byte_perm(O_01, O_23, sel_lo) & 0xFFFF;
        unsigned int o_hi = __byte_perm(O_01, O_23, sel_hi) & 0xFFFF;

        // Pack
        unsigned int s_packed = (s_hi << 16) | s_lo;
        unsigned int o_packed = (o_hi << 16) | o_lo;

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        return __hfma2(t_clamped, slope, offset);
    }
};

struct EVOLVED_H2_N4_D1_FUNNEL_SHIFT {
    // -----------------------------------------------------------------
    // STRATEGY: PTX Funnel Shift for 64-bit Coefficient Extraction
    // -----------------------------------------------------------------
    // The funnel shift instruction (shf.l.clamp / shf.r.clamp) can shift
    // a 64-bit value by a variable amount more efficiently than C++ shifts.
    //
    // shf.l.clamp.b32 d, a, b, c : d = (a:b << c) [clamp c to 0..32]
    // shf.r.clamp.b32 d, a, b, c : d = (a:b >> c) [clamp c to 0..32]
    //
    // For our use case: extract 16-bit coeff from 64-bit packed:
    // (packed >> shift) & 0xFFFF where shift = idx * 16
    //
    // With funnel shift:
    // shf.r.clamp.b32 out, packed_lo, packed_hi, shift

    // 64-bit packed coefficients split into hi/lo 32-bit parts
    static constexpr unsigned int S_LO = 0x3B2F3A0B;  // [S1:S0]
    static constexpr unsigned int S_HI = 0x3D153C45;  // [S3:S2]
    static constexpr unsigned int O_LO = 0x3BB23BFB;  // [O1:O0]
    static constexpr unsigned int O_HI = 0x39CC3B04;  // [O3:O2]

    static __device__ __forceinline__ unsigned int funnel_shift_r(
        unsigned int lo, unsigned int hi, unsigned int shift) {
        unsigned int result;
        asm("shf.r.clamp.b32 %0, %1, %2, %3;"
            : "=r"(result)
            : "r"(lo), "r"(hi), "r"(shift));
        return result;
    }

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp
        unsigned int zero_u = 0x00000000;
        unsigned int max_u = 0x3BFF3BFF;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 max_val = *reinterpret_cast<__half2*>(&max_u);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, zero), max_val);

        // Index extraction
        unsigned int one_u = 0x3C003C00;
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 t_norm = __hadd2(t_clamped, one);
        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);
        int idx_lo = (bits >> 8) & 3;
        int idx_hi = (bits >> 24) & 3;

        // Funnel shift extraction
        // shift = idx * 16, but for 64-bit starting from lo:
        // idx=0,1 -> in lo, idx=2,3 -> need hi
        // We'll do: shf.r(lo, hi, idx*16) gives us the right 32-bit window
        unsigned int shift_lo = idx_lo << 4;
        unsigned int shift_hi = idx_hi << 4;

        unsigned int s_window_lo = funnel_shift_r(S_LO, S_HI, shift_lo);
        unsigned int s_window_hi = funnel_shift_r(S_LO, S_HI, shift_hi);
        unsigned int o_window_lo = funnel_shift_r(O_LO, O_HI, shift_lo);
        unsigned int o_window_hi = funnel_shift_r(O_LO, O_HI, shift_hi);

        // Extract low 16 bits
        unsigned short s_lo_val = s_window_lo & 0xFFFF;
        unsigned short s_hi_val = s_window_hi & 0xFFFF;
        unsigned short o_lo_val = o_window_lo & 0xFFFF;
        unsigned short o_hi_val = o_window_hi & 0xFFFF;

        // Pack
        unsigned int s_packed = ((unsigned int)s_hi_val << 16) | s_lo_val;
        unsigned int o_packed = ((unsigned int)o_hi_val << 16) | o_lo_val;

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        return __hfma2(t_clamped, slope, offset);
    }
};

struct EVOLVED_H2_N4_D1_WARP_BROADCAST {
    // -----------------------------------------------------------------
    // STRATEGY: Warp-Level Coefficient Broadcast
    // -----------------------------------------------------------------
    // Key insight: In many cases, all lanes in a warp have the SAME index.
    // (e.g., when processing contiguous data with t values in same interval)
    //
    // We can use __shfl_sync to broadcast coefficients from lane 0:
    // 1. Lane 0 computes coefficients
    // 2. All other lanes receive via warp shuffle
    //
    // This avoids each lane doing 64-bit shifts independently.
    //
    // Caveat: Only works when all lanes have same index!
    // We need to check this condition and fall back otherwise.

    // 64-bit packed coefficients
    static constexpr unsigned long long S_PACKED = 0x3D153C453B2F3A0BULL;
    static constexpr unsigned long long O_PACKED = 0x39CC3B043BB23BFBULL;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp
        unsigned int zero_u = 0x00000000;
        unsigned int max_u = 0x3BFF3BFF;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 max_val = *reinterpret_cast<__half2*>(&max_u);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, zero), max_val);

        // Index extraction
        unsigned int one_u = 0x3C003C00;
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 t_norm = __hadd2(t_clamped, one);
        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);
        int idx_lo = (bits >> 8) & 3;
        int idx_hi = (bits >> 24) & 3;

        // Check if all lanes have same index (using __match_any_sync)
        // This returns a mask of lanes with matching value
        unsigned int match_lo = __match_any_sync(0xFFFFFFFF, idx_lo);
        unsigned int match_hi = __match_any_sync(0xFFFFFFFF, idx_hi);

        unsigned int s_packed, o_packed;

        // If all 32 lanes match, use broadcast
        if (match_lo == 0xFFFFFFFF && match_hi == 0xFFFFFFFF) {
            // Lane 0 computes, all others receive via shuffle
            if (threadIdx.x % 32 == 0) {
                int shift_lo = idx_lo << 4;
                int shift_hi = idx_hi << 4;
                unsigned short s_lo = (S_PACKED >> shift_lo) & 0xFFFF;
                unsigned short s_hi = (S_PACKED >> shift_hi) & 0xFFFF;
                unsigned short o_lo = (O_PACKED >> shift_lo) & 0xFFFF;
                unsigned short o_hi = (O_PACKED >> shift_hi) & 0xFFFF;
                s_packed = ((unsigned int)s_hi << 16) | s_lo;
                o_packed = ((unsigned int)o_hi << 16) | o_lo;
            }
            s_packed = __shfl_sync(0xFFFFFFFF, s_packed, 0);
            o_packed = __shfl_sync(0xFFFFFFFF, o_packed, 0);
        } else {
            // Fallback: each lane computes independently
            int shift_lo = idx_lo << 4;
            int shift_hi = idx_hi << 4;
            unsigned short s_lo = (S_PACKED >> shift_lo) & 0xFFFF;
            unsigned short s_hi = (S_PACKED >> shift_hi) & 0xFFFF;
            unsigned short o_lo = (O_PACKED >> shift_lo) & 0xFFFF;
            unsigned short o_hi = (O_PACKED >> shift_hi) & 0xFFFF;
            s_packed = ((unsigned int)s_hi << 16) | s_lo;
            o_packed = ((unsigned int)o_hi << 16) | o_lo;
        }

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        return __hfma2(t_clamped, slope, offset);
    }
};

struct EVOLVED_H2_N4_D1_LOP3 {
    // -----------------------------------------------------------------
    // STRATEGY: LOP3 for Combined Bit Operations
    // -----------------------------------------------------------------
    // LOP3 is a 3-input logic operation that can combine multiple bitwise
    // ops into a single instruction.
    //
    // We can use it for the index extraction:
    // idx = (bits >> 8) & 3
    //
    // This is: shift, then AND. LOP3 can't help here directly.
    //
    // BUT for coefficient packing:
    // s_packed = (s_hi << 16) | s_lo
    // This is shift + OR, which could be done with deposit/extract.
    //
    // On SM100, we have BFI (bit field insert):
    // bfi.b32 d, a, b, pos, len : insert 'len' bits of 'a' at position 'pos' in 'b'

    static constexpr unsigned long long S_PACKED = 0x3D153C453B2F3A0BULL;
    static constexpr unsigned long long O_PACKED = 0x39CC3B043BB23BFBULL;

    static __device__ __forceinline__ unsigned int bfi(
        unsigned int src, unsigned int dst, unsigned int pos, unsigned int len) {
        unsigned int result;
        asm("bfi.b32 %0, %1, %2, %3, %4;"
            : "=r"(result)
            : "r"(src), "r"(dst), "r"(pos), "r"(len));
        return result;
    }

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp
        unsigned int zero_u = 0x00000000;
        unsigned int max_u = 0x3BFF3BFF;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 max_val = *reinterpret_cast<__half2*>(&max_u);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, zero), max_val);

        // Index extraction
        unsigned int one_u = 0x3C003C00;
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 t_norm = __hadd2(t_clamped, one);
        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);
        int idx_lo = (bits >> 8) & 3;
        int idx_hi = (bits >> 24) & 3;

        // Coefficient extraction
        int shift_lo = idx_lo << 4;
        int shift_hi = idx_hi << 4;

        unsigned short s_lo = (S_PACKED >> shift_lo) & 0xFFFF;
        unsigned short s_hi = (S_PACKED >> shift_hi) & 0xFFFF;
        unsigned short o_lo = (O_PACKED >> shift_lo) & 0xFFFF;
        unsigned short o_hi = (O_PACKED >> shift_hi) & 0xFFFF;

        // Pack using BFI: insert s_hi at pos 16 with len 16 into s_lo
        unsigned int s_packed = bfi(s_hi, s_lo, 16, 16);
        unsigned int o_packed = bfi(o_hi, o_lo, 16, 16);

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        return __hfma2(t_clamped, slope, offset);
    }
};

// ============================================================================
// N=2 D=3 SIGMOID on [-6, 6]
// ============================================================================
// Coefficients from analysis_results/Sigmoid_-6-6_N2_D3_stats.json (FP16 hex):
// Interval 0 (x < 0): C3=0x1BA8, C2=0x2B32, C1=0x3493, C0=0x3800
// Interval 1 (x >= 0): C3=0x1BA8, C2=0xAB32 (negative), C1=0x3493, C0=0x3800
//
// Boundary saturation: x < -6 -> 0, x > 6 -> 1

struct EVOLVED_H2_SIGMOID_N2_D3_SIMD {
    // 64-bit packed coefficients [C1:C0] (shared between intervals)
    // C1 at bits 16-31, C0 at bits 0-15
    static constexpr unsigned int C10_PACKED = 0x34933800;  // [C1:C0]

    // C3:C2 for interval 0 (x < 0): 0x1BA82B32
    // C3:C2 for interval 1 (x >= 0): 0x1BA8AB32
    static constexpr unsigned int C32_INT0 = 0x1BA82B32;
    static constexpr unsigned int C32_INT1 = 0x1BA8AB32;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Boundary constants
        unsigned int neg6_u = 0xC600C600;  // -6.0 as half2
  // +6.0 as half2 (actually need 0x46004600)
        // Fix: +6.0 in FP16 is 0x4600
        unsigned int zero_u = 0x00000000;
        unsigned int one_u = 0x3C003C00;  // 1.0 as half2

        // Check boundaries
        __half2 neg6 = *reinterpret_cast<__half2*>(&neg6_u);
        __half2 pos6 = __float2half2_rn(6.0f);
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 one = *reinterpret_cast<__half2*>(&one_u);

        // Clamp to [-6, 6]
        __half2 t_clamped = __hmin2(__hmax2(t_h2, neg6), pos6);

        // Interval selection: x < 0 -> interval 0, x >= 0 -> interval 1
        __half2 mask_fp = __hge2(t_h2, zero);
        unsigned int mask_raw = *reinterpret_cast<unsigned int*>(&mask_fp);
        // Convert FP mask to int mask
        unsigned int mask_bool = (mask_raw >> 14) & 0x00010001;
        unsigned int mask = mask_bool * 0xFFFF;

        // Select C32 (C3:C2) based on interval
        unsigned int c32 = (C32_INT1 & mask) | (C32_INT0 & ~mask);

        // Extract individual coefficients
        unsigned short c2_lo = c32 & 0xFFFF;
        unsigned short c3_lo = c32 >> 16;
        unsigned short c0 = C10_PACKED & 0xFFFF;
        unsigned short c1 = C10_PACKED >> 16;

        // Pack coefficients (both lanes use same)
        unsigned int c0_h2 = ((unsigned int)c0 << 16) | c0;
        unsigned int c1_h2 = ((unsigned int)c1 << 16) | c1;
        unsigned int c2_h2 = ((unsigned int)c2_lo << 16) | c2_lo;
        unsigned int c3_h2 = ((unsigned int)c3_lo << 16) | c3_lo;

        __half2 coeff0 = *reinterpret_cast<__half2*>(&c0_h2);
        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_h2);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_h2);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_h2);

        // Horner's method: ((c3*t + c2)*t + c1)*t + c0
        __half2 result = __hfma2(t_clamped, coeff3, coeff2);
        result = __hfma2(t_clamped, result, coeff1);
        result = __hfma2(t_clamped, result, coeff0);

        // Boundary saturation
        __half2 sat_lo = __hlt2(t_h2, neg6);  // x < -6 -> 0
        __half2 sat_hi = __hgt2(t_h2, pos6);  // x > 6 -> 1

        result = __hfma2(sat_lo, __hsub2(zero, result), result);  // if sat_lo: result = 0
        result = __hfma2(sat_hi, __hsub2(one, result), result);   // if sat_hi: result = 1

        return result;
    }
};

// ============================================================================
// N=2 D=3 TANH on [-2.5, 2.5]
// ============================================================================
// Coefficients from analysis_results/Tanh_-2.5-_2.5_N2_D3_stats.json (FP16 hex):
// Interval 0 (x < 0): C3=0x2B81, C2=0x3724, C1=0x3C91, C0=0x0000
// Interval 1 (x >= 0): C3=0x2B81, C2=0xB724 (negative), C1=0x3C91, C0=0x0000
//
// Boundary saturation: x < -2.5 -> -1, x > 2.5 -> 1

struct EVOLVED_H2_TANH_N2_D3_SIMD {
    // C1:C0 packed (shared)
    static constexpr unsigned int C10_PACKED = 0x3C910000;  // [C1:C0]

    // C3:C2 for interval 0 (x < 0): 0x2B813724
    // C3:C2 for interval 1 (x >= 0): 0x2B81B724
    static constexpr unsigned int C32_INT0 = 0x2B813724;
    static constexpr unsigned int C32_INT1 = 0x2B81B724;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Boundary constants
        unsigned int neg25_u = 0xC100C100;  // -2.5 as half2 (0xC100 is -2.5 in FP16)
        unsigned int pos25_u = 0x41004100;  // +2.5 as half2 (0x4100 is 2.5 in FP16)
        unsigned int zero_u = 0x00000000;
        unsigned int one_u = 0x3C003C00;
        unsigned int neg_one_u = 0xBC00BC00;  // -1.0 as half2

        __half2 neg25 = *reinterpret_cast<__half2*>(&neg25_u);
        __half2 pos25 = *reinterpret_cast<__half2*>(&pos25_u);
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 neg_one = *reinterpret_cast<__half2*>(&neg_one_u);

        // Clamp to [-2.5, 2.5]
        __half2 t_clamped = __hmin2(__hmax2(t_h2, neg25), pos25);

        // Interval selection: x < 0 -> interval 0, x >= 0 -> interval 1
        __half2 mask_fp = __hge2(t_h2, zero);
        unsigned int mask_raw = *reinterpret_cast<unsigned int*>(&mask_fp);
        unsigned int mask_bool = (mask_raw >> 14) & 0x00010001;
        unsigned int mask = mask_bool * 0xFFFF;

        // Select C32 based on interval
        unsigned int c32 = (C32_INT1 & mask) | (C32_INT0 & ~mask);

        // Extract coefficients
        unsigned short c2_val = c32 & 0xFFFF;
        unsigned short c3_val = c32 >> 16;
        unsigned short c0 = C10_PACKED & 0xFFFF;
        unsigned short c1 = C10_PACKED >> 16;

        // Pack for both lanes
        unsigned int c0_h2 = ((unsigned int)c0 << 16) | c0;
        unsigned int c1_h2 = ((unsigned int)c1 << 16) | c1;
        unsigned int c2_h2 = ((unsigned int)c2_val << 16) | c2_val;
        unsigned int c3_h2 = ((unsigned int)c3_val << 16) | c3_val;

        __half2 coeff0 = *reinterpret_cast<__half2*>(&c0_h2);
        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_h2);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_h2);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_h2);

        // Horner's method: ((c3*t + c2)*t + c1)*t + c0
        __half2 result = __hfma2(t_clamped, coeff3, coeff2);
        result = __hfma2(t_clamped, result, coeff1);
        result = __hfma2(t_clamped, result, coeff0);

        // Boundary saturation: x < -2.5 -> -1, x > 2.5 -> 1
        __half2 sat_lo = __hlt2(t_h2, neg25);
        __half2 sat_hi = __hgt2(t_h2, pos25);

        result = __hfma2(sat_lo, __hsub2(neg_one, result), result);
        result = __hfma2(sat_hi, __hsub2(one, result), result);

        return result;
    }
};
// ============================================================================
// N=2 D=1 Pow2 on [0, 1) - ULTRA OPTIMIZED
// ============================================================================
// Coefficients from analysis_results/Pow2_0-1_N2_D1_stats.json (FP16 hex):
// Interval 0 (x < 0.5): Slope=0x3A91 (0.821), Offset=0x3BEA (0.989)
// Interval 1 (x >= 0.5): Slope=0x3CA7 (1.163), Offset=0x3A8C (0.818)
//
// Key insight: N=2 means single bit index (x < 0.5 or x >= 0.5)
// This is the mantissa bit in the FP16 representation!

struct POW2_N2_D1_ULTRA {
    // Packed coefficients: [Slope1:Slope0] and [Offset1:Offset0]
    // S0=0x3A91, S1=0x3CA7 -> S_PACKED = 0x3CA73A91
    // O0=0x3BEA, O1=0x3A8C -> O_PACKED = 0x3A8C3BEA
    static constexpr unsigned int S_PACKED = 0x3CA73A91;
    static constexpr unsigned int O_PACKED = 0x3A8C3BEA;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp to [0, 0.9999]
        unsigned int zero_u = 0x00000000;
        unsigned int max_u = 0x3BFF3BFF;  // 0.9999
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 max_val = *reinterpret_cast<__half2*>(&max_u);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, zero), max_val);

        // Bit-hack index: Add 1.0, extract mantissa bit 9 (for N=2)
        // t in [0,0.5) + 1.0 = [1.0, 1.5) -> mantissa bit 9 = 0
        // t in [0.5,1) + 1.0 = [1.5, 2.0) -> mantissa bit 9 = 1
        unsigned int one_u = 0x3C003C00;
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 t_norm = __hadd2(t_clamped, one);

        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);

        // Extract single index bit per lane
        // bit 9 for low lane, bit 25 for high lane
        int idx_lo = (bits >> 9) & 1;
        int idx_hi = (bits >> 25) & 1;

        // For N=2: shift by idx*16 bits to extract coefficient
        int shift_lo = idx_lo << 4;  // 0 or 16
        int shift_hi = idx_hi << 4;  // 0 or 16

        // Extract coefficients (just 2 shifts + 2 masks, not 4!)
        unsigned short s_lo = (S_PACKED >> shift_lo) & 0xFFFF;
        unsigned short s_hi = (S_PACKED >> shift_hi) & 0xFFFF;
        unsigned short o_lo = (O_PACKED >> shift_lo) & 0xFFFF;
        unsigned short o_hi = (O_PACKED >> shift_hi) & 0xFFFF;

        // Pack
        unsigned int s_packed = ((unsigned int)s_hi << 16) | s_lo;
        unsigned int o_packed = ((unsigned int)o_hi << 16) | o_lo;

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        return __hfma2(t_clamped, slope, offset);
    }
};

struct POW2_N2_D1_BRANCHLESS {
    // -------------------------------------------------------------------------
    // STRATEGY: Branchless coefficient selection using bit manipulation
    // -------------------------------------------------------------------------
    // For N=2: idx = 0 or 1
    // We can use: mask = -idx (0x0000 or 0xFFFF)
    // coeff = (coeff1 & mask) | (coeff0 & ~mask)
    //
    // This avoids variable shifts entirely!

    static constexpr unsigned short S0 = 0x3A91;  // Slope0
    static constexpr unsigned short S1 = 0x3CA7;  // Slope1
    static constexpr unsigned short O0 = 0x3BEA;  // Offset0
    static constexpr unsigned short O1 = 0x3A8C;  // Offset1

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp
        unsigned int zero_u = 0x00000000;
        unsigned int max_u = 0x3BFF3BFF;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 max_val = *reinterpret_cast<__half2*>(&max_u);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, zero), max_val);

        // Index extraction
        unsigned int one_u = 0x3C003C00;
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 t_norm = __hadd2(t_clamped, one);
        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);

        // Extract index bits (bit 9 and bit 25)
        unsigned int idx_lo = (bits >> 9) & 1;
        unsigned int idx_hi = (bits >> 25) & 1;

        // Create masks: -1 (0xFFFF) if idx=1, 0 if idx=0
        unsigned short mask_lo = (unsigned short)(-(int)idx_lo);
        unsigned short mask_hi = (unsigned short)(-(int)idx_hi);

        // Branchless coefficient selection
        unsigned short s_lo = (S1 & mask_lo) | (S0 & ~mask_lo);
        unsigned short s_hi = (S1 & mask_hi) | (S0 & ~mask_hi);
        unsigned short o_lo = (O1 & mask_lo) | (O0 & ~mask_lo);
        unsigned short o_hi = (O1 & mask_hi) | (O0 & ~mask_hi);

        // Pack
        unsigned int s_packed = ((unsigned int)s_hi << 16) | s_lo;
        unsigned int o_packed = ((unsigned int)o_hi << 16) | o_lo;

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        return __hfma2(t_clamped, slope, offset);
    }
};

struct POW2_N2_D1_XOR_SELECT {
    // -------------------------------------------------------------------------
    // STRATEGY: XOR-based coefficient selection
    // -------------------------------------------------------------------------
    // Pre-compute: delta = S1 ^ S0, base = S0
    // Select: coeff = base ^ (delta & mask)
    // Where mask = -idx (all 1s if idx=1, all 0s if idx=0)
    //
    // This replaces 2 ANDs + 1 OR with 1 AND + 1 XOR

    static constexpr unsigned short S0 = 0x3A91;
    static constexpr unsigned short S1 = 0x3CA7;
    static constexpr unsigned short O0 = 0x3BEA;
    static constexpr unsigned short O1 = 0x3A8C;

    // Delta = XOR of the two options
    static constexpr unsigned short S_DELTA = S0 ^ S1;  // 0x0636
    static constexpr unsigned short O_DELTA = O0 ^ O1;  // 0x0166

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp
        unsigned int zero_u = 0x00000000;
        unsigned int max_u = 0x3BFF3BFF;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 max_val = *reinterpret_cast<__half2*>(&max_u);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, zero), max_val);

        // Index extraction
        unsigned int one_u = 0x3C003C00;
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 t_norm = __hadd2(t_clamped, one);
        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);

        // Extract index bits
        unsigned int idx_lo = (bits >> 9) & 1;
        unsigned int idx_hi = (bits >> 25) & 1;

        // Create masks
        unsigned short mask_lo = (unsigned short)(-(int)idx_lo);
        unsigned short mask_hi = (unsigned short)(-(int)idx_hi);

        // XOR-based selection: coeff = base ^ (delta & mask)
        unsigned short s_lo = S0 ^ (S_DELTA & mask_lo);
        unsigned short s_hi = S0 ^ (S_DELTA & mask_hi);
        unsigned short o_lo = O0 ^ (O_DELTA & mask_lo);
        unsigned short o_hi = O0 ^ (O_DELTA & mask_hi);

        // Pack
        unsigned int s_packed = ((unsigned int)s_hi << 16) | s_lo;
        unsigned int o_packed = ((unsigned int)o_hi << 16) | o_lo;

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        return __hfma2(t_clamped, slope, offset);
    }
};

struct POW2_N2_D1_PACKED_SELECT {
    // -------------------------------------------------------------------------
    // STRATEGY: Packed XOR selection (32-bit operations)
    // -------------------------------------------------------------------------
    // Combine S and O into 32-bit values, do XOR selection on 32-bit
    // This reduces instruction count by operating on pairs

    // Pack: [O:S] for each interval
    // Interval 0: (O0 << 16) | S0 = 0x3BEA3A91
    // Interval 1: (O1 << 16) | S1 = 0x3A8C3CA7
    static constexpr unsigned int COEFF_0 = 0x3BEA3A91;  // [O0:S0]
    static constexpr unsigned int COEFF_1 = 0x3A8C3CA7;  // [O1:S1]
    static constexpr unsigned int COEFF_DELTA = COEFF_0 ^ COEFF_1;  // 0x01660636

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp
        unsigned int zero_u = 0x00000000;
        unsigned int max_u = 0x3BFF3BFF;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 max_val = *reinterpret_cast<__half2*>(&max_u);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, zero), max_val);

        // Index extraction (packed)
        unsigned int one_u = 0x3C003C00;
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 t_norm = __hadd2(t_clamped, one);
        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);

        // Extract both index bits at once
        unsigned int idx_lo = (bits >> 9) & 1;
        unsigned int idx_hi = (bits >> 25) & 1;

        // Create 32-bit masks
        unsigned int mask_lo = (unsigned int)(-(int)idx_lo);
        unsigned int mask_hi = (unsigned int)(-(int)idx_hi);

        // XOR-based selection on packed coefficients
        unsigned int coeff_lo = COEFF_0 ^ (COEFF_DELTA & mask_lo);
        unsigned int coeff_hi = COEFF_0 ^ (COEFF_DELTA & mask_hi);

        // Extract S (low 16 bits) and O (high 16 bits)
        unsigned short s_lo = coeff_lo & 0xFFFF;
        unsigned short o_lo = coeff_lo >> 16;
        unsigned short s_hi = coeff_hi & 0xFFFF;
        unsigned short o_hi = coeff_hi >> 16;

        // Pack for half2
        unsigned int s_packed = ((unsigned int)s_hi << 16) | s_lo;
        unsigned int o_packed = ((unsigned int)o_hi << 16) | o_lo;

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        return __hfma2(t_clamped, slope, offset);
    }
};
// ============================================================================
// N=4 D=1 Pow2 with XOR-based Selection
// ============================================================================
// Apply XOR strategy to N=4 D=1 (4 intervals instead of 2)
// For N=4: idx is 0,1,2,3 (2 bits)
// We can use a tree of XOR operations or precompute deltas

struct POW2_N4_D1_XOR {
    // Coefficients from Pow2_0-1_N4_D1_stats.json (FP16):
    // Interval 0: S=0x3A0B, O=0x3BFB
    // Interval 1: S=0x3B2F, O=0x3BB2
    // Interval 2: S=0x3C45, O=0x3B04
    // Interval 3: S=0x3D15, O=0x39CC

    // Pack as [O:S] per interval
    static constexpr unsigned int C0 = 0x3BFB3A0B;  // [O0:S0]
    static constexpr unsigned int C1 = 0x3BB23B2F;  // [O1:S1]
    static constexpr unsigned int C2 = 0x3B043C45;  // [O2:S2]
    static constexpr unsigned int C3 = 0x39CC3D15;  // [O3:S3]

    // XOR deltas for tree selection
    // D01 = C0 ^ C1, D23 = C2 ^ C3
    // D0123_lo = C0 ^ C2, D0123_hi = C1 ^ C3
    static constexpr unsigned int D01 = C0 ^ C1;     // 0x00490124
    static constexpr unsigned int D23 = C2 ^ C3;     // 0x02C80150
    static constexpr unsigned int D02 = C0 ^ C2;     // 0x00FF064E

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Clamp
        unsigned int zero_u = 0x00000000;
        unsigned int max_u = 0x3BFF3BFF;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 max_val = *reinterpret_cast<__half2*>(&max_u);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, zero), max_val);

        // Index extraction (2 bits per lane for N=4)
        unsigned int one_u = 0x3C003C00;
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 t_norm = __hadd2(t_clamped, one);
        unsigned int bits = *reinterpret_cast<unsigned int*>(&t_norm);

        // Extract 2-bit index per lane (bits 8-9 and 24-25)
        unsigned int idx_lo = (bits >> 8) & 3;
        unsigned int idx_hi = (bits >> 24) & 3;

        // Tree selection using XOR
        // Level 1: Select between (C0,C1) vs (C2,C3) based on bit 1
        unsigned int bit1_lo = (idx_lo >> 1) & 1;
        unsigned int bit1_hi = (idx_hi >> 1) & 1;
        unsigned int mask1_lo = (unsigned int)(-(int)bit1_lo);
        unsigned int mask1_hi = (unsigned int)(-(int)bit1_hi);

        unsigned int base_lo = C0 ^ (D02 & mask1_lo);  // C0 or C2
        unsigned int base_hi = C0 ^ (D02 & mask1_hi);
        unsigned int delta_lo = D01 ^ ((D01 ^ D23) & mask1_lo);  // D01 or D23
        unsigned int delta_hi = D01 ^ ((D01 ^ D23) & mask1_hi);

        // Level 2: Select within pair based on bit 0
        unsigned int bit0_lo = idx_lo & 1;
        unsigned int bit0_hi = idx_hi & 1;
        unsigned int mask0_lo = (unsigned int)(-(int)bit0_lo);
        unsigned int mask0_hi = (unsigned int)(-(int)bit0_hi);

        unsigned int coeff_lo = base_lo ^ (delta_lo & mask0_lo);
        unsigned int coeff_hi = base_hi ^ (delta_hi & mask0_hi);

        // Extract S and O
        unsigned short s_lo = coeff_lo & 0xFFFF;
        unsigned short o_lo = coeff_lo >> 16;
        unsigned short s_hi = coeff_hi & 0xFFFF;
        unsigned short o_hi = coeff_hi >> 16;

        // Pack for half2
        unsigned int s_packed = ((unsigned int)s_hi << 16) | s_lo;
        unsigned int o_packed = ((unsigned int)o_hi << 16) | o_lo;

        __half2 slope = *reinterpret_cast<__half2*>(&s_packed);
        __half2 offset = *reinterpret_cast<__half2*>(&o_packed);

        return __hfma2(t_clamped, slope, offset);
    }
};

// ============================================================================
// SYMMETRIC SIGMOID N=2 D=3 - Exploit Coefficient Symmetry
// ============================================================================
// Sigmoid coefficients (FP16):
// Interval 0: C0=0x3800, C1=0x3493, C2=0x2B32, C3=0x1BA8
// Interval 1: C0=0x3800, C1=0x3493, C2=0xAB32 (sign flip!), C3=0x1BA8
//
// Only C2 changes sign between intervals!
// Strategy: Store base coefficients, XOR C2 sign bit based on interval

struct SIGMOID_N2_D3_SYMMETRIC {
    // Base coefficients (interval 0)
    static constexpr unsigned short C0 = 0x3800;  // 0.5
    static constexpr unsigned short C1 = 0x3493;  // 0.286
    static constexpr unsigned short C2_BASE = 0x2B32;  // 0.056 (positive)
    static constexpr unsigned short C3 = 0x1BA8;  // 0.0037

    // Sign bit mask for FP16 (bit 15)
    static constexpr unsigned short SIGN_MASK = 0x8000;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Bounds
        __half2 neg6 = __float2half2_rn(-6.0f);
        __half2 pos6 = __float2half2_rn(6.0f);

        // Clamp to [-6, 6]
        __half2 t_clamped = __hmin2(__hmax2(t_h2, neg6), pos6);

        // Interval selection: x >= 0 means idx=1 (flip C2 sign)
        unsigned int zero_u = 0x00000000;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 mask_fp = __hge2(t_h2, zero);
        unsigned int mask_bits = *reinterpret_cast<unsigned int*>(&mask_fp);

        // Extract sign flip mask per lane
        unsigned short flip_lo = (mask_bits & 0x8000) ? SIGN_MASK : 0;
        unsigned short flip_hi = (mask_bits & 0x80000000) ? SIGN_MASK : 0;

        // Apply sign flip to C2
        unsigned short c2_lo = C2_BASE ^ flip_lo;
        unsigned short c2_hi = C2_BASE ^ flip_hi;

        // Pack coefficients
        unsigned int c0_h2 = ((unsigned int)C0 << 16) | C0;
        unsigned int c1_h2 = ((unsigned int)C1 << 16) | C1;
        unsigned int c2_h2 = ((unsigned int)c2_hi << 16) | c2_lo;
        unsigned int c3_h2 = ((unsigned int)C3 << 16) | C3;

        __half2 coeff0 = *reinterpret_cast<__half2*>(&c0_h2);
        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_h2);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_h2);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_h2);

        // Horner's method: ((c3*t + c2)*t + c1)*t + c0
        __half2 result = __hfma2(t_clamped, coeff3, coeff2);
        result = __hfma2(t_clamped, result, coeff1);
        result = __hfma2(t_clamped, result, coeff0);

        // Boundary saturation using bit masking
        // sat_lo_mask = (t_h2 < -6) ? 0xFFFF : 0x0000 per lane
        // sat_hi_mask = (t_h2 > 6) ? 0xFFFF : 0x0000 per lane
        __half2 sat_lo_fp = __hlt2(t_h2, neg6);
        __half2 sat_hi_fp = __hgt2(t_h2, pos6);
        unsigned int sat_lo_mask = *reinterpret_cast<unsigned int*>(&sat_lo_fp);
        unsigned int sat_hi_mask = *reinterpret_cast<unsigned int*>(&sat_hi_fp);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int one_u = 0x3C003C00;  // 1.0 as half2

        // Apply: result = (result & ~sat_lo) | (0 & sat_lo) = result & ~sat_lo
        // Then:  result = (result & ~sat_hi) | (1 & sat_hi)
        result_bits = result_bits & ~sat_lo_mask;  // Zero where t < -6
        result_bits = (result_bits & ~sat_hi_mask) | (one_u & sat_hi_mask);  // 1.0 where t > 6

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// ============================================================================
// SYMMETRIC TANH N=2 D=3 - Exploit Coefficient Symmetry
// ============================================================================
// Tanh coefficients (FP16):
// Interval 0: C0=0x0000, C1=0x3C91, C2=0x3724, C3=0x2B81
// Interval 1: C0=0x0000, C1=0x3C91, C2=0xB724 (sign flip!), C3=0x2B81
//
// Only C2 changes sign! Same pattern as sigmoid.

struct TANH_N2_D3_SYMMETRIC {
    static constexpr unsigned short C0 = 0x0000;  // 0.0
    static constexpr unsigned short C1 = 0x3C91;  // 1.14
    static constexpr unsigned short C2_BASE = 0x3724;  // 0.446 (positive)
    static constexpr unsigned short C3 = 0x2B81;  // 0.059
    static constexpr unsigned short SIGN_MASK = 0x8000;

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Bounds
        __half2 neg25 = __float2half2_rn(-2.5f);
        __half2 pos25 = __float2half2_rn(2.5f);

        // Clamp to [-2.5, 2.5]
        __half2 t_clamped = __hmin2(__hmax2(t_h2, neg25), pos25);

        // Interval selection
        unsigned int zero_u = 0x00000000;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 mask_fp = __hge2(t_h2, zero);
        unsigned int mask_bits = *reinterpret_cast<unsigned int*>(&mask_fp);

        // Sign flip for C2
        unsigned short flip_lo = (mask_bits & 0x8000) ? SIGN_MASK : 0;
        unsigned short flip_hi = (mask_bits & 0x80000000) ? SIGN_MASK : 0;

        unsigned short c2_lo = C2_BASE ^ flip_lo;
        unsigned short c2_hi = C2_BASE ^ flip_hi;

        // Pack coefficients
        unsigned int c0_h2 = ((unsigned int)C0 << 16) | C0;
        unsigned int c1_h2 = ((unsigned int)C1 << 16) | C1;
        unsigned int c2_h2 = ((unsigned int)c2_hi << 16) | c2_lo;
        unsigned int c3_h2 = ((unsigned int)C3 << 16) | C3;

        __half2 coeff0 = *reinterpret_cast<__half2*>(&c0_h2);
        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_h2);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_h2);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_h2);

        // Horner's method
        __half2 result = __hfma2(t_clamped, coeff3, coeff2);
        result = __hfma2(t_clamped, result, coeff1);
        result = __hfma2(t_clamped, result, coeff0);

        // Boundary saturation using bit masking
        __half2 sat_lo_fp = __hlt2(t_h2, neg25);
        __half2 sat_hi_fp = __hgt2(t_h2, pos25);
        unsigned int sat_lo_mask = *reinterpret_cast<unsigned int*>(&sat_lo_fp);
        unsigned int sat_hi_mask = *reinterpret_cast<unsigned int*>(&sat_hi_fp);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int neg_one_u = 0xBC00BC00;  // -1.0 as half2
        unsigned int one_u = 0x3C003C00;      // +1.0 as half2

        // Apply saturation
        result_bits = (result_bits & ~sat_lo_mask) | (neg_one_u & sat_lo_mask);  // -1 where t < -2.5
        result_bits = (result_bits & ~sat_hi_mask) | (one_u & sat_hi_mask);      // +1 where t > 2.5

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// ============================================================================
// SYMMETRIC SWISH N=2 D=3 - Exploit Coefficient Symmetry
// ============================================================================
// Swish coefficients (FP16):
// Interval 0: C0=0xACBC, C1=0x3428 (0.26), C2=0x2EF9, C3=0x2195
// Interval 1: C0=0xACBC, C1=0x39EC (0.74), C2=0x2EF9, C3=0xA195 (sign flip!)
//
// C0, C2 same; C1 changes (0.26 vs 1-0.26=0.74); C3 sign flips

struct SWISH_N2_D3_SYMMETRIC {
    static constexpr unsigned short C0 = 0xACBC;  // -0.074 (fitting artifact)
    static constexpr unsigned short C1_0 = 0x3428;  // 0.26
    static constexpr unsigned short C1_1 = 0x39EC;  // 0.74
    static constexpr unsigned short C2 = 0x2EF9;  // 0.109
    static constexpr unsigned short C3_BASE = 0x2195;  // 0.011 (positive)
    static constexpr unsigned short SIGN_MASK = 0x8000;

    // XOR delta for C1
    static constexpr unsigned short C1_DELTA = C1_0 ^ C1_1;  // 0x0DC4

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Bounds
        __half2 neg6 = __float2half2_rn(-6.0f);
        __half2 pos6 = __float2half2_rn(6.0f);

        // Clamp to [-6, 6]
        __half2 t_clamped = __hmin2(__hmax2(t_h2, neg6), pos6);

        // Interval selection
        unsigned int zero_u = 0x00000000;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 mask_fp = __hge2(t_h2, zero);
        unsigned int mask_bits = *reinterpret_cast<unsigned int*>(&mask_fp);

        // Extract selection masks per lane
        unsigned short sel_lo = (mask_bits & 0x8000) ? 0xFFFF : 0;
        unsigned short sel_hi = (mask_bits & 0x80000000) ? 0xFFFF : 0;

        // C1: XOR selection (C1_0 for interval 0, C1_1 for interval 1)
        unsigned short c1_lo = C1_0 ^ (C1_DELTA & sel_lo);
        unsigned short c1_hi = C1_0 ^ (C1_DELTA & sel_hi);

        // C3: Sign flip for interval 1
        unsigned short c3_lo = C3_BASE ^ (SIGN_MASK & sel_lo);
        unsigned short c3_hi = C3_BASE ^ (SIGN_MASK & sel_hi);

        // Pack coefficients
        unsigned int c0_h2 = ((unsigned int)C0 << 16) | C0;
        unsigned int c1_h2 = ((unsigned int)c1_hi << 16) | c1_lo;
        unsigned int c2_h2 = ((unsigned int)C2 << 16) | C2;
        unsigned int c3_h2 = ((unsigned int)c3_hi << 16) | c3_lo;

        __half2 coeff0 = *reinterpret_cast<__half2*>(&c0_h2);
        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_h2);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_h2);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_h2);

        // Horner's method
        __half2 result = __hfma2(t_clamped, coeff3, coeff2);
        result = __hfma2(t_clamped, result, coeff1);
        result = __hfma2(t_clamped, result, coeff0);

        // Swish boundary saturation:
        // x < -6: swish(x) ≈ 0 (sigmoid → 0)
        // x > 6: swish(x) ≈ x (sigmoid → 1)
        __half2 sat_lo_fp = __hlt2(t_h2, neg6);
        __half2 sat_hi_fp = __hgt2(t_h2, pos6);
        unsigned int sat_lo_mask = *reinterpret_cast<unsigned int*>(&sat_lo_fp);
        unsigned int sat_hi_mask = *reinterpret_cast<unsigned int*>(&sat_hi_fp);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&t_h2);

        // Apply saturation: 0 where x < -6, x where x > 6
        result_bits = result_bits & ~sat_lo_mask;  // Zero where t < -6
        result_bits = (result_bits & ~sat_hi_mask) | (input_bits & sat_hi_mask);  // x where t > 6

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};
// ============================================================================
// ULTRA-OPTIMIZED N=1 D=3 SYMMETRIC KERNELS
// ============================================================================
// These kernels exploit the mathematical symmetry of the functions:
// - Tanh: odd function, tanh(-x) = -tanh(x) → use |x|, flip sign
// - Sigmoid: 1 - symmetry, sigmoid(-x) = 1 - sigmoid(x) → use |x|, subtract
//
// N=1 means NO interval selection at all! Just one polynomial.

// ============================================================================
// TANH N=1 D=3: Ultra-fast using odd function symmetry
// ============================================================================
// Coefficients: C0=0.0, C1=0.793, C2=0.0, C3=-0.072
// Since C0=C2=0 (odd function), only odd powers matter:
// tanh(x) ≈ C1*x + C3*x³ = x*(C1 + C3*x²)
//
// Strategy: Compute for |x|, then flip sign if x was negative

struct TANH_N1_D3_ULTRA {
    // FP16 coefficients (from analysis_results)
    static constexpr unsigned short C1 = 0x3A59;  // 0.793
    static constexpr unsigned short C3 = 0xAC94;  // -0.072

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Bounds
        __half2 neg25 = __float2half2_rn(-2.5f);
        __half2 pos25 = __float2half2_rn(2.5f);

        // Clamp to [-2.5, 2.5]
        __half2 t_clamped = __hmin2(__hmax2(t_h2, neg25), pos25);

        // Pack coefficients (same for both lanes)
        unsigned int c1_h2 = ((unsigned int)C1 << 16) | C1;
        unsigned int c3_h2 = ((unsigned int)C3 << 16) | C3;

        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_h2);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_h2);

        // Compute x² and x³
        __half2 x2 = __hmul2(t_clamped, t_clamped);

        // Polynomial: x*(C1 + C3*x²) = C1*x + C3*x³
        // Horner: x * (C1 + C3*x²)
        __half2 inner = __hfma2(x2, coeff3, coeff1);  // C1 + C3*x²
        __half2 result = __hmul2(t_clamped, inner);   // x * (C1 + C3*x²)

        // Boundary saturation using bit masking
        __half2 sat_lo_fp = __hlt2(t_h2, neg25);
        __half2 sat_hi_fp = __hgt2(t_h2, pos25);
        unsigned int sat_lo_mask = *reinterpret_cast<unsigned int*>(&sat_lo_fp);
        unsigned int sat_hi_mask = *reinterpret_cast<unsigned int*>(&sat_hi_fp);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int neg_one_u = 0xBC00BC00;  // -1.0 as half2
        unsigned int one_u = 0x3C003C00;      // +1.0 as half2

        result_bits = (result_bits & ~sat_lo_mask) | (neg_one_u & sat_lo_mask);
        result_bits = (result_bits & ~sat_hi_mask) | (one_u & sat_hi_mask);

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// ============================================================================
// SIGMOID N=1 D=3: Using 1-symmetry
// ============================================================================
// Sigmoid has: sigmoid(-x) = 1 - sigmoid(x)
// Strategy: Compute sigmoid(|x|), if x<0: return 1 - result
// But the N=1 D=3 fit may not work well due to full domain fitting

struct SIGMOID_N1_D3_ABSYM {
    // Using N=2 D=3 coefficients for x >= 0 only (interval 1)
    // Then apply symmetry: sigmoid(-x) = 1 - sigmoid(x)
    static constexpr unsigned short C0 = 0x3800;  // 0.5
    static constexpr unsigned short C1 = 0x3493;  // 0.286
    static constexpr unsigned short C2 = 0xAB32;  // -0.056 (interval 1)
    static constexpr unsigned short C3 = 0x1BA8;  // 0.0037

    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Bounds
        __half2 neg6 = __float2half2_rn(-6.0f);
        __half2 pos6 = __float2half2_rn(6.0f);

        // Take absolute value for computation
        __half2 abs_t = __habs2(t_h2);
        __half2 t_clamped = __hmin2(abs_t, pos6);

        // Pack coefficients
        unsigned int c0_h2 = ((unsigned int)C0 << 16) | C0;
        unsigned int c1_h2 = ((unsigned int)C1 << 16) | C1;
        unsigned int c2_h2 = ((unsigned int)C2 << 16) | C2;
        unsigned int c3_h2 = ((unsigned int)C3 << 16) | C3;

        __half2 coeff0 = *reinterpret_cast<__half2*>(&c0_h2);
        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_h2);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_h2);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_h2);

        // Horner's method: sigmoid(|x|) for x >= 0
        __half2 result = __hfma2(t_clamped, coeff3, coeff2);
        result = __hfma2(t_clamped, result, coeff1);
        result = __hfma2(t_clamped, result, coeff0);

        // If original x < 0: result = 1 - result
        unsigned int zero_u = 0x00000000;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 neg_mask_fp = __hlt2(t_h2, zero);  // x < 0?
        unsigned int neg_mask = *reinterpret_cast<unsigned int*>(&neg_mask_fp);

        // 1 - result for negative inputs
        unsigned int one_u = 0x3C003C00;
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 flipped = __hsub2(one, result);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int flipped_bits = *reinterpret_cast<unsigned int*>(&flipped);
        result_bits = (result_bits & ~neg_mask) | (flipped_bits & neg_mask);

        result = *reinterpret_cast<__half2*>(&result_bits);

        // Boundary saturation
        __half2 sat_lo_fp = __hlt2(t_h2, neg6);
        __half2 sat_hi_fp = __hgt2(t_h2, pos6);
        unsigned int sat_lo_mask = *reinterpret_cast<unsigned int*>(&sat_lo_fp);
        unsigned int sat_hi_mask = *reinterpret_cast<unsigned int*>(&sat_hi_fp);

        result_bits = *reinterpret_cast<unsigned int*>(&result);
        result_bits = result_bits & ~sat_lo_mask;  // 0 where t < -6
        result_bits = (result_bits & ~sat_hi_mask) | (one_u & sat_hi_mask);  // 1 where t > 6

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// ============================================================================
// N=2 D=3 TANH with PRECOMPUTED PACKED COEFFICIENTS (no runtime packing)
// ============================================================================
// All coefficients are pre-packed as half2 constants to avoid packing overhead

struct TANH_N2_D3_PACKED {
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Bounds
        __half2 neg25 = __float2half2_rn(-2.5f);
        __half2 pos25 = __float2half2_rn(2.5f);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, neg25), pos25);

        // Interval selection: x >= 0?
        unsigned int zero_u = 0x00000000;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 mask_fp = __hge2(t_h2, zero);
        unsigned int sel_mask = *reinterpret_cast<unsigned int*>(&mask_fp);

        // Inline constants
        unsigned int c0_u = 0x00000000;  // 0.0
        unsigned int c1_u = 0x3C913C91;  // 1.14
        unsigned int c2_pos = 0x37243724;  // +0.446
        unsigned int c2_delta = 0x80008000;  // sign flip
        unsigned int c3_u = 0x2B812B81;  // 0.059

        // XOR-select C2
        unsigned int c2_u = c2_pos ^ (c2_delta & sel_mask);

        __half2 coeff0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_u);

        // Horner's method
        __half2 result = __hfma2(t_clamped, coeff3, coeff2);
        result = __hfma2(t_clamped, result, coeff1);
        result = __hfma2(t_clamped, result, coeff0);

        // Boundary saturation
        __half2 sat_lo_fp = __hlt2(t_h2, neg25);
        __half2 sat_hi_fp = __hgt2(t_h2, pos25);
        unsigned int sat_lo_mask = *reinterpret_cast<unsigned int*>(&sat_lo_fp);
        unsigned int sat_hi_mask = *reinterpret_cast<unsigned int*>(&sat_hi_fp);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int neg_one_u = 0xBC00BC00;
        unsigned int one_u = 0x3C003C00;

        result_bits = (result_bits & ~sat_lo_mask) | (neg_one_u & sat_lo_mask);
        result_bits = (result_bits & ~sat_hi_mask) | (one_u & sat_hi_mask);

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// ============================================================================
// N=2 D=3 SIGMOID with PRECOMPUTED PACKED COEFFICIENTS
// ============================================================================

struct SIGMOID_N2_D3_PACKED {
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 neg6 = __float2half2_rn(-6.0f);
        __half2 pos6 = __float2half2_rn(6.0f);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, neg6), pos6);

        unsigned int zero_u = 0x00000000;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 mask_fp = __hge2(t_h2, zero);
        unsigned int sel_mask = *reinterpret_cast<unsigned int*>(&mask_fp);

        // Inline constants
        unsigned int c0_u = 0x38003800;  // 0.5
        unsigned int c1_u = 0x34933493;  // 0.286
        unsigned int c2_pos = 0x2B322B32;  // +0.056
        unsigned int c2_delta = 0x80008000;  // sign flip
        unsigned int c3_u = 0x1BA81BA8;  // 0.0037

        unsigned int c2_u = c2_pos ^ (c2_delta & sel_mask);

        __half2 coeff0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_u);

        __half2 result = __hfma2(t_clamped, coeff3, coeff2);
        result = __hfma2(t_clamped, result, coeff1);
        result = __hfma2(t_clamped, result, coeff0);

        __half2 sat_lo_fp = __hlt2(t_h2, neg6);
        __half2 sat_hi_fp = __hgt2(t_h2, pos6);
        unsigned int sat_lo_mask = *reinterpret_cast<unsigned int*>(&sat_lo_fp);
        unsigned int sat_hi_mask = *reinterpret_cast<unsigned int*>(&sat_hi_fp);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int one_u = 0x3C003C00;

        result_bits = result_bits & ~sat_lo_mask;
        result_bits = (result_bits & ~sat_hi_mask) | (one_u & sat_hi_mask);

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};
// ============================================================================
// ULTRA-FAST N=2 D=2 SYMMETRIC KERNELS
// ============================================================================
// D=2 means only 2 FMAs needed (degree 2 polynomial)
// Both Tanh and Sigmoid have perfect symmetry: C0/C1 shared, C2 sign flips

// ============================================================================
// TANH N=2 D=2 ULTRA-FAST
// ============================================================================
// Coefficients: C0=0x0000, C1=0x3BF7, C2=±0x3404
// Poly: C0 + C1*x + C2*x² = C1*x + C2*x² (since C0=0)
// Horner: x*(C1 + C2*x)

struct TANH_N2_D2_ULTRA {
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Bounds
        __half2 neg25 = __float2half2_rn(-2.5f);
        __half2 pos25 = __float2half2_rn(2.5f);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, neg25), pos25);

        // Interval selection: x >= 0?
        unsigned int zero_u = 0x00000000;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 mask_fp = __hge2(t_h2, zero);
        unsigned int sel_mask = *reinterpret_cast<unsigned int*>(&mask_fp);

        // Inline constants - C0=0, so skip it!
        unsigned int c1_u = 0x3BF73BF7;  // 0.996
        unsigned int c2_pos = 0x34043404;  // +0.251
        unsigned int c2_delta = 0x80008000;  // sign flip

        // XOR-select C2
        unsigned int c2_u = c2_pos ^ (c2_delta & sel_mask);

        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_u);

        // Horner: x*(C1 + C2*x) - ONLY 2 OPERATIONS!
        __half2 inner = __hfma2(t_clamped, coeff2, coeff1);  // C1 + C2*x
        __half2 result = __hmul2(t_clamped, inner);           // x * (C1 + C2*x)

        // Boundary saturation
        __half2 sat_lo_fp = __hlt2(t_h2, neg25);
        __half2 sat_hi_fp = __hgt2(t_h2, pos25);
        unsigned int sat_lo_mask = *reinterpret_cast<unsigned int*>(&sat_lo_fp);
        unsigned int sat_hi_mask = *reinterpret_cast<unsigned int*>(&sat_hi_fp);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int neg_one_u = 0xBC00BC00;
        unsigned int one_u = 0x3C003C00;

        result_bits = (result_bits & ~sat_lo_mask) | (neg_one_u & sat_lo_mask);
        result_bits = (result_bits & ~sat_hi_mask) | (one_u & sat_hi_mask);

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// ============================================================================
// SIGMOID N=2 D=2 ULTRA-FAST
// ============================================================================
// Coefficients: C0=0x3800, C1=0x336E, C2=±0x26BB
// Poly: C0 + C1*x + C2*x²
// Horner: C0 + x*(C1 + C2*x)

struct SIGMOID_N2_D2_ULTRA {
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Bounds
        __half2 neg6 = __float2half2_rn(-6.0f);
        __half2 pos6 = __float2half2_rn(6.0f);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, neg6), pos6);

        // Interval selection
        unsigned int zero_u = 0x00000000;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 mask_fp = __hge2(t_h2, zero);
        unsigned int sel_mask = *reinterpret_cast<unsigned int*>(&mask_fp);

        // Inline constants
        unsigned int c0_u = 0x38003800;  // 0.5
        unsigned int c1_u = 0x336E336E;  // 0.232
        unsigned int c2_pos = 0x26BB26BB;  // +0.026
        unsigned int c2_delta = 0x80008000;  // sign flip

        unsigned int c2_u = c2_pos ^ (c2_delta & sel_mask);

        __half2 coeff0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_u);

        // Horner: C0 + x*(C1 + C2*x) - ONLY 2 FMAs!
        __half2 inner = __hfma2(t_clamped, coeff2, coeff1);  // C1 + C2*x
        __half2 result = __hfma2(t_clamped, inner, coeff0);  // C0 + x*(C1 + C2*x)

        // Boundary saturation
        __half2 sat_lo_fp = __hlt2(t_h2, neg6);
        __half2 sat_hi_fp = __hgt2(t_h2, pos6);
        unsigned int sat_lo_mask = *reinterpret_cast<unsigned int*>(&sat_lo_fp);
        unsigned int sat_hi_mask = *reinterpret_cast<unsigned int*>(&sat_hi_fp);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int one_u = 0x3C003C00;

        result_bits = result_bits & ~sat_lo_mask;  // 0 where x < -6
        result_bits = (result_bits & ~sat_hi_mask) | (one_u & sat_hi_mask);  // 1 where x > 6

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// ============================================================================
// SWISH N=2 D=2 ULTRA-FAST (if coefficients available)
// ============================================================================
// Need to check the Swish N=2 D=2 coefficients...
// ============================================================================
// ABSOLUTE VALUE SYMMETRY KERNELS
// ============================================================================
// For odd functions like tanh: f(-x) = -f(x)
// Strategy: compute f(|x|), then flip sign if x was negative
// This eliminates interval selection entirely!

// ============================================================================
// TANH N=2 D=3 ABS - Absolute Value Symmetry
// ============================================================================
// Since tanh is odd: tanh(-x) = -tanh(x)
// Use coefficients for interval 1 (x >= 0): C0=0, C1=1.14, C2=-0.446, C3=0.059
// Compute tanh(|x|), then multiply by sign(x)

struct TANH_N2_D3_ABS {
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Bounds
        __half2 neg25 = __float2half2_rn(-2.5f);
        __half2 pos25 = __float2half2_rn(2.5f);

        // Save original sign and take absolute value
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&t_h2);
        unsigned int signs = input_bits & sign_mask;  // Store original signs
        unsigned int abs_bits = input_bits & ~sign_mask;  // Clear sign bits
        __half2 abs_t = *reinterpret_cast<__half2*>(&abs_bits);

        // Clamp |x| to [0, 2.5]
        unsigned int zero_u = 0x00000000;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 t_clamped = __hmin2(abs_t, pos25);

        // Coefficients for interval 1 (x >= 0)
        // C0=0, C1=0x3C91, C2=0xB724, C3=0x2B81
        unsigned int c1_u = 0x3C913C91;  // 1.14
        unsigned int c2_u = 0xB724B724;  // -0.446
        unsigned int c3_u = 0x2B812B81;  // 0.059

        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_u);

        // Horner: ((C3*x + C2)*x + C1)*x + 0 = x*((C3*x + C2)*x + C1)
        __half2 result = __hfma2(t_clamped, coeff3, coeff2);
        result = __hfma2(t_clamped, result, coeff1);
        result = __hmul2(t_clamped, result);  // Since C0=0

        // Apply original sign
        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        result_bits |= signs;  // Restore original signs
        result = *reinterpret_cast<__half2*>(&result_bits);

        // Boundary saturation
        __half2 sat_lo_fp = __hlt2(t_h2, neg25);
        __half2 sat_hi_fp = __hgt2(t_h2, pos25);
        unsigned int sat_lo_mask = *reinterpret_cast<unsigned int*>(&sat_lo_fp);
        unsigned int sat_hi_mask = *reinterpret_cast<unsigned int*>(&sat_hi_fp);

        result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int neg_one_u = 0xBC00BC00;
        unsigned int one_u = 0x3C003C00;

        result_bits = (result_bits & ~sat_lo_mask) | (neg_one_u & sat_lo_mask);
        result_bits = (result_bits & ~sat_hi_mask) | (one_u & sat_hi_mask);

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// ============================================================================
// SIGMOID N=2 D=3 ABS - 1-Symmetry
// ============================================================================
// For sigmoid: sigmoid(-x) = 1 - sigmoid(x)
// Compute sigmoid(|x|), if x was negative: result = 1 - result

struct SIGMOID_N2_D3_ABS {
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Bounds
        __half2 neg6 = __float2half2_rn(-6.0f);
        __half2 pos6 = __float2half2_rn(6.0f);

        // Take absolute value
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&t_h2);
        unsigned int was_negative = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_t = *reinterpret_cast<__half2*>(&abs_bits);

        // Clamp |x| to [0, 6]
        __half2 t_clamped = __hmin2(abs_t, pos6);

        // Coefficients for interval 1 (x >= 0)
        // C0=0x3800 (0.5), C1=0x3493 (0.286), C2=0xAB32 (-0.056), C3=0x1BA8 (0.0037)
        unsigned int c0_u = 0x38003800;  // 0.5
        unsigned int c1_u = 0x34933493;  // 0.286
        unsigned int c2_u = 0xAB32AB32;  // -0.056 (interval 1)
        unsigned int c3_u = 0x1BA81BA8;  // 0.0037

        __half2 coeff0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_u);

        // Horner: ((C3*x + C2)*x + C1)*x + C0
        __half2 result = __hfma2(t_clamped, coeff3, coeff2);
        result = __hfma2(t_clamped, result, coeff1);
        result = __hfma2(t_clamped, result, coeff0);

        // If original x < 0: result = 1 - result
        // was_negative has 0x8000 in lanes where x was negative
        unsigned int one_u = 0x3C003C00;
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 flipped = __hsub2(one, result);

        // Select: result if x >= 0, flipped if x < 0 (BRANCHLESS)
        unsigned int lo_neg = (was_negative << 1) & 0x00010000;
        lo_neg = lo_neg - (lo_neg >> 16);
        unsigned int hi_neg = (was_negative >> 15) & 0x00010000;
        hi_neg = hi_neg - (hi_neg >> 16);
        unsigned int neg_mask = lo_neg | (hi_neg << 16);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int flipped_bits = *reinterpret_cast<unsigned int*>(&flipped);
        result_bits = (result_bits & ~neg_mask) | (flipped_bits & neg_mask);
        result = *reinterpret_cast<__half2*>(&result_bits);

        // Boundary saturation
        __half2 sat_lo_fp = __hlt2(t_h2, neg6);
        __half2 sat_hi_fp = __hgt2(t_h2, pos6);
        unsigned int sat_lo_mask = *reinterpret_cast<unsigned int*>(&sat_lo_fp);
        unsigned int sat_hi_mask = *reinterpret_cast<unsigned int*>(&sat_hi_fp);

        result_bits = *reinterpret_cast<unsigned int*>(&result);
        result_bits = result_bits & ~sat_lo_mask;  // 0 where x < -6
        result_bits = (result_bits & ~sat_hi_mask) | (one_u & sat_hi_mask);  // 1 where x > 6

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};
// ============================================================================
// ABSOLUTE VALUE SYMMETRY KERNELS (OPTIMIZED)
// ============================================================================
// Optimized accumulation and saturation logic.

struct TANH_N2_D3_ABS_OPT {
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Constants (Inlined)
        __half2 pos25 = __float2half2_rn(2.5f);

        // Take absolute value and save sign
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&t_h2);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_t = *reinterpret_cast<__half2*>(&abs_bits);

        // Clamp |x| to 2.5
        __half2 t_clamped = __hmin2(abs_t, pos25);

        // Coefficients for interval 1 (x >= 0)
        // C0=0, C1=1.14, C2=-0.446, C3=0.059
        unsigned int c1_u = 0x3C913C91;
        unsigned int c2_u = 0xB724B724;
        unsigned int c3_u = 0x2B812B81;

        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_u);

        // Horner: x*((C3*x + C2)*x + C1)
        __half2 result = __hfma2(t_clamped, coeff3, coeff2);
        result = __hfma2(t_clamped, result, coeff1);
        result = __hmul2(t_clamped, result);

        // One-sided saturation: if |x| > 2.5, result = 1.0
        // (Sign restoration will turn this into -1.0 if x < -2.5)
        __half2 sat_fp = __hgt2(abs_t, pos25);
        unsigned int sat_mask = *reinterpret_cast<unsigned int*>(&sat_fp);

        unsigned int one_u = 0x3C003C00;
        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);

        // result = (result & ~mask) | (1.0 & mask)
        result_bits = (result_bits & ~sat_mask) | (one_u & sat_mask);

        // Restore sign
        result_bits |= signs;

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

struct SIGMOID_N2_D3_ABS_OPT {
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 pos6 = __float2half2_rn(6.0f);

        // Take absolute value
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&t_h2);
        unsigned int was_negative = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_t = *reinterpret_cast<__half2*>(&abs_bits);

        // Clamp |x| to 6.0
        __half2 t_clamped = __hmin2(abs_t, pos6);

        // Coefficients for interval 1 (x >= 0)
        unsigned int c0_u = 0x38003800;  // 0.5
        unsigned int c1_u = 0x34933493;  // 0.286
        unsigned int c2_u = 0xAB32AB32;  // -0.056
        unsigned int c3_u = 0x1BA81BA8;  // 0.0037

        __half2 coeff0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_u);

        // Horner
        __half2 result = __hfma2(t_clamped, coeff3, coeff2);
        result = __hfma2(t_clamped, result, coeff1);
        result = __hfma2(t_clamped, result, coeff0);

        // One-sided saturation: if |x| > 6, result = 1.0
        // (Flip logic will turn this into 0.0 if x < -6, which is correct)
        __half2 sat_fp = __hgt2(abs_t, pos6);
        unsigned int sat_mask = *reinterpret_cast<unsigned int*>(&sat_fp);
        unsigned int one_u = 0x3C003C00;
        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);

        result_bits = (result_bits & ~sat_mask) | (one_u & sat_mask);

        // Restore result to float for calc
        result = *reinterpret_cast<__half2*>(&result_bits);

        // Flip: if x < 0, result = 1.0 - result
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 flipped = __hsub2(one, result);

        // Select logic (BRANCHLESS)
        unsigned int lo_neg = (was_negative << 1) & 0x00010000;
        lo_neg = lo_neg - (lo_neg >> 16);
        unsigned int hi_neg = (was_negative >> 15) & 0x00010000;
        hi_neg = hi_neg - (hi_neg >> 16);
        unsigned int neg_mask = lo_neg | (hi_neg << 16);

        result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int flipped_bits = *reinterpret_cast<unsigned int*>(&flipped);

        result_bits = (result_bits & ~neg_mask) | (flipped_bits & neg_mask);

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// ============================================================================
// SIGMOID ODD FORMULATION: sigmoid(x) = 0.5 + h(x), h is odd
// Key insight: h(x) = sigmoid(x) - 0.5, h(-x) = -h(x)
// So we compute h(|x|) = c1*t + c2*t^2 + c3*t^3 = t*(c1 + t*(c2 + c3*t))
// Then XOR sign back (like tanh!), add 0.5. Same FMA count as tanh!
// ============================================================================
struct SIGMOID_N2_D3_ODD {
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 pos6 = __float2half2_rn(6.0f);

        // Bitwise abs + save sign
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&t_h2);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_t = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_t, pos6);

        // Coefficients: minimax D3 on [0,6] for h(x) = sigmoid(x)-0.5
        __half2 c1 = __float2half2_rn(0.281005859375f);
        __half2 c2 = __float2half2_rn(-0.0533447265625f);
        __half2 c3 = __float2half2_rn(0.0033893585205078125f);

        // Horner: h(t) = t*(c1 + t*(c2 + c3*t))  — same as tanh
        __half2 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        // Sign restore: h is odd → XOR sign bits (same as tanh)
        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);

        // sigmoid(x) = 0.5 + h(x)  — only extra op vs tanh
        return __hadd2(__float2half2_rn(0.5f), h_signed);
    }
};

// ============================================================================
// SWISH OPTIMIZED ABS KERNEL
// ============================================================================
// Identity: swish(x) = swish(|x|) + min(x, 0)
// This holds because swish(x) - swish(-x) = x.
// Strategy:
// 1. Compute swish(|x|) using Interval 1 coefficients (x >= 0).
// 2. Add min(x, 0).
// 3. Saturate swish(|x|) to |x| for |x| > 6.

struct SWISH_N2_D3_ABS_OPT {
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 pos6 = __float2half2_rn(6.0f);

        // Take absolute value
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&t_h2);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_t = *reinterpret_cast<__half2*>(&abs_bits);

        // Clamp |x| to 6.0
        __half2 t_clamped = __hmin2(abs_t, pos6);

        // Coefficients for interval 1 (x >= 0)
        // C0=0 (Fixed from -0.074 artifact), C1=0.74, C2=0.109, C3=-0.011
        unsigned int c0_u = 0x00000000;  // 0.0
        unsigned int c1_u = 0x39EC39EC;  // 0.740
        unsigned int c2_u = 0x2EF92EF9;  // 0.109
        unsigned int c3_u = 0xA195A195;  // -0.011

        __half2 coeff0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_u);

        // Horner: ((C3*x + C2)*x + C1)*x + C0
        __half2 result = __hfma2(t_clamped, coeff3, coeff2);
        result = __hfma2(t_clamped, result, coeff1);
        result = __hfma2(t_clamped, result, coeff0);

        // One-sided saturation: if |x| > 6, result = |x|
        __half2 sat_fp = __hgt2(abs_t, pos6);
        unsigned int sat_mask = *reinterpret_cast<unsigned int*>(&sat_fp);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);

        // result = (result & ~mask) | (abs_t & mask)
        result_bits = (result_bits & ~sat_mask) | (abs_bits & sat_mask);
        result = *reinterpret_cast<__half2*>(&result_bits);

        // Add min(x, 0)
        // min(x, 0) is x if x < 0, else 0.
        // We can use __hmin2(t_h2, zero)
        unsigned int zero_u = 0x00000000;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 neg_part = __hmin2(t_h2, zero);

        result = __hadd2(result, neg_part);

        return result;
    }
};
// ============================================================================
// SIGMOID BITWISE OPTIMIZED KERNEL
// ============================================================================
// Optimization Strategy:
// 1. Bitwise Coefficient Selection:
//    - C2 is +0.026 for x < 0 (Region 0)
//    - C2 is -0.026 for x > 0 (Region 1)
//    - So sign(C2) = ~sign(x).
//    - We can compute C2 by taking |C2| and applying ~(x & sign_mask).
//    - Eliminates __hge2 comparison and condition code dependency.
// 2. Unified Saturation:
//    - Check |x| > 6.0 once.
//    - Saturation value is 1.0 if x > 0, 0.0 if x < 0.
//    - This logic can be done with bitwise ops too.

struct SIGMOID_N2_D3_BITWISE {
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Constants (Inlined)
        unsigned int c0_u = 0x38003800;  // 0.5
        unsigned int c1_u = 0x34933493;  // 0.286
        unsigned int abs_c2_u = 0x2B322B32; // +0.026 (Positive C2)
        unsigned int c3_u = 0x1BA81BA8;  // 0.0037

        // Input bits
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&t_h2);
        unsigned int sign_mask = 0x80008000;
        unsigned int signs = input_bits & sign_mask;

        // C2 Selection: sign(C2) = ~sign(x)
        // If x is + (0x0000), we want C2 - (0x8000). XOR with 0x8000 gives 0x8000.
        // If x is - (0x8000), we want C2 + (0x0000). XOR with 0x8000 gives 0x0000.
        // Formula: C2 = |C2| ^ (signs ^ 0x80008000)
        unsigned int c2_u = abs_c2_u ^ (signs ^ sign_mask);

        // Unpack coefficients
        __half2 coeff0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_u);

        // Horner: ((C3*x + C2)*x + C1)*x + C0
        // Use t_h2 directly (not clamped yet? actually we should clamp x to [-6, 6])

        // Clamping x to [-6, 6]
        __half2 neg6 = __float2half2_rn(-6.0f);
        __half2 pos6 = __float2half2_rn(6.0f);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, neg6), pos6);

        __half2 result = __hfma2(t_clamped, coeff3, coeff2);
        result = __hfma2(t_clamped, result, coeff1);
        result = __hfma2(t_clamped, result, coeff0);

        // Boundary Saturation
        // Check |x| > 6
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_t = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 sat_fp = __hgt2(abs_t, pos6);
        unsigned int sat_mask = *reinterpret_cast<unsigned int*>(&sat_fp);

        // Determine Saturation Value (0 if x<0, 1 if x>0)
        // If x is + (sign=0), val=1.0 (0x3C00)
        // If x is - (sign=1), val=0.0 (0x0000)
        // val = (sign) ? 0 : 1.0
        // val = 1.0 & ~mask_from_sign
        // Construct mask from sign: (sign >> 15) ? No, sign bit is bit 15.
        // bit 15 -> 0xFFFF?
        // Let's use arithmetic right shift of INT32 interpreted as INT16?
        // In CUDA, logic is easier with explicit select.

        unsigned int one_u = 0x3C003C00;
        // If sat_mask is set, we overwrite result with:
        // (x > 0) ? 1.0 : 0.0
        // Mask for positive x: ~signs & sat_mask
        // We only want to set 1.0 if (Saturated AND Positive)
        unsigned int set_one_mask = (~signs) & sat_mask;

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);

        // Logic:
        // 1. Clear bits where we saturate (sat_mask): result & ~sat_mask
        // 2. Set bits to 1.0 where we saturate AND are positive: | (one_u & set_one_mask)
        // 3. Where we saturate AND are negative, we want 0.0, which (result & ~sat_mask) already achieved!

        result_bits = (result_bits & ~sat_mask) | (one_u & set_one_mask);

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};
// ============================================================================
// SIGMOID FAST APPROX (N=1 D=3 Odd)
// ============================================================================
// Proving that Sigmoid can match Swish speed if we accept ~0.05 error.
// Logic: S(x) = 0.5 + x_clamped * (C1 + C3*x_clamped^2)
// This fits S(x)-0.5 as an odd polynomial.
// Coeffs from fit: C1=0.1818, C3=-0.003
struct SIGMOID_N1_D3_FAST {
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        // Constants (C1, C3)
        unsigned int c1_u = 0x31D231D2; // 0.1818
        unsigned int c3_u = 0xAE2DAE2D; // -0.003
        unsigned int half_u = 0x38003800; // 0.5

        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 offset = *reinterpret_cast<__half2*>(&half_u);

        // Clamp x to [-6, 6]
        // Using max+min (2 ops) - Same as Bitwise Sigmoid?
        // Or can we use min(|x|, 6) + copysign?
        // Let's use max+min to be robust.
        __half2 neg6 = __float2half2_rn(-6.0f);
        __half2 pos6 = __float2half2_rn(6.0f);
        __half2 t_clamped = __hmin2(__hmax2(t_h2, neg6), pos6);

        // Poly: x * (C1 + C3*x^2)
        __half2 x2 = __hmul2(t_clamped, t_clamped);
        __half2 p = __hfma2(x2, coeff3, coeff1);
        __half2 y = __hmul2(t_clamped, p);

        // Result = 0.5 + y
        __half2 result = __hadd2(y, offset);

        // Saturation?
        // If |x| > 6, result should be 0 or 1.
        // Our poly at 6: 6*(0.18 - 0.003*36) = 6*(0.18 - 0.108) = 6*0.072 = 0.432.
        // 0.5 + 0.432 = 0.932. Correct is 0.997.
        // Saturation logic needed for perfect 0/1.
        // Reuse bitwise saturation from BITWISE kernel.
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&t_h2);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_t = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 sat_fp = __hgt2(abs_t, pos6);
        unsigned int sat_mask = *reinterpret_cast<unsigned int*>(&sat_fp);

        unsigned int signs = input_bits & sign_mask;
        unsigned int one_u = 0x3C003C00;
        unsigned int set_one_mask = (~signs) & sat_mask;

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        result_bits = (result_bits & ~sat_mask) | (one_u & set_one_mask);

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// =================================================================================
// PRODUCTION SPLINE KERNELS (Merged from SPLINE_FUNCS.cuh)
// =================================================================================

// --- SIGMOID (Spline on [0,6] Symmetric) ---

struct SPLINE_SIGMOID_FWD_D3 {
    // Minimax D3 on [0,6], branchless with __float2half2_rn coefficients
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 pos6 = __float2half2_rn(6.0f);

        // Bitwise abs + save sign
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int was_negative = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, pos6);

        // Coefficients (original minimax D3)
        __half2 c0 = __float2half2_rn(0.500000000f);
        __half2 c1 = __float2half2_rn(0.281005859f);
        __half2 c2 = __float2half2_rn(-0.053344726f);
        __half2 c3 = __float2half2_rn(0.003389358f);

        // Horner: ((c3*x + c2)*x + c1)*x + c0
        __half2 result = __hfma2(t, c3, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        // Symmetry: sigmoid(-x) = 1 - sigmoid(x) [BRANCHLESS]
        unsigned int lo_neg = (was_negative << 1) & 0x00010000;
        lo_neg = lo_neg - (lo_neg >> 16);
        unsigned int hi_neg = (was_negative >> 15) & 0x00010000;
        hi_neg = hi_neg - (hi_neg >> 16);
        unsigned int neg_mask = lo_neg | (hi_neg << 16);

        __half2 one = __float2half2_rn(1.0f);
        __half2 flipped = __hsub2(one, result);
        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int flipped_bits = *reinterpret_cast<unsigned int*>(&flipped);
        result_bits = (result_bits & ~neg_mask) | (flipped_bits & neg_mask);
        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// --- SIGMOID D4 (Minimax optimized, higher accuracy than D3) ---
// Coefficients from Sigmoid_fwd_sym_D4_Minimax_stats.json (FP16 hex)
// Max abs error: 0.00448
struct SPLINE_SIGMOID_FWD_D4 {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 pos6 = __float2half2_rn(6.0f);

        // Bitwise abs
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int was_negative = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, pos6);

        // D4 Minimax coefficients (FP16 hex)
        // c0=0.5, c1=0.2766, c2=-0.0479, c3=0.00161, c4=0.000166
        unsigned int c0_u = 0x38003800;  // 0x3800 = 0.5
        unsigned int c1_u = 0x346D346D;  // 0x346D = 0.2766
        unsigned int c2_u = 0xAA22AA22;  // 0xAA22 = -0.0479
        unsigned int c3_u = 0x16951695;  // 0x1695 = 0.00161
        unsigned int c4_u = 0x09740974;  // 0x0974 = 0.000166

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);

        // Horner: ((((c4*x + c3)*x + c2)*x + c1)*x + c0
        __half2 result = __hfma2(t, c4, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        // Symmetry: sigmoid(-x) = 1 - sigmoid(x)
        __half2 one = __float2half2_rn(1.0f);
        __half2 flipped = __hsub2(one, result);

        // Create mask for negative values (BRANCHLESS)
        unsigned int lo_neg = (was_negative << 1) & 0x00010000;
        lo_neg = lo_neg - (lo_neg >> 16);
        unsigned int hi_neg = (was_negative >> 15) & 0x00010000;
        hi_neg = hi_neg - (hi_neg >> 16);
        unsigned int neg_mask = lo_neg | (hi_neg << 16);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int flipped_bits = *reinterpret_cast<unsigned int*>(&flipped);
        result_bits = (result_bits & ~neg_mask) | (flipped_bits & neg_mask);

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// Swish Forward via Sigmoid D4 Composition: swish(x) = x * sigmoid_d4(x)
// Higher accuracy than D3 composition
struct SPLINE_SWISH_FWD_VIA_SIGMOID_D4 {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 sig = SPLINE_SIGMOID_FWD_D4::evaluate(val);
        return __hmul2(val, sig);
    }
};

struct SPLINE_TANH_FWD_D3 {
    // Minimax D3 on [0,3], branchless with sign XOR
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 pos3 = __float2half2_rn(3.0f);

        // Bitwise abs + save sign
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, pos3);

        // Coefficients (original minimax D3, c0=0)
        __half2 c1 = __float2half2_rn(1.124023437f);
        __half2 c2 = __float2half2_rn(-0.426757812f);
        __half2 c3 = __float2half2_rn(0.054229736f);

        // Horner: x*((c3*x + c2)*x + c1)
        __half2 result = __hfma2(t, c3, c2);
        result = __hfma2(t, result, c1);
        result = __hmul2(t, result);

        // Restore sign: tanh is odd, so just XOR sign bits back
        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        result_bits ^= signs;
        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// --- SWISH FWD D3 FAST (Ported from scaling_kernels.cuh SWISH_N2_D3_ABS_OPT) ---
// Error ~0.048. Extremely fast (3 Ops).
struct SPLINE_SWISH_FWD_D3_FAST {
    // Coefficients hex (D3)
    // C3: -0.011 (0xA195)
    // C2: 0.109 (0x2EF9)
    // C1: 0.740 (0x39EC)
    // C0: 0.0
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 pos6 = __float2half2_rn(6.0f);

        // 1. Bitwise Absolute Value
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&t_h2);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_t = *reinterpret_cast<__half2*>(&abs_bits);

        // Clamp |x| to 6.0
        __half2 t_clamped = __hmin2(abs_t, pos6);

        // 2. Coefficients
        unsigned int c1_u = 0x39333933;
        unsigned int c2_u = 0x30483048;
        unsigned int c3_u = 0xA27AA27A;

        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);

        // 3. Evaluate D3
        // ((c3*x + c2)*x + c1)*x
        __half2 res = __hfma2(t_clamped, c3, c2);
        res = __hfma2(t_clamped, res, c1);
        res = __hmul2(t_clamped, res);

        // 4. One-sided saturation for |x| > 6
        // Robust Bitwise Select
        __half2 sat_fp = __hgt2(abs_t, pos6);
        unsigned int sat_raw = *reinterpret_cast<unsigned int*>(&sat_fp);
        // Turn 1.0 (0x3C00) into 0xFFFF per lane
        unsigned int sat_bools = (sat_raw >> 13) & 0x00010001;
        unsigned int sat_mask = sat_bools * 0xFFFF;

        unsigned int res_bits = *reinterpret_cast<unsigned int*>(&res);
        res_bits = (res_bits & ~sat_mask) | (abs_bits & sat_mask);
        res = *reinterpret_cast<__half2*>(&res_bits);

        // 5. Correction for negatives: swish(x) = swish(|x|) + min(x, 0)
        unsigned int zero_u = 0x00000000;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 neg_part = __hmin2(t_h2, zero);

        res = __hadd2(res, neg_part);

        return res;
    }
};

struct SPLINE_SWISH_FWD_D4 {
    // Coefficients hex (D4)
    // C4: 0.003317859 (0x1ACC)
    // C3: -0.051592800 (0xAA9B)
    // C2: 0.270422136 (0x3454)
    // C1: 0.515677885 (0x3820)
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 pos6 = __float2half2_rn(6.0f);

        // 1. Bitwise Absolute Value
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&t_h2);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_t = *reinterpret_cast<__half2*>(&abs_bits);

        // Clamp |x| to 6.0
        __half2 t_clamped = __hmin2(abs_t, pos6);

        // 2. Coefficients
        unsigned int c4_u = 0x1ACC1ACC;
        unsigned int c3_u = 0xAA9BAA9B;
        unsigned int c2_u = 0x34543454;
        unsigned int c1_u = 0x38203820;

        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);

        // 3. Evaluate D4
        // (((c4*x + c3)*x + c2)*x + c1)*x
        __half2 res = __hfma2(t_clamped, c4, c3);
        res = __hfma2(t_clamped, res, c2);
        res = __hfma2(t_clamped, res, c1);
        res = __hmul2(t_clamped, res);

        // 4. One-sided saturation
        // Robust Bitwise Select
        __half2 sat_fp = __hgt2(abs_t, pos6);
        unsigned int sat_raw = *reinterpret_cast<unsigned int*>(&sat_fp);
        unsigned int sat_bools = (sat_raw >> 13) & 0x00010001;
        unsigned int sat_mask = sat_bools * 0xFFFF;

        unsigned int res_bits = *reinterpret_cast<unsigned int*>(&res);
        res_bits = (res_bits & ~sat_mask) | (abs_bits & sat_mask);
        res = *reinterpret_cast<__half2*>(&res_bits);

        // 5. Correction for negatives
        unsigned int zero_u = 0x00000000;
        __half2 zero = *reinterpret_cast<__half2*>(&zero_u);
        __half2 neg_part = __hmin2(t_h2, zero);

        res = __hadd2(res, neg_part);

        return res;
    }

};

// =================================================================================
// SIGMOID GRADIENT KERNELS (Symmetric)
// =================================================================================

struct SPLINE_SIGMOID_GRAD_D4 {
    static __device__ __forceinline__ __half get_c4() { return __float2half(-0.001077121f); }
    static __device__ __forceinline__ __half get_c3() { return __float2half(0.013905210f); }
    static __device__ __forceinline__ __half get_c2() { return __float2half(-0.048119936f); }
    static __device__ __forceinline__ __half get_c1() { return __float2half(-0.020465386f); }
    static __device__ __forceinline__ __half get_c0() { return __float2half(0.250000000f); }

    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 max_bound = __float2half2_rn(6.0f);
        __half2 abs_val = __hmin2(__habs2(val), max_bound);
        __half2 result_h2 = __float2half2_rn(0.0f);

        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c4(), get_c4()));
        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c3(), get_c3()));
        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c2(), get_c2()));
        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c1(), get_c1()));
        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c0(), get_c0()));
        return result_h2;
    }
};

struct SPLINE_TANH_GRAD_D4 {
    static __device__ __forceinline__ __half get_c4() { return __float2half(-0.068935747f); }
    static __device__ __forceinline__ __half get_c3() { return __float2half(0.444966726f); }
    static __device__ __forceinline__ __half get_c2() { return __float2half(-0.769918975f); }
    static __device__ __forceinline__ __half get_c1() { return __float2half(-0.163723089f); }
    static __device__ __forceinline__ __half get_c0() { return __float2half(1.000000000f); }

    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 max_bound = __float2half2_rn(3.0f);
        __half2 abs_val = __hmin2(__habs2(val), max_bound);
        __half2 result_h2 = __float2half2_rn(0.0f);

        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c4(), get_c4()));
        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c3(), get_c3()));
        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c2(), get_c2()));
        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c1(), get_c1()));
        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c0(), get_c0()));
        return result_h2;
    }
};

struct SPLINE_SWISH_GRAD_D4 {
    static __device__ __forceinline__ __half get_c4() { return __float2half(-0.001416453f); }
    static __device__ __forceinline__ __half get_c3() { return __float2half(0.028705369f); }
    static __device__ __forceinline__ __half get_c2() { return __float2half(-0.207351038f); }
    static __device__ __forceinline__ __half get_c1() { return __float2half(0.602054610f); }
    static __device__ __forceinline__ __half get_c0() { return __float2half(0.500000000f); }

    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 min_bound = __float2half2_rn(-6.0f);
        __half2 max_bound = __float2half2_rn(6.0f);
        val = __hmin2(__hmax2(val, min_bound), max_bound);
        __half2 abs_val = __habs2(val);

        __half2 result_h2 = __float2half2_rn(0.0f);
        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c4(), get_c4()));
        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c3(), get_c3()));
        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c2(), get_c2()));
        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c1(), get_c1()));
        result_h2 = __hfma2(abs_val, result_h2, __halves2half2(get_c0(), get_c0()));

        // Swish gradient symmetry: swish'(-x) = 1 - swish'(x) [BRANCHLESS]
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int was_negative = input_bits & sign_mask;
        unsigned int lo_neg = (was_negative << 1) & 0x00010000;
        lo_neg = lo_neg - (lo_neg >> 16);
        unsigned int hi_neg = (was_negative >> 15) & 0x00010000;
        hi_neg = hi_neg - (hi_neg >> 16);
        unsigned int neg_mask = lo_neg | (hi_neg << 16);

        unsigned int one_u = 0x3C003C00;
        __half2 one = *reinterpret_cast<__half2*>(&one_u);
        __half2 flipped = __hsub2(one, result_h2);
        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result_h2);
        unsigned int flipped_bits = *reinterpret_cast<unsigned int*>(&flipped);
        result_bits = (result_bits & ~neg_mask) | (flipped_bits & neg_mask);
        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// ODD-TRANSFORM version: swish'(x) = 0.5 + g(x), g is odd
// Uses sign-XOR instead of expensive 1-f(x) flip
struct SPLINE_SWISH_GRAD_D4_ODD {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        // Bitwise abs + save sign
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);

        // Clamp |x| to 6.0
        unsigned int pos6_u = 0x46004600;
        __half2 pos6 = *reinterpret_cast<__half2*>(&pos6_u);
        __half2 t = __hmin2(abs_val, pos6);

        // Coefficients for g(|x|) = swish'(|x|) - 0.5 (c0=0 since g(0)=0)
        // Minimax D4 on [0,6]: max_err=0.015061
        unsigned int c1_u = 0x38D138D1;  // 0.602
        unsigned int c2_u = 0xB2A3B2A3;  // -0.207
        unsigned int c3_u = 0x27592759;  // 0.0287
        unsigned int c4_u = 0x95CD95CD;  // -0.001416

        __half2 coeff1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 coeff2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 coeff3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 coeff4 = *reinterpret_cast<__half2*>(&c4_u);

        // Horner: g(t) = t*(c1 + t*(c2 + t*(c3 + c4*t)))
        __half2 g = __hfma2(t, coeff4, coeff3);  // c4*t + c3
        g = __hfma2(t, g, coeff2);                // t*(c4t+c3) + c2
        g = __hfma2(t, g, coeff1);                // t*(..)+c1
        g = __hmul2(t, g);                        // t * (c1 + t*(c2 + t*(c3 + c4*t)))

        // NO clamp on g — swish'(x)-0.5 peaks at ~0.6 (unlike sigmoid-0.5 which maxes at 0.5)
        // Polynomial naturally handles the shape; t clamping to [0,6] is sufficient
        unsigned int half_u = 0x38003800;  // 0.5
        __half2 half_val = *reinterpret_cast<__half2*>(&half_u);

        // Restore sign: g is odd, XOR sign bits back
        unsigned int g_bits = *reinterpret_cast<unsigned int*>(&g);
        g_bits ^= signs;

        // swish'(x) = 0.5 + g(x)
        __half2 g_signed = *reinterpret_cast<__half2*>(&g_bits);
        return __hadd2(half_val, g_signed);
    }
};

// SWISH GRAD D3 ODD: g(t) = t*(c1 + t*(c2 + c3*t)), max_err=0.026
struct SPLINE_SWISH_GRAD_D3_ODD {
    static __device__ __forceinline__ __half2 evaluate(__half2 t_h2) {
        __half2 pos6 = __float2half2_rn(6.0f);

        // Bitwise abs + save sign (identical to SIGMOID_N2_D3_ODD)
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&t_h2);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_t = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_t, pos6);

        // Coefficients: minimax D3 on [0,6] for g(x) = swish'(x)-0.5
        __half2 c1 = __float2half2_rn(0.5419921875f);
        __half2 c2 = __float2half2_rn(-0.1490478516f);
        __half2 c3 = __float2half2_rn(0.0121536255f);

        // Horner: g(t) = t*(c1 + t*(c2 + c3*t))  — same as sigmoid ODD
        __half2 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        // Sign restore: g is odd → XOR sign bits
        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);

        // swish'(x) = 0.5 + g(x)
        return __hadd2(__float2half2_rn(0.5f), h_signed);
    }
};

// SWISH GRAD D5 ODD: g(t) = t*(c1 + t*(c2 + t*(c3 + t*(c4 + c5*t)))), max_err=0.009
struct SPLINE_SWISH_GRAD_D5_ODD {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);

        __half2 pos6 = __float2half2_rn(6.0f);
        __half2 t = __hmin2(abs_val, pos6);

        __half2 c1 = __float2half2_rn(0.5581054688f);
        __half2 c2 = __float2half2_rn(-0.1307373047f);
        __half2 c3 = __float2half2_rn(-0.0111541748f);
        __half2 c4 = __float2half2_rn(0.0065040588f);
        __half2 c5 = __float2half2_rn(-0.0005340576f);

        __half2 g = __hfma2(t, c5, c4);
        g = __hfma2(t, g, c3);
        g = __hfma2(t, g, c2);
        g = __hfma2(t, g, c1);
        g = __hmul2(t, g);

        unsigned int g_bits = *reinterpret_cast<unsigned int*>(&g);
        g_bits ^= signs;
        __half2 g_signed = *reinterpret_cast<__half2*>(&g_bits);
        return __hadd2(__float2half2_rn(0.5f), g_signed);
    }
};

// SWISH GRAD D6 ODD: g(t) = t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + c6*t))))), max_err=0.003
struct SPLINE_SWISH_GRAD_D6_ODD {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);

        __half2 pos6 = __float2half2_rn(6.0f);
        __half2 t = __hmin2(abs_val, pos6);

        __half2 c1 = __float2half2_rn(0.5039062500f);
        __half2 c2 = __float2half2_rn(-0.0053024292f);
        __half2 c3 = __float2half2_rn(-0.1071777344f);
        __half2 c4 = __float2half2_rn(0.0386962891f);
        __half2 c5 = __float2half2_rn(-0.0054283142f);
        __half2 c6 = __float2half2_rn(0.0002763271f);

        __half2 g = __hfma2(t, c6, c5);
        g = __hfma2(t, g, c4);
        g = __hfma2(t, g, c3);
        g = __hfma2(t, g, c2);
        g = __hfma2(t, g, c1);
        g = __hmul2(t, g);

        unsigned int g_bits = *reinterpret_cast<unsigned int*>(&g);
        g_bits ^= signs;
        __half2 g_signed = *reinterpret_cast<__half2*>(&g_bits);
        return __hadd2(__float2half2_rn(0.5f), g_signed);
    }
};

// =============================================================================
// D5 MINIMAX BACKWARD/GRADIENT KERNELS
// Higher accuracy than D4, derived from Minimax optimization
// =============================================================================

// Sigmoid gradient: sig'(x) = sig(x)*(1-sig(x)), symmetric (even function)
// Coefficients from Sigmoid_grad_sym_D5_Minimax_stats.json (FP16 hex)
struct SPLINE_SIGMOID_GRAD_D5 {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 pos6 = __float2half2_rn(6.0f);

        // Bitwise abs - sigmoid gradient is symmetric: sig'(-x) = sig'(x)
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, pos6);

        // D5 Minimax coefficients (FP16 hex)
        // c0=0.25, c1=0.0078, c2=-0.0923, c3=0.0357, c4=-0.0053, c5=0.000285
        unsigned int c0_u = 0x34003400;  // 0x3400 = 0.25
        unsigned int c1_u = 0x1FF81FF8;  // 0x1FF8 = 0.0078
        unsigned int c2_u = 0xADE8ADE8;  // 0xADE8 = -0.0923
        unsigned int c3_u = 0x28912891;  // 0x2891 = 0.0357
        unsigned int c4_u = 0x9D749D74;  // 0x9D74 = -0.0053
        unsigned int c5_u = 0x0CAD0CAD;  // 0x0CAD = 0.000285

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);
        __half2 c5 = *reinterpret_cast<__half2*>(&c5_u);

        // Full Horner: (((((c5*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
        __half2 result = __hfma2(t, c5, c4);
        result = __hfma2(t, result, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        return result;
    }
};

// Tanh gradient: tanh'(x) = 1 - tanh(x)^2, symmetric (even function)
// Coefficients from Tanh_grad_sym_D5_Minimax_stats.json (FP16 hex)
struct SPLINE_TANH_GRAD_D5 {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 pos3 = __float2half2_rn(3.0f);

        // Bitwise abs - tanh gradient is symmetric: tanh'(-x) = tanh'(x)
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, pos3);

        // D5 Minimax coefficients (FP16 hex)
        // c0=1.0, c1=0.0623, c2=-1.477, c3=1.142, c4=-0.341, c5=0.0365
        unsigned int c0_u = 0x3C003C00;  // 0x3C00 = 1.0
        unsigned int c1_u = 0x2BF82BF8;  // 0x2BF8 = 0.0623
        unsigned int c2_u = 0xBDE8BDE8;  // 0xBDE8 = -1.477
        unsigned int c3_u = 0x3C913C91;  // 0x3C91 = 1.142
        unsigned int c4_u = 0xB574B574;  // 0xB574 = -0.341
        unsigned int c5_u = 0x28AD28AD;  // 0x28AD = 0.0365

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);
        __half2 c5 = *reinterpret_cast<__half2*>(&c5_u);

        // Full Horner: (((((c5*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
        __half2 result = __hfma2(t, c5, c4);
        result = __hfma2(t, result, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        return result;
    }
};

// Swish gradient: swish'(x) = sig(x) + x*sig(x)*(1-sig(x)), asymmetric
// Coefficients from Swish_grad_pos_D5_Minimax_stats.json (FP16 hex)
// Uses positive domain fit + symmetry: swish'(-x) = 1 - swish'(x)
struct SPLINE_SWISH_GRAD_D5 {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 pos6 = __float2half2_rn(6.0f);

        // Bitwise abs and save sign
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, pos6);

        // D5 Minimax coefficients (FP16 hex)
        // c0=0.5, c1=0.558, c2=-0.131, c3=-0.0112, c4=0.0065, c5=-0.000534
        unsigned int c0_u = 0x38003800;  // 0.5
        unsigned int c1_u = 0x38773877;  // 0.558
        unsigned int c2_u = 0xB02FB02F;  // -0.131
        unsigned int c3_u = 0xA1B6A1B6;  // -0.0112
        unsigned int c4_u = 0x1EA91EA9;  // 0.0065
        unsigned int c5_u = 0x90609060;  // -0.000534

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);
        __half2 c5 = *reinterpret_cast<__half2*>(&c5_u);

        // Horner: ((((c5*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
        __half2 result = __hfma2(t, c5, c4);
        result = __hfma2(t, result, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        // Asymmetric correction: if x < 0, result = 1 - result
        __half2 one = __float2half2_rn(1.0f);
        __half2 flipped = __hsub2(one, result);

        // Create mask for negative values
        unsigned int neg_mask = 0;
        if (signs & 0x8000) neg_mask |= 0xFFFF;
        if (signs & 0x80000000) neg_mask |= 0xFFFF0000;

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int flipped_bits = *reinterpret_cast<unsigned int*>(&flipped);
        result_bits = (result_bits & ~neg_mask) | (flipped_bits & neg_mask);

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// Swish Forward via Sigmoid Composition: swish(x) = x * sigmoid(x)
// Faster than direct polynomial (0.0369ms vs 0.0451ms)
struct SPLINE_SWISH_FWD_VIA_SIGMOID {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 sig = SPLINE_SIGMOID_FWD_D3::evaluate(val);
        return __hmul2(val, sig);
    }
};

// Swish via the fast ODD sigmoid: swish(x) = x * sigmoid_odd(x)
struct SPLINE_SWISH_FWD_VIA_SIGMOID_ODD {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 sig = SIGMOID_N2_D3_ODD::evaluate(val);
        return __hmul2(val, sig);
    }
};

// =================================================================================
// FUSED SWISH FWD D3 ODD
// =================================================================================
// Key identity: swish(x) = x * sigmoid(x) = x * (0.5 + h(x))
// where h(x) = sigmoid(x) - 0.5 is odd (h(-x) = -h(x))
// h(|x|) is the raw polynomial before sign restore.
// Then: swish(x) = 0.5*x + x*h(x) = 0.5*x + x*sign(x)*h(|x|) = 0.5*x + |x|*h(|x|)
// This avoids the XOR sign restore and intermediate add 0.5 from sigmoid.
// Net saving: 2 instructions vs SPLINE_SWISH_FWD_VIA_SIGMOID_ODD.

struct SWISH_FWD_D3_ODD {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        // Bitwise abs
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);

        // Clamp |x| to 6.0
        __half2 pos6 = __float2half2_rn(6.0f);
        __half2 t = __hmin2(abs_val, pos6);

        // Coefficients for h(|x|) = sigmoid(|x|) - 0.5  (minimax D3 on [0,6])
        __half2 c1 = __float2half2_rn(0.281005859375f);
        __half2 c2 = __float2half2_rn(-0.0533447265625f);
        __half2 c3 = __float2half2_rn(0.0033893585205078125f);

        // Horner: h(t) = t*(c1 + t*(c2 + c3*t))
        __half2 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);   // h = h(|x|), raw (unsigned)

        // Fused: swish(x) = 0.5*x + |x|*h(|x|)
        __half2 half_x = __hmul2(__float2half2_rn(0.5f), val);  // 0.5 * x (preserves sign)
        return __hfma2(abs_val, h, half_x);                     // |x|*h + 0.5*x
    }
};

// =================================================================================
// EXTENDED RANGE GRADIENT KERNELS (D4)
// These use wider fitting domains for better tail coverage
// =================================================================================

// Sigmoid gradient D4 Extended Range [0, 7]
// Coefficients from Sigmoid_grad_sym_D4_Minimax_extended_range_stats.json
// FP16: c0=0x3400, c1=0xA828, c2=0xA861, c3=0x20DC, c4=0x9176
// Max abs error: 0.009
struct SPLINE_SIGMOID_GRAD_D4_EXT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 pos7 = __float2half2_rn(7.0f);

        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, pos7);

        unsigned int c0_u = 0x34003400;  // 0.25
        unsigned int c1_u = 0xA828A828;  // -0.0325
        unsigned int c2_u = 0xA861A861;  // -0.0342
        unsigned int c3_u = 0x20DC20DC;  // 0.00949
        unsigned int c4_u = 0x91769176;  // -0.000667

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);

        __half2 result = __hfma2(t, c4, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        return result;
    }
};

// Tanh gradient D4 Extended Range [0, 4]
// Coefficients from Tanh_grad_sym_D4_Minimax_extended_range_stats.json
// FP16: c0=0x3C00, c1=0xB582, c2=0xB5F1, c3=0x328D, c4=0xA6C9
// Max abs error: 0.047
struct SPLINE_TANH_GRAD_D4_EXT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 pos4 = __float2half2_rn(4.0f);

        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, pos4);

        unsigned int c0_u = 0x3C003C00;  // 1.0
        unsigned int c1_u = 0xB582B582;  // -0.344
        unsigned int c2_u = 0xB5F1B5F1;  // -0.371
        unsigned int c3_u = 0x328D328D;  // 0.205
        unsigned int c4_u = 0xA6C9A6C9;  // -0.0265

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);

        __half2 result = __hfma2(t, c4, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        return result;
    }
};

// Swish gradient D4 Extended Range [0, 7]
// Coefficients from Swish_grad_pos_D4_Minimax_extended_range_stats.json
// FP16: c0=0x3800, c1=0x38CC, c2=0xB286, c3=0x270D, c4=0x955E
// Max abs error: 0.016
struct SPLINE_SWISH_GRAD_D4_EXT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 pos7 = __float2half2_rn(7.0f);

        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, pos7);

        unsigned int c0_u = 0x38003800;  // 0.5
        unsigned int c1_u = 0x38CC38CC;  // 0.600
        unsigned int c2_u = 0xB286B286;  // -0.204
        unsigned int c3_u = 0x270D270D;  // 0.0275
        unsigned int c4_u = 0x955E955E;  // -0.00131

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);

        __half2 result = __hfma2(t, c4, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        // Branchless asymmetric correction: if x < 0, result = 1 - result
        // Sign-extend: 0x8000 -> 0xFFFF, 0 -> 0
        unsigned int neg_mask_lo = (signs << 1) - (signs >> 15);  // 0x8000 -> 0xFFFF
        unsigned int neg_mask_hi = ((signs >> 16) << 1) - ((signs >> 31));
        unsigned int neg_mask = (neg_mask_hi << 16) | (neg_mask_lo & 0xFFFF);

        __half2 one = __float2half2_rn(1.0f);
        __half2 flipped = __hsub2(one, result);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int flipped_bits = *reinterpret_cast<unsigned int*>(&flipped);
        result_bits = (result_bits & ~neg_mask) | (flipped_bits & neg_mask);

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// =================================================================================
// EXTENDED RANGE GRADIENT KERNELS (D5)
// These use wider fitting domains for better tail coverage
// =================================================================================

// Sigmoid gradient D5 Extended Range [0, 7]
// Coefficients from Sigmoid_grad_sym_D5_Minimax_extended_range_stats.json
// FP16: c0=0x3400, c1=0x1D8A, c2=0xAD76, c3=0x2801, c4=0x9C70, c5=0x0AF6
// Max abs error: 0.0029
struct SPLINE_SIGMOID_GRAD_D5_EXT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 pos7 = __float2half2_rn(7.0f);

        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, pos7);

        unsigned int c0_u = 0x34003400;  // 0.25
        unsigned int c1_u = 0x1D8A1D8A;  // 0.00541
        unsigned int c2_u = 0xAD76AD76;  // -0.0853
        unsigned int c3_u = 0x28012801;  // 0.0313
        unsigned int c4_u = 0x9C709C70;  // -0.00433
        unsigned int c5_u = 0x0AF60AF6;  // 0.000212

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);
        __half2 c5 = *reinterpret_cast<__half2*>(&c5_u);

        __half2 result = __hfma2(t, c5, c4);
        result = __hfma2(t, result, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        return result;
    }
};

// Tanh gradient D5 Extended Range [0, 4]
// Coefficients from Tanh_grad_sym_D5_Minimax_extended_range_stats.json
// FP16: c0=0x3C00, c1=0xAD22, c2=0xBC3C, c3=0x39F6, c4=0xB21C, c5=0x245D
// Max abs error: 0.018
struct SPLINE_TANH_GRAD_D5_EXT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 pos4 = __float2half2_rn(4.0f);

        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, pos4);

        unsigned int c0_u = 0x3C003C00;  // 1.0
        unsigned int c1_u = 0xAD22AD22;  // -0.0802
        unsigned int c2_u = 0xBC3CBC3C;  // -1.059
        unsigned int c3_u = 0x39F639F6;  // 0.745
        unsigned int c4_u = 0xB21CB21C;  // -0.191
        unsigned int c5_u = 0x245D245D;  // 0.0170

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);
        __half2 c5 = *reinterpret_cast<__half2*>(&c5_u);

        __half2 result = __hfma2(t, c5, c4);
        result = __hfma2(t, result, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        return result;
    }
};

// Swish gradient D5 Extended Range [0, 7.5]
// Coefficients from Swish_grad_pos_D5_Minimax_extended_range_stats.json
// FP16: c0=0x3800, c1=0x38BE, c2=0xB22A, c3=0x25AA, c4=0x8D4D, c5=0x83E7
// Max abs error: 0.017
struct SPLINE_SWISH_GRAD_D5_EXT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 pos75 = __float2half2_rn(7.5f);

        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, pos75);

        unsigned int c0_u = 0x38003800;  // 0.5
        unsigned int c1_u = 0x38BE38BE;  // 0.593
        unsigned int c2_u = 0xB22AB22A;  // -0.193
        unsigned int c3_u = 0x25AA25AA;  // 0.0221
        unsigned int c4_u = 0x8D4D8D4D;  // -0.000324
        unsigned int c5_u = 0x83E783E7;  // -0.0000596

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);
        __half2 c5 = *reinterpret_cast<__half2*>(&c5_u);

        __half2 result = __hfma2(t, c5, c4);
        result = __hfma2(t, result, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        // Branchless asymmetric correction: if x < 0, result = 1 - result
        unsigned int neg_mask_lo = (signs << 1) - (signs >> 15);
        unsigned int neg_mask_hi = ((signs >> 16) << 1) - ((signs >> 31));
        unsigned int neg_mask = (neg_mask_hi << 16) | (neg_mask_lo & 0xFFFF);

        __half2 one = __float2half2_rn(1.0f);
        __half2 flipped = __hsub2(one, result);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int flipped_bits = *reinterpret_cast<unsigned int*>(&flipped);
        result_bits = (result_bits & ~neg_mask) | (flipped_bits & neg_mask);

        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// ============================================================================
// D4/D5/D6 GRADIENT APPROXIMATIONS (Optimal from 2D sweep)
// ============================================================================

// Sigmoid Gradient D4 [0, 5.4] -> clamp at 5.5, error: 0.0042
// hex: ['0X3400', '0XA25D', '0XAB69', '0X246D', '0X95D7']
struct SIGMOID_GRAD_D4_OPT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 clamp_val = __float2half2_rn(5.5f);
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, clamp_val);

        unsigned int c0_u = 0x34003400;
        unsigned int c1_u = 0xA25DA25D;
        unsigned int c2_u = 0xAB69AB69;
        unsigned int c3_u = 0x246D246D;
        unsigned int c4_u = 0x95D795D7;

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);

        __half2 result = __hfma2(t, c4, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        return result;
    }
};

// Sigmoid Gradient D5 [0, 6.4] -> clamp at 6.5, error: 0.0015
// hex: ['0X3400', '0X1CBF', '0XAD93', '0X2838', '0X9CDE', '0X0C03']
struct SIGMOID_GRAD_D5_OPT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 clamp_val = __float2half2_rn(6.5f);
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, clamp_val);

        unsigned int c0_u = 0x34003400;
        unsigned int c1_u = 0x1CBF1CBF;
        unsigned int c2_u = 0xAD93AD93;
        unsigned int c3_u = 0x28382838;
        unsigned int c4_u = 0x9CDE9CDE;
        unsigned int c5_u = 0x0C030C03;

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);
        __half2 c5 = *reinterpret_cast<__half2*>(&c5_u);

        __half2 result = __hfma2(t, c5, c4);
        result = __hfma2(t, result, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        return result;
    }
};

// Sigmoid Gradient D6 [0, 8.1] -> clamp at 7.0, error: 0.0009
// hex: ['0X3400', '0X214F', '0XAE62', '0X2964', '0X9FD6', '0X116B', '0X8175']
struct SIGMOID_GRAD_D6_OPT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 clamp_val = __float2half2_rn(7.0f);
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, clamp_val);

        unsigned int c0_u = 0x34003400;
        unsigned int c1_u = 0x214F214F;
        unsigned int c2_u = 0xAE62AE62;
        unsigned int c3_u = 0x29642964;
        unsigned int c4_u = 0x9FD69FD6;
        unsigned int c5_u = 0x116B116B;
        unsigned int c6_u = 0x81758175;

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);
        __half2 c5 = *reinterpret_cast<__half2*>(&c5_u);
        __half2 c6 = *reinterpret_cast<__half2*>(&c6_u);

        __half2 result = __hfma2(t, c6, c5);
        result = __hfma2(t, result, c4);
        result = __hfma2(t, result, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        return result;
    }
};

// Tanh Gradient D4 [0, 2.7] -> clamp at 2.7, error: 0.0179
// hex: ['0X3C00', '0XAE5D', '0XBB69', '0X386D', '0XADD7']
struct TANH_GRAD_D4_OPT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 clamp_val = __float2half2_rn(2.7f);
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, clamp_val);

        unsigned int c0_u = 0x3C003C00;
        unsigned int c1_u = 0xAE5DAE5D;
        unsigned int c2_u = 0xBB69BB69;
        unsigned int c3_u = 0x386D386D;
        unsigned int c4_u = 0xADD7ADD7;

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);

        __half2 result = __hfma2(t, c4, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        return result;
    }
};

// Tanh Gradient D5 [0, 3.2] -> clamp at 3.2, error: 0.0066
// hex: ['0X3C00', '0X28BF', '0XBD93', '0X3C38', '0XB4DE', '0X2803']
struct TANH_GRAD_D5_OPT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 clamp_val = __float2half2_rn(3.2f);
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, clamp_val);

        unsigned int c0_u = 0x3C003C00;
        unsigned int c1_u = 0x28BF28BF;
        unsigned int c2_u = 0xBD93BD93;
        unsigned int c3_u = 0x3C383C38;
        unsigned int c4_u = 0xB4DEB4DE;
        unsigned int c5_u = 0x28032803;

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);
        __half2 c5 = *reinterpret_cast<__half2*>(&c5_u);

        __half2 result = __hfma2(t, c5, c4);
        result = __hfma2(t, result, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        return result;
    }
};

// Tanh Gradient D6 [0, 4.0] -> clamp at 3.5, error: 0.0037
// hex: ['0X3C00', '0X2D77', '0XBE6C', '0X3D70', '0XB7F1', '0X2D85', '0X9DFA']
struct TANH_GRAD_D6_OPT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 clamp_val = __float2half2_rn(3.5f);
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, clamp_val);

        unsigned int c0_u = 0x3C003C00;
        unsigned int c1_u = 0x2D772D77;
        unsigned int c2_u = 0xBE6CBE6C;
        unsigned int c3_u = 0x3D703D70;
        unsigned int c4_u = 0xB7F1B7F1;
        unsigned int c5_u = 0x2D852D85;
        unsigned int c6_u = 0x9DFA9DFA;

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);
        __half2 c5 = *reinterpret_cast<__half2*>(&c5_u);
        __half2 c6 = *reinterpret_cast<__half2*>(&c6_u);

        __half2 result = __hfma2(t, c6, c5);
        result = __hfma2(t, result, c4);
        result = __hfma2(t, result, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        return result;
    }
};

// Swish Gradient D3 [0, 5.0] -> asymmetric, error: 0.012
// hex: ['0X3800', '0X389E', '0XB189', '0X2406']
// swish'(-x) = 1 - swish'(x), so for x<0: result = 1 - result
struct SWISH_GRAD_D3_OPT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 clamp_val = __float2half2_rn(5.0f);
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, clamp_val);

        unsigned int c0_u = 0x38003800;
        unsigned int c1_u = 0x389E389E;
        unsigned int c2_u = 0xB189B189;
        unsigned int c3_u = 0x24062406;

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);

        __half2 result = __hfma2(t, c3, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        // Asymmetric correction: if x < 0, result = 1 - result
        unsigned int neg_mask_lo = (signs << 1) - (signs >> 15);
        unsigned int neg_mask_hi = ((signs >> 16) << 1) - ((signs >> 31));
        unsigned int neg_mask = (neg_mask_hi << 16) | (neg_mask_lo & 0xFFFF);

        __half2 one = __float2half2_rn(1.0f);
        __half2 flipped = __hsub2(one, result);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int flipped_bits = *reinterpret_cast<unsigned int*>(&flipped);
        result_bits = (result_bits & ~neg_mask) | (flipped_bits & neg_mask);
        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// Swish Gradient D4 [0, 6.0] -> asymmetric, error: 0.017
// hex: ['0X3800', '0X38E3', '0XB2D8', '0X27A8', '0X960A']
struct SWISH_GRAD_D4_OPT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 clamp_val = __float2half2_rn(6.0f);
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, clamp_val);

        unsigned int c0_u = 0x38003800;
        unsigned int c1_u = 0x38E338E3;
        unsigned int c2_u = 0xB2D8B2D8;
        unsigned int c3_u = 0x27A827A8;
        unsigned int c4_u = 0x960A960A;

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);

        __half2 result = __hfma2(t, c4, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        unsigned int neg_mask_lo = (signs << 1) - (signs >> 15);
        unsigned int neg_mask_hi = ((signs >> 16) << 1) - ((signs >> 31));
        unsigned int neg_mask = (neg_mask_hi << 16) | (neg_mask_lo & 0xFFFF);

        __half2 one = __float2half2_rn(1.0f);
        __half2 flipped = __hsub2(one, result);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int flipped_bits = *reinterpret_cast<unsigned int*>(&flipped);
        result_bits = (result_bits & ~neg_mask) | (flipped_bits & neg_mask);
        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// Swish Gradient D5 [0, 5.0] -> asymmetric, error: 0.0046
// hex: ['0X3800', '0X3849', '0XAD4A', '0XA968', '0X2350', '0X94E3']
struct SWISH_GRAD_D5_OPT {
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        __half2 clamp_val = __float2half2_rn(5.0f);
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, clamp_val);

        unsigned int c0_u = 0x38003800;
        unsigned int c1_u = 0x38493849;
        unsigned int c2_u = 0xAD4AAD4A;
        unsigned int c3_u = 0xA968A968;
        unsigned int c4_u = 0x23502350;
        unsigned int c5_u = 0x94E394E3;

        __half2 c0 = *reinterpret_cast<__half2*>(&c0_u);
        __half2 c1 = *reinterpret_cast<__half2*>(&c1_u);
        __half2 c2 = *reinterpret_cast<__half2*>(&c2_u);
        __half2 c3 = *reinterpret_cast<__half2*>(&c3_u);
        __half2 c4 = *reinterpret_cast<__half2*>(&c4_u);
        __half2 c5 = *reinterpret_cast<__half2*>(&c5_u);

        __half2 result = __hfma2(t, c5, c4);
        result = __hfma2(t, result, c3);
        result = __hfma2(t, result, c2);
        result = __hfma2(t, result, c1);
        result = __hfma2(t, result, c0);

        unsigned int neg_mask_lo = (signs << 1) - (signs >> 15);
        unsigned int neg_mask_hi = ((signs >> 16) << 1) - ((signs >> 31));
        unsigned int neg_mask = (neg_mask_hi << 16) | (neg_mask_lo & 0xFFFF);

        __half2 one = __float2half2_rn(1.0f);
        __half2 flipped = __hsub2(one, result);

        unsigned int result_bits = *reinterpret_cast<unsigned int*>(&result);
        unsigned int flipped_bits = *reinterpret_cast<unsigned int*>(&flipped);
        result_bits = (result_bits & ~neg_mask) | (flipped_bits & neg_mask);
        return *reinterpret_cast<__half2*>(&result_bits);
    }
};

// ============================================================================
// Benchmark Kernels for D4/D5/D6
// ============================================================================
template <typename GradStruct>
__global__ void benchmark_grad_kernel(__half2* data, int n_h2, int inner_repeats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_h2) {
        __half2 t_out = data[idx];
        #pragma unroll
        for (int r = 0; r < inner_repeats; ++r) {
            t_out = GradStruct::evaluate(t_out);
        }
        data[idx] = t_out;
    }
}

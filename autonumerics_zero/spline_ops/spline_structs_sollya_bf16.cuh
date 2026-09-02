// spline_structs_sollya_bf16.cuh — BF16 Sollya fpminimax activation structs
// AUTO-GENERATED with the same runtime clamps and evaluation shapes as spline_structs_odd_bf16.cuh
// Copyright (c) 2026 Graphcore Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified in 2026 for the standalone fast-polynomial-transcendentals release.

#pragma once
#include <cuda_bf16.h>

// =============================================================================
// sigmoid_fwd — Sollya BF16 variants
// =============================================================================

struct SIGMOID_FWD_D3_ODD_SOLLYA_BF16 {
    // Clamp=6.0000, max_err=0.005475 (Sollya BF16, centered odd)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(6.0000000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.2812500000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0534667969f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0033874512f);

        __nv_bfloat162 h = c3;
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct SIGMOID_FWD_D4_ODD_SOLLYA_BF16 {
    // Clamp=5.2812, max_err=0.003457 (Sollya BF16, centered odd)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(5.2812500000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.2695312500f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0380859375f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.0020904541f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0005722046f);

        __nv_bfloat162 h = c4;
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct SIGMOID_FWD_D5_ODD_SOLLYA_BF16 {
    // Clamp=5.4062, max_err=0.001136 (Sollya BF16, centered odd)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(5.4062500000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.2558593750f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0115356445f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.0179443359f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0042724609f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0002956390f);

        __nv_bfloat162 h = c5;
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct SIGMOID_FWD_D6_ODD_SOLLYA_BF16 {
    // Clamp=4.9062, max_err=0.000239 (Sollya BF16, centered odd)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(4.9062500000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.2490234375f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(0.0062561035f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.0327148438f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0095825195f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0011596680f);
        __nv_bfloat162 c6 = __float2bfloat162_rn(0.0000522137f);

        __nv_bfloat162 h = c6;
        h = __hfma2(t, h, c5);
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

// =============================================================================
// tanh_fwd — Sollya BF16 variants
// =============================================================================

struct TANH_FWD_D3_ODD_SOLLYA_BF16 {
    // Clamp=2.7500, max_err=0.014698 (Sollya BF16, odd factorized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.7500000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(1.1171875000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.4160156250f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0505371094f);

        __nv_bfloat162 h = c3;
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);
        h = __hmin2(h, __float2bfloat162_rn(1.0f));

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return h_signed;
    }
};

struct TANH_FWD_D4_ODD_SOLLYA_BF16 {
    // Clamp=2.7500, max_err=0.007217 (Sollya BF16, odd factorized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.7500000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(1.0859375000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.3281250000f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.0150146484f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0140991211f);

        __nv_bfloat162 h = c4;
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);
        h = __hmin2(h, __float2bfloat162_rn(1.0f));

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return h_signed;
    }
};

struct TANH_FWD_D5_ODD_SOLLYA_BF16 {
    // Clamp=2.2500, max_err=0.000780 (Sollya BF16, odd factorized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.2500000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(1.0078125000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0157470703f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.3964843750f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.1962890625f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0299072266f);

        __nv_bfloat162 h = c5;
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);
        h = __hmin2(h, __float2bfloat162_rn(1.0f));

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return h_signed;
    }
};

struct TANH_FWD_D6_ODD_SOLLYA_BF16 {
    // Clamp=2.2500, max_err=0.000957 (Sollya BF16, odd factorized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.2500000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(1.0000000000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(0.0214843750f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.4570312500f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.2402343750f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0444335938f);
        __nv_bfloat162 c6 = __float2bfloat162_rn(0.0017700195f);

        __nv_bfloat162 h = c6;
        h = __hfma2(t, h, c5);
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);
        h = __hmin2(h, __float2bfloat162_rn(1.0f));

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return h_signed;
    }
};

// =============================================================================
// sigmoid_bwd — Sollya BF16 variants
// =============================================================================

struct SIGMOID_BWD_D3_EVEN_SOLLYA_BF16 {
    // Clamp=4.0000, max_err=0.005859 (Sollya BF16, even)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(4.0000000000f));

        __nv_bfloat162 c0 = __float2bfloat162_rn(0.2558593750f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(-0.0385742188f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0292968750f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0061035156f);

        __nv_bfloat162 r = c3;
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

struct SIGMOID_BWD_D4_EVEN_SOLLYA_BF16 {
    // Clamp=4.7500, max_err=0.001969 (Sollya BF16, even)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(4.7500000000f));

        __nv_bfloat162 c0 = __float2bfloat162_rn(0.2519531250f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(-0.0089721680f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0668945312f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0213623047f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(-0.0019302368f);

        __nv_bfloat162 r = c4;
        r = __hfma2(t, r, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

struct SIGMOID_BWD_D5_EVEN_SOLLYA_BF16 {
    // Clamp=4.7500, max_err=0.000977 (Sollya BF16, even)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(4.7500000000f));

        __nv_bfloat162 c0 = __float2bfloat162_rn(0.2490234375f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(0.0139770508f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.1015625000f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0407714844f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(-0.0064697266f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(0.0003757477f);

        __nv_bfloat162 r = c5;
        r = __hfma2(t, r, c4);
        r = __hfma2(t, r, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

struct SIGMOID_BWD_D6_EVEN_SOLLYA_BF16 {
    // Clamp=4.2500, max_err=0.000297 (Sollya BF16, even)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(4.2500000000f));

        __nv_bfloat162 c0 = __float2bfloat162_rn(0.2500000000f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(0.0023956299f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0742187500f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0156250000f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0042724609f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0017700195f);
        __nv_bfloat162 c6 = __float2bfloat162_rn(0.0001621246f);

        __nv_bfloat162 r = c6;
        r = __hfma2(t, r, c5);
        r = __hfma2(t, r, c4);
        r = __hfma2(t, r, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

// =============================================================================
// tanh_bwd — Sollya BF16 variants
// =============================================================================

struct TANH_BWD_D3_EVEN_SOLLYA_BF16 {
    // Clamp=2.0000, max_err=0.023438 (Sollya BF16, even)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.0000000000f));

        __nv_bfloat162 c0 = __float2bfloat162_rn(1.0234375000f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(-0.3085937500f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.4687500000f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.1953125000f);

        __nv_bfloat162 r = c3;
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

struct TANH_BWD_D4_EVEN_SOLLYA_BF16 {
    // Clamp=2.2500, max_err=0.007812 (Sollya BF16, even)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.2500000000f));

        __nv_bfloat162 c0 = __float2bfloat162_rn(1.0078125000f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(-0.0600585938f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-1.1093750000f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.7187500000f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(-0.1328125000f);

        __nv_bfloat162 r = c4;
        r = __hfma2(t, r, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

struct TANH_BWD_D5_EVEN_SOLLYA_BF16 {
    // Clamp=2.2500, max_err=0.003906 (Sollya BF16, even)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.2500000000f));

        __nv_bfloat162 c0 = __float2bfloat162_rn(0.9960937500f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(0.1005859375f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-1.5703125000f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(1.2265625000f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(-0.3710937500f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(0.0400390625f);

        __nv_bfloat162 r = c5;
        r = __hfma2(t, r, c4);
        r = __hfma2(t, r, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

struct TANH_BWD_D6_EVEN_SOLLYA_BF16 {
    // Clamp=2.2500, max_err=0.001415 (Sollya BF16, even)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.2500000000f));

        __nv_bfloat162 c0 = __float2bfloat162_rn(1.0000000000f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(0.0312500000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-1.2578125000f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.6484375000f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.1289062500f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.1611328125f);
        __nv_bfloat162 c6 = __float2bfloat162_rn(0.0303955078f);

        __nv_bfloat162 r = c6;
        r = __hfma2(t, r, c5);
        r = __hfma2(t, r, c4);
        r = __hfma2(t, r, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

// =============================================================================
// swish_bwd — Sollya BF16 variants
// =============================================================================

struct SWISH_BWD_D3_ODD_SOLLYA_BF16 {
    // Clamp=4.7500, max_err=0.012147 (Sollya BF16, centered odd)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(4.7500000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.5781250000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.1738281250f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0158691406f);

        __nv_bfloat162 h = c3;
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct SWISH_BWD_D4_ODD_SOLLYA_BF16 {
    // Clamp=5.2500, max_err=0.011308 (Sollya BF16, centered odd)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(5.2500000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.5820312500f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.1777343750f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0172119141f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(-0.0001506805f);

        __nv_bfloat162 h = c4;
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct SWISH_BWD_D5_ODD_SOLLYA_BF16 {
    // Clamp=4.7500, max_err=0.004591 (Sollya BF16, centered odd)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(4.7500000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.5234375000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0537109375f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.0629882812f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0200195312f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0017318726f);

        __nv_bfloat162 h = c5;
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct SWISH_BWD_D6_ODD_SOLLYA_BF16 {
    // Clamp=5.0000, max_err=0.001411 (Sollya BF16, centered odd)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(5.0000000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.5039062500f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(0.0106201172f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.1289062500f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0490722656f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0075073242f);
        __nv_bfloat162 c6 = __float2bfloat162_rn(0.0004253387f);

        __nv_bfloat162 h = c6;
        h = __hfma2(t, h, c5);
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

// =============================================================================
// gelu_fwd — Sollya BF16 variants
// =============================================================================

struct GELU_FWD_D3_ODD_SOLLYA_BF16 {
    // Clamp=3.0000, max_err=0.006232 (Sollya BF16, direct GeLU)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0000000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.4726562500f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.1464843750f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0147094727f);

        __nv_bfloat162 h = c3;
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        __nv_bfloat162 phi = __hadd2(__float2bfloat162_rn(0.5f), h_signed);
        return __hmul2(val, phi);
    }
};

struct GELU_FWD_D4_ODD_SOLLYA_BF16 {
    // Clamp=3.0000, max_err=0.002758 (Sollya BF16, direct GeLU)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0000000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.4414062500f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0908203125f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.0148315430f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0048522949f);

        __nv_bfloat162 h = c4;
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        __nv_bfloat162 phi = __hadd2(__float2bfloat162_rn(0.5f), h_signed);
        return __hmul2(val, phi);
    }
};

struct GELU_FWD_D5_ODD_SOLLYA_BF16 {
    // Clamp=3.0000, max_err=0.000502 (Sollya BF16, direct GeLU)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0000000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.4003906250f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(0.0086059570f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.0981445312f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0339355469f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0036163330f);

        __nv_bfloat162 h = c5;
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        __nv_bfloat162 phi = __hadd2(__float2bfloat162_rn(0.5f), h_signed);
        return __hmul2(val, phi);
    }
};

struct GELU_FWD_D6_ODD_SOLLYA_BF16 {
    // Clamp=3.0000, max_err=0.000251 (Sollya BF16, direct GeLU)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0000000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.3906250000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(0.0393066406f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.1337890625f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0534667969f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0087280273f);
        __nv_bfloat162 c6 = __float2bfloat162_rn(0.0005149841f);

        __nv_bfloat162 h = c6;
        h = __hfma2(t, h, c5);
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        __nv_bfloat162 phi = __hadd2(__float2bfloat162_rn(0.5f), h_signed);
        return __hmul2(val, phi);
    }
};

// =============================================================================
// gelu_bwd — Sollya BF16 variants
// =============================================================================

struct GELU_BWD_D3_ODD_SOLLYA_BF16 {
    // Clamp=3.0000, max_err=0.017943 (Sollya BF16, centered odd)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0000000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.9648437500f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.4628906250f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0654296875f);

        __nv_bfloat162 h = c3;
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct GELU_BWD_D4_ODD_SOLLYA_BF16 {
    // Clamp=3.0000, max_err=0.011488 (Sollya BF16, centered odd)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0000000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.9140625000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.3476562500f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.0084228516f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0140380859f);

        __nv_bfloat162 h = c4;
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct GELU_BWD_D5_ODD_SOLLYA_BF16 {
    // Clamp=3.0000, max_err=0.002363 (Sollya BF16, centered odd)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0000000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.8046875000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(0.0143432617f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.3789062500f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.1630859375f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0206298828f);

        __nv_bfloat162 h = c5;
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct GELU_BWD_D6_ODD_SOLLYA_BF16 {
    // Clamp=3.0000, max_err=0.001261 (Sollya BF16, centered odd)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0000000000f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.7812500000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(0.1166992188f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.5312500000f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.2636718750f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0510253906f);
        __nv_bfloat162 c6 = __float2bfloat162_rn(0.0034332275f);

        __nv_bfloat162 h = c6;
        h = __hfma2(t, h, c5);
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

// =============================================================================
// swish_fwd — Sollya-composed BF16 variants
// =============================================================================

struct SWISH_FWD_D3_ODD_SOLLYA_BF16 {
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        return __hmul2(val, SIGMOID_FWD_D3_ODD_SOLLYA_BF16::evaluate(val));
    }
};

struct SWISH_FWD_D4_ODD_SOLLYA_BF16 {
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        return __hmul2(val, SIGMOID_FWD_D4_ODD_SOLLYA_BF16::evaluate(val));
    }
};

struct SWISH_FWD_D5_ODD_SOLLYA_BF16 {
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        return __hmul2(val, SIGMOID_FWD_D5_ODD_SOLLYA_BF16::evaluate(val));
    }
};

struct SWISH_FWD_D6_ODD_SOLLYA_BF16 {
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        return __hmul2(val, SIGMOID_FWD_D6_ODD_SOLLYA_BF16::evaluate(val));
    }
};

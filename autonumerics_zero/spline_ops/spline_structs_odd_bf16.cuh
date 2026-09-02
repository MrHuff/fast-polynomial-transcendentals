// spline_structs_odd_bf16.cuh — D3-D6 ODD/EVEN activation structs (BFloat16)
// AUTO-GENERATED with BF16-optimized coefficients via fit_all_degrees_bf16.py
// Uses __nv_bfloat162 vectorized type for 2-wide BF16 operations.
#pragma once
#include <cuda_bf16.h>

// =============================================================================
// SIGMOID FWD — ODD: sigmoid(x) = 0.5 + sign(x)*h(|x|)
// =============================================================================

struct SIGMOID_FWD_D3_ODD_BF16 {
    // Li=6.00, Lc=6.00, Err=0.007420 (BF16-optimized; canonical FA4 sigmoid alignment)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(6.0f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.2810058594f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0533447266f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0033893585f);

        __nv_bfloat162 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct SIGMOID_FWD_D4_ODD_BF16 {
    // Li=5.00, Lc=5.28125, Err=0.005891 (BF16-optimized, exact saturated tail)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(5.28125f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.2714843750f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0402832031f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.0014419556f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0005149841f);

        __nv_bfloat162 h = __hfma2(t, c4, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct SIGMOID_FWD_D5_ODD_BF16 {
    // Li=6.25, Lc=5.40625, Err=0.005891 (BF16-optimized, exact saturated tail)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(5.40625f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.2617187500f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0240478516f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.0107421875f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0026550293f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0001697540f);

        __nv_bfloat162 h = __hfma2(t, c5, c4);
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

struct SIGMOID_FWD_D6_ODD_BF16 {
    // Li=8.00, Lc=4.90625, Err=0.008847 (BF16-optimized, exact saturated tail)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(4.90625f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.2578125000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0134277344f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.0185546875f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0051879883f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0005455017f);
        __nv_bfloat162 c6 = __float2bfloat162_rn(0.0000208616f);

        __nv_bfloat162 h = __hfma2(t, c6, c5);
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
// TANH FWD — ODD: tanh(x) = sign(x)*h(|x|)
// =============================================================================

struct TANH_FWD_D3_ODD_BF16 {
    // Li=2.50, Lc=2.75, Err=0.014538 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.75f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(1.1406250000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.4472656250f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0595703125f);

        __nv_bfloat162 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);
        h = __hmin2(h, __float2bfloat162_rn(1.0f));

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return h_signed;
    }
};

struct TANH_FWD_D4_ODD_BF16 {
    // Li=2.50, Lc=2.75, Err=0.013285 (BF16-optimized; FA4 device refit)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.75f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(1.0859375000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.3222656250f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.0230712891f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0164794922f);

        __nv_bfloat162 h = __hfma2(t, c4, c3);
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

struct TANH_FWD_D5_ODD_BF16 {
    // Li=4.25, Lc=2.25, Err=0.016504 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.25f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(1.1171875000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.4023437500f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0245361328f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0129394531f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0018615723f);

        __nv_bfloat162 h = __hfma2(t, c5, c4);
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

struct TANH_FWD_D6_ODD_BF16 {
    // Li=4.25, Lc=2.25, Err=0.014477 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.25f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(1.0390625000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.1523437500f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.2412109375f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.1367187500f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0278320312f);
        __nv_bfloat162 c6 = __float2bfloat162_rn(0.0020446777f);

        __nv_bfloat162 h = __hfma2(t, c6, c5);
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
// SWISH FWD — ODD: swish(x) = x * sigmoid(x)
// Composed: uses sigmoid spline internally
// =============================================================================

struct SWISH_FWD_D3_ODD_BF16 {
    // Direct odd/even form: swish(x) = 0.5*x + |x|*h(|x|)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(6.0f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.2810058594f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0533447266f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0033893585f);

        __nv_bfloat162 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        return __hfma2(abs_val, h, __hmul2(__float2bfloat162_rn(0.5f), val));
    }
};

struct SWISH_FWD_D4_ODD_BF16 {
    // Direct odd/even form with an exact ReLU-like tail at |x| >= 5.28125.
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(5.28125f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.2714843750f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0402832031f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.0014419556f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0005149841f);

        __nv_bfloat162 h = __hfma2(t, c4, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        return __hfma2(abs_val, h, __hmul2(__float2bfloat162_rn(0.5f), val));
    }
};

struct SWISH_FWD_D5_ODD_BF16 {
    // Composed: swish(x) = x * sigmoid(x), uses SIGMOID_FWD_D5_ODD_BF16
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        return __hmul2(val, SIGMOID_FWD_D5_ODD_BF16::evaluate(val));
    }
};

struct SWISH_FWD_D6_ODD_BF16 {
    // Composed: swish(x) = x * sigmoid(x), uses SIGMOID_FWD_D6_ODD_BF16
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        return __hmul2(val, SIGMOID_FWD_D6_ODD_BF16::evaluate(val));
    }
};

// =============================================================================
// SIGMOID BWD — EVEN: sigmoid'(|x|)
// =============================================================================

struct SIGMOID_BWD_D3_EVEN_BF16 {
    // Li=4.50, Lc=4.00, Err=0.011549 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(4.0f));

        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0040893555f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0207519531f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(-0.0424804688f);
        __nv_bfloat162 c0 = __float2bfloat162_rn(0.2500000000f);

        __nv_bfloat162 r = __hfma2(t, c3, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

struct SIGMOID_BWD_D4_EVEN_BF16 {
    // Li=4.75, Lc=4.75, Err=0.007216 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(4.75f));

        __nv_bfloat162 c4 = __float2bfloat162_rn(-0.0019531250f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0218505859f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0698242188f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(-0.0037994385f);
        __nv_bfloat162 c0 = __float2bfloat162_rn(0.2500000000f);

        __nv_bfloat162 r = __hfma2(t, c4, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

struct SIGMOID_BWD_D5_EVEN_BF16 {
    // Li=7.50, Lc=4.75, Err=0.008373 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(4.75f));

        __nv_bfloat162 c5 = __float2bfloat162_rn(0.0001573563f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(-0.0034027100f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0257568359f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0722656250f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(-0.0046386719f);
        __nv_bfloat162 c0 = __float2bfloat162_rn(0.2500000000f);

        __nv_bfloat162 r = __hfma2(t, c5, c4);
        r = __hfma2(t, r, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

struct SIGMOID_BWD_D6_EVEN_BF16 {
    // Li=7.50, Lc=4.25, Err=0.009759 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(4.25f));

        __nv_bfloat162 c6 = __float2bfloat162_rn(-0.0000252724f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(0.0007286072f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(-0.0081787109f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0439453125f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.1025390625f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(0.0117797852f);
        __nv_bfloat162 c0 = __float2bfloat162_rn(0.2500000000f);

        __nv_bfloat162 r = __hfma2(t, c6, c5);
        r = __hfma2(t, r, c4);
        r = __hfma2(t, r, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

// =============================================================================
// TANH BWD — EVEN: tanh'(|x|) = 1 - tanh(|x|)^2
// =============================================================================

struct TANH_BWD_D3_EVEN_BF16 {
    // Li=2.25, Lc=2.00, Err=0.046194 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.0f));

        __nv_bfloat162 c3 = __float2bfloat162_rn(0.1308593750f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.3320312500f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(-0.3398437500f);
        __nv_bfloat162 c0 = __float2bfloat162_rn(1.0000000000f);

        __nv_bfloat162 r = __hfma2(t, c3, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

struct TANH_BWD_D4_EVEN_BF16 {
    // Li=2.25, Lc=2.25, Err=0.032650 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.25f));

        __nv_bfloat162 c4 = __float2bfloat162_rn(-0.1396484375f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.7578125000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-1.1875000000f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(-0.0078735352f);
        __nv_bfloat162 c0 = __float2bfloat162_rn(1.0000000000f);

        __nv_bfloat162 r = __hfma2(t, c4, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

struct TANH_BWD_D5_EVEN_BF16 {
    // Li=4.25, Lc=2.25, Err=0.038063 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.25f));

        __nv_bfloat162 c5 = __float2bfloat162_rn(0.0129394531f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(-0.1533203125f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.6289062500f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.9257812500f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(-0.1210937500f);
        __nv_bfloat162 c0 = __float2bfloat162_rn(1.0000000000f);

        __nv_bfloat162 r = __hfma2(t, c5, c4);
        r = __hfma2(t, r, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

struct TANH_BWD_D6_EVEN_BF16 {
    // Li=5.75, Lc=2.25, Err=0.036270 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.25f));

        __nv_bfloat162 c6 = __float2bfloat162_rn(-0.0016632080f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(0.0319824219f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(-0.2324218750f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.7773437500f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-1.0468750000f);
        __nv_bfloat162 c1 = __float2bfloat162_rn(-0.0903320312f);
        __nv_bfloat162 c0 = __float2bfloat162_rn(1.0000000000f);

        __nv_bfloat162 r = __hfma2(t, c6, c5);
        r = __hfma2(t, r, c4);
        r = __hfma2(t, r, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);
        return r;
    }
};

// =============================================================================
// SWISH BWD — ODD: swish'(x) = 0.5 + sign(x)*h(|x|)
// =============================================================================

struct SWISH_BWD_D3_ODD_BF16 {
    // Li=4.75, Lc=4.75, Err=0.021872 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(4.75f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.5820312500f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.1777343750f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0164794922f);

        __nv_bfloat162 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct SWISH_BWD_D4_ODD_BF16 {
    // Li=7.25, Lc=5.25, Err=0.018683 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(5.25f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.6093750000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.2128906250f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0296630859f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(-0.0014572144f);

        __nv_bfloat162 h = __hfma2(t, c4, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct SWISH_BWD_D5_ODD_BF16 {
    // Li=6.25, Lc=4.75, Err=0.019464 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(4.75f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.5742187500f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.1582031250f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0034484863f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0034790039f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0003204346f);

        __nv_bfloat162 h = __hfma2(t, c5, c4);
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

struct SWISH_BWD_D6_ODD_BF16 {
    // Li=10.00, Lc=5.00, Err=0.019176 (BF16-optimized)
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(5.0f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.6015625000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.2050781250f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0260009766f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(-0.0004787445f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0001363754f);
        __nv_bfloat162 c6 = __float2bfloat162_rn(0.0000074804f);

        __nv_bfloat162 h = __hfma2(t, c6, c5);
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
// ERF FWD — ODD: erf(x) = sign(x)*h(|x|)
// NOTE: Using FP16-derived coefficients (no BF16 fitter yet)
// =============================================================================

struct ERF_FWD_D3_ODD_BF16 {
    // Li=2.0, Lc=2.0, Err=0.015391
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.0f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(1.2919921875f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.5253906250f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0639648438f);

        __nv_bfloat162 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        return *reinterpret_cast<__nv_bfloat162*>(&h_bits);
    }
};

struct ERF_FWD_D4_ODD_BF16 {
    // Li=2.0, Lc=2.0, Err=0.004819
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.0f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(1.1767578125f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.1896972656f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.2137451172f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0694580078f);

        __nv_bfloat162 h = __hfma2(t, c4, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        return *reinterpret_cast<__nv_bfloat162*>(&h_bits);
    }
};

struct ERF_FWD_D5_ODD_BF16 {
    // Li=2.25, Lc=2.0, Err=0.003449
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.0f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(1.1240234375f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(0.0565795898f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.5727539062f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.2763671875f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0411682129f);

        __nv_bfloat162 h = __hfma2(t, c5, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        return *reinterpret_cast<__nv_bfloat162*>(&h_bits);
    }
};

struct ERF_FWD_D6_ODD_BF16 {
    // Li=2.25, Lc=2.25, Err=0.003705
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(2.25f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(1.1171875000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(0.0983276367f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.6562500000f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.3493652344f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0702514648f);
        __nv_bfloat162 c6 = __float2bfloat162_rn(0.0043144226f);

        __nv_bfloat162 h = __hfma2(t, c6, c5);
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        return *reinterpret_cast<__nv_bfloat162*>(&h_bits);
    }
};

// =============================================================================
// GELU FWD — ODD: gelu(x) = 0.5*x*(1 + erf(x/sqrt(2)))
// NOTE: Using FP16-derived coefficients (no BF16 fitter yet)
// =============================================================================

struct GELU_FWD_D3_ODD_BF16 {
    // Li=4.5, Lc=3.0, Err=0.008349
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.4592285156f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.1364746094f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0131378174f);

        __nv_bfloat162 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        __nv_bfloat162 phi = __hadd2(__float2bfloat162_rn(0.5f), h_signed);
        return __hmul2(val, phi);
    }
};

struct GELU_FWD_D4_ODD_BF16 {
    // Li=3.0, Lc=3.0, Err=0.002767
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.4208984375f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.0592346191f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.0298614502f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0071029663f);

        __nv_bfloat162 h = __hfma2(t, c4, c3);
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

struct GELU_FWD_D5_ODD_BF16 {
    // Li=3.0, Lc=3.0, Err=0.001600
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.3962402344f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(0.0178222656f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.1052856445f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0362548828f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0038852692f);

        __nv_bfloat162 h = __hfma2(t, c5, c4);
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

struct GELU_FWD_D6_ODD_BF16 {
    // Li=3.5, Lc=3.0, Err=0.001976
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.3935546875f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(0.0309448242f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.1257324219f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0500793457f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0081100464f);
        __nv_bfloat162 c6 = __float2bfloat162_rn(0.0004782677f);

        __nv_bfloat162 h = __hfma2(t, c6, c5);
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
// GELU BWD — ODD
// NOTE: Using FP16-derived coefficients (no BF16 fitter yet)
// =============================================================================

struct GELU_BWD_D3_ODD_BF16 {
    // Li=3.25, Lc=3.0, Err=0.018993
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.9707031250f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.4719238281f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0680541992f);

        __nv_bfloat162 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct GELU_BWD_D4_ODD_BF16 {
    // Li=3.0, Lc=3.0, Err=0.013880
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.9306640625f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.3867187500f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(0.0167694092f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0092239380f);

        __nv_bfloat162 h = __hfma2(t, c4, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __nv_bfloat162 h_signed = *reinterpret_cast<__nv_bfloat162*>(&h_bits);
        return __hadd2(__float2bfloat162_rn(0.5f), h_signed);
    }
};

struct GELU_BWD_D5_ODD_BF16 {
    // Li=3.75, Lc=3.0, Err=0.010052
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.8706054688f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(-0.2055664062f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.1563720703f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.0755004883f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0088424683f);

        __nv_bfloat162 h = __hfma2(t, c5, c4);
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

struct GELU_BWD_D6_ODD_BF16 {
    // Li=4.5, Lc=3.0, Err=0.010358
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __nv_bfloat162 abs_val = *reinterpret_cast<__nv_bfloat162*>(&abs_bits);
        __nv_bfloat162 t = __hmin2(abs_val, __float2bfloat162_rn(3.0f));

        __nv_bfloat162 c1 = __float2bfloat162_rn(0.8125000000f);
        __nv_bfloat162 c2 = __float2bfloat162_rn(0.0046691895f);
        __nv_bfloat162 c3 = __float2bfloat162_rn(-0.4064941406f);
        __nv_bfloat162 c4 = __float2bfloat162_rn(0.2064208984f);
        __nv_bfloat162 c5 = __float2bfloat162_rn(-0.0401306152f);
        __nv_bfloat162 c6 = __float2bfloat162_rn(0.0027942657f);

        __nv_bfloat162 h = __hfma2(t, c6, c5);
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
// ALGEBRAIC BACKWARD PASSES
// =============================================================================

struct SIGMOID_BWD_ALGEBRAIC_BF16 {
    // gi = go * y * (1-y), where y = sigmoid(x) is cached from forward
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 y) {
        __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
        return __hmul2(y, __hsub2(one, y));
    }
};

struct TANH_BWD_ALGEBRAIC_BF16 {
    // gi = go * (1 - y²), where y = tanh(x) is cached from forward
    static __device__ __forceinline__ __nv_bfloat162 evaluate(__nv_bfloat162 y) {
        __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
        return __hsub2(one, __hmul2(y, y));
    }
};

// =============================================================================
// BACKWARD COMPATIBILITY ALIASES
// =============================================================================
#ifndef SPLINE_STRUCTS_NO_ALIASES
using SIGMOID_N2_D3_ODD_BF16 = SIGMOID_FWD_D3_ODD_BF16;
using SPLINE_TANH_FWD_D3_BF16 = TANH_FWD_D3_ODD_BF16;
using SPLINE_SIGMOID_GRAD_D4_BF16 = SIGMOID_BWD_D4_EVEN_BF16;
using SPLINE_TANH_GRAD_D4_BF16 = TANH_BWD_D4_EVEN_BF16;
using SPLINE_SWISH_GRAD_D3_ODD_BF16 = SWISH_BWD_D3_ODD_BF16;
using SPLINE_SWISH_GRAD_D4_ODD_BF16 = SWISH_BWD_D4_ODD_BF16;
using SPLINE_SWISH_GRAD_D5_ODD_BF16 = SWISH_BWD_D5_ODD_BF16;
using SPLINE_SWISH_GRAD_D6_ODD_BF16 = SWISH_BWD_D6_ODD_BF16;
using SWISH_FWD_D3_FUSED_ODD_BF16 = SWISH_FWD_D3_ODD_BF16;
#endif // SPLINE_STRUCTS_NO_ALIASES

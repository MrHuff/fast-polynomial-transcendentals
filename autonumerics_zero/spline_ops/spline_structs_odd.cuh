// spline_structs_odd.cuh — D3-D6 ODD/EVEN activation structs
// Generated from 2D (Li,Lc) sweep with FP16 Horner simulation.
#pragma once
#include <cuda_fp16.h>

// =============================================================================
// SIGMOID FWD — ODD: sigmoid(x) = 0.5 + sign(x)*h(|x|)
// =============================================================================

struct SIGMOID_FWD_D3_ODD {
    // Li=6.5, Lc=6.25, Err=0.005562
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(6.25f));

        __half2 c1 = __float2half2_rn(0.2812500000f);
        __half2 c2 = __float2half2_rn(-0.0534362793f);
        __half2 c3 = __float2half2_rn(0.0033779144f);

        __half2 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        return __hadd2(__float2half2_rn(0.5f), h_signed);
    }
};

struct SIGMOID_FWD_D4_ODD {
    // Li=4.75, Lc=5.0, Err=0.004261
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(5.0f));

        __half2 c1 = __float2half2_rn(0.2690429688f);
        __half2 c2 = __float2half2_rn(-0.0365905762f);
        __half2 c3 = __float2half2_rn(-0.0029220581f);
        __half2 c4 = __float2half2_rn(0.0006918907f);

        __half2 h = __hfma2(t, c4, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        return __hadd2(__float2half2_rn(0.5f), h_signed);
    }
};

struct SIGMOID_FWD_D5_ODD {
    // Li=6.0, Lc=5.5, Err=0.002594
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(5.5f));

        __half2 c1 = __float2half2_rn(0.2607421875f);
        __half2 c2 = __float2half2_rn(-0.0205383301f);
        __half2 c3 = __float2half2_rn(-0.0126800537f);
        __half2 c4 = __float2half2_rn(0.0030593872f);
        __half2 c5 = __float2half2_rn(-0.0001997948f);

        __half2 h = __hfma2(t, c5, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        return __hadd2(__float2half2_rn(0.5f), h_signed);
    }
};

struct SIGMOID_FWD_D6_ODD {
    // Li=7.25, Lc=5.5, Err=0.003168
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(5.5f));

        __half2 c1 = __float2half2_rn(0.2539062500f);
        __half2 c2 = __float2half2_rn(-0.0053977966f);
        __half2 c3 = __float2half2_rn(-0.0239105225f);
        __half2 c4 = __float2half2_rn(0.0067253113f);
        __half2 c5 = __float2half2_rn(-0.0007462502f);
        __half2 c6 = __float2half2_rn(0.0000304580f);

        __half2 h = __hfma2(t, c6, c5);
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        return __hadd2(__float2half2_rn(0.5f), h_signed);
    }
};

// =============================================================================
// TANH FWD — ODD: tanh(x) = sign(x)*h(|x|)
// =============================================================================

struct TANH_FWD_D3_ODD {
    // Li=3.25, Lc=3.25, Err=0.011089
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(3.25f));

        __half2 c1 = __float2half2_rn(1.1250000000f);
        __half2 c2 = __float2half2_rn(-0.4274902344f);
        __half2 c3 = __float2half2_rn(0.0540466309f);

        __half2 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        return *reinterpret_cast<__half2*>(&h_bits);
    }
};

struct TANH_FWD_D4_ODD {
    // Li=2.75, Lc=2.75, Err=0.009193
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(2.75f));

        __half2 c1 = __float2half2_rn(1.1064453125f);
        __half2 c2 = __float2half2_rn(-0.3747558594f);
        __half2 c3 = __float2half2_rn(0.0160827637f);
        __half2 c4 = __float2half2_rn(0.0078811646f);

        __half2 h = __hfma2(t, c4, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        return *reinterpret_cast<__half2*>(&h_bits);
    }
};

struct TANH_FWD_D5_ODD {
    // Li=3.0, Lc=2.75, Err=0.005188
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(2.75f));

        __half2 c1 = __float2half2_rn(1.0429687500f);
        __half2 c2 = __float2half2_rn(-0.1643066406f);
        __half2 c3 = __float2half2_rn(-0.2028808594f);
        __half2 c4 = __float2half2_rn(0.0979003906f);
        __half2 c5 = __float2half2_rn(-0.0127868652f);

        __half2 h = __hfma2(t, c5, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        return *reinterpret_cast<__half2*>(&h_bits);
    }
};

struct TANH_FWD_D6_ODD {
    // Li=4.5, Lc=3.0, Err=0.006221
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(3.0f));

        __half2 c1 = __float2half2_rn(1.0556640625f);
        __half2 c2 = __float2half2_rn(-0.1966552734f);
        __half2 c3 = __float2half2_rn(-0.1898193359f);
        __half2 c4 = __float2half2_rn(0.1103515625f);
        __half2 c5 = __float2half2_rn(-0.0219879150f);
        __half2 c6 = __float2half2_rn(0.0015487671f);

        __half2 h = __hfma2(t, c6, c5);
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        return *reinterpret_cast<__half2*>(&h_bits);
    }
};

// =============================================================================
// SWISH FWD — Fused: swish(x) = 0.5*x + |x|*h(|x|)
// =============================================================================

struct SWISH_FWD_D3_ODD {
    // Reuses sigmoid ODD, Li=6.5, Lc=6.25
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(6.25f));

        __half2 c1 = __float2half2_rn(0.2812500000f);
        __half2 c2 = __float2half2_rn(-0.0534362793f);
        __half2 c3 = __float2half2_rn(0.0033779144f);

        __half2 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        return __hfma2(abs_val, h, __hmul2(__float2half2_rn(0.5f), val));
    }
};

struct SWISH_FWD_D4_ODD {
    // Reuses sigmoid ODD, Li=4.75, Lc=5.0
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(5.0f));

        __half2 c1 = __float2half2_rn(0.2690429688f);
        __half2 c2 = __float2half2_rn(-0.0365905762f);
        __half2 c3 = __float2half2_rn(-0.0029220581f);
        __half2 c4 = __float2half2_rn(0.0006918907f);

        __half2 h = __hfma2(t, c4, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        return __hfma2(abs_val, h, __hmul2(__float2half2_rn(0.5f), val));
    }
};

struct SWISH_FWD_D5_ODD {
    // Reuses sigmoid ODD, Li=6.0, Lc=5.5
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(5.5f));

        __half2 c1 = __float2half2_rn(0.2607421875f);
        __half2 c2 = __float2half2_rn(-0.0205383301f);
        __half2 c3 = __float2half2_rn(-0.0126800537f);
        __half2 c4 = __float2half2_rn(0.0030593872f);
        __half2 c5 = __float2half2_rn(-0.0001997948f);

        __half2 h = __hfma2(t, c5, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        return __hfma2(abs_val, h, __hmul2(__float2half2_rn(0.5f), val));
    }
};

struct SWISH_FWD_D6_ODD {
    // Reuses sigmoid ODD, Li=7.25, Lc=5.5
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(5.5f));

        __half2 c1 = __float2half2_rn(0.2539062500f);
        __half2 c2 = __float2half2_rn(-0.0053977966f);
        __half2 c3 = __float2half2_rn(-0.0239105225f);
        __half2 c4 = __float2half2_rn(0.0067253113f);
        __half2 c5 = __float2half2_rn(-0.0007462502f);
        __half2 c6 = __float2half2_rn(0.0000304580f);

        __half2 h = __hfma2(t, c6, c5);
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        return __hfma2(abs_val, h, __hmul2(__float2half2_rn(0.5f), val));
    }
};

// =============================================================================
// SIGMOID BWD — EVEN: sigmoid'(x) = poly(|x|)
// =============================================================================

struct SIGMOID_BWD_D3_EVEN {
    // Li=4.5, Lc=4.0, Err=0.011002
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(4.0f));

        __half2 c0 = __float2half2_rn(0.2500000000f);
        __half2 c1 = __float2half2_rn(-0.0423889160f);
        __half2 c2 = __float2half2_rn(-0.0207824707f);
        __half2 c3 = __float2half2_rn(0.0040893555f);

        __half2 r = __hfma2(t, c3, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);

        return r;
    }
};

struct SIGMOID_BWD_D4_EVEN {
    // Li=5.25, Lc=5.25, Err=0.004201
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(5.25f));

        __half2 c0 = __float2half2_rn(0.2500000000f);
        __half2 c1 = __float2half2_rn(-0.0103073120f);
        __half2 c2 = __float2half2_rn(-0.0606994629f);
        __half2 c3 = __float2half2_rn(0.0183105469f);
        __half2 c4 = __float2half2_rn(-0.0015373230f);

        __half2 r = __hfma2(t, c4, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);

        return r;
    }
};

struct SIGMOID_BWD_D5_EVEN {
    // Li=5.75, Lc=5.5, Err=0.002840
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(5.5f));

        __half2 c0 = __float2half2_rn(0.2500000000f);
        __half2 c1 = __float2half2_rn(0.0081405640f);
        __half2 c2 = __float2half2_rn(-0.0934448242f);
        __half2 c3 = __float2half2_rn(0.0364685059f);
        __half2 c4 = __float2half2_rn(-0.0055160522f);
        __half2 c5 = __float2half2_rn(0.0003011227f);

        __half2 r = __hfma2(t, c5, c4);
        r = __hfma2(t, r, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);

        return r;
    }
};

struct SIGMOID_BWD_D6_EVEN {
    // Li=6.25, Lc=5.25, Err=0.002963
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(5.25f));

        __half2 c0 = __float2half2_rn(0.2500000000f);
        __half2 c1 = __float2half2_rn(0.0112609863f);
        __half2 c2 = __float2half2_rn(-0.1012573242f);
        __half2 c3 = __float2half2_rn(0.0428466797f);
        __half2 c4 = __float2half2_rn(-0.0077819824f);
        __half2 c5 = __float2half2_rn(0.0006651878f);
        __half2 c6 = __float2half2_rn(-0.0000216961f);

        __half2 r = __hfma2(t, c6, c5);
        r = __hfma2(t, r, c4);
        r = __hfma2(t, r, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);

        return r;
    }
};

// =============================================================================
// TANH BWD — EVEN: tanh'(x) = poly(|x|)
// =============================================================================

struct TANH_BWD_D3_EVEN {
    // Li=2.25, Lc=2.0, Err=0.044007
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(2.0f));

        __half2 c0 = __float2half2_rn(1.0000000000f);
        __half2 c1 = __float2half2_rn(-0.3391113281f);
        __half2 c2 = __float2half2_rn(-0.3325195312f);
        __half2 c3 = __float2half2_rn(0.1308593750f);

        __half2 r = __hfma2(t, c3, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);

        return r;
    }
};

struct TANH_BWD_D4_EVEN {
    // Li=2.5, Lc=2.5, Err=0.019043
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(2.5f));

        __half2 c0 = __float2half2_rn(1.0000000000f);
        __half2 c1 = __float2half2_rn(-0.0553894043f);
        __half2 c2 = __float2half2_rn(-1.0458984375f);
        __half2 c3 = __float2half2_rn(0.6425781250f);
        __half2 c4 = __float2half2_rn(-0.1112060547f);

        __half2 r = __hfma2(t, c4, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);

        return r;
    }
};

struct TANH_BWD_D5_EVEN {
    // Li=3.25, Lc=2.75, Err=0.015758
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(2.75f));

        __half2 c0 = __float2half2_rn(1.0000000000f);
        __half2 c1 = __float2half2_rn(0.0315551758f);
        __half2 c2 = __float2half2_rn(-1.3740234375f);
        __half2 c3 = __float2half2_rn(1.0341796875f);
        __half2 c4 = __float2half2_rn(-0.2961425781f);
        __half2 c5 = __float2half2_rn(0.0302276611f);

        __half2 r = __hfma2(t, c5, c4);
        r = __hfma2(t, r, c3);
        r = __hfma2(t, r, c2);
        r = __hfma2(t, r, c1);
        r = __hfma2(t, r, c0);

        return r;
    }
};

struct TANH_BWD_D6_EVEN {
    // Li=4.75, Lc=3.25, Err=0.017399
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(3.25f));

        __half2 c0 = __float2half2_rn(1.0000000000f);
        __half2 c1 = __float2half2_rn(0.0289306641f);
        __half2 c2 = __float2half2_rn(-1.4033203125f);
        __half2 c3 = __float2half2_rn(1.1250000000f);
        __half2 c4 = __float2half2_rn(-0.3789062500f);
        __half2 c5 = __float2half2_rn(0.0596923828f);
        __half2 c6 = __float2half2_rn(-0.0036144257f);

        __half2 r = __hfma2(t, c6, c5);
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

struct SWISH_BWD_D3_ODD {
    // Li=5.25, Lc=5.0, Err=0.017511
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(5.0f));

        __half2 c1 = __float2half2_rn(0.5703125000f);
        __half2 c2 = __float2half2_rn(-0.1676025391f);
        __half2 c3 = __float2half2_rn(0.0148391724f);

        __half2 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        return __hadd2(__float2half2_rn(0.5f), h_signed);
    }
};

struct SWISH_BWD_D4_ODD {
    // Li=8.0, Lc=5.5, Err=0.014858
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(5.5f));

        __half2 c1 = __float2half2_rn(0.5996093750f);
        __half2 c2 = __float2half2_rn(-0.2036132812f);
        __half2 c3 = __float2half2_rn(0.0272979736f);
        __half2 c4 = __float2half2_rn(-0.0012779236f);

        __half2 h = __hfma2(t, c4, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        return __hadd2(__float2half2_rn(0.5f), h_signed);
    }
};

struct SWISH_BWD_D5_ODD {
    // Li=6.25, Lc=6.25, Err=0.010503
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(6.25f));

        __half2 c1 = __float2half2_rn(0.5742187500f);
        __half2 c2 = __float2half2_rn(-0.1582031250f);
        __half2 c3 = __float2half2_rn(0.0034561157f);
        __half2 c4 = __float2half2_rn(0.0034828186f);
        __half2 c5 = __float2half2_rn(-0.0003206730f);

        __half2 h = __hfma2(t, c5, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        return __hadd2(__float2half2_rn(0.5f), h_signed);
    }
};

struct SWISH_BWD_D6_ODD {
    // Li=7.25, Lc=5.75, Err=0.010162
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(5.75f));

        __half2 c1 = __float2half2_rn(0.5415039062f);
        __half2 c2 = __float2half2_rn(-0.0898437500f);
        __half2 c3 = __float2half2_rn(-0.0439147949f);
        __half2 c4 = __float2half2_rn(0.0179443359f);
        __half2 c5 = __float2half2_rn(-0.0023403168f);
        __half2 c6 = __float2half2_rn(0.0001055002f);

        __half2 h = __hfma2(t, c6, c5);
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        return __hadd2(__float2half2_rn(0.5f), h_signed);
    }
};

// =============================================================================
// ALGEBRAIC BACKWARD — uses cached forward output y, not raw input x
// sigmoid_bwd: go * y * (1-y)    — evaluate returns y*(1-y)
// tanh_bwd:    go * (1 - y²)     — evaluate returns 1-y²
// =============================================================================

struct SIGMOID_BWD_ALGEBRAIC {
    // gi = go * y * (1-y), where y = sigmoid(x) is cached from forward
    static __device__ __forceinline__ __half2 evaluate(__half2 y) {
        __half2 one = __float2half2_rn(1.0f);
        return __hmul2(y, __hsub2(one, y));
    }
};

struct TANH_BWD_ALGEBRAIC {
    // gi = go * (1 - y²), where y = tanh(x) is cached from forward
    static __device__ __forceinline__ __half2 evaluate(__half2 y) {
        __half2 one = __float2half2_rn(1.0f);
        return __hsub2(one, __hmul2(y, y));
    }
};

// =============================================================================
// ERF FWD — ODD: erf(x) = sign(x)*p(|x|)
// =============================================================================

struct ERF_FWD_D3_ODD {
    // Li=2.0, Lc=2.0, Err=0.015391
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(2.0f));

        __half2 c1 = __float2half2_rn(1.2919921875f);
        __half2 c2 = __float2half2_rn(-0.5253906250f);
        __half2 c3 = __float2half2_rn(0.0639648438f);

        __half2 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        return *reinterpret_cast<__half2*>(&h_bits);
    }
};

struct ERF_FWD_D4_ODD {
    // Li=2.0, Lc=2.0, Err=0.004819
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(2.0f));

        __half2 c1 = __float2half2_rn(1.1767578125f);
        __half2 c2 = __float2half2_rn(-0.1896972656f);
        __half2 c3 = __float2half2_rn(-0.2137451172f);
        __half2 c4 = __float2half2_rn(0.0694580078f);

        __half2 h = __hfma2(t, c4, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        return *reinterpret_cast<__half2*>(&h_bits);
    }
};

struct ERF_FWD_D5_ODD {
    // Li=2.25, Lc=2.0, Err=0.003449
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(2.0f));

        __half2 c1 = __float2half2_rn(1.1240234375f);
        __half2 c2 = __float2half2_rn(0.0565795898f);
        __half2 c3 = __float2half2_rn(-0.5727539062f);
        __half2 c4 = __float2half2_rn(0.2763671875f);
        __half2 c5 = __float2half2_rn(-0.0411682129f);

        __half2 h = __hfma2(t, c5, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        return *reinterpret_cast<__half2*>(&h_bits);
    }
};

struct ERF_FWD_D6_ODD {
    // Li=2.25, Lc=2.25, Err=0.003705
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(2.25f));

        __half2 c1 = __float2half2_rn(1.1171875000f);
        __half2 c2 = __float2half2_rn(0.0983276367f);
        __half2 c3 = __float2half2_rn(-0.6562500000f);
        __half2 c4 = __float2half2_rn(0.3493652344f);
        __half2 c5 = __float2half2_rn(-0.0702514648f);
        __half2 c6 = __float2half2_rn(0.0043144226f);

        __half2 h = __hfma2(t, c6, c5);
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        return *reinterpret_cast<__half2*>(&h_bits);
    }
};

// =============================================================================
// GELU FWD — ODD: Phi(x) = 0.5 + sign(x)*h(|x|), GELU(x) = x*Phi(x)
// h(|x|) = 0.5*erf(|x|/sqrt(2))
// =============================================================================

struct GELU_FWD_D3_ODD {
    // Li=4.5, Lc=3.0, Err=0.008349
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(3.0f));

        __half2 c1 = __float2half2_rn(0.4592285156f);
        __half2 c2 = __float2half2_rn(-0.1364746094f);
        __half2 c3 = __float2half2_rn(0.0131378174f);

        __half2 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        __half2 phi = __hadd2(__float2half2_rn(0.5f), h_signed);
        return __hmul2(val, phi);
    }
};

struct GELU_FWD_D4_ODD {
    // Li=3.0, Lc=3.0, Err=0.002767
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(3.0f));

        __half2 c1 = __float2half2_rn(0.4208984375f);
        __half2 c2 = __float2half2_rn(-0.0592346191f);
        __half2 c3 = __float2half2_rn(-0.0298614502f);
        __half2 c4 = __float2half2_rn(0.0071029663f);

        __half2 h = __hfma2(t, c4, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        __half2 phi = __hadd2(__float2half2_rn(0.5f), h_signed);
        return __hmul2(val, phi);
    }
};

struct GELU_FWD_D5_ODD {
    // Li=3.0, Lc=3.0, Err=0.001600
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(3.0f));

        __half2 c1 = __float2half2_rn(0.3962402344f);
        __half2 c2 = __float2half2_rn(0.0178222656f);
        __half2 c3 = __float2half2_rn(-0.1052856445f);
        __half2 c4 = __float2half2_rn(0.0362548828f);
        __half2 c5 = __float2half2_rn(-0.0038852692f);

        __half2 h = __hfma2(t, c5, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        __half2 phi = __hadd2(__float2half2_rn(0.5f), h_signed);
        return __hmul2(val, phi);
    }
};

struct GELU_FWD_D6_ODD {
    // Li=3.5, Lc=3.0, Err=0.001976
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(3.0f));

        __half2 c1 = __float2half2_rn(0.3935546875f);
        __half2 c2 = __float2half2_rn(0.0309448242f);
        __half2 c3 = __float2half2_rn(-0.1257324219f);
        __half2 c4 = __float2half2_rn(0.0500793457f);
        __half2 c5 = __float2half2_rn(-0.0081100464f);
        __half2 c6 = __float2half2_rn(0.0004782677f);

        __half2 h = __hfma2(t, c6, c5);
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        __half2 phi = __hadd2(__float2half2_rn(0.5f), h_signed);
        return __hmul2(val, phi);
    }
};

// =============================================================================
// GELU BWD — ODD: GELU'(x) = 0.5 + sign(x)*h(|x|)
// h(|x|) = GELU'(|x|) - 0.5 = 0.5*erf(|x|/√2) + |x|*exp(-|x|²/2)/√(2π)
// =============================================================================

struct GELU_BWD_D3_ODD {
    // Li=3.25, Lc=3.0, Err=0.018993
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(3.0f));

        __half2 c1 = __float2half2_rn(0.9707031250f);
        __half2 c2 = __float2half2_rn(-0.4719238281f);
        __half2 c3 = __float2half2_rn(0.0680541992f);

        __half2 h = __hfma2(t, c3, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        return __hadd2(__float2half2_rn(0.5f), h_signed);
    }
};

struct GELU_BWD_D4_ODD {
    // Li=3.0, Lc=3.0, Err=0.013880
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(3.0f));

        __half2 c1 = __float2half2_rn(0.9306640625f);
        __half2 c2 = __float2half2_rn(-0.3867187500f);
        __half2 c3 = __float2half2_rn(0.0167694092f);
        __half2 c4 = __float2half2_rn(0.0092239380f);

        __half2 h = __hfma2(t, c4, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        return __hadd2(__float2half2_rn(0.5f), h_signed);
    }
};

struct GELU_BWD_D5_ODD {
    // Li=3.75, Lc=3.0, Err=0.010052
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(3.0f));

        __half2 c1 = __float2half2_rn(0.8706054688f);
        __half2 c2 = __float2half2_rn(-0.2055664062f);
        __half2 c3 = __float2half2_rn(-0.1563720703f);
        __half2 c4 = __float2half2_rn(0.0755004883f);
        __half2 c5 = __float2half2_rn(-0.0088424683f);

        __half2 h = __hfma2(t, c5, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        return __hadd2(__float2half2_rn(0.5f), h_signed);
    }
};

struct GELU_BWD_D6_ODD {
    // Li=4.5, Lc=3.0, Err=0.010358
    static __device__ __forceinline__ __half2 evaluate(__half2 val) {
        unsigned int sign_mask = 0x80008000;
        unsigned int input_bits = *reinterpret_cast<unsigned int*>(&val);
        unsigned int signs = input_bits & sign_mask;
        unsigned int abs_bits = input_bits & ~sign_mask;
        __half2 abs_val = *reinterpret_cast<__half2*>(&abs_bits);
        __half2 t = __hmin2(abs_val, __float2half2_rn(3.0f));

        __half2 c1 = __float2half2_rn(0.8125000000f);
        __half2 c2 = __float2half2_rn(0.0046691895f);
        __half2 c3 = __float2half2_rn(-0.4064941406f);
        __half2 c4 = __float2half2_rn(0.2064208984f);
        __half2 c5 = __float2half2_rn(-0.0401306152f);
        __half2 c6 = __float2half2_rn(0.0027942657f);

        __half2 h = __hfma2(t, c6, c5);
        h = __hfma2(t, h, c4);
        h = __hfma2(t, h, c3);
        h = __hfma2(t, h, c2);
        h = __hfma2(t, h, c1);
        h = __hmul2(t, h);

        unsigned int h_bits = *reinterpret_cast<unsigned int*>(&h);
        h_bits ^= signs;
        __half2 h_signed = *reinterpret_cast<__half2*>(&h_bits);
        return __hadd2(__float2half2_rn(0.5f), h_signed);
    }
};

// =============================================================================
// BACKWARD COMPATIBILITY ALIASES
// =============================================================================
#ifndef SPLINE_STRUCTS_NO_ALIASES
using SIGMOID_N2_D3_ODD = SIGMOID_FWD_D3_ODD;
using SPLINE_TANH_FWD_D3 = TANH_FWD_D3_ODD;
using SPLINE_SIGMOID_GRAD_D4 = SIGMOID_BWD_D4_EVEN;
using SPLINE_TANH_GRAD_D4 = TANH_BWD_D4_EVEN;
using SPLINE_SWISH_GRAD_D3_ODD = SWISH_BWD_D3_ODD;
using SPLINE_SWISH_GRAD_D4_ODD = SWISH_BWD_D4_ODD;
using SPLINE_SWISH_GRAD_D5_ODD = SWISH_BWD_D5_ODD;
using SPLINE_SWISH_GRAD_D6_ODD = SWISH_BWD_D6_ODD;
using SWISH_FWD_D3_FUSED_ODD = SWISH_FWD_D3_ODD;
#endif // SPLINE_STRUCTS_NO_ALIASES

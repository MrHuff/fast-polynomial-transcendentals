#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kBlockSize = 256;
constexpr int kValuesPerVector = 4;
constexpr int kMaxBlocks = 4096;

__device__ __forceinline__ float fma_rn(float a, float b, float c) {
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}

__device__ __forceinline__ float xor_sign(float value, unsigned int sign) {
    return __uint_as_float(__float_as_uint(value) ^ sign);
}

__device__ __forceinline__ void reduce_to_quarter_turn(
    float angle,
    float& reduced,
    int& quadrant) {
    constexpr float kInvHalfPi = 0.63661977236758134308f;
    constexpr float kHalfPiHigh = 1.57079625129699707031f;
    constexpr float kHalfPiLow = 7.54978941586159635335e-08f;

    const int quarter_turn = __float2int_rn(angle * kInvHalfPi);
    const float quarter_turn_f = static_cast<float>(quarter_turn);
    reduced = fma_rn(quarter_turn_f, -kHalfPiHigh, angle);
    reduced = fma_rn(quarter_turn_f, -kHalfPiLow, reduced);
    quadrant = quarter_turn & 3;
}

__device__ __forceinline__ void reduce_to_quarter_turn_magic_bias(
    float angle,
    float& reduced,
    int& quadrant) {
    constexpr float kInvHalfPi = 0.63661977236758134308f;
    constexpr float kRoundingBias = 12582912.0f;
    constexpr float kHalfPiHigh = 1.57079625129699707031f;
    constexpr float kHalfPiLow = 7.54978941586159635335e-08f;

    // The low mantissa bits encode the rounded integer while the biased sum
    // remains at unit precision (|angle| < 2^22 * pi/2). CUDA uses the same
    // conversion-free reduction in its simplified half-precision trig path.
    const float biased = fma_rn(angle, kInvHalfPi, kRoundingBias);
    const float quarter_turn = biased - kRoundingBias;
    reduced = fma_rn(quarter_turn, -kHalfPiHigh, angle);
    reduced = fma_rn(quarter_turn, -kHalfPiLow, reduced);
    quadrant = static_cast<int>(__float_as_uint(biased));
}

__device__ __forceinline__ void reduce_to_quarter_turn_magic_bias_fast(
    float angle,
    float& reduced,
    unsigned int& quadrant) {
    constexpr float kInvHalfPi = 0.63661977236758134308f;
    constexpr float kRoundingBias = 12582912.0f;
    constexpr float kHalfPi = 1.57079637050628662109375f;

    const float biased = fma_rn(angle, kInvHalfPi, kRoundingBias);
    const float quarter_turn = biased - kRoundingBias;
    reduced = fma_rn(quarter_turn, -kHalfPi, angle);
    quadrant = __float_as_uint(biased);
}

__device__ __forceinline__ void reduce_to_half_turn_magic_bias(
    float angle,
    float& reduced,
    unsigned int& sign) {
    constexpr float kInvPi = 0.31830988618379067154f;
    constexpr float kRoundingBias = 12582912.0f;
    constexpr float kPiHigh = 3.141592502593994140625f;
    constexpr float kPiLow = 1.50995788317231927067e-07f;

    const float biased = fma_rn(angle, kInvPi, kRoundingBias);
    const float half_turn = biased - kRoundingBias;
    reduced = fma_rn(half_turn, -kPiHigh, angle);
    reduced = fma_rn(half_turn, -kPiLow, reduced);
    sign = (__float_as_uint(biased) & 1U) << 31;
}

__device__ __forceinline__ void reduce_to_half_turn_magic_bias_fast(
    float angle,
    float& reduced,
    unsigned int& sign) {
    constexpr float kInvPi = 0.31830988618379067154f;
    constexpr float kRoundingBias = 12582912.0f;
    constexpr float kPi = 3.1415927410125732421875f;

    // One-term pi is the RoPE-specific speed path. At 8K positions its range
    // error stays inside the approximately 1e-3 phase budget.
    const float biased = fma_rn(angle, kInvPi, kRoundingBias);
    const float half_turn = biased - kRoundingBias;
    reduced = fma_rn(half_turn, -kPi, angle);
    sign = (__float_as_uint(biased) & 1U) << 31;
}

__device__ __forceinline__ void restore_quadrant(
    float sin_reduced,
    float cos_reduced,
    int quadrant,
    float& sin_value,
    float& cos_value) {
    const bool swap = (quadrant & 1) != 0;
    const float sin_magnitude = swap ? cos_reduced : sin_reduced;
    const float cos_magnitude = swap ? sin_reduced : cos_reduced;
    const unsigned int sin_sign = static_cast<unsigned int>(quadrant & 2) << 30;
    const unsigned int cos_sign =
        static_cast<unsigned int>((quadrant + 1) & 2) << 30;
    sin_value = xor_sign(sin_magnitude, sin_sign);
    cos_value = xor_sign(cos_magnitude, cos_sign);
}

struct NativeSincos {
    static __device__ __forceinline__ void evaluate(
        float angle,
        float& cos_value,
        float& sin_value) {
        sincosf(angle, &sin_value, &cos_value);
    }
};

struct PolynomialD3D4 {
    static __device__ __forceinline__ void evaluate(
        float angle,
        float& cos_value,
        float& sin_value) {
        float reduced;
        int quadrant;
        reduce_to_quarter_turn(angle, reduced, quadrant);
        const float squared = reduced * reduced;

        // Sollya fpminimax on [0, pi/4], with FP32 coefficients.
        const float sin_polynomial = fma_rn(
            -0.16034401953220367431640625f,
            squared,
            0.99903142452239990234375f);
        const float sin_reduced = reduced * sin_polynomial;
        float cos_reduced = fma_rn(
            0.0403986163437366485595703125f,
            squared,
            -0.4997082054615020751953125f);
        cos_reduced = fma_rn(
            cos_reduced,
            squared,
            0.999990046024322509765625f);
        restore_quadrant(
            sin_reduced,
            cos_reduced,
            quadrant,
            sin_value,
            cos_value);
    }
};

struct PolynomialD3D4Cycle {
    static __device__ __forceinline__ void evaluate(
        float angle,
        float& cos_value,
        float& sin_value) {
        constexpr float kInvTwoPi = 0.15915494309189533577f;

        // Work in turns, then use sin/cos symmetry to fold into [0, 1/8].
        // This avoids the integer quadrant conversion and two-term pi/2
        // subtraction used by the more accurate Cody-Waite reducer above.
        const float cycles = angle * kInvTwoPi;
        const float wrapped = cycles - nearbyintf(cycles);
        const unsigned int sin_sign =
            __float_as_uint(wrapped) & 0x80000000U;
        float reduced = fabsf(wrapped);

        const bool outer_quarter = reduced > 0.25f;
        reduced = outer_quarter ? 0.5f - reduced : reduced;
        const bool swap = reduced > 0.125f;
        reduced = swap ? 0.25f - reduced : reduced;

        const float squared = reduced * reduced;
        const float sin_polynomial = fma_rn(
            -39.77336883544921875f,
            squared,
            6.277099609375f);
        const float sin_reduced = reduced * sin_polynomial;
        float cos_reduced = fma_rn(
            62.96308135986328125f,
            squared,
            -19.7276897430419921875f);
        cos_reduced = fma_rn(
            cos_reduced,
            squared,
            0.999990046024322509765625f);

        const float sin_magnitude = swap ? cos_reduced : sin_reduced;
        const float cos_magnitude = swap ? sin_reduced : cos_reduced;
        const unsigned int cos_sign =
            static_cast<unsigned int>(outer_quarter) << 31;
        sin_value = xor_sign(sin_magnitude, sin_sign);
        cos_value = xor_sign(cos_magnitude, cos_sign);
    }
};

struct PolynomialD3D4MagicBias {
    static __device__ __forceinline__ void evaluate(
        float angle,
        float& cos_value,
        float& sin_value) {
        float reduced;
        int quadrant;
        reduce_to_quarter_turn_magic_bias(angle, reduced, quadrant);
        const float squared = reduced * reduced;

        const float sin_polynomial = fma_rn(
            -0.16034401953220367431640625f,
            squared,
            0.99903142452239990234375f);
        const float sin_reduced = reduced * sin_polynomial;
        float cos_reduced = fma_rn(
            0.0403986163437366485595703125f,
            squared,
            -0.4997082054615020751953125f);
        cos_reduced = fma_rn(
            cos_reduced,
            squared,
            0.999990046024322509765625f);
        restore_quadrant(
            sin_reduced,
            cos_reduced,
            quadrant,
            sin_value,
            cos_value);
    }
};

__device__ __forceinline__ void evaluate_half_turn_d5_d4(
    float angle,
    float sin_c0,
    float sin_c1,
    float sin_c2,
    float cos_c0,
    float cos_c1,
    float cos_c2,
    float& cos_value,
    float& sin_value) {
    float reduced;
    unsigned int sign;
    reduce_to_half_turn_magic_bias(angle, reduced, sign);
    const float squared = reduced * reduced;

    float sin_polynomial = fma_rn(sin_c2, squared, sin_c1);
    sin_polynomial = fma_rn(sin_polynomial, squared, sin_c0);
    const float sin_reduced = reduced * sin_polynomial;
    float cos_reduced = fma_rn(cos_c2, squared, cos_c1);
    cos_reduced = fma_rn(cos_reduced, squared, cos_c0);
    sin_value = xor_sign(sin_reduced, sign);
    cos_value = xor_sign(cos_reduced, sign);
}

struct PolynomialD5D4HalfTurnLS {
    static __device__ __forceinline__ void evaluate(
        float angle,
        float& cos_value,
        float& sin_value) {
        evaluate_half_turn_d5_d4(
            angle,
            0.999771416187286376953125f,
            -0.16582702100276947021484375f,
            0.00757423602044582366943359375f,
            0.999579489231109619140625f,
            -0.4963922202587127685546875f,
            0.0372092612087726593017578125f,
            cos_value,
            sin_value);
    }
};

struct PolynomialD5D4HalfTurnSollya {
    static __device__ __forceinline__ void evaluate(
        float angle,
        float& cos_value,
        float& sin_value) {
        evaluate_half_turn_d5_d4(
            angle,
            0.999696791172027587890625f,
            -0.1656731069087982177734375f,
            0.0075143859721720218658447265625f,
            0.9994032382965087890625f,
            -0.495580852031707763671875f,
            0.03679168224334716796875f,
            cos_value,
            sin_value);
    }
};

struct PolynomialD5D4HalfTurnSollyaFast {
    static __device__ __forceinline__ void evaluate(
        float angle,
        float& cos_value,
        float& sin_value) {
        float reduced;
        unsigned int sign;
        reduce_to_half_turn_magic_bias_fast(angle, reduced, sign);
        const float squared = reduced * reduced;

        float sin_polynomial = fma_rn(
            0.0075143859721720218658447265625f,
            squared,
            -0.1656731069087982177734375f);
        sin_polynomial = fma_rn(
            sin_polynomial,
            squared,
            0.999696791172027587890625f);
        const float sin_reduced = reduced * sin_polynomial;
        float cos_reduced = fma_rn(
            0.03679168224334716796875f,
            squared,
            -0.495580852031707763671875f);
        cos_reduced = fma_rn(
            cos_reduced,
            squared,
            0.9994032382965087890625f);
        sin_value = xor_sign(sin_reduced, sign);
        cos_value = xor_sign(cos_reduced, sign);
    }
};

struct PolynomialD5D4 {
    static __device__ __forceinline__ void evaluate(
        float angle,
        float& cos_value,
        float& sin_value) {
        float reduced;
        int quadrant;
        reduce_to_quarter_turn(angle, reduced, quadrant);
        const float squared = reduced * reduced;

        float sin_polynomial = fma_rn(
            0.008137642405927181f,
            squared,
            -0.16661198437213898f);
        sin_polynomial = fma_rn(
            sin_polynomial,
            squared,
            0.9999962449073792f);
        const float sin_reduced = reduced * sin_polynomial;
        float cos_reduced = fma_rn(
            0.04051213711500168f,
            squared,
            -0.499763548374176f);
        cos_reduced = fma_rn(
            cos_reduced,
            squared,
            0.999993085861206f);
        restore_quadrant(
            sin_reduced,
            cos_reduced,
            quadrant,
            sin_value,
            cos_value);
    }
};

struct PolynomialD7D6 {
    static __device__ __forceinline__ void evaluate(
        float angle,
        float& cos_value,
        float& sin_value) {
        float reduced;
        int quadrant;
        reduce_to_quarter_turn(angle, reduced, quadrant);
        const float squared = reduced * reduced;

        float sin_polynomial = fma_rn(
            -0.00019484198128338903f,
            squared,
            0.00833179522305727f);
        sin_polynomial = fma_rn(
            sin_polynomial,
            squared,
            -0.1666664183139801f);
        sin_polynomial = fma_rn(
            sin_polynomial,
            squared,
            1.0f);
        const float sin_reduced = reduced * sin_polynomial;
        float cos_reduced = fma_rn(
            -0.0013605882413685322f,
            squared,
            0.041656624525785446f);
        cos_reduced = fma_rn(
            cos_reduced,
            squared,
            -0.49999886751174927f);
        cos_reduced = fma_rn(cos_reduced, squared, 1.0f);
        restore_quadrant(
            sin_reduced,
            cos_reduced,
            quadrant,
            sin_value,
            cos_value);
    }
};

__device__ __forceinline__ unsigned int bfloat162_bits(__nv_bfloat162 value) {
    return *reinterpret_cast<unsigned int*>(&value);
}

__device__ __forceinline__ __nv_bfloat162 bits_bfloat162(unsigned int bits) {
    return *reinterpret_cast<__nv_bfloat162*>(&bits);
}

__device__ __forceinline__ unsigned int half2_bits(__half2 value) {
    return *reinterpret_cast<unsigned int*>(&value);
}

__device__ __forceinline__ __half2 bits_half2(unsigned int bits) {
    return *reinterpret_cast<__half2*>(&bits);
}

__device__ __forceinline__ void restore_quadrant_packed16(
    unsigned int sin_reduced,
    unsigned int cos_reduced,
    unsigned int quadrant_low,
    unsigned int quadrant_high,
    unsigned int& sin_value,
    unsigned int& cos_value) {
    const unsigned int swap_mask =
        ((0U - (quadrant_low & 1U)) & 0x0000ffffU) |
        ((0U - (quadrant_high & 1U)) & 0xffff0000U);
    const unsigned int exchanged =
        (sin_reduced ^ cos_reduced) & swap_mask;
    const unsigned int sin_sign =
        ((quadrant_low & 2U) << 14) | ((quadrant_high & 2U) << 30);
    const unsigned int cos_sign =
        (((quadrant_low + 1U) & 2U) << 14) |
        (((quadrant_high + 1U) & 2U) << 30);
    sin_value = sin_reduced ^ exchanged ^ sin_sign;
    cos_value = cos_reduced ^ exchanged ^ cos_sign;
}

__device__ __forceinline__ void restore_quadrant_half2_pair(
    __half2 sin_reduced,
    __half2 cos_reduced,
    unsigned int quadrant_low,
    unsigned int quadrant_high,
    __half2& sin_value,
    __half2& cos_value) {
    unsigned int sin_bits;
    unsigned int cos_bits;
    restore_quadrant_packed16(
        half2_bits(sin_reduced),
        half2_bits(cos_reduced),
        quadrant_low,
        quadrant_high,
        sin_bits,
        cos_bits);
    sin_value = bits_half2(sin_bits);
    cos_value = bits_half2(cos_bits);
}

__device__ __forceinline__ void restore_quadrant_bf16_lane(
    unsigned int sin_reduced,
    unsigned int cos_reduced,
    int quadrant,
    unsigned int& sin_value,
    unsigned int& cos_value) {
    const bool swap = (quadrant & 1) != 0;
    const unsigned int sin_magnitude = swap ? cos_reduced : sin_reduced;
    const unsigned int cos_magnitude = swap ? sin_reduced : cos_reduced;
    const unsigned int sin_sign = static_cast<unsigned int>(quadrant & 2) << 14;
    const unsigned int cos_sign =
        static_cast<unsigned int>((quadrant + 1) & 2) << 14;
    sin_value = sin_magnitude ^ sin_sign;
    cos_value = cos_magnitude ^ cos_sign;
}

__device__ __forceinline__ void restore_quadrant_bf16_pair(
    __nv_bfloat162 sin_reduced,
    __nv_bfloat162 cos_reduced,
    int quadrant_low,
    int quadrant_high,
    __nv_bfloat162& sin_value,
    __nv_bfloat162& cos_value) {
    const unsigned int sin_bits = bfloat162_bits(sin_reduced);
    const unsigned int cos_bits = bfloat162_bits(cos_reduced);
    unsigned int sin_low;
    unsigned int cos_low;
    unsigned int sin_high;
    unsigned int cos_high;
    restore_quadrant_bf16_lane(
        sin_bits & 0xffffU,
        cos_bits & 0xffffU,
        quadrant_low,
        sin_low,
        cos_low);
    restore_quadrant_bf16_lane(
        sin_bits >> 16,
        cos_bits >> 16,
        quadrant_high,
        sin_high,
        cos_high);
    sin_value = bits_bfloat162(sin_low | (sin_high << 16));
    cos_value = bits_bfloat162(cos_low | (cos_high << 16));
}

struct NativeSincosBF16 {
    static __device__ __forceinline__ void evaluate(
        float angle_low,
        float angle_high,
        __nv_bfloat162& cos_value,
        __nv_bfloat162& sin_value) {
        float cos_low;
        float sin_low;
        float cos_high;
        float sin_high;
        sincosf(angle_low, &sin_low, &cos_low);
        sincosf(angle_high, &sin_high, &cos_high);
        cos_value = __floats2bfloat162_rn(cos_low, cos_high);
        sin_value = __floats2bfloat162_rn(sin_low, sin_high);
    }
};

struct NativeSincosFP16 {
    static __device__ __forceinline__ void evaluate(
        float angle_low,
        float angle_high,
        __half2& cos_value,
        __half2& sin_value) {
        float cos_low;
        float sin_low;
        float cos_high;
        float sin_high;
        sincosf(angle_low, &sin_low, &cos_low);
        sincosf(angle_high, &sin_high, &cos_high);
        cos_value = __floats2half2_rn(cos_low, cos_high);
        sin_value = __floats2half2_rn(sin_low, sin_high);
    }
};

__device__ __forceinline__ void evaluate_quarter_turn_d3_d4_fp16(
    __half2 reduced,
    unsigned int quadrant_low,
    unsigned int quadrant_high,
    __half2& cos_value,
    __half2& sin_value) {
    const __half2 squared = __hmul2(reduced, reduced);
    const __half2 sin_polynomial = __hfma2(
        __float2half2_rn(-0.1602783203125f),
        squared,
        __float2half2_rn(0.9990234375f));
    const __half2 sin_reduced = __hmul2(reduced, sin_polynomial);
    __half2 cos_reduced = __hfma2(
        __float2half2_rn(0.040435791015625f),
        squared,
        __float2half2_rn(-0.499755859375f));
    cos_reduced = __hfma2(
        cos_reduced,
        squared,
        __float2half2_rn(1.0f));
    restore_quadrant_half2_pair(
        sin_reduced,
        cos_reduced,
        quadrant_low,
        quadrant_high,
        sin_value,
        cos_value);
}

struct PolynomialD3D4QuarterTurnFP16 {
    static __device__ __forceinline__ void evaluate(
        float angle_low,
        float angle_high,
        __half2& cos_value,
        __half2& sin_value) {
        float reduced_low;
        float reduced_high;
        unsigned int quadrant_low;
        unsigned int quadrant_high;
        reduce_to_quarter_turn_magic_bias_fast(
            angle_low, reduced_low, quadrant_low);
        reduce_to_quarter_turn_magic_bias_fast(
            angle_high, reduced_high, quadrant_high);

        evaluate_quarter_turn_d3_d4_fp16(
            __floats2half2_rn(reduced_low, reduced_high),
            quadrant_low,
            quadrant_high,
            cos_value,
            sin_value);
    }
};

struct PolynomialD3D4QuarterTurnBF16 {
    static __device__ __forceinline__ void evaluate(
        float angle_low,
        float angle_high,
        __nv_bfloat162& cos_value,
        __nv_bfloat162& sin_value) {
        float reduced_low;
        float reduced_high;
        unsigned int quadrant_low;
        unsigned int quadrant_high;
        reduce_to_quarter_turn_magic_bias_fast(
            angle_low, reduced_low, quadrant_low);
        reduce_to_quarter_turn_magic_bias_fast(
            angle_high, reduced_high, quadrant_high);

        const __nv_bfloat162 reduced =
            __floats2bfloat162_rn(reduced_low, reduced_high);
        const __nv_bfloat162 squared = __hmul2(reduced, reduced);
        const __nv_bfloat162 sin_polynomial = __hfma2(
            __float2bfloat162_rn(-0.162109375f),
            squared,
            __float2bfloat162_rn(1.0f));
        const __nv_bfloat162 sin_reduced =
            __hmul2(reduced, sin_polynomial);
        __nv_bfloat162 cos_reduced = __hfma2(
            __float2bfloat162_rn(0.041015625f),
            squared,
            __float2bfloat162_rn(-0.5f));
        cos_reduced = __hfma2(
            cos_reduced,
            squared,
            __float2bfloat162_rn(1.0f));

        unsigned int sin_bits;
        unsigned int cos_bits;
        restore_quadrant_packed16(
            bfloat162_bits(sin_reduced),
            bfloat162_bits(cos_reduced),
            quadrant_low,
            quadrant_high,
            sin_bits,
            cos_bits);
        sin_value = bits_bfloat162(sin_bits);
        cos_value = bits_bfloat162(cos_bits);
    }
};

struct PolynomialD3D4HalfTurnFP16 {
    static __device__ __forceinline__ void evaluate(
        float angle_low,
        float angle_high,
        __half2& cos_value,
        __half2& sin_value) {
        float reduced_low;
        float reduced_high;
        unsigned int sign_low;
        unsigned int sign_high;
        reduce_to_half_turn_magic_bias_fast(
            angle_low, reduced_low, sign_low);
        reduce_to_half_turn_magic_bias_fast(
            angle_high, reduced_high, sign_high);

        const __half2 reduced = __floats2half2_rn(reduced_low, reduced_high);
        const __half2 squared = __hmul2(reduced, reduced);
        const __half2 sin_polynomial = __hfma2(
            __float2half2_rn(-0.14501953125f),
            squared,
            __float2half2_rn(0.98876953125f));
        const __half2 sin_reduced = __hmul2(reduced, sin_polynomial);
        __half2 cos_reduced = __hfma2(
            __float2half2_rn(0.037200927734375f),
            squared,
            __float2half2_rn(-0.496337890625f));
        cos_reduced = __hfma2(
            cos_reduced,
            squared,
            __float2half2_rn(0.99951171875f));

        const unsigned int signs = (sign_low >> 16) | sign_high;
        sin_value = bits_half2(half2_bits(sin_reduced) ^ signs);
        cos_value = bits_half2(half2_bits(cos_reduced) ^ signs);
    }
};

struct PolynomialD3D4HalfTurnBF16 {
    static __device__ __forceinline__ void evaluate(
        float angle_low,
        float angle_high,
        __nv_bfloat162& cos_value,
        __nv_bfloat162& sin_value) {
        float reduced_low;
        float reduced_high;
        unsigned int sign_low;
        unsigned int sign_high;
        reduce_to_half_turn_magic_bias_fast(
            angle_low, reduced_low, sign_low);
        reduce_to_half_turn_magic_bias_fast(
            angle_high, reduced_high, sign_high);

        const __nv_bfloat162 reduced =
            __floats2bfloat162_rn(reduced_low, reduced_high);
        const __nv_bfloat162 squared = __hmul2(reduced, reduced);
        const __nv_bfloat162 sin_polynomial = __hfma2(
            __float2bfloat162_rn(-0.1455078125f),
            squared,
            __float2bfloat162_rn(0.98828125f));
        const __nv_bfloat162 sin_reduced =
            __hmul2(reduced, sin_polynomial);
        __nv_bfloat162 cos_reduced = __hfma2(
            __float2bfloat162_rn(0.037109375f),
            squared,
            __float2bfloat162_rn(-0.49609375f));
        cos_reduced = __hfma2(
            cos_reduced,
            squared,
            __float2bfloat162_rn(1.0f));

        const unsigned int signs = (sign_low >> 16) | sign_high;
        sin_value = bits_bfloat162(bfloat162_bits(sin_reduced) ^ signs);
        cos_value = bits_bfloat162(bfloat162_bits(cos_reduced) ^ signs);
    }
};

__device__ __forceinline__ void evaluate_half_turn_d5_d4_fp16(
    float angle_low,
    float angle_high,
    __half2& cos_value,
    __half2& sin_value) {
    float reduced_low;
    float reduced_high;
    unsigned int sign_low;
    unsigned int sign_high;
    reduce_to_half_turn_magic_bias_fast(
        angle_low, reduced_low, sign_low);
    reduce_to_half_turn_magic_bias_fast(
        angle_high, reduced_high, sign_high);

    const __half2 reduced = __floats2half2_rn(reduced_low, reduced_high);
    const __half2 squared = __hmul2(reduced, reduced);

    // Coefficients are locally optimized on the FP16 lattice for the
    // rounded HFMA2 evaluation, rather than rounded from the FP32 fit.
    __half2 sin_polynomial = __hfma2(
        __float2half2_rn(0.0074920654296875f),
        squared,
        __float2half2_rn(-0.16552734375f));
    sin_polynomial = __hfma2(
        sin_polynomial,
        squared,
        __float2half2_rn(0.99951171875f));
    __half2 sin_reduced = __hmul2(reduced, sin_polynomial);

    __half2 cos_reduced = __hfma2(
        __float2half2_rn(0.036773681640625f),
        squared,
        __float2half2_rn(-0.49560546875f));
    cos_reduced = __hfma2(
        cos_reduced,
        squared,
        __float2half2_rn(0.99951171875f));

    const unsigned int signs = (sign_low >> 16) | sign_high;
    sin_value = bits_half2(half2_bits(sin_reduced) ^ signs);
    cos_value = bits_half2(half2_bits(cos_reduced) ^ signs);
}

struct PolynomialD5D4HalfTurnFP16 {
    static __device__ __forceinline__ void evaluate(
        float angle_low,
        float angle_high,
        __half2& cos_value,
        __half2& sin_value) {
        evaluate_half_turn_d5_d4_fp16(
            angle_low, angle_high, cos_value, sin_value);
    }
};

struct PolynomialD7D6HalfTurnFP16 {
    static __device__ __forceinline__ void evaluate(
        float angle_low,
        float angle_high,
        __half2& cos_value,
        __half2& sin_value) {
        float reduced_low;
        float reduced_high;
        unsigned int sign_low;
        unsigned int sign_high;
        reduce_to_half_turn_magic_bias_fast(
            angle_low, reduced_low, sign_low);
        reduce_to_half_turn_magic_bias_fast(
            angle_high, reduced_high, sign_high);

        const __half2 reduced = __floats2half2_rn(reduced_low, reduced_high);
        const __half2 squared = __hmul2(reduced, reduced);

        // FP16-lattice coefficients minimize max phase/component error on
        // the complete 8K, theta=500000 RoPE angle table.
        __half2 sin_polynomial = __hfma2(
            __float2half2_rn(-0.00018227100372314453f),
            squared,
            __float2half2_rn(0.00829315185546875f));
        sin_polynomial = __hfma2(
            sin_polynomial,
            squared,
            __float2half2_rn(-0.1666259765625f));
        sin_polynomial = __hfma2(
            sin_polynomial,
            squared,
            __float2half2_rn(1.0f));
        __half2 sin_reduced = __hmul2(reduced, sin_polynomial);

        __half2 cos_reduced = __hfma2(
            __float2half2_rn(-0.0012674331665039062f),
            squared,
            __float2half2_rn(0.041412353515625f));
        cos_reduced = __hfma2(
            cos_reduced,
            squared,
            __float2half2_rn(-0.499755859375f));
        cos_reduced = __hfma2(
            cos_reduced,
            squared,
            __float2half2_rn(1.0f));

        const unsigned int signs = (sign_low >> 16) | sign_high;
        sin_value = bits_half2(half2_bits(sin_reduced) ^ signs);
        cos_value = bits_half2(half2_bits(cos_reduced) ^ signs);
    }
};

struct PolynomialD5D6HalfTurnFP16 {
    static __device__ __forceinline__ void evaluate(
        float angle_low,
        float angle_high,
        __half2& cos_value,
        __half2& sin_value) {
        float reduced_low;
        float reduced_high;
        unsigned int sign_low;
        unsigned int sign_high;
        reduce_to_half_turn_magic_bias_fast(
            angle_low, reduced_low, sign_low);
        reduce_to_half_turn_magic_bias_fast(
            angle_high, reduced_high, sign_high);

        const __half2 reduced = __floats2half2_rn(reduced_low, reduced_high);
        const __half2 squared = __hmul2(reduced, reduced);

        __half2 sin_polynomial = __hfma2(
            __float2half2_rn(0.0074920654296875f),
            squared,
            __float2half2_rn(-0.16552734375f));
        sin_polynomial = __hfma2(
            sin_polynomial,
            squared,
            __float2half2_rn(0.99951171875f));
        __half2 sin_reduced = __hmul2(reduced, sin_polynomial);

        __half2 cos_reduced = __hfma2(
            __float2half2_rn(-0.0012674331665039062f),
            squared,
            __float2half2_rn(0.041412353515625f));
        cos_reduced = __hfma2(
            cos_reduced,
            squared,
            __float2half2_rn(-0.499755859375f));
        cos_reduced = __hfma2(
            cos_reduced,
            squared,
            __float2half2_rn(1.0f));

        const unsigned int signs = (sign_low >> 16) | sign_high;
        sin_value = bits_half2(half2_bits(sin_reduced) ^ signs);
        cos_value = bits_half2(half2_bits(cos_reduced) ^ signs);
    }
};

struct PolynomialD5D6HalfTurnBF16 {
    static __device__ __forceinline__ void evaluate(
        float angle_low,
        float angle_high,
        __nv_bfloat162& cos_value,
        __nv_bfloat162& sin_value) {
        float reduced_low;
        float reduced_high;
        unsigned int sign_low;
        unsigned int sign_high;
        reduce_to_half_turn_magic_bias_fast(
            angle_low, reduced_low, sign_low);
        reduce_to_half_turn_magic_bias_fast(
            angle_high, reduced_high, sign_high);

        const __nv_bfloat162 reduced =
            __floats2bfloat162_rn(reduced_low, reduced_high);
        const __nv_bfloat162 squared = __hmul2(reduced, reduced);
        __nv_bfloat162 sin_polynomial = __hfma2(
            __float2bfloat162_rn(0.007598876953125f),
            squared,
            __float2bfloat162_rn(-0.166015625f));
        sin_polynomial = __hfma2(
            sin_polynomial,
            squared,
            __float2bfloat162_rn(1.0f));
        const __nv_bfloat162 sin_reduced =
            __hmul2(reduced, sin_polynomial);

        __nv_bfloat162 cos_reduced = __hfma2(
            __float2bfloat162_rn(-0.00125885009765625f),
            squared,
            __float2bfloat162_rn(0.04150390625f));
        cos_reduced = __hfma2(
            cos_reduced,
            squared,
            __float2bfloat162_rn(-0.5f));
        cos_reduced = __hfma2(
            cos_reduced,
            squared,
            __float2bfloat162_rn(1.0f));

        const unsigned int signs = (sign_low >> 16) | sign_high;
        sin_value = bits_bfloat162(bfloat162_bits(sin_reduced) ^ signs);
        cos_value = bits_bfloat162(bfloat162_bits(cos_reduced) ^ signs);
    }
};

struct PolynomialD5D4HalfTurnFP16BF16 {
    static __device__ __forceinline__ void evaluate(
        float angle_low,
        float angle_high,
        __nv_bfloat162& cos_value,
        __nv_bfloat162& sin_value) {
        __half2 cos_half;
        __half2 sin_half;
        evaluate_half_turn_d5_d4_fp16(
            angle_low, angle_high, cos_half, sin_half);

        const float2 sin_pair = __half22float2(sin_half);
        const float2 cos_pair = __half22float2(cos_half);
        sin_value = __floats2bfloat162_rn(sin_pair.x, sin_pair.y);
        cos_value = __floats2bfloat162_rn(cos_pair.x, cos_pair.y);
    }
};

struct PolynomialD3D4BF16 {
    static __device__ __forceinline__ void evaluate(
        float angle_low,
        float angle_high,
        __nv_bfloat162& cos_value,
        __nv_bfloat162& sin_value) {
        float reduced_low;
        float reduced_high;
        int quadrant_low;
        int quadrant_high;
        reduce_to_quarter_turn(angle_low, reduced_low, quadrant_low);
        reduce_to_quarter_turn(angle_high, reduced_high, quadrant_high);

        const __nv_bfloat162 reduced =
            __floats2bfloat162_rn(reduced_low, reduced_high);
        const __nv_bfloat162 squared = __hmul2(reduced, reduced);
        const __nv_bfloat162 sin_polynomial = __hfma2(
            __float2bfloat162_rn(-0.16034401953220367431640625f),
            squared,
            __float2bfloat162_rn(0.99903142452239990234375f));
        const __nv_bfloat162 sin_reduced =
            __hmul2(reduced, sin_polynomial);
        __nv_bfloat162 cos_reduced = __hfma2(
            __float2bfloat162_rn(0.0403986163437366485595703125f),
            squared,
            __float2bfloat162_rn(-0.4997082054615020751953125f));
        cos_reduced = __hfma2(
            cos_reduced,
            squared,
            __float2bfloat162_rn(0.999990046024322509765625f));
        restore_quadrant_bf16_pair(
            sin_reduced,
            cos_reduced,
            quadrant_low,
            quadrant_high,
            sin_value,
            cos_value);
    }
};

template <typename Evaluator>
__global__ void __launch_bounds__(kBlockSize) sincos_vec_kernel(
    const float4* __restrict__ angles,
    float4* __restrict__ cos_output,
    float4* __restrict__ sin_output,
    int vector_count) {
    for (int index = threadIdx.x + blockIdx.x * blockDim.x;
         index < vector_count;
         index += blockDim.x * gridDim.x) {
        const float4 input = angles[index];
        float4 cos_values;
        float4 sin_values;
        Evaluator::evaluate(input.x, cos_values.x, sin_values.x);
        Evaluator::evaluate(input.y, cos_values.y, sin_values.y);
        Evaluator::evaluate(input.z, cos_values.z, sin_values.z);
        Evaluator::evaluate(input.w, cos_values.w, sin_values.w);
        cos_output[index] = cos_values;
        sin_output[index] = sin_values;
    }
}

template <typename Evaluator>
__global__ void __launch_bounds__(kBlockSize) sincos_scalar_kernel(
    const float* __restrict__ angles,
    float* __restrict__ cos_output,
    float* __restrict__ sin_output,
    int element_count) {
    for (int index = threadIdx.x + blockIdx.x * blockDim.x;
         index < element_count;
         index += blockDim.x * gridDim.x) {
        Evaluator::evaluate(angles[index], cos_output[index], sin_output[index]);
    }
}

template <typename Evaluator>
__global__ void __launch_bounds__(kBlockSize) sincos_compute_vec_kernel(
    const float4* __restrict__ angles,
    float4* __restrict__ cos_output,
    float4* __restrict__ sin_output,
    int vector_count,
    int iterations) {
    constexpr float kProbeScale = 0.999999940395355224609375f;
    constexpr float kProbeOffset = 0.0001220703125f;
    for (int index = threadIdx.x + blockIdx.x * blockDim.x;
         index < vector_count;
         index += blockDim.x * gridDim.x) {
        float4 probe = angles[index];
        float4 cos_sum = {};
        float4 sin_sum = {};
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            float4 cos_values;
            float4 sin_values;
            Evaluator::evaluate(probe.x, cos_values.x, sin_values.x);
            Evaluator::evaluate(probe.y, cos_values.y, sin_values.y);
            Evaluator::evaluate(probe.z, cos_values.z, sin_values.z);
            Evaluator::evaluate(probe.w, cos_values.w, sin_values.w);
            cos_sum.x += cos_values.x;
            cos_sum.y += cos_values.y;
            cos_sum.z += cos_values.z;
            cos_sum.w += cos_values.w;
            sin_sum.x += sin_values.x;
            sin_sum.y += sin_values.y;
            sin_sum.z += sin_values.z;
            sin_sum.w += sin_values.w;
            probe.x = fma_rn(probe.x, kProbeScale, kProbeOffset);
            probe.y = fma_rn(probe.y, kProbeScale, kProbeOffset);
            probe.z = fma_rn(probe.z, kProbeScale, kProbeOffset);
            probe.w = fma_rn(probe.w, kProbeScale, kProbeOffset);
        }
        cos_output[index] = cos_sum;
        sin_output[index] = sin_sum;
    }
}

template <typename Evaluator>
__global__ void __launch_bounds__(kBlockSize) sincos_compute_scalar_kernel(
    const float* __restrict__ angles,
    float* __restrict__ cos_output,
    float* __restrict__ sin_output,
    int element_count,
    int iterations) {
    constexpr float kProbeScale = 0.999999940395355224609375f;
    constexpr float kProbeOffset = 0.0001220703125f;
    for (int index = threadIdx.x + blockIdx.x * blockDim.x;
         index < element_count;
         index += blockDim.x * gridDim.x) {
        float probe = angles[index];
        float cos_sum = 0.0f;
        float sin_sum = 0.0f;
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            float cos_value;
            float sin_value;
            Evaluator::evaluate(probe, cos_value, sin_value);
            cos_sum += cos_value;
            sin_sum += sin_value;
            probe = fma_rn(probe, kProbeScale, kProbeOffset);
        }
        cos_output[index] = cos_sum;
        sin_output[index] = sin_sum;
    }
}

template <typename Evaluator>
__global__ void __launch_bounds__(kBlockSize) sincos_bf16_vec_kernel(
    const float4* __restrict__ angles,
    uint2* __restrict__ cos_output,
    uint2* __restrict__ sin_output,
    int vector_count) {
    for (int index = threadIdx.x + blockIdx.x * blockDim.x;
         index < vector_count;
         index += blockDim.x * gridDim.x) {
        const float4 input = angles[index];
        __nv_bfloat162 cos_low;
        __nv_bfloat162 sin_low;
        __nv_bfloat162 cos_high;
        __nv_bfloat162 sin_high;
        Evaluator::evaluate(input.x, input.y, cos_low, sin_low);
        Evaluator::evaluate(input.z, input.w, cos_high, sin_high);
        cos_output[index] = make_uint2(
            bfloat162_bits(cos_low), bfloat162_bits(cos_high));
        sin_output[index] = make_uint2(
            bfloat162_bits(sin_low), bfloat162_bits(sin_high));
    }
}

template <typename Evaluator>
__global__ void __launch_bounds__(kBlockSize) sincos_bf16_pair_kernel(
    const float2* __restrict__ angles,
    __nv_bfloat162* __restrict__ cos_output,
    __nv_bfloat162* __restrict__ sin_output,
    int pair_count) {
    for (int index = threadIdx.x + blockIdx.x * blockDim.x;
         index < pair_count;
         index += blockDim.x * gridDim.x) {
        const float2 input = angles[index];
        Evaluator::evaluate(
            input.x, input.y, cos_output[index], sin_output[index]);
    }
}

template <typename Evaluator>
__global__ void __launch_bounds__(kBlockSize) sincos_bf16_compute_vec_kernel(
    const float4* __restrict__ angles,
    uint2* __restrict__ cos_output,
    uint2* __restrict__ sin_output,
    int vector_count,
    int iterations) {
    constexpr float kProbeScale = 0.999999940395355224609375f;
    constexpr float kProbeOffset = 0.0001220703125f;
    for (int index = threadIdx.x + blockIdx.x * blockDim.x;
         index < vector_count;
         index += blockDim.x * gridDim.x) {
        float4 probe = angles[index];
        __nv_bfloat162 cos_sum_low = __float2bfloat162_rn(0.0f);
        __nv_bfloat162 sin_sum_low = __float2bfloat162_rn(0.0f);
        __nv_bfloat162 cos_sum_high = __float2bfloat162_rn(0.0f);
        __nv_bfloat162 sin_sum_high = __float2bfloat162_rn(0.0f);
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            __nv_bfloat162 cos_low;
            __nv_bfloat162 sin_low;
            __nv_bfloat162 cos_high;
            __nv_bfloat162 sin_high;
            Evaluator::evaluate(probe.x, probe.y, cos_low, sin_low);
            Evaluator::evaluate(probe.z, probe.w, cos_high, sin_high);
            cos_sum_low = __hadd2(cos_sum_low, cos_low);
            sin_sum_low = __hadd2(sin_sum_low, sin_low);
            cos_sum_high = __hadd2(cos_sum_high, cos_high);
            sin_sum_high = __hadd2(sin_sum_high, sin_high);
            probe.x = fma_rn(probe.x, kProbeScale, kProbeOffset);
            probe.y = fma_rn(probe.y, kProbeScale, kProbeOffset);
            probe.z = fma_rn(probe.z, kProbeScale, kProbeOffset);
            probe.w = fma_rn(probe.w, kProbeScale, kProbeOffset);
        }
        cos_output[index] = make_uint2(
            bfloat162_bits(cos_sum_low), bfloat162_bits(cos_sum_high));
        sin_output[index] = make_uint2(
            bfloat162_bits(sin_sum_low), bfloat162_bits(sin_sum_high));
    }
}

template <typename Evaluator>
__global__ void __launch_bounds__(kBlockSize) sincos_fp16_vec_kernel(
    const float4* __restrict__ angles,
    uint2* __restrict__ cos_output,
    uint2* __restrict__ sin_output,
    int vector_count) {
    for (int index = threadIdx.x + blockIdx.x * blockDim.x;
         index < vector_count;
         index += blockDim.x * gridDim.x) {
        const float4 input = angles[index];
        __half2 cos_low;
        __half2 sin_low;
        __half2 cos_high;
        __half2 sin_high;
        Evaluator::evaluate(input.x, input.y, cos_low, sin_low);
        Evaluator::evaluate(input.z, input.w, cos_high, sin_high);
        cos_output[index] = make_uint2(
            half2_bits(cos_low), half2_bits(cos_high));
        sin_output[index] = make_uint2(
            half2_bits(sin_low), half2_bits(sin_high));
    }
}

template <typename Evaluator>
__global__ void __launch_bounds__(kBlockSize) sincos_fp16_pair_kernel(
    const float2* __restrict__ angles,
    __half2* __restrict__ cos_output,
    __half2* __restrict__ sin_output,
    int pair_count) {
    for (int index = threadIdx.x + blockIdx.x * blockDim.x;
         index < pair_count;
         index += blockDim.x * gridDim.x) {
        const float2 input = angles[index];
        Evaluator::evaluate(
            input.x, input.y, cos_output[index], sin_output[index]);
    }
}

template <typename Evaluator>
__global__ void __launch_bounds__(kBlockSize) sincos_fp16_compute_vec_kernel(
    const float4* __restrict__ angles,
    uint2* __restrict__ cos_output,
    uint2* __restrict__ sin_output,
    int vector_count,
    int iterations) {
    constexpr float kProbeScale = 0.999999940395355224609375f;
    constexpr float kProbeOffset = 0.0001220703125f;
    for (int index = threadIdx.x + blockIdx.x * blockDim.x;
         index < vector_count;
         index += blockDim.x * gridDim.x) {
        float4 probe = angles[index];
        __half2 cos_sum_low = __float2half2_rn(0.0f);
        __half2 sin_sum_low = __float2half2_rn(0.0f);
        __half2 cos_sum_high = __float2half2_rn(0.0f);
        __half2 sin_sum_high = __float2half2_rn(0.0f);
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            __half2 cos_low;
            __half2 sin_low;
            __half2 cos_high;
            __half2 sin_high;
            Evaluator::evaluate(probe.x, probe.y, cos_low, sin_low);
            Evaluator::evaluate(probe.z, probe.w, cos_high, sin_high);
            cos_sum_low = __hadd2(cos_sum_low, cos_low);
            sin_sum_low = __hadd2(sin_sum_low, sin_low);
            cos_sum_high = __hadd2(cos_sum_high, cos_high);
            sin_sum_high = __hadd2(sin_sum_high, sin_high);
            probe.x = fma_rn(probe.x, kProbeScale, kProbeOffset);
            probe.y = fma_rn(probe.y, kProbeScale, kProbeOffset);
            probe.z = fma_rn(probe.z, kProbeScale, kProbeOffset);
            probe.w = fma_rn(probe.w, kProbeScale, kProbeOffset);
        }
        cos_output[index] = make_uint2(
            half2_bits(cos_sum_low), half2_bits(cos_sum_high));
        sin_output[index] = make_uint2(
            half2_bits(sin_sum_low), half2_bits(sin_sum_high));
    }
}

__device__ __forceinline__ void reduce_rope_phase_to_quarter_turn_q15(
    unsigned int position,
    unsigned int phase_increment,
    short& reduced_q15,
    unsigned int& quadrant) {
    const unsigned int phase = position * phase_increment;
    quadrant = (phase + 0x20000000U) >> 30;
    const unsigned int residual_bits = phase - (quadrant << 30);
    const int residual = static_cast<int>(residual_bits);
    reduced_q15 = static_cast<short>(residual >> 14);
}

__device__ __forceinline__ void evaluate_quarter_turn_q15_d3_d4_fp16(
    short reduced_low_q15,
    short reduced_high_q15,
    unsigned int quadrant_low,
    unsigned int quadrant_high,
    __half2& cos_value,
    __half2& sin_value) {
    const __half2 reduced = __hmul2(
        __halves2half2(
            __short2half_rn(reduced_low_q15),
            __short2half_rn(reduced_high_q15)),
        __float2half2_rn(0x1p-15f));
    const __half2 squared = __hmul2(reduced, reduced);
    const __half2 sin_polynomial = __hfma2(
        __float2half2_rn(-0.07763671875f),
        squared,
        __float2half2_rn(0.78466796875f));
    const __half2 sin_reduced = __hmul2(reduced, sin_polynomial);
    __half2 cos_reduced = __hfma2(
        __float2half2_rn(0.01538848876953125f),
        squared,
        __float2half2_rn(-0.308349609375f));
    cos_reduced = __hfma2(
        cos_reduced,
        squared,
        __float2half2_rn(1.0f));
    restore_quadrant_half2_pair(
        sin_reduced,
        cos_reduced,
        quadrant_low,
        quadrant_high,
        sin_value,
        cos_value);
}

__device__ __forceinline__ void evaluate_half_turn_q15_d5_d6_fp16(
    short reduced_low_q15,
    short reduced_high_q15,
    unsigned int signs,
    __half2& cos_value,
    __half2& sin_value) {
    const __half2 reduced = __hmul2(
        __halves2half2(
            __short2half_rn(reduced_low_q15),
            __short2half_rn(reduced_high_q15)),
        __float2half2_rn(0x1p-15f));
    const __half2 squared = __hmul2(reduced, reduced);

    __half2 sin_polynomial = __hfma2(
        __float2half2_rn(0.072021484375f),
        squared,
        __float2half2_rn(-0.64208984375f));
    sin_polynomial = __hfma2(
        sin_polynomial,
        squared,
        __float2half2_rn(1.5703125f));
    const __half2 sin_reduced = __hmul2(reduced, sin_polynomial);

    __half2 cos_reduced = __hfma2(
        __float2half2_rn(-0.0190887451171875f),
        squared,
        __float2half2_rn(0.25244140625f));
    cos_reduced = __hfma2(
        cos_reduced,
        squared,
        __float2half2_rn(-1.2333984375f));
    cos_reduced = __hfma2(
        cos_reduced,
        squared,
        __float2half2_rn(1.0f));

    sin_value = bits_half2(half2_bits(sin_reduced) ^ signs);
    cos_value = bits_half2(half2_bits(cos_reduced) ^ signs);
}

__device__ __forceinline__ void evaluate_rope_half_turn_q15_d5_d6_fp16(
    unsigned int position,
    uint2 phase_increment,
    __half2& cos_value,
    __half2& sin_value) {
    const unsigned int phase_low = position * phase_increment.x;
    const unsigned int phase_high = position * phase_increment.y;
    const short reduced_low_q15 = static_cast<short>(phase_low >> 15);
    const short reduced_high_q15 = static_cast<short>(phase_high >> 15);
    const unsigned int upper_halves =
        __byte_perm(phase_low, phase_high, 0x7632);
    const unsigned int signs =
        (upper_halves ^ (upper_halves << 1)) & 0x80008000U;
    evaluate_half_turn_q15_d5_d6_fp16(
        reduced_low_q15,
        reduced_high_q15,
        signs,
        cos_value,
        sin_value);
}

__device__ __forceinline__ void evaluate_rope_phase_lut_fp16(
    unsigned int position,
    uint2 phase_increment,
    const unsigned int* __restrict__ phase_table,
    __half2& cos_value,
    __half2& sin_value) {
    constexpr float kRadiansPerPhaseUnit = 0x1.921fb6p-30f;
    constexpr unsigned int kTableIndexShift = 25;
    constexpr unsigned int kTableRound = 1U << (kTableIndexShift - 1);
    constexpr unsigned int kTableIndexMask = 127U;

    const unsigned int phase_low = position * phase_increment.x;
    const unsigned int phase_high = position * phase_increment.y;
    const unsigned int index_low =
        ((phase_low + kTableRound) >> kTableIndexShift) & kTableIndexMask;
    const unsigned int index_high =
        ((phase_high + kTableRound) >> kTableIndexShift) & kTableIndexMask;
    const int residual_low = static_cast<int>(
        phase_low - (index_low << kTableIndexShift));
    const int residual_high = static_cast<int>(
        phase_high - (index_high << kTableIndexShift));
    const __half2 delta = __floats2half2_rn(
        static_cast<float>(residual_low) * kRadiansPerPhaseUnit,
        static_cast<float>(residual_high) * kRadiansPerPhaseUnit);

    const unsigned int base_low = phase_table[index_low];
    const unsigned int base_high = phase_table[index_high];
    const __half2 sin_base = bits_half2(
        __byte_perm(base_low, base_high, 0x5410));
    const __half2 cos_base = bits_half2(
        __byte_perm(base_low, base_high, 0x7632));
    const __half2 negative_delta = bits_half2(
        half2_bits(delta) ^ 0x80008000U);
    sin_value = __hfma2(delta, cos_base, sin_base);
    cos_value = __hfma2(negative_delta, sin_base, cos_base);
}

__global__ void __launch_bounds__(128) rope_sincos_native_fp16_kernel(
    const float2* __restrict__ frequencies,
    __half2* __restrict__ cos_output,
    __half2* __restrict__ sin_output,
    int sequence_length,
    int frequency_pair_count) {
    const int frequency_pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int position = blockIdx.y * blockDim.y + threadIdx.y;
    if (frequency_pair >= frequency_pair_count || position >= sequence_length) {
        return;
    }

    const float2 frequency = frequencies[frequency_pair];
    const float position_f = static_cast<float>(position);
    const int output_index = position * frequency_pair_count + frequency_pair;
    NativeSincosFP16::evaluate(
        position_f * frequency.x,
        position_f * frequency.y,
        cos_output[output_index],
        sin_output[output_index]);
}

__global__ void __launch_bounds__(128) rope_sincos_fixed_d3_d4_fp16_kernel(
    const uint2* __restrict__ phase_increments,
    __half2* __restrict__ cos_output,
    __half2* __restrict__ sin_output,
    int sequence_length,
    int frequency_pair_count) {
    const int frequency_pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int position = blockIdx.y * blockDim.y + threadIdx.y;
    if (frequency_pair >= frequency_pair_count || position >= sequence_length) {
        return;
    }

    const uint2 increment = phase_increments[frequency_pair];
    short reduced_low_q15;
    short reduced_high_q15;
    unsigned int quadrant_low;
    unsigned int quadrant_high;
    reduce_rope_phase_to_quarter_turn_q15(
        static_cast<unsigned int>(position),
        increment.x,
        reduced_low_q15,
        quadrant_low);
    reduce_rope_phase_to_quarter_turn_q15(
        static_cast<unsigned int>(position),
        increment.y,
        reduced_high_q15,
        quadrant_high);

    const int output_index = position * frequency_pair_count + frequency_pair;
    evaluate_quarter_turn_q15_d3_d4_fp16(
        reduced_low_q15,
        reduced_high_q15,
        quadrant_low,
        quadrant_high,
        cos_output[output_index],
        sin_output[output_index]);
}

__global__ void __launch_bounds__(128)
rope_sincos_fixed_half_turn_d5_d6_fp16_kernel(
    const uint4* __restrict__ phase_increments,
    uint2* __restrict__ cos_output,
    uint2* __restrict__ sin_output,
    int sequence_length,
    int frequency_group_count) {
    const int frequency_group = blockIdx.x * blockDim.x + threadIdx.x;
    const int position = blockIdx.y * blockDim.y + threadIdx.y;
    if (frequency_group >= frequency_group_count || position >= sequence_length) {
        return;
    }

    const uint4 increment = phase_increments[frequency_group];
    __half2 cos_low;
    __half2 sin_low;
    __half2 cos_high;
    __half2 sin_high;
    evaluate_rope_half_turn_q15_d5_d6_fp16(
        static_cast<unsigned int>(position),
        make_uint2(increment.x, increment.y),
        cos_low,
        sin_low);
    evaluate_rope_half_turn_q15_d5_d6_fp16(
        static_cast<unsigned int>(position),
        make_uint2(increment.z, increment.w),
        cos_high,
        sin_high);
    const int output_index = position * frequency_group_count + frequency_group;
    cos_output[output_index] =
        make_uint2(half2_bits(cos_low), half2_bits(cos_high));
    sin_output[output_index] =
        make_uint2(half2_bits(sin_low), half2_bits(sin_high));
}

__global__ void __launch_bounds__(128) rope_sincos_fixed_lut_fp16_kernel(
    const uint2* __restrict__ phase_increments,
    const unsigned int* __restrict__ phase_table,
    __half2* __restrict__ cos_output,
    __half2* __restrict__ sin_output,
    int sequence_length,
    int frequency_pair_count) {
    const int frequency_pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int position = blockIdx.y * blockDim.y + threadIdx.y;
    if (frequency_pair >= frequency_pair_count || position >= sequence_length) {
        return;
    }

    const int output_index = position * frequency_pair_count + frequency_pair;
    evaluate_rope_phase_lut_fp16(
        static_cast<unsigned int>(position),
        phase_increments[frequency_pair],
        phase_table,
        cos_output[output_index],
        sin_output[output_index]);
}

__device__ __forceinline__ uint4 rotate_rope_four_fp16(
    uint4 values,
    __half2 cos_low,
    __half2 sin_low,
    __half2 cos_high,
    __half2 sin_high) {
    const __half2 even_low = bits_half2(
        __byte_perm(values.x, values.y, 0x5410));
    const __half2 odd_low = bits_half2(
        __byte_perm(values.x, values.y, 0x7632));
    const __half2 even_high = bits_half2(
        __byte_perm(values.z, values.w, 0x5410));
    const __half2 odd_high = bits_half2(
        __byte_perm(values.z, values.w, 0x7632));
    const __half2 negative_sin_low = bits_half2(
        half2_bits(sin_low) ^ 0x80008000U);
    const __half2 negative_sin_high = bits_half2(
        half2_bits(sin_high) ^ 0x80008000U);

    const __half2 rotated_even_low = __hfma2(
        negative_sin_low,
        odd_low,
        __hmul2(cos_low, even_low));
    const __half2 rotated_odd_low = __hfma2(
        sin_low,
        even_low,
        __hmul2(cos_low, odd_low));
    const __half2 rotated_even_high = __hfma2(
        negative_sin_high,
        odd_high,
        __hmul2(cos_high, even_high));
    const __half2 rotated_odd_high = __hfma2(
        sin_high,
        even_high,
        __hmul2(cos_high, odd_high));

    return make_uint4(
        __byte_perm(
            half2_bits(rotated_even_low),
            half2_bits(rotated_odd_low),
            0x5410),
        __byte_perm(
            half2_bits(rotated_even_low),
            half2_bits(rotated_odd_low),
            0x7632),
        __byte_perm(
            half2_bits(rotated_even_high),
            half2_bits(rotated_odd_high),
            0x5410),
        __byte_perm(
            half2_bits(rotated_even_high),
            half2_bits(rotated_odd_high),
            0x7632));
}

__device__ __forceinline__ void apply_rope_qk_fp16(
    const uint4* __restrict__ q_input,
    const uint4* __restrict__ k_input,
    uint4* __restrict__ q_output,
    uint4* __restrict__ k_output,
    int batch,
    int position,
    int sequence_length,
    int q_head_count,
    int k_head_count,
    int frequency_group,
    int frequency_group_count,
    __half2 cos_low,
    __half2 sin_low,
    __half2 cos_high,
    __half2 sin_high) {
    const int token = batch * sequence_length + position;
    const int q_input_base =
        token * q_head_count * frequency_group_count + frequency_group;
    const int k_input_base =
        token * k_head_count * frequency_group_count + frequency_group;

    #pragma unroll 1
    for (int head = 0; head < q_head_count; ++head) {
        const int input_index =
            q_input_base + head * frequency_group_count;
        const int output_index =
            ((batch * q_head_count + head) * sequence_length + position)
                * frequency_group_count
            + frequency_group;
        q_output[output_index] = rotate_rope_four_fp16(
            q_input[input_index],
            cos_low,
            sin_low,
            cos_high,
            sin_high);
    }

    #pragma unroll 1
    for (int head = 0; head < k_head_count; ++head) {
        const int input_index =
            k_input_base + head * frequency_group_count;
        const int output_index =
            ((batch * k_head_count + head) * sequence_length + position)
                * frequency_group_count
            + frequency_group;
        k_output[output_index] = rotate_rope_four_fp16(
            k_input[input_index],
            cos_low,
            sin_low,
            cos_high,
            sin_high);
    }
}

__global__ void __launch_bounds__(128) rope_apply_cached_fp16_kernel(
    const uint4* __restrict__ q_input,
    const uint4* __restrict__ k_input,
    const uint2* __restrict__ cos_table,
    const uint2* __restrict__ sin_table,
    uint4* __restrict__ q_output,
    uint4* __restrict__ k_output,
    int sequence_length,
    int q_head_count,
    int k_head_count,
    int frequency_group_count) {
    const int frequency_group = blockIdx.x * blockDim.x + threadIdx.x;
    const int position = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;
    if (frequency_group >= frequency_group_count || position >= sequence_length) {
        return;
    }

    const int table_index =
        position * frequency_group_count + frequency_group;
    const uint2 cos_values = cos_table[table_index];
    const uint2 sin_values = sin_table[table_index];
    apply_rope_qk_fp16(
        q_input,
        k_input,
        q_output,
        k_output,
        batch,
        position,
        sequence_length,
        q_head_count,
        k_head_count,
        frequency_group,
        frequency_group_count,
        bits_half2(cos_values.x),
        bits_half2(sin_values.x),
        bits_half2(cos_values.y),
        bits_half2(sin_values.y));
}

__global__ void __launch_bounds__(128) rope_apply_native_fp16_kernel(
    const uint4* __restrict__ q_input,
    const uint4* __restrict__ k_input,
    const float4* __restrict__ frequencies,
    uint4* __restrict__ q_output,
    uint4* __restrict__ k_output,
    int sequence_length,
    int q_head_count,
    int k_head_count,
    int frequency_group_count) {
    const int frequency_group = blockIdx.x * blockDim.x + threadIdx.x;
    const int position = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;
    if (frequency_group >= frequency_group_count || position >= sequence_length) {
        return;
    }

    const float4 frequency = frequencies[frequency_group];
    const float position_f = static_cast<float>(position);
    __half2 cos_low;
    __half2 sin_low;
    __half2 cos_high;
    __half2 sin_high;
    NativeSincosFP16::evaluate(
        position_f * frequency.x,
        position_f * frequency.y,
        cos_low,
        sin_low);
    NativeSincosFP16::evaluate(
        position_f * frequency.z,
        position_f * frequency.w,
        cos_high,
        sin_high);
    apply_rope_qk_fp16(
        q_input,
        k_input,
        q_output,
        k_output,
        batch,
        position,
        sequence_length,
        q_head_count,
        k_head_count,
        frequency_group,
        frequency_group_count,
        cos_low,
        sin_low,
        cos_high,
        sin_high);
}

__global__ void __launch_bounds__(128)
rope_apply_fixed_half_turn_d5_d6_fp16_kernel(
    const uint4* __restrict__ q_input,
    const uint4* __restrict__ k_input,
    const uint4* __restrict__ phase_increments,
    uint4* __restrict__ q_output,
    uint4* __restrict__ k_output,
    int sequence_length,
    int q_head_count,
    int k_head_count,
    int frequency_group_count) {
    const int frequency_group = blockIdx.x * blockDim.x + threadIdx.x;
    const int position = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;
    if (frequency_group >= frequency_group_count || position >= sequence_length) {
        return;
    }

    const uint4 increment = phase_increments[frequency_group];
    __half2 cos_low;
    __half2 sin_low;
    __half2 cos_high;
    __half2 sin_high;
    evaluate_rope_half_turn_q15_d5_d6_fp16(
        static_cast<unsigned int>(position),
        make_uint2(increment.x, increment.y),
        cos_low,
        sin_low);
    evaluate_rope_half_turn_q15_d5_d6_fp16(
        static_cast<unsigned int>(position),
        make_uint2(increment.z, increment.w),
        cos_high,
        sin_high);
    apply_rope_qk_fp16(
        q_input,
        k_input,
        q_output,
        k_output,
        batch,
        position,
        sequence_length,
        q_head_count,
        k_head_count,
        frequency_group,
        frequency_group_count,
        cos_low,
        sin_low,
        cos_high,
        sin_high);
}

__global__ void __launch_bounds__(128) rope_sincos_native_fp16_compute_kernel(
    const float2* __restrict__ frequencies,
    __half2* __restrict__ cos_output,
    __half2* __restrict__ sin_output,
    int sequence_length,
    int frequency_pair_count,
    int iterations) {
    const int frequency_pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int position = blockIdx.y * blockDim.y + threadIdx.y;
    if (frequency_pair >= frequency_pair_count || position >= sequence_length) {
        return;
    }

    const float2 frequency = frequencies[frequency_pair];
    __half2 cos_sum = __float2half2_rn(0.0f);
    __half2 sin_sum = __float2half2_rn(0.0f);
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const float position_f = static_cast<float>(position + iteration);
        __half2 cos_value;
        __half2 sin_value;
        NativeSincosFP16::evaluate(
            position_f * frequency.x,
            position_f * frequency.y,
            cos_value,
            sin_value);
        cos_sum = __hadd2(cos_sum, cos_value);
        sin_sum = __hadd2(sin_sum, sin_value);
    }

    const int output_index = position * frequency_pair_count + frequency_pair;
    cos_output[output_index] = cos_sum;
    sin_output[output_index] = sin_sum;
}

__global__ void __launch_bounds__(128)
rope_sincos_fixed_d3_d4_fp16_compute_kernel(
    const uint2* __restrict__ phase_increments,
    __half2* __restrict__ cos_output,
    __half2* __restrict__ sin_output,
    int sequence_length,
    int frequency_pair_count,
    int iterations) {
    const int frequency_pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int position = blockIdx.y * blockDim.y + threadIdx.y;
    if (frequency_pair >= frequency_pair_count || position >= sequence_length) {
        return;
    }

    const uint2 increment = phase_increments[frequency_pair];
    __half2 cos_sum = __float2half2_rn(0.0f);
    __half2 sin_sum = __float2half2_rn(0.0f);
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const unsigned int probe_position =
            static_cast<unsigned int>(position + iteration);
        short reduced_low_q15;
        short reduced_high_q15;
        unsigned int quadrant_low;
        unsigned int quadrant_high;
        reduce_rope_phase_to_quarter_turn_q15(
            probe_position,
            increment.x,
            reduced_low_q15,
            quadrant_low);
        reduce_rope_phase_to_quarter_turn_q15(
            probe_position,
            increment.y,
            reduced_high_q15,
            quadrant_high);
        __half2 cos_value;
        __half2 sin_value;
        evaluate_quarter_turn_q15_d3_d4_fp16(
            reduced_low_q15,
            reduced_high_q15,
            quadrant_low,
            quadrant_high,
            cos_value,
            sin_value);
        cos_sum = __hadd2(cos_sum, cos_value);
        sin_sum = __hadd2(sin_sum, sin_value);
    }

    const int output_index = position * frequency_pair_count + frequency_pair;
    cos_output[output_index] = cos_sum;
    sin_output[output_index] = sin_sum;
}

__global__ void __launch_bounds__(128)
rope_sincos_fixed_half_turn_d5_d6_fp16_compute_kernel(
    const uint4* __restrict__ phase_increments,
    uint2* __restrict__ cos_output,
    uint2* __restrict__ sin_output,
    int sequence_length,
    int frequency_group_count,
    int iterations) {
    const int frequency_group = blockIdx.x * blockDim.x + threadIdx.x;
    const int position = blockIdx.y * blockDim.y + threadIdx.y;
    if (frequency_group >= frequency_group_count || position >= sequence_length) {
        return;
    }

    const uint4 increment = phase_increments[frequency_group];
    __half2 cos_sum_low = __float2half2_rn(0.0f);
    __half2 sin_sum_low = __float2half2_rn(0.0f);
    __half2 cos_sum_high = __float2half2_rn(0.0f);
    __half2 sin_sum_high = __float2half2_rn(0.0f);
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const unsigned int probe_position =
            static_cast<unsigned int>(position + iteration);
        __half2 cos_low;
        __half2 sin_low;
        __half2 cos_high;
        __half2 sin_high;
        evaluate_rope_half_turn_q15_d5_d6_fp16(
            probe_position,
            make_uint2(increment.x, increment.y),
            cos_low,
            sin_low);
        evaluate_rope_half_turn_q15_d5_d6_fp16(
            probe_position,
            make_uint2(increment.z, increment.w),
            cos_high,
            sin_high);
        cos_sum_low = __hadd2(cos_sum_low, cos_low);
        sin_sum_low = __hadd2(sin_sum_low, sin_low);
        cos_sum_high = __hadd2(cos_sum_high, cos_high);
        sin_sum_high = __hadd2(sin_sum_high, sin_high);
    }

    const int output_index = position * frequency_group_count + frequency_group;
    cos_output[output_index] =
        make_uint2(half2_bits(cos_sum_low), half2_bits(cos_sum_high));
    sin_output[output_index] =
        make_uint2(half2_bits(sin_sum_low), half2_bits(sin_sum_high));
}

__global__ void __launch_bounds__(128) rope_sincos_fixed_lut_fp16_compute_kernel(
    const uint2* __restrict__ phase_increments,
    const unsigned int* __restrict__ phase_table,
    __half2* __restrict__ cos_output,
    __half2* __restrict__ sin_output,
    int sequence_length,
    int frequency_pair_count,
    int iterations) {
    const int frequency_pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int position = blockIdx.y * blockDim.y + threadIdx.y;
    if (frequency_pair >= frequency_pair_count || position >= sequence_length) {
        return;
    }

    const uint2 increment = phase_increments[frequency_pair];
    __half2 cos_sum = __float2half2_rn(0.0f);
    __half2 sin_sum = __float2half2_rn(0.0f);
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        __half2 cos_value;
        __half2 sin_value;
        evaluate_rope_phase_lut_fp16(
            static_cast<unsigned int>(position + iteration),
            increment,
            phase_table,
            cos_value,
            sin_value);
        cos_sum = __hadd2(cos_sum, cos_value);
        sin_sum = __hadd2(sin_sum, sin_value);
    }

    const int output_index = position * frequency_pair_count + frequency_pair;
    cos_output[output_index] = cos_sum;
    sin_output[output_index] = sin_sum;
}

bool is_aligned_16(const void* pointer) {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
}

template <typename Evaluator>
void launch_sincos(
    float* output,
    const float* angles,
    int element_count,
    cudaStream_t stream) {
    if (element_count == 0) {
        return;
    }
    float* cos_output = output;
    float* sin_output = output + element_count;
    if ((element_count % kValuesPerVector) == 0
        && is_aligned_16(angles)
        && is_aligned_16(cos_output)
        && is_aligned_16(sin_output)) {
        const int vector_count = element_count / kValuesPerVector;
        const int blocks = std::min(
            (vector_count + kBlockSize - 1) / kBlockSize,
            kMaxBlocks);
        sincos_vec_kernel<Evaluator><<<blocks, kBlockSize, 0, stream>>>(
            reinterpret_cast<const float4*>(angles),
            reinterpret_cast<float4*>(cos_output),
            reinterpret_cast<float4*>(sin_output),
            vector_count);
        return;
    }

    const int blocks = std::min(
        (element_count + kBlockSize - 1) / kBlockSize,
        kMaxBlocks);
    sincos_scalar_kernel<Evaluator><<<blocks, kBlockSize, 0, stream>>>(
        angles,
        cos_output,
        sin_output,
        element_count);
}

template <typename Evaluator>
void launch_sincos_compute(
    float* output,
    const float* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    if (element_count == 0) {
        return;
    }
    float* cos_output = output;
    float* sin_output = output + element_count;
    if ((element_count % kValuesPerVector) == 0
        && is_aligned_16(angles)
        && is_aligned_16(cos_output)
        && is_aligned_16(sin_output)) {
        const int vector_count = element_count / kValuesPerVector;
        const int blocks = std::min(
            (vector_count + kBlockSize - 1) / kBlockSize,
            kMaxBlocks);
        sincos_compute_vec_kernel<Evaluator><<<blocks, kBlockSize, 0, stream>>>(
            reinterpret_cast<const float4*>(angles),
            reinterpret_cast<float4*>(cos_output),
            reinterpret_cast<float4*>(sin_output),
            vector_count,
            iterations);
        return;
    }

    const int blocks = std::min(
        (element_count + kBlockSize - 1) / kBlockSize,
        kMaxBlocks);
    sincos_compute_scalar_kernel<Evaluator><<<blocks, kBlockSize, 0, stream>>>(
        angles,
        cos_output,
        sin_output,
        element_count,
        iterations);
}

template <typename Evaluator>
void launch_sincos_bf16(
    __nv_bfloat16* output,
    const float* angles,
    int element_count,
    cudaStream_t stream) {
    if (element_count == 0) {
        return;
    }
    __nv_bfloat16* cos_output = output;
    __nv_bfloat16* sin_output = output + element_count;
    if ((element_count % kValuesPerVector) == 0
        && is_aligned_16(angles)) {
        const int vector_count = element_count / kValuesPerVector;
        const int blocks = std::min(
            (vector_count + kBlockSize - 1) / kBlockSize,
            kMaxBlocks);
        sincos_bf16_vec_kernel<Evaluator><<<blocks, kBlockSize, 0, stream>>>(
            reinterpret_cast<const float4*>(angles),
            reinterpret_cast<uint2*>(cos_output),
            reinterpret_cast<uint2*>(sin_output),
            vector_count);
        return;
    }

    const int pair_count = element_count / 2;
    const int blocks = std::min(
        (pair_count + kBlockSize - 1) / kBlockSize,
        kMaxBlocks);
    sincos_bf16_pair_kernel<Evaluator><<<blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const float2*>(angles),
        reinterpret_cast<__nv_bfloat162*>(cos_output),
        reinterpret_cast<__nv_bfloat162*>(sin_output),
        pair_count);
}

template <typename Evaluator>
void launch_sincos_bf16_compute(
    __nv_bfloat16* output,
    const float* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    const int vector_count = element_count / kValuesPerVector;
    const int blocks = std::min(
        (vector_count + kBlockSize - 1) / kBlockSize,
        kMaxBlocks);
    sincos_bf16_compute_vec_kernel<Evaluator><<<blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const float4*>(angles),
        reinterpret_cast<uint2*>(output),
        reinterpret_cast<uint2*>(output + element_count),
        vector_count,
        iterations);
}

template <typename Evaluator>
void launch_sincos_fp16(
    __half* output,
    const float* angles,
    int element_count,
    cudaStream_t stream) {
    if (element_count == 0) {
        return;
    }
    __half* cos_output = output;
    __half* sin_output = output + element_count;
    if ((element_count % kValuesPerVector) == 0
        && is_aligned_16(angles)) {
        const int vector_count = element_count / kValuesPerVector;
        const int blocks = std::min(
            (vector_count + kBlockSize - 1) / kBlockSize,
            kMaxBlocks);
        sincos_fp16_vec_kernel<Evaluator><<<blocks, kBlockSize, 0, stream>>>(
            reinterpret_cast<const float4*>(angles),
            reinterpret_cast<uint2*>(cos_output),
            reinterpret_cast<uint2*>(sin_output),
            vector_count);
        return;
    }

    const int pair_count = element_count / 2;
    const int blocks = std::min(
        (pair_count + kBlockSize - 1) / kBlockSize,
        kMaxBlocks);
    sincos_fp16_pair_kernel<Evaluator><<<blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const float2*>(angles),
        reinterpret_cast<__half2*>(cos_output),
        reinterpret_cast<__half2*>(sin_output),
        pair_count);
}

template <typename Evaluator>
void launch_sincos_fp16_compute(
    __half* output,
    const float* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    const int vector_count = element_count / kValuesPerVector;
    const int blocks = std::min(
        (vector_count + kBlockSize - 1) / kBlockSize,
        kMaxBlocks);
    sincos_fp16_compute_vec_kernel<Evaluator><<<blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const float4*>(angles),
        reinterpret_cast<uint2*>(output),
        reinterpret_cast<uint2*>(output + element_count),
        vector_count,
        iterations);
}

}  // namespace

extern "C" {

void launch_sincos_native_f32(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos<NativeSincos>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d3_d4_f32(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos<PolynomialD3D4>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d3_d4_cycle_f32(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos<PolynomialD3D4Cycle>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d3_d4_magic_bias_f32(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos<PolynomialD3D4MagicBias>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d5_d4_half_turn_ls_f32(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos<PolynomialD5D4HalfTurnLS>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d5_d4_half_turn_sollya_f32(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos<PolynomialD5D4HalfTurnSollya>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d5_d4_half_turn_sollya_fast_f32(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos<PolynomialD5D4HalfTurnSollyaFast>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d5_d4_f32(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos<PolynomialD5D4>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d7_d6_f32(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos<PolynomialD7D6>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_native_compute_f32(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_compute<NativeSincos>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d3_d4_compute_f32(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_compute<PolynomialD3D4>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d3_d4_cycle_compute_f32(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_compute<PolynomialD3D4Cycle>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d3_d4_magic_bias_compute_f32(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_compute<PolynomialD3D4MagicBias>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d5_d4_half_turn_ls_compute_f32(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_compute<PolynomialD5D4HalfTurnLS>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d5_d4_half_turn_sollya_compute_f32(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_compute<PolynomialD5D4HalfTurnSollya>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d5_d4_half_turn_sollya_fast_compute_f32(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_compute<PolynomialD5D4HalfTurnSollyaFast>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d5_d4_compute_f32(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_compute<PolynomialD5D4>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d7_d6_compute_f32(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_compute<PolynomialD7D6>(
        static_cast<float*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_native_bf16(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos_bf16<NativeSincosBF16>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d3_d4_bf16(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos_bf16<PolynomialD3D4BF16>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d3_d4_quarter_turn_bf16(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos_bf16<PolynomialD3D4QuarterTurnBF16>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d3_d4_half_turn_bf16(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos_bf16<PolynomialD3D4HalfTurnBF16>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d5_d6_half_turn_bf16(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos_bf16<PolynomialD5D6HalfTurnBF16>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d5_d4_half_turn_fp16_bf16(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos_bf16<PolynomialD5D4HalfTurnFP16BF16>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_native_bf16_compute(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_bf16_compute<NativeSincosBF16>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d3_d4_bf16_compute(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_bf16_compute<PolynomialD3D4BF16>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d3_d4_quarter_turn_bf16_compute(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_bf16_compute<PolynomialD3D4QuarterTurnBF16>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d3_d4_half_turn_bf16_compute(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_bf16_compute<PolynomialD3D4HalfTurnBF16>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d5_d6_half_turn_bf16_compute(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_bf16_compute<PolynomialD5D6HalfTurnBF16>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d5_d4_half_turn_fp16_bf16_compute(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_bf16_compute<PolynomialD5D4HalfTurnFP16BF16>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_native_fp16(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos_fp16<NativeSincosFP16>(
        static_cast<__half*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d3_d4_quarter_turn_fp16(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos_fp16<PolynomialD3D4QuarterTurnFP16>(
        static_cast<__half*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d3_d4_half_turn_fp16(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos_fp16<PolynomialD3D4HalfTurnFP16>(
        static_cast<__half*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d5_d4_half_turn_fp16(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos_fp16<PolynomialD5D4HalfTurnFP16>(
        static_cast<__half*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d7_d6_half_turn_fp16(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos_fp16<PolynomialD7D6HalfTurnFP16>(
        static_cast<__half*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_d5_d6_half_turn_fp16(
    void* output,
    const void* angles,
    int element_count,
    cudaStream_t stream) {
    launch_sincos_fp16<PolynomialD5D6HalfTurnFP16>(
        static_cast<__half*>(output),
        static_cast<const float*>(angles),
        element_count,
        stream);
}

void launch_sincos_native_fp16_compute(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_fp16_compute<NativeSincosFP16>(
        static_cast<__half*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d3_d4_quarter_turn_fp16_compute(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_fp16_compute<PolynomialD3D4QuarterTurnFP16>(
        static_cast<__half*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d3_d4_half_turn_fp16_compute(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_fp16_compute<PolynomialD3D4HalfTurnFP16>(
        static_cast<__half*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d5_d4_half_turn_fp16_compute(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_fp16_compute<PolynomialD5D4HalfTurnFP16>(
        static_cast<__half*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d7_d6_half_turn_fp16_compute(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_fp16_compute<PolynomialD7D6HalfTurnFP16>(
        static_cast<__half*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_sincos_d5_d6_half_turn_fp16_compute(
    void* output,
    const void* angles,
    int element_count,
    int iterations,
    cudaStream_t stream) {
    launch_sincos_fp16_compute<PolynomialD5D6HalfTurnFP16>(
        static_cast<__half*>(output),
        static_cast<const float*>(angles),
        element_count,
        iterations,
        stream);
}

void launch_rope_sincos_native_fp16(
    void* output,
    const void* frequencies,
    int sequence_length,
    int frequency_count,
    cudaStream_t stream) {
    if (sequence_length == 0 || frequency_count == 0) {
        return;
    }
    const int frequency_pair_count = frequency_count / 2;
    const int element_count = sequence_length * frequency_count;
    __half* half_output = static_cast<__half*>(output);
    const dim3 block(32, 4);
    const dim3 grid(
        (frequency_pair_count + block.x - 1) / block.x,
        (sequence_length + block.y - 1) / block.y);
    rope_sincos_native_fp16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float2*>(frequencies),
        reinterpret_cast<__half2*>(half_output),
        reinterpret_cast<__half2*>(half_output + element_count),
        sequence_length,
        frequency_pair_count);
}

void launch_rope_sincos_fixed_d3_d4_fp16(
    void* output,
    const void* phase_increments,
    int sequence_length,
    int frequency_count,
    cudaStream_t stream) {
    if (sequence_length == 0 || frequency_count == 0) {
        return;
    }
    const int frequency_pair_count = frequency_count / 2;
    const int element_count = sequence_length * frequency_count;
    __half* half_output = static_cast<__half*>(output);
    const dim3 block(32, 4);
    const dim3 grid(
        (frequency_pair_count + block.x - 1) / block.x,
        (sequence_length + block.y - 1) / block.y);
    rope_sincos_fixed_d3_d4_fp16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint2*>(phase_increments),
        reinterpret_cast<__half2*>(half_output),
        reinterpret_cast<__half2*>(half_output + element_count),
        sequence_length,
        frequency_pair_count);
}

void launch_rope_sincos_fixed_half_turn_d5_d6_fp16(
    void* output,
    const void* phase_increments,
    int sequence_length,
    int frequency_count,
    cudaStream_t stream) {
    if (sequence_length == 0 || frequency_count == 0) {
        return;
    }
    const int frequency_group_count = frequency_count / 4;
    const int element_count = sequence_length * frequency_count;
    __half* half_output = static_cast<__half*>(output);
    const dim3 block(16, 8);
    const dim3 grid(
        (frequency_group_count + block.x - 1) / block.x,
        (sequence_length + block.y - 1) / block.y);
    rope_sincos_fixed_half_turn_d5_d6_fp16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint4*>(phase_increments),
        reinterpret_cast<uint2*>(half_output),
        reinterpret_cast<uint2*>(half_output + element_count),
        sequence_length,
        frequency_group_count);
}

void launch_rope_sincos_fixed_lut_fp16(
    void* output,
    const void* phase_increments,
    const void* phase_table,
    int sequence_length,
    int frequency_count,
    cudaStream_t stream) {
    if (sequence_length == 0 || frequency_count == 0) {
        return;
    }
    const int frequency_pair_count = frequency_count / 2;
    const int element_count = sequence_length * frequency_count;
    __half* half_output = static_cast<__half*>(output);
    const dim3 block(32, 4);
    const dim3 grid(
        (frequency_pair_count + block.x - 1) / block.x,
        (sequence_length + block.y - 1) / block.y);
    rope_sincos_fixed_lut_fp16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint2*>(phase_increments),
        static_cast<const unsigned int*>(phase_table),
        reinterpret_cast<__half2*>(half_output),
        reinterpret_cast<__half2*>(half_output + element_count),
        sequence_length,
        frequency_pair_count);
}

void launch_rope_apply_cached_fp16(
    void* q_output,
    void* k_output,
    const void* q_input,
    const void* k_input,
    const void* rope_table,
    int batch_size,
    int sequence_length,
    int q_head_count,
    int k_head_count,
    int head_dim,
    cudaStream_t stream) {
    if (batch_size == 0 || sequence_length == 0 || head_dim == 0) {
        return;
    }
    const int frequency_group_count = head_dim / 8;
    const int table_group_count = sequence_length * frequency_group_count;
    const uint2* table = static_cast<const uint2*>(rope_table);
    const dim3 block(16, 8);
    const dim3 grid(
        (frequency_group_count + block.x - 1) / block.x,
        (sequence_length + block.y - 1) / block.y,
        batch_size);
    rope_apply_cached_fp16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint4*>(q_input),
        static_cast<const uint4*>(k_input),
        table,
        table + table_group_count,
        static_cast<uint4*>(q_output),
        static_cast<uint4*>(k_output),
        sequence_length,
        q_head_count,
        k_head_count,
        frequency_group_count);
}

void launch_rope_apply_native_fp16(
    void* q_output,
    void* k_output,
    const void* q_input,
    const void* k_input,
    const void* frequencies,
    int batch_size,
    int sequence_length,
    int q_head_count,
    int k_head_count,
    int head_dim,
    cudaStream_t stream) {
    if (batch_size == 0 || sequence_length == 0 || head_dim == 0) {
        return;
    }
    const int frequency_group_count = head_dim / 8;
    const dim3 block(16, 8);
    const dim3 grid(
        (frequency_group_count + block.x - 1) / block.x,
        (sequence_length + block.y - 1) / block.y,
        batch_size);
    rope_apply_native_fp16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint4*>(q_input),
        static_cast<const uint4*>(k_input),
        static_cast<const float4*>(frequencies),
        static_cast<uint4*>(q_output),
        static_cast<uint4*>(k_output),
        sequence_length,
        q_head_count,
        k_head_count,
        frequency_group_count);
}

void launch_rope_apply_fixed_half_turn_d5_d6_fp16(
    void* q_output,
    void* k_output,
    const void* q_input,
    const void* k_input,
    const void* phase_increments,
    int batch_size,
    int sequence_length,
    int q_head_count,
    int k_head_count,
    int head_dim,
    cudaStream_t stream) {
    if (batch_size == 0 || sequence_length == 0 || head_dim == 0) {
        return;
    }
    const int frequency_group_count = head_dim / 8;
    const dim3 block(16, 8);
    const dim3 grid(
        (frequency_group_count + block.x - 1) / block.x,
        (sequence_length + block.y - 1) / block.y,
        batch_size);
    rope_apply_fixed_half_turn_d5_d6_fp16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint4*>(q_input),
        static_cast<const uint4*>(k_input),
        static_cast<const uint4*>(phase_increments),
        static_cast<uint4*>(q_output),
        static_cast<uint4*>(k_output),
        sequence_length,
        q_head_count,
        k_head_count,
        frequency_group_count);
}

void launch_rope_sincos_native_fp16_compute(
    void* output,
    const void* frequencies,
    int sequence_length,
    int frequency_count,
    int iterations,
    cudaStream_t stream) {
    if (sequence_length == 0 || frequency_count == 0) {
        return;
    }
    const int frequency_pair_count = frequency_count / 2;
    const int element_count = sequence_length * frequency_count;
    __half* half_output = static_cast<__half*>(output);
    const dim3 block(32, 4);
    const dim3 grid(
        (frequency_pair_count + block.x - 1) / block.x,
        (sequence_length + block.y - 1) / block.y);
    rope_sincos_native_fp16_compute_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float2*>(frequencies),
        reinterpret_cast<__half2*>(half_output),
        reinterpret_cast<__half2*>(half_output + element_count),
        sequence_length,
        frequency_pair_count,
        iterations);
}

void launch_rope_sincos_fixed_d3_d4_fp16_compute(
    void* output,
    const void* phase_increments,
    int sequence_length,
    int frequency_count,
    int iterations,
    cudaStream_t stream) {
    if (sequence_length == 0 || frequency_count == 0) {
        return;
    }
    const int frequency_pair_count = frequency_count / 2;
    const int element_count = sequence_length * frequency_count;
    __half* half_output = static_cast<__half*>(output);
    const dim3 block(32, 4);
    const dim3 grid(
        (frequency_pair_count + block.x - 1) / block.x,
        (sequence_length + block.y - 1) / block.y);
    rope_sincos_fixed_d3_d4_fp16_compute_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint2*>(phase_increments),
        reinterpret_cast<__half2*>(half_output),
        reinterpret_cast<__half2*>(half_output + element_count),
        sequence_length,
        frequency_pair_count,
        iterations);
}

void launch_rope_sincos_fixed_half_turn_d5_d6_fp16_compute(
    void* output,
    const void* phase_increments,
    int sequence_length,
    int frequency_count,
    int iterations,
    cudaStream_t stream) {
    if (sequence_length == 0 || frequency_count == 0) {
        return;
    }
    const int frequency_group_count = frequency_count / 4;
    const int element_count = sequence_length * frequency_count;
    __half* half_output = static_cast<__half*>(output);
    const dim3 block(16, 8);
    const dim3 grid(
        (frequency_group_count + block.x - 1) / block.x,
        (sequence_length + block.y - 1) / block.y);
    rope_sincos_fixed_half_turn_d5_d6_fp16_compute_kernel<<<
        grid, block, 0, stream>>>(
        static_cast<const uint4*>(phase_increments),
        reinterpret_cast<uint2*>(half_output),
        reinterpret_cast<uint2*>(half_output + element_count),
        sequence_length,
        frequency_group_count,
        iterations);
}

void launch_rope_sincos_fixed_lut_fp16_compute(
    void* output,
    const void* phase_increments,
    const void* phase_table,
    int sequence_length,
    int frequency_count,
    int iterations,
    cudaStream_t stream) {
    if (sequence_length == 0 || frequency_count == 0) {
        return;
    }
    const int frequency_pair_count = frequency_count / 2;
    const int element_count = sequence_length * frequency_count;
    __half* half_output = static_cast<__half*>(output);
    const dim3 block(32, 4);
    const dim3 grid(
        (frequency_pair_count + block.x - 1) / block.x,
        (sequence_length + block.y - 1) / block.y);
    rope_sincos_fixed_lut_fp16_compute_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint2*>(phase_increments),
        static_cast<const unsigned int*>(phase_table),
        reinterpret_cast<__half2*>(half_output),
        reinterpret_cast<__half2*>(half_output + element_count),
        sequence_length,
        frequency_pair_count,
        iterations);
}

}  // extern "C"

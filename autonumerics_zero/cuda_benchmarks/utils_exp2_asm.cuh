// Copyright (c) 2026 Graphcore Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified in 2026 for the standalone fast-polynomial-transcendentals release.

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

struct utils_exp2_asm {
    static __forceinline__ __device__ float2 evaluate(float2 val) {
        float x = val.x;
        float y = val.y;
        int r7, r8;

        // Ported from FlashAttention e2e_asm2
        // Input: %2 = x, %3 = y
        // Output: %0 = r7, %1 = r8
        asm volatile (
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
            : "=r"(r7), "=r"(r8)
            : "f"(x), "f"(y)
        );

        return make_float2(__int_as_float(r7), __int_as_float(r8));
    }
};

struct utils_exp2_asm_half2 {
    static __forceinline__ __device__ __half2 evaluate(__half2 h) {
        float2 f = __half22float2(h);
        float2 res = utils_exp2_asm::evaluate(f);
        return __float22half2_rn(res);
    }
};

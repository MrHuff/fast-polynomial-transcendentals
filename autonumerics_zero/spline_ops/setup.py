# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified in 2026 for the standalone fast-polynomial-transcendentals release.

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


HERE = Path(__file__).resolve().parent
spline_cuda_arch = os.environ.get("SPLINE_OPS_CUDA_ARCH", "sm_100").strip()
nvcc_args = ["-O3", "--use_fast_math"]
if spline_cuda_arch and spline_cuda_arch.lower() not in {"auto", "none"}:
    nvcc_args.insert(1, f"-arch={spline_cuda_arch}")

setup(
    ext_modules=[
        CUDAExtension(
            name="spline_ops",
            sources=[
                "spline_ops.cpp",
                "spline_kernels.cu",
                "spline_kernels_bf16.cu",
                "sincos_kernels.cu",
            ],
            include_dirs=[str(HERE)],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": nvcc_args,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

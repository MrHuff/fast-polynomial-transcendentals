
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Path to cuda_benchmarks for SPLINE_FUNCS.cuh
benchmarks_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../cuda_benchmarks"))
spline_cuda_arch = os.environ.get("SPLINE_OPS_CUDA_ARCH", "sm_100").strip()
nvcc_args = ["-O3", "--use_fast_math"]
if spline_cuda_arch and spline_cuda_arch.lower() not in {"auto", "none"}:
    nvcc_args.insert(1, f"-arch={spline_cuda_arch}")

setup(
    name='spline_ops',
    packages=[],  # Explicitly disable package discovery
    ext_modules=[
        CUDAExtension(
            name='spline_ops',
            sources=[
                'spline_ops.cpp',
                'spline_kernels.cu',
                'spline_kernels_bf16.cu',
                'sincos_kernels.cu',
            ],
            include_dirs=[benchmarks_dir, os.path.dirname(__file__) or '.'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': nvcc_args,
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

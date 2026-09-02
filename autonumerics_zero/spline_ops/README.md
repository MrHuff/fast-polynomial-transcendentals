# SFU spline operators

This distribution builds the CUDA extension and Python wrapper used by the
fast-polynomial-transcendentals experiments. It installs two top-level modules:
the compiled `spline_ops` extension and the `spline_compile` Python API.

The build imports PyTorch's CUDA extension tooling and compiles against the
installed CUDA toolkit. Install the intended PyTorch build first, then build
without isolation so that the extension uses that exact environment:

```bash
python -m pip install --no-build-isolation .
```

The default NVCC target is `sm_100`, matching the GB200 experiments. Set
`SPLINE_OPS_CUDA_ARCH=auto` to let PyTorch choose architectures for the
visible GPUs, or set it to another NVCC architecture such as `sm_90`.

The source distribution contains the C++, CUDA, and generated coefficient
headers required to rebuild the extension.

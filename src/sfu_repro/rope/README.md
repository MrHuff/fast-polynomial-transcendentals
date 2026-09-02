<!--
Copyright (c) 2026 Graphcore Ltd. All rights reserved.
SPDX-License-Identifier: Apache-2.0
Modified 2026-09-02 for the standalone SFU reproduction package.
-->

# Paired sine/cosine and fused RoPE reruns

This directory contains the standalone fitting and benchmark programs for the
paired sine/cosine experiment described in the paper's RoPE appendix. The
programs were adapted from the Apache-2.0-licensed experiment at
`low-bits-training` commit
`1461fd63fdcddb9ef27367a60036dee8e1a11159`; they no longer require that
training framework.

The commands below define **new public reruns**. The historical experiment did
not retain its raw per-trial timing record or a complete runtime manifest, and
the paper does not report a final quantitative RoPE comparison. New output
must therefore be identified by its own repository state, software stack,
hardware, clock policy, and command line. It must not be presented as an exact
replay of historical timing.

## What each program covers

- `fit_polynomial_sincos.py` fits parity-constrained or unrestricted sine and
  cosine pairs with uniform or RoPE-angle weighting. Sollya minimax fitting is
  optional.
- `polynomial_sincos.py` contains the portable PyTorch reference evaluator and
  retained coefficient sets.
- `benchmark_polynomial_sincos.py` performs a portable numerical and table
  generation check. It can run on a CPU.
- `benchmark_cuda_sincos.py` sweeps cache-resident through HBM-resident working
  sets and includes repeated-evaluation controls.
- `benchmark_cuda_rope_sincos.py` checks the fixed-point RoPE phase path and
  its repeated evaluator.
- `benchmark_cuda_fused_rope.py` compares cached-table, native-SFU, and
  polynomial sine/cosine inside the same fused Q/K rotation boundary.
- `autonumerics_zero/spline_ops/sincos_kernels.cu` contains the CUDA kernels;
  `audit_sincos_sass.py` checks the compiled instruction paths.

The retained packed-FP16 kernel coefficients were locally optimized on the
FP16 lattice. The historical lattice-search driver was not retained. The fit
commands below reproduce the checked-in least-squares and Sollya fitting
procedures, not that missing search process.

## CPU setup and tests

Run commands from the repository root. Install a suitable PyTorch build first;
the project metadata deliberately does not choose a PyTorch wheel.

```bash
python -m pip install -e '.[analysis,test]'
python -m pytest tests/test_rope_polynomial_sincos.py
mkdir -p results/rope
```

The Sollya-specific test is skipped when `sollya` is unavailable.

## Fit the polynomial pairs

Fit the parity-constrained D7/D6 uniform control on the quarter-turn interval:

```bash
python -m sfu_repro.rope.fit_polynomial_sincos \
  --basis parity --weighting uniform \
  --sin-terms 4 --cos-terms 4 \
  --output results/rope/quarter-turn-uniform-d7-d6.json
```

Fit the same basis against the Llama RoPE angle distribution used by the
experiment:

```bash
python -m sfu_repro.rope.fit_polynomial_sincos \
  --basis parity --weighting rope \
  --sin-terms 4 --cos-terms 4 \
  --head-dim 128 --max-seq-len 8192 --theta 500000 \
  --output results/rope/quarter-turn-rope-d7-d6.json
```

Run the unrestricted D7/D6 analysis control on the same distribution. This
uses 15 coefficients instead of the parity form's eight and is not a kernel
candidate.

```bash
python -m sfu_repro.rope.fit_polynomial_sincos \
  --basis full --weighting rope \
  --sin-terms 4 --cos-terms 4 \
  --head-dim 128 --max-seq-len 8192 --theta 500000 \
  --output results/rope/quarter-turn-rope-full-d7-d6.json
```

Fit the quarter-turn D3/D4 Sollya control with float32 coefficients:

```bash
python -m sfu_repro.rope.fit_polynomial_sincos \
  --basis parity --reduction quarter-turn --fit-method sollya \
  --weighting uniform --sin-terms 2 --cos-terms 3 \
  --coefficient-dtype float32 \
  --output results/rope/quarter-turn-sollya-d3-d4.json
```

Compare half-turn least-squares and Sollya D5/D4 fits with identical bases:

```bash
python -m sfu_repro.rope.fit_polynomial_sincos \
  --basis parity --reduction half-turn --fit-method least-squares \
  --weighting uniform --sin-terms 3 --cos-terms 3 \
  --coefficient-dtype float32 \
  --output results/rope/half-turn-least-squares-d5-d4.json

python -m sfu_repro.rope.fit_polynomial_sincos \
  --basis parity --reduction half-turn --fit-method sollya \
  --weighting uniform --sin-terms 3 --cos-terms 3 \
  --coefficient-dtype float32 \
  --output results/rope/half-turn-sollya-d5-d4.json
```

Every fit output includes the coefficients, uniform and RoPE-distribution
errors, and the reduced-angle distribution summary. Sollya is required only
for commands that select `--fit-method sollya`.

## Portable numerical check

This CPU command evaluates the retained polynomial pairs on the 8K,
128-dimensional RoPE table and compares them with the PyTorch references:

```bash
python -m sfu_repro.rope.benchmark_polynomial_sincos \
  --device cpu --head-dim 128 --max-seq-len 8192 --theta 500000 \
  --output results/rope/cpu-table-check.json
```

CPU timings are a smoke test, not a paper-aligned hardware measurement.

## Build and inspect the CUDA kernels

The paper-aligned target is SM100. Build the standalone extension, run its
functional tests, and audit the resulting SASS:

```bash
SPLINE_OPS_CUDA_ARCH=sm_100 \
  python -m pip install -v --no-build-isolation \
  ./autonumerics_zero/spline_ops

python -m pytest autonumerics_zero/spline_ops/test_spline_ops.py

python autonumerics_zero/spline_ops/audit_sincos_sass.py \
  "$(python -c 'import spline_ops; print(spline_ops.__file__)')" \
  --json-out results/rope/sincos-sass-audit.json
```

The audit requires NVIDIA's binary utilities. It checks, among other things,
that the polynomial paths contain the intended packed FMA instructions and no
unexpected `MUFU` instructions. Its JSON hashes the exact extension binary,
the sine/cosine CUDA source, the audit program, and the `cuobjdump` binary, and
records the `cuobjdump` version and sanitized command. This is a binary-bound
opcode audit. The extension hash identifies exactly what was inspected, while
the record does not include a build receipt proving that binary was compiled
from the recorded clean source revision.

## Cache-to-HBM and repeated-evaluation rerun

This is the paper's isolated cache/HBM sweep plus compute-saturated repeated
evaluation. It requires the compiled extension and an NVIDIA GPU:

```bash
python -m sfu_repro.rope.benchmark_cuda_sincos \
  --element-counts 4096,65536,1048576,16777216,134217728 \
  --head-dim 128 --theta 500000 \
  --warmup 10 --repeats 50 --trials 5 \
  --compute-element-count 262144 --compute-iterations 64 \
  --output results/rope/cache-hbm-and-repeated.json
```

## RoPE-table and repeated-evaluator rerun

This command exercises the Q0.32 phase-increment path used by the fused
candidate and compares its table output and repeated evaluator with native
sine/cosine:

```bash
python -m sfu_repro.rope.benchmark_cuda_rope_sincos \
  --sequence-length 8192 --head-dim 128 --theta 500000 \
  --warmup 20 --repeats 500 --trials 11 \
  --compute-iterations 64 --compute-repeats 300 \
  --output results/rope/rope-table-and-repeated.json
```

## Fused RoPE rerun

The fused comparison keeps Q/K loads, packed complex rotation, transpose, and
stores common across cached-table, native-SFU, and polynomial variants:

```bash
python -m sfu_repro.rope.benchmark_cuda_fused_rope \
  --batch-size 1 --sequence-length 8192 --head-dim 128 --theta 500000 \
  --head-configs 1:1,8:2,32:8 \
  --warmup 10 --repeats 100 --trials 9 --seed 1234 \
  --output results/rope/fused-rope.json
```

Each fit and benchmark JSON records the repository revision and dirty state,
complete command, Python/PyTorch/CUDA runtime, loaded extension origin and hash
when applicable, raw timing samples, and measurement order. The drivers reject
a dirty checkout or a `spline_ops` build that is not bound to the local source;
`--allow-unbound-source` is diagnostic only. Also retain the compiler/driver
versions and clock/power policy used for every GPU rerun. Measurements on
another GPU or software stack are valid new experiments, but are not
paper-aligned GB200 timings.

# Reproducing the experiments

This guide distinguishes portable code execution, paper-aligned hardware
measurements, access-gated model evaluation, and offline inspection of
historical evidence. These are different reproduction levels.
The same workflow statuses are available to tools in
[`repro/experiments.json`](../repro/experiments.json).
The [experiment matrix](EXPERIMENT_MATRIX.md) maps each paper scope to its
kernel, fitter, public command, and exact historical replay boundary.

## Reproduction levels

1. **Software check:** import `sfu_repro`, validate configuration, and run tests.
2. **Numerical reproduction:** re-fit a function or rebuild a packed kernel and
   compare its numerical output.
3. **Paper-aligned timing:** repeat the stated workload and measurement protocol
   on NVIDIA GB200/SM100 hardware.
4. **Public model rerun:** run a B1--B4 native/polynomial model probe or the B5
   three-arm routed-`exp2` probe through the pinned TorchTitan adapter.
5. **Public training rerun:** run a B1--B4 native/polynomial pair, optionally
   with held-out validation.
6. **Downstream model evaluation:** patch a supported, legally obtained
   open-weight model and run the stated quality or throughput protocol.
7. **Historical evidence audit:** verify and plot retained results whose exact
   original runtime is not fully reconstructable.

Levels 1--6 create new observations. Level 7 does not. Timing from a different
GPU, driver, clock policy, tensor shape, or software revision should be reported
as a new experiment.

## Clone and create the base environment

The repository is public and can be cloned anonymously. If using an
authenticated Git transport, do not embed a token in the URL. The base package
stays lightweight: PyTorch and Transformers are declared only
by the optional `test` extra used below. If the default PyTorch distribution is
not appropriate for the intended CPU or CUDA platform, install the desired
build first; the extra will reuse any compatible installation.

```bash
git clone --recurse-submodules https://github.com/MrHuff/fast-polynomial-transcendentals.git
cd fast-polynomial-transcendentals
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[analysis,test]'
python -m pytest
```

Inspect optional dependencies without launching a benchmark:

```bash
sfu-doctor --profile analysis
sfu-doctor --profile component
```

If the repository was cloned without submodules, initialize them explicitly:

```bash
git submodule update --init --recursive
```

For B2, B3, and the integrated fractional-`exp2` probes, install the patched
FA4 package from the pinned submodule rather than a different `fa4` wheel:

```bash
python -m pip install -e ./flash-attention/flash_attn/cute
python -c 'from flash_attn.cute import interface, polynomial_manifest'
```

The import check must resolve to this checkout. An upstream installation that
lacks `flash_attn.cute` or the polynomial-manifest APIs is not sufficient.

Before a measured run, record the top-level and submodule revisions and whether
either tree is dirty. Do not describe an artifact as commit-bound when it was
produced from an unrecorded local modification.

## Re-run the fitting and arithmetic checks

The repository contains several distinct fitting paths. Preserve the command,
tool versions, generated JSON, and repository state for every new fit. Dense
sampled maximum errors are empirical grid estimates unless a producing program
explicitly emits a formal certificate.

The `fit_all_degrees*.py`, `fit_gelu_fp16.py`,
`fit_fa4_tanh_backends.py`, `fit_exp2_softmax.py`, and
`fit_flash_sigmoid_exp2_d2.py` entry points add an `_provenance` object to each
JSON output. The Sollya header generator does the same and also hashes its
generated header. These records contain a machine-prefix-free command, the
repository revision and dirty state, tool versions, and SHA-256 hashes of the
fitting sources and canonical numerical payload. Existing coefficient and
metric keys remain unchanged. A fit record identifies a coefficient-search
run; it does not promote coefficients into a checked-in kernel or attest the
accuracy of compiled device arithmetic.

### FP16 and BF16 activation fits

The retained general sweeps cover D3--D6 sigmoid, tanh, sigmoid derivative,
tanh derivative, and SiLU derivative forms:

```bash
mkdir -p outputs
python autonumerics_zero/evolution/fit_all_degrees.py \
  --json-out outputs/all_degree_coefficients_fp16.json
python autonumerics_zero/evolution/fit_all_degrees_bf16.py \
  --json-out outputs/all_degree_coefficients_bf16.json
```

Render the new BF16 fit as a reviewable candidate header without modifying the
deployed checked-in header:

```bash
python autonumerics_zero/evolution/generate_bf16_structs.py \
  --bf16-input outputs/all_degree_coefficients_bf16.json \
  --fallback-header autonumerics_zero/spline_ops/spline_structs_odd_bf16.cuh \
  --output outputs/spline_structs_odd_bf16.generated.cuh \
  --receipt-out outputs/spline_structs_odd_bf16.generated.cuh.provenance.json
```

The generator derives the sigmoid, tanh, and SiLU families from the new BF16
fit. It copies ERF and GELU structs from the explicitly named reviewed fallback
because that BF16 sweep does not fit those families. The receipt hashes the fit
JSON, fallback header, generator, and result. Promoting the candidate into the
CUDA extension is a separate code-review step. A source-bound receipt requires
the BF16 fit to have been generated by `fit_all_degrees_bf16.py` at the same
clean revision with unchanged source hashes and numerical payload;
`--allow-unbound-source` produces a diagnostic receipt instead.

The explicit output arguments leave the checked-in historical coefficient
files unchanged. Omitting `--json-out` preserves the original scripts'
conventional paths under `autonumerics_zero/cuda_benchmarks/analysis_results/`.
The BF16 same-form comparison additionally requires PyTorch. The separate
generated-header comparison requires a local Sollya installation. Write its
header and measurements to new paths so the retained artifacts remain intact:

```bash
python autonumerics_zero/spline_ops/generate_sollya_structs_bf16.py \
  --current-header autonumerics_zero/spline_ops/spline_structs_odd_bf16.cuh \
  --header-out outputs/spline_structs_sollya_bf16.generated.cuh \
  --json-out outputs/sollya_device_bf16.generated.json
```

This produces a fresh host-arithmetic comparison with the retained JSON shape;
it is not a device-error run. The producer parses current deployed-header
literals without first BF16-rounding them, explicitly BF16-rounds Sollya
coefficients, and evaluates both sides with NumPy real arithmetic. The source
artifact also does not establish endpoint-constrained least-squares lineage
for every current row. See the
[function-table review](../extras/paper/FUNCTION_TABLE_REVIEW.md) before using
this output to interpret manuscript Table 2.

`fit_all_degrees_bf16.py` deliberately preserves the historical fit-selection
surrogate: it rounds the multiply and add of each Horner stage separately. The
deployed CUDA kernels use packed `__hfma2`, which rounds a fused multiply-add
once. Use the compiled GPU accuracy programs for device error; the historical
search surrogate is not an exact SASS emulator.

### Recovered ERF and GELU fitter

The historical FP16 ERF/GELU D3--D6 sweep is available through a standalone,
caller-directed adapter:

```bash
mkdir -p results
python autonumerics_zero/evolution/fit_gelu_fp16.py \
  --json-out results/gelu-coefficients-fp16.json
```

It preserves the original two-dimensional interpolation/clamp sweep and FP16
replay. The deployed BF16 ERF/GELU structs were mechanically initialized from
these FP16-derived coefficients. The [provenance guide](PROVENANCE.md) records
the recovered source commit and modification boundary.

### B2 tanh and fractional-exp2 fits

Refit the separate B2 CuTe-FP32 and handwritten-device-BF16 targets without
overwriting a retained artifact:

```bash
python autonumerics_zero/evolution/fit_fa4_tanh_backends.py \
  --output results/fa4-tanh-backend-fits.json
```

The fractional-`exp2` fit uses NumPy and writes a new JSON artifact:

```bash
mkdir -p results
python autonumerics_zero/evolution/fit_exp2_softmax.py \
  --samples 1000001 \
  --json-out results/exp2-fit.json
```

The sequence-length-specific two-FMA D2 search prints its coefficients and
error metrics:

```bash
python autonumerics_zero/evolution/fit_flash_sigmoid_exp2_d2.py \
  --sequence-length 4096 --score-sigma 1.0 --seed 1234 --maxiter 300 \
  --json-out results/flash-sigmoid-d2-fit.json
```

### B3 deployed-fit audit

The exact B3 direct-D3 forward and factored-D4 derivative constants are in the
pinned FlashAttention-4 checkout. The program that selected them was not
retained, including its objective, weighting, and sample set. Audit the
deployed constants and packed-BF16 arithmetic instead:

```bash
python scripts/audit_b3_deployed_fit.py \
  --grid-min -6 --grid-max 6 --grid-points 24577 \
  --json-out results/b3-deployed-fit-audit.json
```

The audit fails if the FlashAttention-4 Gitlink, checkout, or audited source
files do not match the pinned revision. It is a source-bound verification, not
a reconstructed fitter. See [the B3 audit note](B3_DEPLOYED_FIT_AUDIT.md) for
the precise emulated rounding sequence.

## Build the packed CUDA extension

The paper-aligned extension target is SM100. From the repository root:

```bash
cd autonumerics_zero/spline_ops
SPLINE_OPS_CUDA_ARCH=sm_100 python -m pip install -v --no-build-isolation .
cd ../..
python -m pytest autonumerics_zero/spline_ops/test_spline_ops.py
```

This step requires a CUDA compiler compatible with the installed PyTorch build.
Compilation for another architecture is supported only as a new validation
target; it does not reproduce the reported GB200 timing.

## Run the isolated FP16 function-speed benchmark

The standalone function sweep compares native PyTorch functions with the
paper's selected packed polynomial programs: D3 sigmoid, D4 tanh, D3 SiLU, and
D5 GELU in forward; and D4 sigmoid, D4 tanh, D3 SiLU, and D5 GELU in backward.
It uses FP16 so the polynomial kernels evaluate pairs of values with `half2`
vectorized fused multiply-add instructions. Every polynomial backward entry is
a separately fitted direct derivative program evaluated from the original
input. These isolated rows do not encode the case-specific backward
construction used by every B1--B4 integration.

After building `spline_ops`, reproduce the paper-aligned endpoints and timing
protocol on a GB200:

```bash
mkdir -p outputs
python autonumerics_zero/experiments/benchmark_report_function_speed.py \
  --sizes 4194304,268435456 \
  --warmup 1000 --repetitions 1000 --rounds 9 --seed 1234 \
  --output outputs/isolated-function-speed.json
```

The 4,194,304-element workload is L2-cache resident under the benchmark's
two-array forward and three-array backward working-set model. The
268,435,456-element workload is HBM resident. Each reported speedup is the
median native CUDA-event time divided by the median polynomial time across
nine rounds whose measurement order alternates. For native sigmoid and tanh
backward, the forward output is precomputed outside the timed region and the
backward uses the framework's algebraic derivative. Those two rows compare a
direct derivative fit with native algebraic backward; they are not
SFU-versus-FMA backward comparisons.

The retained summary is
`evidence/report-data/isolated_function_speedups_fp16.csv`; its protocol and
lineage are in the adjacent `.provenance.json`, and the complete per-round
GB200 measurements are in
`evidence/report-data/isolated_function_speedups_fp16_gb200.json`. A new run
records the current repository revision, dirty-tree state, benchmark-driver
hash, loaded extension binary and hash, CUDA/PyTorch versions, and GPU
properties.

## Run the B1--B4 component probe

The standalone driver replaces the original training-repository harness. B1
and B4 require the compiled `spline_ops` extension. B2 and B3 require the
patched FlashAttention-4 submodule. B3 is restricted to the paper geometry at
sequence length 4096.

```bash
mkdir -p results
python scripts/benchmark_components.py \
  --cases b1,b2,b3,b4 \
  --json-out results/components.json
```

Inspect the complete interface before changing geometry or measurement counts:

```bash
python scripts/benchmark_components.py --help
```

Preserve warm-up, repetitions, measurement order, shapes, dtype, and clock
policy when making a paper-aligned comparison. Report per-round observations in
addition to aggregate statistics. The driver refuses a dirty checkout or a
`spline_ops`/FA4 import that is not bound to the declared local source. The
`--allow-unbound-source` override is diagnostic and labels its JSON accordingly.

## Run the isolated fractional-exp2 benchmark

The pure-CUDA driver compiles the fixed SM100 source in a temporary directory,
runs its accuracy and timing sweeps, and records all nine timing samples in
source order for every result:

```bash
python autonumerics_zero/experiments/benchmark_exp2_pwl2.py \
  --json-out outputs/exp2_pwl2_isolated.json
```

The JSON binds the source, driver, compiler binary, compiled temporary binary,
compiler version, compile command, GPU name, fixed workload geometry, and raw
samples. It rejects a dirty or unversioned checkout unless
`--allow-unbound-source` is supplied, which labels the output diagnostic.

## Run the FlashAttention-4 fractional-exp2 probe

The mixed schedule uses
`name=backend:forward_frequency:backward_frequency` variant syntax. The
paper-aligned apples-to-apples workload is explicit here and in
`repro/experiments.json`:

```bash
python scripts/benchmark_fa4_exp2_mix.py \
  --batch-size 1 --sequence-length 4096 \
  --heads 32 --kv-heads 8 --head-dim 128 \
  --warmup 10 --iterations 100 --rounds 15 --seed 1234 \
  --variants \
  sfu=d3:0:0,d3_default=d3:auto:0,d3_matched=d3:12:32,pwl2_matched=pwl2_safe_f16:12:32,d2_matched=d2_safe:12:32 \
  --json-out results/fa4-exp2.json
```

```bash
python scripts/benchmark_fa4_exp2_mix.py --help
```

This path depends on behavior in the pinned patched FlashAttention-4 revision.
An unpatched upstream installation is not interchangeable. The driver records
the imported FA4 file and hash and rejects a dirty or foreign source by default.

## Re-run the paired sine/cosine and fused RoPE experiments

The RoPE subtree contains the fitting, portable reference, CUDA table,
cache-to-HBM, repeated-evaluation, SASS-audit, and fused Q/K rotation paths.
Start with the CPU fit and numerical check:

```bash
mkdir -p results/rope

python -m sfu_repro.rope.fit_polynomial_sincos \
  --basis parity --weighting rope \
  --sin-terms 4 --cos-terms 4 \
  --head-dim 128 --max-seq-len 8192 --theta 500000 \
  --output results/rope/quarter-turn-rope-d7-d6.json

python -m sfu_repro.rope.benchmark_polynomial_sincos \
  --device cpu --head-dim 128 --max-seq-len 8192 --theta 500000 \
  --output results/rope/cpu-table-check.json
```

After building `spline_ops`, run the cache/HBM and fused comparisons:

```bash
python -m sfu_repro.rope.benchmark_cuda_sincos \
  --element-counts 4096,65536,1048576,16777216,134217728 \
  --head-dim 128 --theta 500000 \
  --warmup 10 --repeats 50 --trials 5 \
  --compute-element-count 262144 --compute-iterations 64 \
  --output results/rope/cache-hbm-and-repeated.json

python -m sfu_repro.rope.benchmark_cuda_fused_rope \
  --batch-size 1 --sequence-length 8192 --head-dim 128 --theta 500000 \
  --head-configs 1:1,8:2,32:8 \
  --warmup 10 --repeats 100 --trials 9 --seed 1234 \
  --output results/rope/fused-rope.json
```

The complete [RoPE rerun guide](../src/sfu_repro/rope/README.md) also provides
the uniform, unrestricted, half-turn, and Sollya fits; fixed-point RoPE table
and repeated-evaluator benchmark; extension build; and SASS audit commands.
The historical local FP16-lattice search, raw per-trial timing record, and
complete runtime manifest were not retained. These commands produce new public
measurements. New fit and benchmark JSON records the source revision, dirty
state, runtime, complete invocation, module origin when applicable, all timing
samples, and measurement order. A dirty or foreign `spline_ops` source requires
the explicit diagnostic-only `--allow-unbound-source` override.

## Run public TorchTitan model probes, training, and validation

The public training bridge uses the upstream TorchTitan v0.2.2 submodule at
commit `73a0e6979dd10b6b1904098eb3c8f62c18ab87ce`. The official release names
PyTorch `2.12.0.dev20260220+cu126` and TorchAO
`0.17.0.dev20260220+cu126`. Install those wheels, TorchTitan, the packed spline
extension, and patched FA4 in a dedicated environment:

```bash
python -m pip install --pre \
  --index-url https://download.pytorch.org/whl/nightly/cu126 \
  -r configs/torchtitan/pytorch-v0.2.2-cu126.requirements.txt
python -m pip install -e ./torchtitan
python -m pip install -v --no-build-isolation ./autonumerics_zero/spline_ops
python -m pip install -e ./flash-attention/flash_attn/cute
python scripts/prepare_torchtitan_assets.py --execute
sfu-doctor --profile train
```

`prepare_torchtitan_assets.py` is an offline dry run unless `--execute` is
present. It fetches tokenizer files only from the immutable Llama 3.1 8B and
DeepSeek-V3.1-Base commits in `sfu_repro.torchtitan.pins`, then writes
`tokenizer-manifest.json` with every file digest. Authentication comes from the
standard Hugging Face environment or credential store; the helper has no token
argument. The training launcher verifies this manifest before execution.

The repository's CPU/static tests validate the eleven TOML files, token
horizons, matched configuration equality, B5 routing schedules, source pin,
and generated launcher arguments. Import and runtime validation require the
matching release stack; paper-scale execution remains hardware-gated.

`configs/torchtitan/paper_runs.json` binds B1--B4 to matched native and
polynomial configs and records their model-probe and 100B-token geometry. It
also binds B5 to the native, PWL2-safe-FP16, and D2-safe routed-`exp2` configs.
The launcher prints commands by default and runs only when `--execute` is
present. For example, preview the B1 model probe:

```bash
python scripts/run_torchtitan.py \
  --case b1 --variant native --phase model-probe
```

Every executed arm writes an immutable `sfu-launch-receipt.json` in its
effective TorchTitan dump folder before launch. The receipt embeds the selected
base TOML bytes and hash together with every effective override, sanitized
effective arguments, dataset and tokenizer pins, verified tokenizer-manifest
hash, source revisions and dirty states, runtime versions, and launch topology.
Rendezvous hosts and transport identifiers are redacted; scientific options
remain exact. Pretraining
receipts also bind a recursive SHA-256 identity for every regular file in the
shared seed checkpoint. The exporter below finds the nearest receipt
automatically and rejects a missing, unbound, or mismatched native/candidate
pair.

Run both arms, then export the one TensorBoard event directory produced for
each arm:

```bash
python scripts/export_torchtitan_metrics.py \
  --case b1 --phase model-probe \
  --native-dir outputs/torchtitan/probes/b1/native/tb/RUN_TIMESTAMP \
  --polynomial-dir outputs/torchtitan/probes/b1/polynomial/tb/RUN_TIMESTAMP \
  --output results/b1_torchtitan_model_probe.json
```

Replace each `RUN_TIMESTAMP` with the corresponding directory name emitted by
TorchTitan.

The exporter hashes both event files and both launch receipts and reports
medians over steps 20--100 inclusive. Its speedup is based on TorchTitan's
wall-clock end-to-end iteration metric. It is a new public-protocol statistic,
distinct from the paper's synchronized CUDA-event forward, backward, and
optimizer phase timings. `--allow-unbound-receipts` is diagnostic only and
marks the exported result unbound.

The B1 path fuses the gate and up projections in both arms and calls the packed
native or D3 polynomial SwiGLU kernel. B2 and B3 patch all 32 Llama attention
modules. B3 enforces sequence length 4096 and deterministically initializes
the added query/key norms. B4 patches exactly the 29 routed grouped-expert
blocks; its router, dense block, and shared experts remain native. Tensor
parallelism is disabled for B1 because the pinned upstream Llama tensor plan
addresses the unfused projection names.

Preview the three B5 model-probe arms:

```bash
python scripts/run_torchtitan.py \
  --case b5 --variant native --phase model-probe
python scripts/run_torchtitan.py \
  --case b5 --variant pwl2_safe_f16 --phase model-probe
python scripts/run_torchtitan.py \
  --case b5 --variant d2_safe --phase model-probe
```

All three use a Llama-8B-shaped model, batch size 1, sequence length 4096, 80
steps, and disabled compilation; steady-state summaries begin at step 20. The
native arm uses `d3:0:0`, which routes no lanes through a polynomial evaluator.
The two candidate arms use
`pwl2_safe_f16:12:32` and `d2_safe:12:32`; each routes four lanes per group,
giving nominal polynomial fractions of one third in the forward pass and one
eighth in the backward pass. Preserve each arm's TensorBoard event directory
as raw output.

Export each named candidate against the native arm. The exporter hashes both
source event files, keeps the candidate label, and reads the inclusive 20--80
window from the run manifest:

```bash
python scripts/export_torchtitan_metrics.py \
  --case b5 --phase model-probe \
  --native-dir outputs/torchtitan/probes/b5/native/tb/RUN_TIMESTAMP \
  --candidate-dir outputs/torchtitan/probes/b5/pwl2_safe_f16/tb/RUN_TIMESTAMP \
  --candidate-name pwl2_safe_f16 \
  --output results/b5_pwl2_safe_f16_torchtitan_model_probe.json

python scripts/export_torchtitan_metrics.py \
  --case b5 --phase model-probe \
  --native-dir outputs/torchtitan/probes/b5/native/tb/RUN_TIMESTAMP \
  --candidate-dir outputs/torchtitan/probes/b5/d2_safe/tb/RUN_TIMESTAMP \
  --candidate-name d2_safe \
  --output results/b5_d2_safe_torchtitan_model_probe.json
```

This B5 recipe is a new public TorchTitan rerun of the retained probe shape and
schedule. The compact historical model-probe artifacts attest the shape, step
count, and route fractions, but they do not contain the invocation, resolved
configuration, raw logs, seed, dataset, tokenizer, or source revision. The
compile-disabled setting and integer route frequencies are source-derived; the
step-20 boundary is retained for D2 and a declared new aggregation choice for
PWL2. The seed and C4 test fixture are also declared public choices. The
proprietary `gc-training` plumbing and historical initialization were not
retained. B5 defines a model probe only; it does not define a long-run training,
validation, or downstream checkpoint protocol.

Create one case-specific seed checkpoint, then load it into both arms:

```bash
python scripts/run_torchtitan.py \
  --case b1 --variant native --phase seed-checkpoint --execute

python scripts/run_torchtitan.py \
  --case b1 --variant native --phase pretraining \
  --seed-checkpoint outputs/torchtitan/seeds/b1_llama3_8b/checkpoint/step-0

python scripts/run_torchtitan.py \
  --case b1 --variant polynomial --phase pretraining \
  --seed-checkpoint outputs/torchtitan/seeds/b1_llama3_8b/checkpoint/step-0
```

The last two commands remain dry runs until `--execute` is added. Multi-node
execution requires the ordinary `torchrun` rendezvous arguments; repeat
`--torchrun-arg` to pass them. Preserve the seed checkpoint, resolved config,
dataset snapshot or local data hash, tokenizer assets, logs, and package/GPU
metadata with every result. On execution, the launcher hashes every regular
file in the seed-checkpoint tree and stores the aggregate identity in the
launch receipt. The paired exporter requires the native and polynomial
receipts to contain the same identity.

Validation was disabled in the historical campaigns. `--validation` enables a
new public held-out protocol for a pretraining run; it is not a reconstruction
of a historical validation result:

```bash
python scripts/run_torchtitan.py \
  --case b1 --variant native --phase pretraining \
  --seed-checkpoint outputs/torchtitan/seeds/b1_llama3_8b/checkpoint/step-0 \
  --validation
```

Run this command for both native and polynomial arms of B1, B2, B3, and B4,
using the same case-specific seed checkpoint for each pair. The manifest
`repro/experiments.json` enumerates all eight validation launches and all four
paired exports. A bound validation export requires held-out loss series from
both arms. Its commands pass `--require-validation`, which also rejects
receipts from launches that did not enable validation.

For a completed native/polynomial pretraining pair, run the same exporter with
`--phase pretraining`. It preserves the full logged training-loss and token
series, available end-to-end throughput/timing series, and any validation-loss
and validation-throughput series. It does not compute a pretraining-equivalence
claim.

The public long-run seed is 1234. That value and the deterministic B3 norm
reset are declared public protocol choices because the corresponding historical
initialization details were not retained. B1--B3 training and all held-out
validation use `gmongaras/SlimPajama-627B_Reupload` at commit
`c34c22dbb10ae6b264a2f357a909d1a537141b36`. The original Cerebras Hub
repository is no longer anonymously available, and byte identity with the
historical input has not been established. The public loader uses deterministic
streaming order rather than the historical shuffled loader. B4 training uses
`allenai/olmo-mix-1124` at commit
`8162bd79c6dc4fea470506531a8d791badc06b4b`; the historical B4 corpus and order
remain unresolved. These recipes reproduce the code paths and stated
experimental geometry with declared public inputs; they do not claim
bit-for-bit identity with the historical 100B-token trajectories.

## Run a downstream open-weight evaluation

Three standalone entry points replace the former training-module integration:

```bash
python scripts/benchmark_open_weights.py --help
python scripts/run_open_weight_suite.py --help
python scripts/check_open_weight_environment.py --help
```

The five paper cases and source-derived protocol are declarative in
`configs/open_weight_paper.json`. Preview a credential-free command without
loading a model:

```bash
python scripts/run_open_weight_suite.py \
  --models glm4p7_flash \
  --mode eval \
  --quality-eval \
  --dry-run
```

Run quality evaluation without the synthetic throughput phases:

```bash
python scripts/run_open_weight_suite.py \
  --models glm4p7_flash \
  --mode eval \
  --quality-eval
```

Run prefill and decode throughput without quality evaluation:

```bash
python scripts/run_open_weight_suite.py \
  --models glm4p7_flash \
  --mode both \
  --no-eval
```

Use a separate environment for each software profile. The historical records
name Transformers 4.48.2 for Qwen2.5; 5.9.0 for Qwen3, GLM, and GPT-OSS; and
4.57.6 for Kimi. Exact source-derived package sets and their case mapping are
under `configs/eval_environments/`. Install a PyTorch build appropriate for the
local CUDA stack, create one environment per profile, install its requirements
file, and install the pinned evaluator checkout in editable mode:

```bash
python -m pip install -r \
  configs/eval_environments/hf_5_9_moe.requirements.txt
python -m pip install -e ./lm-evaluation-harness
python -m pip install -v --no-build-isolation \
  ./autonumerics_zero/spline_ops
```

The Kimi profile also requires the pinned full FlashAttention checkout. Force
a local build: its upstream installer may otherwise fetch a prebuilt wheel
whose CUDA extension was not compiled from this gitlink.

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE \
  python -m pip install -v --no-build-isolation ./flash-attention
```

Use the matching requirements file for a different case. Check the selected
profile without loading a model:

```bash
python scripts/check_open_weight_environment.py \
  --models glm4p7_flash --check-installed
```

The full five-model matrix must be split across its three incompatible Python
environments. Substitute the actual interpreter from each environment:

```bash
python scripts/run_open_weight_suite.py \
  --python /path/to/hf-4.48-python \
  --models qwen2p5_7b --mode eval --quality-eval \
  --output-dir outputs/open_weight/quality

python scripts/run_open_weight_suite.py \
  --python /path/to/hf-5.9-python \
  --models qwen3_30b_a3b_base,glm4p7_flash,gpt_oss_120b \
  --mode eval --quality-eval --output-dir outputs/open_weight/quality

python scripts/run_open_weight_suite.py \
  --python /path/to/hf-4.57-python \
  --models kimi_linear_48b_a3b_base --mode eval --quality-eval \
  --output-dir outputs/open_weight/quality
```

Repeat those three commands with `--mode both --no-eval` and
`--output-dir outputs/open_weight/throughput` for prefill and decode timing.
These are the exact command groups listed in `repro/experiments.json`; a single
`--models paper` execution cannot represent all three dependency profiles.

Every non-dry suite run performs this installed-package, module, model-revision,
task-protocol, and local-checkout preflight automatically. A mismatch stops the
run before model loading. `--skip-environment-check` is reserved for
diagnostics; output produced with it is outside the declared paper-quality
public protocol. The child artifact records whether that parent preflight
passed, was skipped, or was not run. The paper-quality summarizer accepts only
`passed`.

Each non-native row must patch its declared activation, router, or softcap
scope; zero-target patches fail before evaluation or timing. Results retain all
prefill/decode repetition samples and attest the loaded `spline_ops` and FA4
module files, hashes, package metadata, source revisions, and clean state. The
result envelope also records the complete path-sanitized command, evaluation
limit, evaluator batch size, model load options, and every measurement control.
It records the sanitized paths and SHA-256 digests of the suite and environment
configurations, plus the task configuration for quality runs. The paper-quality
summarizer rejects native-like patch scope, reordered variants, incomplete
samples, unbound module origins, limited quality runs, skipped preflights,
changed configuration content, and any dtype, shape, batch, warm-up, or
repetition setting that differs from `configs/open_weight_paper.json`.

The historical Qwen2.5 and Qwen3 checkpoint revisions were not retained. The
public protocol therefore selects immutable replacements in
`configs/open_weight_paper.json`; each is labeled
`public-protocol-selection`. The other three revisions are reconstructed from
the retained source. The suite passes every selected revision to both model and
tokenizer loading and records it in the output. `--model-revision` can select a
different full commit for a clearly labeled new run.

The quality command also loads `configs/lm_eval_paper_tasks.json`. That file
pins lm-eval v0.4.12 at commit
`6d642546f4688648fced259eb3302efd36ece5af`, all eight task-dataset commits,
and the four evaluator seeds. The runner imports the pinned checkout, verifies
its clean Git state, and rejects undeclared or mismatched dataset revisions.
Built-in task definitions at that commit fix prompts, splits, filters, and
metrics. Paper-quality runs log per-example documents, prompt and target
hashes, model responses, and metrics by default; preserve the complete result
JSON. Each expanded leaf task must report positive integral `n-samples` counts,
with `effective` equal to `original` and to the retained sample-list length.
This independently rejects truncated sample output and limited task
populations. Use `--no-eval-log-samples` only for an explicitly non-paper
diagnostic.

The historical campaign did not retain task-dataset commits or its sampled
few-shot identities. The pinned datasets and deterministic seeds therefore
define a new public downstream protocol. They do not turn the retained table
into an exact replay of the historical evaluation.

Use only a model identifier and cache location that you are authorized to
access. The scripts do not grant model or dataset rights and should not be used
to package or upload weights. When authentication is required, the child
process reads `HF_TOKEN` from its environment; the suite does not accept it as
a command-line argument or print it during a dry run. Keep provider credentials
out of every JSON result and committed configuration.

The historical paper table is not runtime-attested by these new scripts. Its
source-derived protocol is retained in
`evidence/report-data/open_weight_eval_protocol.json`. A new run should state
its exact model revision, tokenizer revision, evaluation-harness revision,
tasks, few-shot settings, dtype, throughput lengths, warm-up, repetition count,
and hardware.

After completing both the quality and throughput passes for every configured
case and variant, validate and combine their raw JSON files:

```bash
python scripts/summarize_open_weight_results.py \
  outputs/open_weight/quality outputs/open_weight/throughput \
  --config configs/open_weight_paper.json \
  --task-config configs/lm_eval_paper_tasks.json \
  --environments configs/eval_environments/profiles.json \
  --experiment-manifest repro/experiments.json \
  --mode combined \
  --csv-out results/open_weight_summary.csv \
  --markdown-out results/open_weight_summary.md
```

The summarizer joins records by immutable model revision and variant, checks
the task metrics and dataset/evaluator protocol, requires the declared package
profile, and verifies each source binding used by that artifact. Quality
artifacts require the pinned lm-eval checkout. Kimi artifacts also require the
pinned FlashAttention checkout because that profile imports it. Other
throughput artifacts require neither checkout, and TorchTitan is unrelated to
this workflow. The summarizer emits a comprehensive CSV and a compact Markdown
table. Missing cases, variants, quality tasks, or throughput phases fail a
combined summary. It also requires exact content hashes for the selected
configuration files and the exact public batch, dtype, sequence lengths, seed,
warm-ups, repetition counts, decode length, evaluator batch sizes, task
configuration, and unlimited task population. Diagnostic CLI overrides remain
useful for smoke tests, but their JSON cannot enter a paper-quality summary.
`--allow-unbound-source` relaxes source binding only; it does not relax these
protocol checks.

The summary consumes new public-run JSON only. The exact historical Table 5
inputs and outputs are incomplete: two Qwen checkpoint revisions,
task-dataset commits, sampled few-shot identities, raw per-model result shards,
and the complete runtime were not retained. The public pins make future runs
repeatable; they do not reconstruct that historical table.

## Replot retained paired pre-training histories

This offline operation reads the materialized CSV. It does not contact an
experiment service or rerun training:

```bash
mkdir -p results
python evidence/pretraining/plot_paired_loss.py \
  --input evidence/report-data/b1_b4_paired_loss_curves.csv \
  --output results/b1_b4_paired_loss_curves.pdf
```

The CSV records sampled points from single historical trajectories. The
repository omits the original distributed launcher, full configuration,
datasets, checkpoints, and the information needed to prove identical
initialization and data order. No command in this repository exactly repeats
the reported 100B-token pre-training runs.

## Rebuild the paper extras

`extras/paper/` contains a local-input-only reconstruction layer for the
paper-facing artifacts. The exact paired-loss layout, including common token
horizons and final-25-billion-token insets, is generated with:

```bash
python extras/paper/plot_paired_loss_curves.py \
  --data-path evidence/report-data/b1_b4_paired_loss_curves.csv \
  --input-provenance evidence/report-data/b1_b4_paired_loss_curves.provenance.json \
  --output-path outputs/paper/b1_b4_paired_loss_curves.pdf \
  --output-provenance outputs/paper/b1_b4_paired_loss_curves.figure.json
```

The method figures are an offline CPU transformation. The six accuracy plots
are fresh measurements of the explicitly loaded CUDA extension:

```bash
python extras/paper/generate_method_figures.py \
  --coefficients autonumerics_zero/cuda_benchmarks/analysis_results/all_degree_coefficients_bf16.json \
  --output-dir outputs/paper/method

python extras/paper/generate_accuracy_figures.py \
  --extension-dir autonumerics_zero/spline_ops \
  --output-dir outputs/paper/accuracy
```

Generate a fresh deployed-header-shaped Sollya control and audit the
quantitative tables:

```bash
python autonumerics_zero/spline_ops/generate_sollya_structs_bf16.py \
  --current-header autonumerics_zero/spline_ops/spline_structs_odd_bf16.cuh \
  --header-out outputs/paper/spline_structs_sollya_bf16.generated.cuh \
  --json-out outputs/paper/sollya_device_bf16.generated.json

python extras/paper/check_paper_tables.py \
  --function-comparison outputs/paper/sollya_device_bf16.generated.json \
  --function-lineage extras/paper/function_table_lineage.json \
  --allow-review-required \
  --output outputs/paper/table_audit.json
```

The retained deployed-header comparison and released isolated-function,
integration, complete-model, downstream, and pre-training evidence reproduce
the manuscript's rounded table cells. Running the checker without overrides
audits all six retained sources without requiring Sollya. For Table 2, this is
a numeric source-to-typeset audit only.

`extras/paper/generate_sollya_comparison.py` is an auxiliary D3--D6
coefficient-sweep sensitivity control, not the Table 2 source. Seven selected
cells differ after manuscript rounding because (1) its sigmoid D3 row uses the
earlier fit and clamp 4.75 instead of the later deployed fit and clamp 6.0,
(2) the later tanh D4 device refit is absent from the sweep, (3) its Sollya
side directly fits the SiLU residual while the table composes both columns
from sigmoid coefficients, and (4) its GELU paths use the FP16 sweep and an
11-bit Sollya budget rather than the table's deployed BF16 header and
8-bit/BF16 control.

Those workflow divergences are distinct from the retained Table 2 semantic
review: asymmetric coefficient rounding, host NumPy rather than device error,
and incomplete endpoint-constrained least-squares lineage. The default checker
reports `review-required` and exits 2 even when all numeric cells match. Add
`--allow-review-required` only to acknowledge the recorded open items while
inspecting the numeric audit; it does not resolve them and cannot permit a
numeric or lineage mismatch. A newly generated comparison uses its embedded
`measurement` semantics instead of the retained artifact SHA. The
[function-table review](../extras/paper/FUNCTION_TABLE_REVIEW.md) records the
publication check.

The self-contained manuscript source is in `extras/paper/manuscript`. Its
allowlist contains 21 arXiv upload files and excludes raw evidence, internal
review archives, and proprietary fonts:

```bash
cd extras/paper/manuscript
sha256sum -c SOURCE_MANIFEST.sha256
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The source compiles with TeX Live 2023. Graphcore branding and the manuscript's
arXiv license remain rights-holder decisions; they are not granted by the
repository's Apache-2.0 software license.

## Result metadata

For a new result, preserve at least:

- repository commit and dirty-tree state;
- external component commits, including FlashAttention-4;
- GPU name, compute capability, memory, driver, CUDA, and PyTorch versions;
- clock and power policy when controlled;
- tensor/model shapes, dtype, seed, and model/tokenizer revisions;
- warm-up, repetitions, measurement order, and synchronization policy;
- all variants, raw samples, summary statistic, and numerical comparisons; and
- an explicit provenance class such as `new-measurement`,
  `diagnostic-unbound-source`, `historical-materialized`, or
  `historical-source-derived`.

[`schemas/result-v1.json`](../schemas/result-v1.json) defines the portable
envelope used by the component and standalone benchmark drivers. Fitting tools
and the TorchTitan exporter retain documented task-specific JSON layouts. For
an envelope-compatible result, experiment-specific fields belong under
`workload`, `protocol`, and each observation's `metrics`.

Validate one or more envelope-compatible results with the installed command:

```bash
sfu-validate results/components.json results/fa4-exp2.json
```

## Credentials, outputs, and publication

Do not commit tokens, credential files, model caches, checkpoints, dataset
copies, hostnames, scheduler configuration, or private storage locations.
Review JSON, CSV, logs, PDF metadata, and Git history before sharing an
artifact. Complete the [public-release checklist](PUBLIC_RELEASE_CHECKLIST.md)
before changing repository visibility.

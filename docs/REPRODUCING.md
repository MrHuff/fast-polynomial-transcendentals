# Reproducing the experiments

This guide distinguishes portable code execution, paper-aligned hardware
measurements, access-gated model evaluation, and offline inspection of
historical evidence. These are different reproduction levels.
The same workflow statuses are available to tools in
[`repro/experiments.json`](../repro/experiments.json).

## Reproduction levels

1. **Software check:** import `sfu_repro`, validate configuration, and run tests.
2. **Numerical reproduction:** re-fit a function or rebuild a packed kernel and
   compare its numerical output.
3. **Paper-aligned timing:** repeat the stated workload and measurement protocol
   on NVIDIA GB200/SM100 hardware.
4. **Model evaluation:** patch a supported, legally obtained open-weight model
   and run the stated quality or throughput protocol.
5. **Historical evidence audit:** verify and plot retained results whose exact
   original runtime is not fully reconstructable.

Levels 1--4 create new observations. Level 5 does not. Timing from a different
GPU, driver, clock policy, tensor shape, or software revision should be reported
as a new experiment.

## Clone and create the base environment

The review repository is private. Use an account that has been granted access,
and let Git or SSH handle authentication without embedding a token in the URL.
Install a PyTorch build appropriate for the intended CPU or CUDA platform
before running the full test suite; the project metadata deliberately does not
select a PyTorch wheel.

```bash
git clone --recurse-submodules git@github.com:MrHuff/fast-polynomial-transcendentals.git
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

For B2, B3, and the integrated fractional-`exp2` probe, install the patched
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

## Re-run a CPU fitting program

The fractional-`exp2` fit uses NumPy and writes a new JSON artifact:

```bash
mkdir -p results
python autonumerics_zero/evolution/fit_exp2_softmax.py \
  --samples 1000001 \
  --json-out results/exp2-fit.json
```

The BF16 same-form comparison additionally requires PyTorch and a local Sollya
installation:

```bash
python autonumerics_zero/spline_ops/generate_sollya_structs_bf16.py
```

Tool versions must accompany newly reported results. Dense sampled minimax
values are empirical grid estimates unless the producing program explicitly
emits a formal certificate.

## Build the packed CUDA extension

The paper-aligned extension target is SM100. From the repository root:

```bash
cd autonumerics_zero/spline_ops
SPLINE_OPS_CUDA_ARCH=sm_100 python -m pip install -v .
cd ../..
python -m pytest autonumerics_zero/spline_ops/test_spline_ops.py
```

This step requires a CUDA compiler compatible with the installed PyTorch build.
Compilation for another architecture is supported only as a new validation
target; it does not reproduce the reported GB200 timing.

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
addition to aggregate statistics.

## Run the FlashAttention-4 fractional-exp2 probe

The mixed schedule uses
`name=backend:forward_frequency:backward_frequency` variant syntax. Use the
driver's defaults to reproduce the retained workload, or spell out every
variant in the result record:

```bash
python scripts/benchmark_fa4_exp2_mix.py \
  --json-out results/fa4-exp2.json
```

```bash
python scripts/benchmark_fa4_exp2_mix.py --help
```

This path depends on behavior in the pinned patched FlashAttention-4 revision.
An unpatched upstream installation is not interchangeable.

## Run an open-weight evaluation

Two standalone entry points replace the former training-module integration:

```bash
python scripts/benchmark_open_weights.py --help
python scripts/run_open_weight_suite.py --help
```

The five paper cases and source-derived protocol are declarative in
`configs/open_weight_paper.json`. Preview a credential-free command without
loading a model:

```bash
python scripts/run_open_weight_suite.py \
  --models glm4p7_flash \
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
4.57.6 for Kimi, with additional packages listed per case in the config. These
declarative versions have not yet been published as tested lockfiles. Install a
PyTorch build appropriate for the local CUDA stack, then install the selected
profile rather than forcing all five cases into one aggregate environment.

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
  `historical-materialized`, or `source-derived`.

[`schemas/result-v1.json`](../schemas/result-v1.json) defines a portable common
envelope. Experiment-specific fields belong under `workload`, `protocol`, and
each observation's `metrics` rather than being silently omitted.

Validate one or more result envelopes with the installed command:

```bash
sfu-validate results/components.json results/fa4-exp2.json
```

## Credentials, outputs, and publication

Do not commit tokens, credential files, model caches, checkpoints, dataset
copies, hostnames, scheduler configuration, or private storage locations.
Review JSON, CSV, logs, PDF metadata, and Git history before sharing an
artifact. Complete the [public-release checklist](PUBLIC_RELEASE_CHECKLIST.md)
before changing repository visibility.

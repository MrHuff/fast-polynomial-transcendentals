# Fast Polynomial Transcendentals for LLMs

This repository is the standalone code and evidence companion to *Fast
Polynomial Transcendentals for LLMs*. It studies low-order polynomial programs
that replace selected special-function-unit (SFU) operations in large language
model (LLM) components.

> **Status:** private release candidate. The executable research code,
> third-party attributions, public TorchTitan bridge, and pinned downstream-
> evaluation protocol are present. Complete the remaining checks in
> `docs/PUBLIC_RELEASE_CHECKLIST.md` before changing repository visibility.

The repository deliberately does not depend on the original
`low-bits-training` module. The reusable activation, attention, TorchTitan, and
open-weight evaluation plumbing has been extracted into `sfu_repro`. The public
training path uses a pinned upstream TorchTitan submodule. Cluster launchers,
credentials, and organization-specific configuration remain outside this
repository.

Repository: <https://github.com/MrHuff/fast-polynomial-transcendentals>

## Scope

The paper evaluates four primary integration sites:

| Case | Integration site | Native comparison |
|---|---|---|
| B1 | Dense SiLU activation | Native SiLU |
| B2 | FlashAttention-4 attention softcap | SFU tanh |
| B3 | FlashAttention-4 sigmoid attention | SFU sigmoid |
| B4 | Routed-expert SwiGLU activation | Native expert SiLU |

The code also contains fitting and exploratory fractional-`exp2` and paired
sine/cosine work. B5 tests three routed-`exp2` schedules in FlashAttention-4,
including a public 80-step Llama-8B-shaped TorchTitan probe. Reported timings
were collected on NVIDIA GB200/SM100 hardware. A run on another GPU generation
is a new measurement, not a reproduction of the paper's timing result.

## What is reproducible here

| Workflow | Status |
|---|---|
| Import and unit-test the standalone configuration and patching layer | CPU |
| Re-run retained activation, GELU/ERF, tanh, fractional-`exp2`, and sine/cosine fitting programs | CPU; some comparisons require Sollya |
| Audit the exact deployed B3 constants and packed-BF16 arithmetic | CPU; the original B3 coefficient-selection program was not retained |
| Build and test the packed polynomial extension | CUDA toolchain and supported GPU |
| Re-run isolated B1--B4 and FlashAttention-4 probes | Patched FlashAttention-4, CUDA, and supported GPU |
| Re-run cache/HBM, repeated-evaluation, and fused-RoPE probes | Compiled extension, CUDA, and supported GPU |
| Re-run B1--B4 model probes and declared 100B-token recipes | Pinned TorchTitan, CUDA, model tokenizer, corpus access, and 1--32 supported GPUs |
| Re-run the B5 routed-`exp2` model probe | Pinned TorchTitan and FlashAttention-4, CUDA, tokenizer access, and one supported GPU |
| Run held-out validation for a new paired training run | Pinned TorchTitan and the selected validation split |
| Re-run and summarize supported downstream open-weight evaluations | Pinned lm-eval tasks, datasets, per-model environments, model/data access, CUDA, and supported GPU |
| Replot the retained paired pre-training histories | Offline from materialized evidence |
| Build the review manuscript and audit its released figures and numeric table cells | Offline paper extras; the accuracy plots additionally require the compiled CUDA extension and a supported GPU. Table 2 method and arithmetic semantics remain under review |
| Bit-for-bit repeat the historical 100B-token trajectories | Not possible from retained evidence |

The public TorchTitan recipes expose the model shapes, intervention code,
optimizer schedule, parallel layout, token horizon, seed-checkpoint workflow,
immutable public corpus and tokenizer selections, and optional validation path.
They are new, reviewable rerun protocols. The retained evidence does not
identify every historical seed, data order, checkpoint, software image, or B4
corpus choice, so those recipes do not claim bit-for-bit identity with the
historical trajectories.

The same distinction applies to downstream evaluation. The public protocol
pins five model/tokenizer revisions, lm-eval v0.4.12, eight task-dataset
revisions, evaluator seeds, few-shot counts, and per-model package profiles.
The historical paper table cannot be replayed exactly because its two Qwen
checkpoint revisions, task-dataset commits, sampled few-shot identities, raw
per-model result shards, and complete runtime were not retained.

See the [experiment matrix](docs/EXPERIMENT_MATRIX.md) for each kernel, fitter,
model workflow, output path, and historical boundary.

## Quick start

The repository is private during review, so cloning requires access granted by
the owner. No credential belongs in a command, configuration file, or result
artifact. The base package does not install PyTorch or Transformers. The
optional `test` extra includes both so a clean development environment can
collect and run the complete CPU test suite. To select a particular CPU or CUDA
PyTorch build, install that build first; the extra will reuse any compatible
installation.

```bash
git clone --recurse-submodules https://github.com/MrHuff/fast-polynomial-transcendentals.git
cd fast-polynomial-transcendentals
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[analysis,test]'
python -m pytest
```

The training bridge pins TorchTitan v0.2.2 at commit
`73a0e6979dd10b6b1904098eb3c8f62c18ab87ce`. Its release names a matching
PyTorch/TorchAO nightly pair; install that pair and the editable submodule
before using the training recipes:

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

The asset helper downloads tokenizer files only, at the commits declared in
`src/sfu_repro/torchtitan/pins.py`, and writes a local SHA-256 manifest. It reads
provider authentication through the standard Hugging Face environment or
credential store and never accepts a token on its command line.

Run the standalone component probe after installing its CUDA dependencies:

```bash
python scripts/benchmark_components.py \
  --cases b1,b2,b3,b4 \
  --json-out results/components.json
```

B3's paper configuration is defined only for sequence length 4096; the driver
rejects other lengths. Its exact deployed constants and arithmetic are
auditable, but its original coefficient-selection program was not retained.
See [the reproduction guide](docs/REPRODUCING.md) for the fitting,
extension-build, FlashAttention-4, TorchTitan training and validation,
open-weight downstream evaluation and aggregation, RoPE, and offline-figure
workflows.

## Repository map

- `src/sfu_repro/`: standalone activation, attention, result-validation, and
  TorchTitan adapter plumbing.
- `scripts/`: portable component, FlashAttention-4, open-weight model-patching,
  evaluation drivers, and a provenance-safe TorchTitan metrics exporter.
- `configs/`: reviewable experiment configuration without credentials.
- `torchtitan/`: upstream TorchTitan v0.2.2 submodule used by the public
  distributed training and validation recipes.
- `lm-evaluation-harness/`: upstream lm-eval v0.4.12 submodule used by the
  public downstream-quality protocol.
- `autonumerics_zero/`: selected fitting programs, packed CUDA kernels, and
  isolated benchmarks.
- `evidence/report-data/`: materialized paper evidence retained for audit.
- `evidence/pretraining/`: offline plotting code for retained trajectories.
- `extras/paper/`: arXiv-safe manuscript source plus offline figure and table
  generation/audit tools; manuscript and branding rights remain separate from
  the software license.
- `repro/experiments.json`: machine-readable workflow and status map.
- `schemas/result-v1.json`: metadata envelope used by source-bound benchmark
  drivers; fitter and TorchTitan exports retain documented task-specific JSON.
- `docs/EXPERIMENT_MATRIX.md`: per-experiment code, command, and replay status.
- `docs/PROVENANCE.md`: extraction and historical evidence map.

## External dependencies and access

The CUDA experiments require a compatible PyTorch/CUDA toolchain, the compiled
`spline_ops` extension, and the paper's patched FlashAttention-4 revision.
Install FA4 from `./flash-attention/flash_attn/cute` so the polynomial-manifest
and handwritten-path APIs come from the pinned checkout; a different upstream
`flash_attn` package is not interchangeable.
Open-weight experiments additionally require the relevant model, tokenizer,
evaluation datasets, and their Python runtimes. The public quality protocol
pins the lm-eval source, its eight built-in task definitions, evaluator seeds,
few-shot counts, and every Hugging Face dataset revision. This repository does
not redistribute model or dataset assets. Obtain each asset from its provider
and comply with its license, acceptable-use terms, and access controls.

The public training protocol pins an immutable SlimPajama community-reupload
snapshot for B1--B3 training and held-out validation, because the original
Cerebras Hub repository is no longer anonymously available. Byte identity with
the historical corpus and the historical shuffled sample order are not
established. B4 training uses a separately pinned OLMo Mix snapshot; its public
validation uses the pinned SlimPajama validation split.

Authentication must use the provider's normal secure credential mechanism.
Never commit access tokens, cached model credentials, service configuration,
or model files.

PyTorch is intentionally not selected by the base Python package because its
installation must match the target CUDA stack. Install the appropriate PyTorch
build before a component or model workflow, then use `sfu-doctor --profile
component` or `sfu-doctor --profile eval` to check optional dependencies.

## Evidence and claims

Materialized evidence is preserved separately from executable code. Historical
artifacts do not become newly generated results merely because they are stored
here. Each new benchmark should record the repository commit, dirty-tree state,
hardware and software versions, complete workload shape, warm-up, repetitions,
measurement order, and per-variant observations. Source-bound benchmark
drivers use the metadata shape in
[`schemas/result-v1.json`](schemas/result-v1.json). Fitter outputs and the
TorchTitan exporter use task-specific JSON because their native records do not
share the observation layout.

See [provenance](docs/PROVENANCE.md) for known source bindings and gaps. In
particular, some historical artifacts came from dirty worktrees or do not
retain their complete invocation. Claims should remain scoped to the evidence
class recorded there.

The table checker binds the packaged `main.tex` by SHA-256 and establishes
that every mapped Table 2 number comes from the retained comparison JSON at
the stated rounding. It does not establish that
both coefficient columns received the same BF16 treatment, that the values are
device errors, or that every current row has the method lineage named in the
manuscript. See the [function-table review](extras/paper/FUNCTION_TABLE_REVIEW.md).
The checker therefore reports `review-required` by default even when the
numeric cells match; an explicit acknowledgement retains those review items in
the output without resolving them.

## License and attribution

The repository's Apache-2.0 license and [NOTICE](NOTICE) cover the extracted
and modified code as stated in those files. The `autonumerics_zero` subtree
also retains its source license. TorchTitan and FlashAttention are
BSD-3-Clause; lm-evaluation-harness is MIT licensed. Python dependencies,
models, tokenizers, datasets, and evaluation tasks have separate terms; see
[third-party notices](THIRD_PARTY_NOTICES.md).

Public release remains contingent on the checks in
[`docs/PUBLIC_RELEASE_CHECKLIST.md`](docs/PUBLIC_RELEASE_CHECKLIST.md).

## Citation and contact

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

Correspondence: [robert.stats.hu@gmail.com](mailto:robert.stats.hu@gmail.com)

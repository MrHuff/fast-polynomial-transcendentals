# Fast Polynomial Transcendentals for LLMs

This repository is the standalone code and evidence companion to *Fast
Polynomial Transcendentals for LLMs*. It studies low-order polynomial programs
that replace selected special-function-unit (SFU) operations in large language
model (LLM) components.

> **Status:** private review and cleanup candidate. The repository is usable for
> review by authorized collaborators, but it has not completed the model/data,
> third-party-license, provenance, and privacy checks required for public
> release. Private visibility is not a substitute for those checks.

The repository deliberately does not depend on the original
`low-bits-training` module. The reusable activation, attention, and open-weight
evaluation plumbing has been extracted into `sfu_repro`. Cluster launchers,
training-framework adapters, credentials, and organization-specific
configuration are outside this repository.

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
sine/cosine work. Reported timings were collected on NVIDIA GB200/SM100
hardware. A run on another GPU generation is a new measurement, not a
reproduction of the paper's timing result.

## What is reproducible here

| Workflow | Status |
|---|---|
| Import and unit-test the standalone configuration and patching layer | CPU |
| Re-run selected numerical fitting programs | CPU; some comparisons require Sollya |
| Build and test the packed polynomial extension | CUDA toolchain and supported GPU |
| Re-run isolated B1--B4 and FlashAttention-4 probes | Patched FlashAttention-4, CUDA, and supported GPU |
| Re-run supported open-weight evaluations | Model/data access, evaluator dependencies, CUDA, and supported GPU |
| Replot the retained paired pre-training histories | Offline from materialized evidence |
| Exactly repeat the reported 100B-token pre-training trajectories | Not provided |

The retained pre-training CSV and provenance record support auditing and
offline plotting. This repository does not contain the original distributed
launcher, complete training configuration, datasets, checkpoints, or evidence
of identical initialization and data order for every arm. It therefore does
not present a full or exact pre-training rerun path.

## Quick start

The repository is private during review, so cloning requires access granted by
the owner. No credential belongs in a command, configuration file, or result
artifact. The package deliberately does not select a PyTorch wheel; install a
CPU or CUDA build appropriate for the target platform before running the full
test suite.

```bash
git clone --recurse-submodules git@github.com:MrHuff/fast-polynomial-transcendentals.git
cd fast-polynomial-transcendentals
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[analysis,test]'
python -m pytest
```

Run the standalone component probe after installing its CUDA dependencies:

```bash
python scripts/benchmark_components.py \
  --cases b1,b2,b3,b4 \
  --json-out results/components.json
```

B3's paper configuration is defined only for sequence length 4096; the driver
rejects other lengths. See [the reproduction guide](docs/REPRODUCING.md) for
the fitting, extension-build, FlashAttention-4, open-weight, and offline-figure
workflows.

## Repository map

- `src/sfu_repro/`: standalone activation, attention, result-validation, and
  diagnostic plumbing.
- `scripts/`: portable component, FlashAttention-4, open-weight model-patching,
  and evaluation drivers.
- `configs/`: reviewable experiment configuration without credentials.
- `autonumerics_zero/`: selected fitting programs, packed CUDA kernels, and
  isolated benchmarks.
- `evidence/report-data/`: materialized paper evidence retained for audit.
- `evidence/pretraining/`: offline plotting code for retained trajectories.
- `repro/experiments.json`: machine-readable workflow and status map.
- `schemas/result-v1.json`: common metadata envelope for new measurements.
- `docs/PROVENANCE.md`: extraction and historical evidence map.

## External dependencies and access

The CUDA experiments require a compatible PyTorch/CUDA toolchain, the compiled
`spline_ops` extension, and the paper's patched FlashAttention-4 revision.
Install FA4 from `./flash-attention/flash_attn/cute` so the polynomial-manifest
and handwritten-path APIs come from the pinned checkout; a different upstream
`flash_attn` package is not interchangeable.
Open-weight experiments additionally require the relevant model, tokenizer,
evaluation datasets, and their Python runtimes. This repository does not
redistribute those assets. Obtain each asset from its provider and comply with
its license, acceptable-use terms, and access controls.

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
measurement order, and per-variant observations. The recommended metadata
shape is defined in [`schemas/result-v1.json`](schemas/result-v1.json).

See [provenance](docs/PROVENANCE.md) for known source bindings and gaps. In
particular, some historical artifacts came from dirty worktrees or do not
retain their complete invocation. Claims should remain scoped to the evidence
class recorded there.

## License and attribution

The repository's Apache-2.0 license and [NOTICE](NOTICE) cover the extracted
and modified code as stated in those files. The `autonumerics_zero` subtree
also retains its source license. The FlashAttention submodule, Python
dependencies, models, tokenizers, datasets, and evaluation tasks have separate
terms that continue to apply.

Public release remains contingent on the checks in
[`docs/PUBLIC_RELEASE_CHECKLIST.md`](docs/PUBLIC_RELEASE_CHECKLIST.md).

## Citation and contact

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

Correspondence: [robert.stats.hu@gmail.com](mailto:robert.stats.hu@gmail.com)

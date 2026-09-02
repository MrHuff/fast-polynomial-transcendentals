# Experiment and reproduction matrix

This matrix is the shortest route from a paper result to its public code,
inputs, and provenance boundary. It distinguishes three properties that are
easy to conflate:

- **Kernel available** means the implementation used by the corresponding
  intervention is present in this repository or a pinned submodule.
- **Public rerun available** means a documented command can produce a new
  measurement once the stated public dependencies, assets, and hardware are
  supplied.
- **Historical replay** means the original source state, command, external
  inputs, runtime, and raw output are complete enough to repeat the reported
  run. Most model-scale experiments do not meet this stronger standard.

New output must record its own repository state, submodule revisions, runtime,
hardware, command, and input revisions. A public rerun does not retroactively
fill a gap in a historical artifact.

## Fitting and arithmetic

| Target | Public implementation | What can be rerun | Historical boundary |
|---|---|---|---|
| FP16 sigmoid, tanh, sigmoid derivative, tanh derivative, and SiLU derivative fits | `autonumerics_zero/evolution/fit_all_degrees.py` | The retained D3--D6 constrained least-squares sweep and FP16 Horner replay, with a caller-selected JSON output | The default path preserves historical behavior; the manifest writes a new output without altering checked-in evidence |
| BF16 same-form activation fits | `autonumerics_zero/evolution/fit_all_degrees_bf16.py`, `autonumerics_zero/evolution/generate_bf16_structs.py`, and `autonumerics_zero/spline_ops/generate_sollya_structs_bf16.py` | The retained search surrogate, a caller-directed candidate-header generation receipt, and the Sollya comparison | The candidate generator fits sigmoid/tanh/SiLU families and copies ERF/GELU from the named fallback header; promotion is a separate review step. The historical search rounds multiplication and addition separately, while packed device FMA has different rounding |
| ERF and GELU FP16 fits | `autonumerics_zero/evolution/fit_gelu_fp16.py` | The original D3--D6 fit/clamp sweep, with output redirected to a caller-selected JSON file | The deployed BF16 ERF/GELU structs were initialized from these FP16-derived coefficients; this does not make the fitter a BF16 instruction emulator |
| B2 softcap tanh | `autonumerics_zero/evolution/fit_fa4_tanh_backends.py` | Separate CuTe-FP32 and handwritten-device-BF16 fitting targets | A new fit is source-bound to its recorded arguments and environment; device accuracy still requires the compiled path |
| B3 direct sigmoid attention | `flash-attention/flash_attn/cute/polynomial_manifest.py`, `flash-attention/flash_attn/cute/handwritten_spline_ptx.py`, and `scripts/audit_b3_deployed_fit.py` | The exact deployed constants and an emulation of the packed-BF16 arithmetic can be audited on a declared CPU grid | **Blocked as a historical fit replay:** the program, objective, weighting, and sample set that selected the deployed D3/D4 coefficients were not retained |
| Fractional-`exp2` | `autonumerics_zero/evolution/fit_exp2_softmax.py` and `autonumerics_zero/evolution/fit_flash_sigmoid_exp2_d2.py` | Endpoint-constrained PWL2/D2 fitting and the sequence-length-specific D2 search | Dense sampled minimax values are empirical unless the producing program emits a formal certificate |
| Paired sine/cosine | `src/sfu_repro/rope/fit_polynomial_sincos.py` | Checked-in least-squares and optional Sollya fits, plus portable numerical checks | The local FP16-lattice search that produced the retained packed-kernel coefficients was not retained |

Run the safe, caller-directed fit and audit entry points from the repository
root:

```bash
mkdir -p results

python autonumerics_zero/evolution/fit_gelu_fp16.py \
  --json-out results/gelu-coefficients-fp16.json

python autonumerics_zero/evolution/fit_fa4_tanh_backends.py \
  --output results/fa4-tanh-backend-fits.json

python autonumerics_zero/evolution/fit_exp2_softmax.py \
  --samples 1000001 \
  --json-out results/exp2-fit.json

python scripts/audit_b3_deployed_fit.py \
  --grid-min -6 --grid-max 6 --grid-points 24577 \
  --json-out results/b3-deployed-fit-audit.json
```

The [B3 audit note](B3_DEPLOYED_FIT_AUDIT.md) defines the emulated arithmetic.
The [RoPE guide](../src/sfu_repro/rope/README.md) gives every sine/cosine fit,
numerical, SASS-audit, cache/HBM, repeated-evaluation, and fused-RoPE command.

## Kernels and isolated measurements

| Paper scope | Kernel source | Public rerun | Historical replay status |
|---|---|---|---|
| B1 dense SiLU | `autonumerics_zero/spline_ops/spline_kernels_bf16.cu` and generated coefficient headers | `python scripts/benchmark_components.py --cases b1` | New runs use the retained kernel boundary; the materialized historical component artifact records a dirty source tree whose full delta is unavailable |
| B2 FA4 softcap tanh | Pinned patched FlashAttention-4 plus `src/sfu_repro/fa4.py` | `python scripts/benchmark_components.py --cases b2` | Paper-aligned timing requires the recorded GB200/SM100 class, workload, and measurement protocol |
| B3 FA4 sigmoid attention | Pinned patched FlashAttention-4 manifest and handwritten PTX generator | `python scripts/benchmark_components.py --cases b3` at sequence length 4096 | The deployed kernel is present; its original coefficient-selection run is blocked as described above |
| B4 routed-expert SwiGLU | `autonumerics_zero/spline_ops/spline_kernels_bf16.cu` and generated coefficient headers | `python scripts/benchmark_components.py --cases b4` | The historical distributed probe used an uncommitted harness; retained aggregates and raw-log hashes do not restore that source delta |
| B5 fractional-`exp2` | Pinned patched FlashAttention-4 and `scripts/benchmark_fa4_exp2_mix.py` | Isolated and integrated CUDA reruns are available | Several retained JSON files lack a clean immutable source binding or complete invocation; each rerun is new evidence |
| Paired sine/cosine and fused RoPE | `autonumerics_zero/spline_ops/sincos_kernels.cu` | Cache-to-HBM, repeated evaluator, SASS audit, and fused Q/K rotation commands are public | The historical raw per-trial timing record and complete runtime manifest were not retained |

All CUDA timing requires a compatible PyTorch/CUDA toolchain. GB200/SM100 is
the paper-aligned target. Measurements on other hardware are useful new data,
but they do not repeat the reported timing environment.

## Model-scale training and validation

| Scope | Public protocol | Public inputs | Historical replay status |
|---|---|---|---|
| B1--B4 100-step probes | Pinned TorchTitan v0.2.2, `configs/torchtitan/paper_runs.json`, and `scripts/run_torchtitan.py` | Pinned tokenizer assets and the declared public dataset fixtures | New source-bound runs; the original complete invocations and clean runtime state were not retained for every case |
| B1--B4 100B-token training | Matched native/polynomial configs, one shared case seed checkpoint, fixed geometry, optimizer schedule, and token horizon | Pinned replacement corpus/tokenizer revisions | **Blocked as a bit-for-bit historical replay:** historical checkpoints, full data order, some dataset choices, initialization details, and the complete software image are missing |
| B1--B4 held-out validation | `--validation` on the public TorchTitan pretraining workflow | Pinned SlimPajama validation split | This is a new public protocol. Historical campaigns had validation disabled, so there is no historical validation result to replay |
| B5 80-step model probe | Three TorchTitan configs for native, PWL2-safe-FP16, and D2-safe routing | Public C4 test fixture, tokenizer, seed, and TorchTitan runtime | The shape, batch, sequence length, step count, and routes are retained. The compact historical artifacts omit the full command, resolved runtime, raw logs, source revision, dataset/tokenizer identity, seed, and initialization |

B5 defines an 80-step model probe only. No B5 long-run training, validation,
or downstream checkpoint protocol is claimed.

The training launcher previews by default and executes only with `--execute`.
The [reproduction guide](REPRODUCING.md#run-public-torchtitan-model-probes-training-and-validation)
documents seed-checkpoint creation, paired launches, B5 exports, validation,
and result preservation.

## Downstream open-weight evaluation

| Scope | Public protocol | Output handling | Historical replay status |
|---|---|---|---|
| Five models, eight quality tasks | `configs/open_weight_paper.json`, `configs/lm_eval_paper_tasks.json`, pinned lm-eval v0.4.12, and three environment-specific `scripts/run_open_weight_suite.py --mode eval --quality-eval` commands | Per-example sample logging, positive integral full-population `n-samples` counts, patch scope, and loaded-module attestations are required; preserve each raw JSON | **Blocked as an exact replay of the paper table:** historical Qwen checkpoint revisions, task-dataset commits, sampled few-shot identities, raw per-model result shards, and the complete runtime were not retained |
| Prefill and decode throughput | The same five pinned public model revisions and three per-model environment profiles | Raw timing JSON records every repetition, warm-up, shapes, source, runtime, and loaded-module binding | Public Qwen revisions are declared replacements; hardware and runtime determine a new measurement |
| Combined summary | `scripts/summarize_open_weight_results.py` | Validates content hashes for all selected configs, parent preflight status, complete invocation metadata, exact configured quality/throughput controls, and only the external sources used by each artifact (lm-eval for quality and FlashAttention for Kimi) | The summarizer consumes new public-run JSON only; TorchTitan is unrelated to this workflow, and the output does not reconstruct or certify the historical table |

An actual downstream run performs the pinned environment/source preflight
automatically. Run it directly when preparing an environment:

```bash
python scripts/check_open_weight_environment.py \
  --models glm4p7_flash --check-installed
```

After both the quality and throughput passes have completed for every selected
case and variant, summarize the raw result directory:

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

The summarizer rejects dirty, missing, or non-manifest source bindings by
default. It also rejects limited evaluations, skipped environment preflights,
changed configuration content, and measurement settings that differ from the
pinned public configuration. Throughput artifacts do not require lm-eval;
FlashAttention is required only for the Kimi profile that imports it.
`--allow-unbound-source` exists for diagnostic summaries and places the output
outside the declared paper-quality public protocol; it does not weaken the
measurement-protocol checks.

## Offline historical evidence

The retained paired pretraining CSV can be plotted without network, model
assets, or a GPU:

```bash
python evidence/pretraining/plot_paired_loss.py \
  --input evidence/report-data/b1_b4_paired_loss_curves.csv \
  --output results/b1_b4_paired_loss_curves.pdf
```

This reproduces the figure from the materialized samples. It does not rerun
training or turn checkpoints and minibatches from one trajectory into
independent repetitions. See [provenance](PROVENANCE.md) for the binding of
each retained artifact.

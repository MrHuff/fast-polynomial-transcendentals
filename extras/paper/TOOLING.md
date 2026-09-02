# Paper artifact tooling

These scripts reconstruct paper-facing artifacts from released source data.
They do not contact Weights & Biases, Hugging Face, or another remote service,
and they never write into `evidence/` by default.

## Method figures

The two conceptual method figures are deterministic CPU reconstructions from
the checked-in BF16 coefficient sweep:

```bash
python extras/paper/generate_method_figures.py \
  --output-dir outputs/paper/method
```

The output directory contains both figures and a receipt binding the generator,
coefficient input, parameters, and output hashes. `--coefficients`,
`--output-dir`, `--format`, and `--receipt` can all be selected explicitly.

## Accuracy figures

The six forward/backward accuracy figures replay the compiled BF16 CUDA
kernels at BF16-representable inputs. Build `autonumerics_zero/spline_ops`
first, then run on a CUDA GPU:

```bash
python extras/paper/generate_accuracy_figures.py \
  --extension-dir autonumerics_zero/spline_ops \
  --output-dir outputs/paper/accuracy
```

The receipt hashes the loaded extension binary and available CUDA/C++ sources.
This is a fresh measurement of the selected build, not proof that a binary was
built from those source files and not a bit-identical historical replay.

## Table 2 and Sollya coefficient controls

The numeric cells in manuscript Table 2 are bound to the retained
deployed-header comparison. The table checker reproduces every cell from that
JSON after the manuscript's scaling and rounding. This establishes numeric
lineage only; [FUNCTION_TABLE_REVIEW.md](FUNCTION_TABLE_REVIEW.md) records the
open semantic review.

The core generator can produce a new artifact with the same shape:

```bash
python autonumerics_zero/spline_ops/generate_sollya_structs_bf16.py \
  --current-header autonumerics_zero/spline_ops/spline_structs_odd_bf16.cuh \
  --header-out outputs/paper/spline_structs_sollya_bf16.generated.cuh \
  --json-out outputs/paper/sollya_device_bf16.generated.json
```

This is the source shape represented by the retained
`autonumerics_zero/cuda_benchmarks/analysis_results/sollya_device_bf16.json`
artifact and used by the table checker below. Its current side parses deployed
header literals as host floats without first applying BF16 rounding, while its
Sollya side explicitly BF16-rounds coefficients. It evaluates both sides with
host NumPy real arithmetic on a 20,001-point grid. It is not a packed-CUDA
device-error measurement, and its retained inputs do not establish
endpoint-constrained least-squares lineage for every selected current row.

### Auxiliary coefficient-sweep sensitivity control

`generate_sollya_comparison.py` is auxiliary. It starts from the D3--D6
coefficient-sweep JSON rather than the deployed headers:

With Sollya installed locally:

```bash
python extras/paper/generate_sollya_comparison.py \
  --output outputs/paper/sollya_comparison.json \
  --tex-output outputs/paper/sollya_comparison.tex
```

This recreates D3--D6 same-form controls at 20,001 points. It is not a Table 2
regenerator. Seven selected cells differ after manuscript rounding for four
specific reasons:

1. sigmoid D3 uses the earlier sweep fit and clamp 4.75 rather than the later
   deployed fit and clamp 6.0;
2. the later tanh D4 device refit used by Table 2 is absent from the sweep;
3. this tool's Sollya side directly fits the reduced SiLU residual, whereas
   the retained table composes both columns from the corresponding sigmoid
   rows; and
4. its GELU and GELU-derivative paths use the FP16 sweep and an 11-bit Sollya
   budget, whereas the retained table uses the deployed BF16 header and an
   8-bit/BF16 control.

Within the auxiliary calculation, degree, sparse monomials, clamp, precision
budget, and algebraic reconstruction are held fixed for each sweep row.

Those four workflow differences are distinct from the retained table's
semantic review: asymmetric BF16 handling of current and Sollya coefficients,
host NumPy rather than device error, and incomplete endpoint-constrained
least-squares lineage.

## Quantitative table audit

The checked-in claim map transcribes the rounded values in the packaged
manuscript and binds that `main.tex` by SHA-256. The checker derives the
function, component, complete-model, downstream, and paired-pretraining rows
from released evidence while ignoring historical run identifier columns:

```bash
python extras/paper/check_paper_tables.py \
  --function-lineage extras/paper/function_table_lineage.json \
  --output outputs/paper/table_audit.json
```

To audit a newly regenerated deployed-header comparison, select it explicitly:

```bash
python extras/paper/check_paper_tables.py \
  --function-comparison outputs/paper/sollya_device_bf16.generated.json \
  --function-lineage extras/paper/function_table_lineage.json \
  --output outputs/paper/table_audit.json
```

A non-retained comparison need not match the retained artifact SHA, but it must
carry the generator's equivalent top-level `measurement` object. Missing or
altered embedded semantics fail even when all numeric cells match.

Any value that changes after the manuscript's stated rounding is a mismatch.
The audit does not treat the claims map as evidence: each row must be recovered
from the named source artifact. The retained deployed-header comparison and
the other four released evidence groups currently pass numerically. Because
the validated lineage sidecar records unresolved Table 2 semantics, the
default command writes `review-required` and exits 2. This keeps a release
check from treating a numeric match as scientific approval. To inspect the
otherwise passing report while explicitly acknowledging those open items, add
`--allow-review-required`; the report keeps the review items and records the
acknowledgement. That flag never permits a numeric or lineage mismatch. A Table 2
numeric pass does not validate symmetric BF16 handling, device error, or the
endpoint-constrained least-squares column label.

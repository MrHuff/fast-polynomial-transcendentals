# Function-table lineage review

Status: the Table 2 numbers are reproducible from the retained JSON, but the
column semantics require manuscript review before publication.

`check_paper_tables.py` recovers every selected `current_max_error` and
`sollya_max_error` cell from
`autonumerics_zero/cuda_benchmarks/analysis_results/sollya_device_bf16.json`
and matches the manuscript after scaling by `1e3` and rounding to two decimal
places. This establishes source-to-typeset consistency only. It does not
establish device error, symmetric coefficient quantization, or the fitting
lineage named by the column headings.

`function_table_lineage.json` records the retained artifact hash, audited
generator snapshot, current-header hash, arithmetic model, and review items in
machine-readable form.

The table checker reports `review-required` and exits 2 by default while these
semantic items remain open. `--allow-review-required` records an explicit
acknowledgement so the remaining numeric tables can be checked; it does not
resolve the review.

## Why the auxiliary sweep differs

Seven auxiliary cells differ after manuscript rounding for four concrete
reasons:

1. **Sigmoid D3 selection and clamp.** Table 2 uses the later deployed fit with
   clamp 6.0. The auxiliary sweep uses the earlier coefficient row with clamp
   4.75.
2. **Tanh D4 device refit.** Table 2 reads the later device-refit row from the
   deployed header. That row is absent from the auxiliary sweep input.
3. **SiLU construction.** The retained table lineage composes both SiLU
   columns from the corresponding deployed sigmoid rows. The auxiliary
   program's Sollya side directly fits the reduced SiLU residual; its current
   side still composes the sweep's sigmoid coefficients.
4. **GELU precision.** The auxiliary GELU and GELU-derivative controls use the
   FP16 sweep and an 11-bit Sollya coefficient budget. The retained table uses
   the deployed BF16 header and an 8-bit/BF16 Sollya control.

These are differences between two coefficient-control workflows. They are
separate from the manuscript semantic review below.

## Manuscript semantic review

- **Asymmetric coefficient handling.** The retained producer parses current
  header literals as host floats and sends them directly to its error
  evaluator. It explicitly rounds the generated Sollya coefficients to BF16.
  The deployed header converts its current literals to BF16 at runtime, but
  that conversion is not applied to the current side of the retained JSON
  calculation.
- **Host arithmetic rather than device error.** The retained errors use a
  20,001-point NumPy grid and host real-valued Horner evaluation. They do not
  execute the packed BF16 CUDA program, replay intermediate device rounding,
  or measure the compiled kernel. The CUDA accuracy workflow is the relevant
  path for a new device-error measurement.
- **Incomplete method lineage.** The retained JSON and deployed header do not
  establish endpoint-constrained least-squares provenance for every selected
  current row. A numeric match cannot validate that method label.

The auxiliary generator is useful as a sensitivity analysis. It is not a
Table 2 regenerator and must not be used to replace the retained table cells.
Before publication, review the Table 2 caption and column labels against these
boundaries or regenerate a comparison with symmetric coefficient handling,
an explicitly named arithmetic model, and row-level fitter provenance.

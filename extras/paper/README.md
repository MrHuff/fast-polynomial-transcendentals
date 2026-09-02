# Paper extras

This directory contains the self-contained manuscript source and the offline
transformations used to regenerate or audit its figures and quantitative
tables. These tools do not fetch experiment histories, models, or datasets.
See [TOOLING.md](TOOLING.md) for the method, accuracy, Sollya, and table-audit
commands.

The quantitative table checker binds the packaged `main.tex` and reproduces
all mapped manuscript cells from the released evidence. For Table 2, that
result is a numeric transcription check, not approval of its method labels or
arithmetic description. Read
[FUNCTION_TABLE_REVIEW.md](FUNCTION_TABLE_REVIEW.md) before citing or changing
that table. The adjacent `function_table_lineage.json` is the machine-readable
lineage sidecar.

The arXiv-safe source set is under [manuscript](manuscript/README.md). It
includes the current manuscript, bibliography, appendices, style, logo, and
all required PDF figures, together with an upload allowlist and SHA-256
manifest. The repository's Apache-2.0 license covers the software in this
directory; manuscript text, branding, and figures require their own
publication and rights approval.

## Paired B1--B4 loss figure

`plot_paired_loss_curves.py` consumes a local materialized CSV and its local
provenance JSON. The input provenance must bind the CSV by SHA-256 and declare
the retained token-domain smoothing protocol. The script reads only `case`,
`role`, `curve_label`, `tokens_seen`, and `loss`; service metadata and other
historical identifiers are neither needed nor copied.

```bash
python extras/paper/plot_paired_loss_curves.py \
  --data-path evidence/report-data/b1_b4_paired_loss_curves.csv \
  --input-provenance evidence/report-data/b1_b4_paired_loss_curves.provenance.json \
  --output-path outputs/paper/b1_b4_paired_loss_curves.pdf \
  --output-provenance outputs/paper/b1_b4_paired_loss_curves.figure.json
```

All paths are caller-selected. The generated receipt binds both inputs, the
script, the output PDF, the plotting protocol, per-arm row counts, and common
token horizons. Its command record replaces path values with role placeholders
and copies no fields from the source provenance.

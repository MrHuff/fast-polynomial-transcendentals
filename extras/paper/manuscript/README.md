# Manuscript source

This directory contains the self-contained source package for *Fast Polynomial
Transcendentals for LLMs*. `main.tex` is the top-level manuscript. The source
uses PDFLaTeX and standard TeX Live packages; no proprietary fonts are bundled
or required.

## Build

Use TeX Live 2023 and `latexmk`:

```bash
cd extras/paper/manuscript
sha256sum -c SOURCE_MANIFEST.sha256
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The checked-in `main.bbl` allows the arXiv build to resolve references without
running BibTeX. To refresh it after changing `main.bib`, run `latexmk` in the
same directory and include the updated `main.bbl` in review.

## arXiv source set

`ARXIV_FILES.txt` is the upload allowlist. It intentionally excludes compiled
paper output, LaTeX scratch files, internal-review archives, experiment
credentials, raw job metadata, and local filesystem paths. `00README.json`
declares `main.tex` as the top-level source and selects TeX Live 2023.
The checked-in `main.pdf` is a reader copy and is not part of that upload
allowlist.

`SOURCE_MANIFEST.sha256` binds every file in the upload allowlist. Recompute it
only after reviewing an intentional manuscript or asset change.

The Graphcore report style and symbol are included because the manuscript uses
them directly. Their public use and the final arXiv license require approval by
the relevant rights holder; the repository's Apache-2.0 software license does
not by itself grant rights to the manuscript branding or third-party figures.

Before submission, inspect the complete generated PDF and confirm the author,
title, abstract, correspondence address, code URL, AI-use disclosure,
citations, figures, and selected arXiv license.

# Third-party notices

This repository is Apache-2.0 licensed except where a file or submodule states
otherwise. The following components retain their own licenses and terms.

## TorchTitan

- Project: <https://github.com/pytorch/torchtitan>
- Pinned revision: `73a0e6979dd10b6b1904098eb3c8f62c18ab87ce`
- Upstream tag: `v0.2.2`
- License: BSD-3-Clause
- Copyright: Meta Platforms, Inc. and affiliates
- License copies: [`torchtitan/LICENSE`](torchtitan/LICENSE) and
  [`licenses/TORCHTITAN-BSD-3-CLAUSE.txt`](licenses/TORCHTITAN-BSD-3-CLAUSE.txt)

The TorchTitan Git submodule is an independent work. Portions of
`src/sfu_repro/torchtitan/plugin.py` follow its public grouped-expert control
flow and retain the applicable notice. No endorsement by Meta or the PyTorch
project is implied.

## LM Evaluation Harness

- Project: <https://github.com/EleutherAI/lm-evaluation-harness>
- Pinned revision: `6d642546f4688648fced259eb3302efd36ece5af`
- Upstream tag: `v0.4.12`
- License: MIT
- Copyright: EleutherAI
- License copies: [`lm-evaluation-harness/LICENSE.md`](lm-evaluation-harness/LICENSE.md)
  and [`licenses/LM-EVALUATION-HARNESS-MIT.txt`](licenses/LM-EVALUATION-HARNESS-MIT.txt)

The LM Evaluation Harness Git submodule is an independent work. Its built-in
task definitions and evaluation implementation are executed at the pinned
revision for public downstream reruns.

## FlashAttention

- Project checkout: [`flash-attention`](flash-attention)
- Pinned revision: `38afdedda24b0bf26e6904d3bed7807c19a6906e`
- License: BSD-3-Clause; see [`flash-attention/LICENSE`](flash-attention/LICENSE)

## External models and datasets

Model weights, tokenizers, training corpora, and evaluation datasets are not
redistributed. Their licenses, access conditions, acceptable-use policies, and
attribution requirements remain with their providers. Configuration names do
not grant access or alter those terms.

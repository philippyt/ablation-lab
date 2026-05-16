---
id: norm.layernorm
level: component
title: LayerNorm
short: Center to zero mean, rescale to unit variance, then learn an affine.
related_concepts: []
sources:
  - "Ba, Kiros, Hinton, Layer Normalization (2016)"
  - "Original Transformer (Vaswani et al., 2017)"
  - "GPT-2, GPT-3 (Radford et al.)"
  - "BERT (Devlin et al., 2019)"
---

For each token's hidden vector `x ∈ R^d`, LayerNorm computes:

`y = γ * (x - mean(x)) / sqrt(var(x) + ε) + β`

It learns two parameters per channel: a scale `γ` and a shift `β`. The mean-subtraction step centers activations; `γ, β` let the model push activations off-center wherever useful.

Parameter count per norm: `2 * d_model`.

**Why it matters:** LayerNorm is the conservative default. It is well-understood, stable, and shipped in GPT-2, GPT-3, BERT, and the original transformer paper. It is slightly more expensive than RMSNorm (one extra mean-subtraction, twice as many parameters per norm). Switching to RMSNorm drops total params slightly and saves a small but compounding amount of compute per token.

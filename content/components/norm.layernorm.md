---
id: norm.layernorm
level: component
title: LayerNorm
short: Center to zero mean, rescale to unit variance, then learn an affine.
related_concepts: []
---

For each token's hidden vector `x ∈ R^d`, LayerNorm computes:

`y = γ * (x - mean(x)) / sqrt(var(x) + ε) + β`

It learns two parameters per channel: a scale `γ` and a shift `β`. The mean-subtraction step centers activations. The learned `γ` and `β` then let the model push activations off-center wherever useful.

Parameter count per norm is `2 * d_model`.

**Why it matters:** LayerNorm is the conservative default. It is well understood, numerically stable, and shipped in GPT-2, GPT-3, BERT, and the original transformer paper. It is slightly more expensive than RMSNorm because it does one extra mean-subtraction and holds twice as many parameters per norm. Switching to RMSNorm drops total params slightly and saves a small but compounding amount of compute per token at serving time.

---
id: norm.deepnorm
level: component
title: DeepNorm (post-norm with α-scaled residual)
short: Post-norm placement plus a tuned residual scale, for very deep nets.
related_concepts: []
sources:
  - "Wang et al., DeepNet: Scaling Transformers to 1,000 Layers (2022)"
  - "GLM-130B (Tsinghua, 2022)"
---

A post-norm variant designed to make very deep transformers (1,000 layers or more) trainable. The residual update is rescaled by a depth-dependent constant `α`:

`x_{l+1} = LayerNorm(α · x_l + Sublayer(x_l))`

with `α = (2N)^(1/4)` for an N-layer encoder. The scaling keeps gradient magnitudes bounded across depth so stacking does not blow up training.

Parameter count is the same as LayerNorm. DeepNorm is a recipe (placement plus scale), not a new norm primitive.

**Why it matters:** Most modern open weights are pre-norm. The norm sits before each sublayer and the residual is added after. Pre-norm is easier to train without tricks but has its own pathologies at extreme depth. DeepNorm was the recipe behind GLM-130B and other deep-stack experiments. Mainstream open weights stayed at pre-norm depths of roughly 32 to 80 layers where the trick is not needed. DeepNorm remains the proof-of-concept that extreme-depth transformers can work with the right normalization scheme.

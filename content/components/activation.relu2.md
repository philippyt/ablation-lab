---
id: activation.relu2
level: component
title: ReLU² (Squared ReLU)
short: ReLU(x)². Sparse and cheap. Favored for sparse MoE routers.
related_concepts: []
sources:
  - "So et al., Primer: Searching for Efficient Transformers (2021)"
  - "Used in some MoE experiments and follow-up activation studies"
---

ReLU² is exactly what it sounds like:

`ReLU²(x) = max(0, x)²`

Sparse (zero output for negative inputs), monotonic, and strictly cheaper to compute than SiLU or GELU. The squaring gives a steeper-than-linear slope for positive inputs, which sharpens contrast between the active channels.

**Why it matters:** Primer's neural architecture search picked ReLU² as a winning ingredient. It has shown up in MoE routers and in research on sparse, fast feed-forward layers. It is not standard in the big open weights. It does appear in efficiency-focused work because it composes well with sparsity. A token whose FFN gate is zero contributes nothing downstream, so downstream layers can skip computation in principle. In this lab the parameter cost is identical to other activations because the gated FFN has the same weight shapes regardless. The real value of ReLU² is compute-side and benchmark-sensitive.

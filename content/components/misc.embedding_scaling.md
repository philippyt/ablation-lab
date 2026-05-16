---
id: misc.embedding_scaling
level: component
title: Embedding Scaling (Gemma style)
short: Multiply token embeddings by sqrt(d_model) before the first block.
related_concepts: []
sources:
  - "Vaswani et al., Attention Is All You Need (2017): original `sqrt(d)` scaling"
  - "Gemma 1/2/3 (Google, 2024)"
---

The token embedding matrix is initialized at `O(1/sqrt(d_model))` scale (Xavier-style). The first attention layer then sees inputs whose scale is much smaller than the residual-stream activations later in the network. Scaling embeddings by `sqrt(d_model)` brings the input scale in line.

Gemma applies this as:

`x = tok_embed[ids] * sqrt(d_model)`

before adding any positional information.

Parameter count: 0. Embedding scaling is a forward-pass multiplier, not a learned op.

**Why it matters:** The original transformer paper specified `sqrt(d_model)` scaling for the same reason. Most modern Llama-lineage models drop it; Gemma kept it. In practice the difference is small and detectable mostly in early training dynamics. In this lab embedding scaling does not change cost-panel params; it is documented so that when you read a Gemma config and see the scaling factor in the forward, you know what it is and why.

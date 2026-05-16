---
id: misc.logit_soft_cap
level: component
title: Logit Soft-Capping (Gemma 2)
short: Squash logits through tanh at the end to bound their magnitude.
related_concepts: []
sources:
  - "Gemma 2 (Google, 2024)"
  - "Also applied to attention logits inside the block in Gemma 2"
---

Before the final softmax over the vocabulary, Gemma 2 applies:

`logits = soft_cap * tanh(logits / soft_cap)`

with `soft_cap = 30.0` for output logits and `soft_cap = 50.0` for attention scores. The transform is the identity near zero but smoothly clips magnitude to `± soft_cap`.

Parameter count: 0. The soft cap is a forward-pass nonlinearity.

**Why it matters:** Unbounded logits can cause training instability and inference issues like extreme softmax saturation. Soft-capping is a mild, differentiable alternative to hard clipping that Gemma 2 found valuable. Most other open weights skip it; if your training is stable enough, it adds a small inference cost (one tanh) for safety you may not need. The cost panel does not register soft-cap; the value is stability, not memory.

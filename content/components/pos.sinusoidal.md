---
id: pos.sinusoidal
level: component
title: Sinusoidal Position Embeddings
short: Fixed sine/cosine pattern added to token embeddings.
related_concepts: []
sources:
  - "Vaswani et al., Attention Is All You Need (2017)"
  - "Original transformer encoder/decoder, T5 (early)"
---

The original transformer's position scheme. Each position gets a fixed vector of paired sines and cosines at geometric frequencies:

`PE[m, 2i]   = sin(m / 10000^(2i / d_model))`
`PE[m, 2i+1] = cos(m / 10000^(2i / d_model))`

These vectors are added to the token embedding. No learned position parameters.

Parameter count: 0. The tables are deterministic.

**Why it matters:** Sinusoidal positions were the elegant first try and a stepping stone to RoPE. The intuition is that paired sin and cos at geometric frequencies give each position a unique fingerprint. That intuition carried over directly to RoPE. RoFormer's improvement was to apply the rotation inside the dot product rather than as an additive bias, which gives a true relative-position property. Modern open weights skipped sinusoidal in favor of either learned absolute (GPT-2 and GPT-3) or RoPE (everyone else). Sinusoidal is zero-parameter like RoPE, but without RoPE's relative-position guarantee.

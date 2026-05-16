---
id: pos.rope
level: component
title: Rotary Position Embedding (RoPE)
short: Rotate Q and K by a position-dependent angle, no learned position params.
related_concepts: [kv_cache]
sources:
  - "Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)"
  - "Llama 1/2/3, Mistral, Mixtral, Qwen 2/2.5, Gemma, DeepSeek-V2/V3"
  - "Llama 3 raised base from 10,000 to 500,000 for long context; Qwen uses 1,000,000"
---

Treat each pair of channels in Q and K as a 2D point. Rotate that point by an angle `m · θ_i`, where `m` is the token position and `θ_i = base^(-2i / head_dim)` indexes the channel pair. The dot product Q·K then depends only on the *relative* offset between two positions, even though the rotation was applied with absolute positions.

For a query at position `m` and a key at position `n`, the dot product reduces to a function of `(m - n)`. This relative-position property is why RoPE generalizes (with the right base) to contexts longer than seen at training time.

Parameter count: 0. RoPE has no learned position embeddings; the cos/sin tables are deterministic from `base` and `head_dim`. The `base` (or `theta`) is the main lever: 10,000 was the original RoFormer value; Llama 3 uses 500,000 and Qwen uses 1,000,000 to support longer contexts without retraining.

**Why it matters:** Almost every modern open weight uses RoPE (Llama, Mistral, Qwen, Gemma, DeepSeek). It costs zero parameters, extrapolates better than absolute embeddings, and is what makes context-length extension recipes like YaRN and LongRoPE possible. In this lab the cost panel does not change when you pick RoPE because it adds no params; the value is that everything else in the cost panel still works.

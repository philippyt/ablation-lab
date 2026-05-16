---
id: norm.rmsnorm
level: component
title: RMSNorm
short: LayerNorm without the mean-subtraction or bias. Just a rescale.
related_concepts: []
sources:
  - "Zhang, Sennrich, Root Mean Square Layer Normalization (2019)"
  - "Llama 1/2/3 (Meta)"
  - "Mistral, Mixtral (Mistral)"
  - "Qwen 2, Qwen 2.5 (Alibaba)"
  - "Gemma 1/2/3 (Google)"
---

Strips the mean-subtraction and bias from LayerNorm. The formula becomes:

`y = γ * x / sqrt(mean(x²) + ε)`

It learns one parameter per channel (`γ`) instead of two. Parameter count per norm: `d_model`.

Empirically, the mean-subtraction step in LayerNorm contributes little; what matters is the rescaling. RMSNorm preserves the rescaling and drops the rest, leaving a simpler op.

**Why it matters:** Llama, Mistral, Qwen, and Gemma all use RMSNorm because the simplification compounds: half the params per norm, one fewer reduction per token, and no measured quality loss at scale. Swapping LayerNorm to RMSNorm in this lab drops total params slightly; the bigger effect (faster compute) does not show in the cost panel but matters in real serving.

---
id: activation.silu
level: component
title: SiLU / Swish (used inside SwiGLU)
short: x · sigmoid(x). Smooth, non-monotonic, the gate in SwiGLU.
related_concepts: []
sources:
  - "Hendrycks, Gimpel, Gaussian Error Linear Units (2016)"
  - "Ramachandran et al., Searching for Activation Functions (2017)"
  - "Llama, Mistral, Qwen, Gemma all use SiLU inside SwiGLU"
---

SiLU (also called Swish) is:

`SiLU(x) = x · sigmoid(x)`

It is smooth, near-linear for large positive `x`, and slightly non-monotonic for moderately negative `x` (small dip below zero, then recovers). In SwiGLU it is applied to the gate branch: `SiLU(w_gate(x)) * w_up(x)`.

**Why it matters:** SiLU paired with the GLU gating structure is the activation that ships in almost every modern open weight (Llama, Mistral, Qwen, Gemma). The original transformer used ReLU. BERT and GPT-2 used GELU. PaLM and Llama settled on SiLU inside SwiGLU. The differences in raw quality between SiLU and GELU are small at scale. SwiGLU's gating mechanism is what matters more than which nonlinearity you pick inside it. In this lab the parameter count does not differ across activations because the gated FFN has the same weight shapes regardless. Runtime differences are also small at this size.

---
id: activation.gelu
level: component
title: GELU
short: x · Φ(x), a smooth ReLU-like activation. Popular pre-SwiGLU.
related_concepts: []
sources:
  - "Hendrycks, Gimpel, Gaussian Error Linear Units (2016)"
  - "BERT (Devlin et al., 2019)"
  - "GPT-2, GPT-3 (Radford et al., 2019/2020)"
  - "T5 1.1 uses GeGLU (GLU with GELU gate)"
---

GELU is `x · Φ(x)` where `Φ` is the Gaussian CDF, often approximated as:

`GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`

Smooth, near-linear for large `x`, smoothly suppresses moderately negative `x`. Conceptually similar to SiLU but with slightly different shape.

**Why it matters:** GELU was the de facto activation for BERT-era and GPT-2/3-era transformers. It still ships in many models (encoder transformers, smaller decoders, T5 1.1's GeGLU variant). The shift to SwiGLU was driven by the GLU gating structure rather than dissatisfaction with GELU itself: a GeGLU layer (GLU with GELU gate) is also a strong choice, used in T5 1.1 and earlier PaLM experiments.

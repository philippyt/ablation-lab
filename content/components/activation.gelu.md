---
id: activation.gelu
level: component
title: GELU
short: x · Φ(x). A smooth, ReLU-like activation. Popular before SwiGLU.
related_concepts: []
---

GELU is `x · Φ(x)` where `Φ` is the Gaussian CDF. It is often approximated as:

`GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`

Smooth, near-linear for large positive `x`, and it smoothly suppresses moderately negative `x` instead of clipping at zero. Conceptually similar to SiLU but with a slightly different shape near the origin.

**Why it matters:** GELU was the de facto activation for BERT-era and GPT-2 and GPT-3-era transformers. It still ships in many models including encoder transformers, smaller decoders, and T5 1.1's GeGLU variant. The shift to SwiGLU was driven by the GLU gating structure rather than any dissatisfaction with GELU itself. A GeGLU layer (a GLU with a GELU gate) is also a strong choice and was used in T5 1.1 and earlier PaLM experiments.

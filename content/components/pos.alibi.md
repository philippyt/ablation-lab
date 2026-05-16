---
id: pos.alibi
level: component
title: ALiBi (Attention with Linear Biases)
short: Add a fixed per-head linear bias to attention scores. No learned params.
related_concepts: []
sources:
  - "Press, Smith, Lewis, Train Short, Test Long: ALiBi (2022)"
  - "BLOOM (BigScience, 2022), MPT 7B/30B (MosaicML)"
  - "Falcon-Mamba uses ALiBi-style biases"
---

Instead of modifying Q/K, ALiBi adds a static, position-dependent bias to the attention logits before the softmax:

`scores[i, j] += -m_h * |i - j|`

where `m_h` is a per-head slope (a fixed geometric sequence). Closer tokens get a smaller penalty; far-away tokens get a large negative bias and are effectively ignored.

Parameter count: 0. The slopes `m_h` are baked in by formula. ALiBi extrapolates to longer-than-training contexts naturally because the bias is defined for any offset.

**Why it matters:** BLOOM and MPT shipped ALiBi because it gives strong length extrapolation with no extra params and a simpler implementation than RoPE. The trade-off, observed in later studies, is somewhat weaker in-distribution quality versus RoPE at matched compute, which is why the field largely settled on RoPE plus a tuned base. In this lab, ALiBi adds zero params and is honest-untrained: pick it and you see the "no checkpoint" banner.

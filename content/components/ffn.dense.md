---
id: ffn.dense
level: component
title: Dense FFN (SwiGLU)
short: One feed-forward per block. Every parameter runs for every token.
related_concepts: [active_vs_total_params]
sources:
  - "Shazeer, GLU Variants Improve Transformer (2020)"
  - "PaLM (Chowdhery et al., 2022)"
  - "Llama 1/2/3 (Meta), Mistral, Qwen 2, Gemma"
---

Each block has a feed-forward network after attention. The variant used here is SwiGLU (Shazeer 2020):

`FFN(x) = w_down(SiLU(w_gate(x)) * w_up(x))`

with `w_gate, w_up: R^d → R^h` and `w_down: R^h → R^d`. The hidden dim `h` is set to `round_up_64(2/3 * ffn_mult * d_model)`. SwiGLU's gating roughly doubles the param count of a vanilla ReLU MLP at matched hidden size, so the `2/3` factor compensates to keep the param count comparable.

In a dense FFN every parameter participates in every forward pass: `active_params_per_token == total_params`.

**Why it matters:** Dense SwiGLU is the workhorse of modern open weights (Llama, Mistral, Qwen, Gemma). Most of the parameter budget of a transformer lives in the FFN. When you swap dense to MoE, total params jump while active params per token stays close to one FFN's worth. That gap is the MoE proposition, and dense is the baseline it argues against.

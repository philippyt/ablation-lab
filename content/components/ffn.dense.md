---
id: ffn.dense
level: component
title: Dense FFN
short: One feed-forward per block. Every parameter runs for every token.
related_concepts: [active_vs_total_params]
---

Each block has a feed-forward network after attention. The shape is gated:

`FFN(x) = w_down(act(w_gate(x)) * w_up(x))`

with `w_gate, w_up: R^d → R^h` and `w_down: R^h → R^d`. The activation in the middle is whatever you pick on the Activation row (SiLU, GELU, ReLU²). The hidden dim `h` is set to `round_up_64(2/3 * ffn_mult * d_model)`. The `2/3` factor compensates for the extra `w_gate` matrix so total params stay comparable to a vanilla two-matrix MLP at matched hidden size.

In a dense FFN every parameter participates in every forward pass, so `active_params_per_token == total_params`.

**Why it matters:** A dense gated FFN is the workhorse of modern open weights (Llama, Mistral, Qwen, Gemma). Most of the parameter budget of a transformer lives here. When you swap dense to MoE, total params jump while active params per token stay close to one FFN's worth. That gap is the MoE proposition, and dense is the baseline it argues against.

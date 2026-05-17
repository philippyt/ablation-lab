---
id: ffn.moe
level: component
title: Mixture-of-Experts (MoE)
short: N parallel FFNs. Each token routes through top-k of them.
related_concepts: [active_vs_total_params]
---

Instead of one FFN, MoE has `n_experts` parallel "expert" FFNs and a small router. For each token:

1. Router computes scores: `g = softmax(W_g · x)` with `W_g ∈ R^{d × n_experts}`.
2. Top-k experts are selected: `S_k = topk(g, k)`.
3. Token output is the weighted sum: `y = Σ_{i ∈ S_k} g_i * Expert_i(x)`.

Total params grow linearly with `n_experts`. [Active params per token](concept:moe_total_vs_active) stay close to a single expert's cost plus the router, because only `top_k` experts run per token. Mixtral runs top-2 of 8 experts. DeepSeek-V3 runs top-8 of 256 routed experts plus 1 shared, landing at 671B total parameters and 37B active per token.

**Why it matters:** MoE scales total capacity without proportionally scaling compute per token. Mixtral, DeepSeek-V2 and V3, Qwen 2 MoE, and OLMoE all bet on this shape. The catch is that experts need enough data to specialize. On this lab's 1.1 MB corpus the router collapses to a single expert and the quality gain does not show up. The cost shape (total parameters much greater than active parameters) is real and visible in the panel. The quality story requires much more data than we have here.

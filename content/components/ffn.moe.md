---
id: ffn.moe
level: component
title: Mixture-of-Experts (MoE)
short: N parallel FFNs. Each token routes through top-k of them.
related_concepts: [active_vs_total_params]
sources:
  - "Shazeer et al., Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer (2017)"
  - "Switch Transformer (Fedus et al., 2021)"
  - "Mixtral 8x7B / 8x22B (Mistral)"
  - "DeepSeek-V2, DeepSeek-V3 (DeepSeek)"
  - "Qwen 2 MoE, OLMoE (AI2)"
---

Instead of one FFN, MoE has `n_experts` parallel "expert" FFNs and a small router. For each token:

1. Router computes scores: `g = softmax(W_g · x)` with `W_g ∈ R^{d × n_experts}`.
2. Top-k experts are selected: `S_k = topk(g, k)`.
3. Token output is the weighted sum: `y = Σ_{i ∈ S_k} g_i * Expert_i(x)`.

Total params grow with `n_experts`. [Active params per token](concept:moe_total_vs_active) stay close to a single expert's cost plus the router, because only `top_k` experts run per token. Mixtral runs top-2 of 8; DeepSeek-V3 runs top-8 of 256 routed experts (plus 1 shared) at 671B total / 37B active.

**Why it matters:** MoE scales total capacity without proportionally scaling compute per token. Mixtral, DeepSeek-V2/V3, Qwen 2 MoE, and OLMoE all bet on this shape. The catch is that experts need enough data to specialize; on this lab's 1.1 MB corpus the router collapses to one expert and the quality gain does not show. The cost shape (total much greater than active) is real and visible in the panel; the quality story requires much more data than we have here.

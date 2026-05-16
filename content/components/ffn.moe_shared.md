---
id: ffn.moe_shared
level: component
title: MoE with Shared Experts (DeepSeek style)
short: Some experts run for every token; the rest are routed.
related_concepts: [active_vs_total_params]
sources:
  - "Dai et al., DeepSeekMoE: Towards Ultimate Expert Specialization (2024)"
  - "DeepSeek-V2 / V3 (DeepSeek-AI, 2024)"
  - "Qwen 2 MoE has a related shared-expert design"
---

A variant of MoE used by DeepSeek. The FFN layer is split into two pools:

1. A small number of *shared* experts that always run for every token (typically 1–2).
2. A large pool of *routed* experts; the router picks top-k per token.

Token output is `Σ(shared experts) + Σ_{i ∈ topk} g_i · routed_i(x)`. The shared experts capture common, broadly-useful transformations the router would otherwise have to re-discover in every routed expert. The fine-grained routed pool can then specialize narrowly.

DeepSeek-V3 uses 1 shared + 256 routed experts, top-8 routed, at 671B total / 37B active. The shared-expert idea also lets DeepSeek-V3 drop the auxiliary load-balance loss by replacing it with a bias-update routing scheme that is less hostile to optimization.

**Why it matters:** Shared-expert MoE is part of why DeepSeek-V3 ships competitive quality at a much lower active-param count than dense Llama-class models. The cost panel here would split into "always-active params" (the shared experts) and "topk-active routed params". In this lab it is not yet implemented; the explainer documents the cost shape (more common compute, deeper specialization) so the conceptual map matches what real frontier models do.

---
id: ffn.moe_shared
level: component
title: MoE with Shared Experts (DeepSeek style)
short: Some experts run for every token. The rest are routed.
related_concepts: [active_vs_total_params]
sources:
  - "Dai et al., DeepSeekMoE: Towards Ultimate Expert Specialization (2024)"
  - "DeepSeek-V2 / V3 (DeepSeek-AI, 2024)"
  - "Qwen 2 MoE has a related shared-expert design"
---

A variant of MoE used by DeepSeek. The FFN layer is split into two pools:

1. A small number of *shared* experts that always run for every token (typically 1 or 2).
2. A large pool of *routed* experts. The router picks top-k per token.

Token output is `Σ(shared experts) + Σ_{i ∈ topk} g_i · routed_i(x)`. The shared experts capture common, broadly useful transformations the router would otherwise have to re-discover inside every routed expert. The fine-grained routed pool can then specialize narrowly without burning capacity on shared baseline behavior.

DeepSeek-V3 uses 1 shared expert and 256 routed experts with top-8 routing, landing at 671B total parameters and 37B active per token. The shared-expert idea also lets DeepSeek-V3 drop the auxiliary load-balance loss. It replaces that loss with a bias-update routing scheme that is less hostile to optimization.

**Why it matters:** Shared-expert MoE is part of why DeepSeek-V3 ships competitive quality at a much lower active-parameter count than dense Llama-class models. A faithful cost panel for this design splits into "always-active params" (the shared experts plus attention) and "top-k active routed params". This shape is not yet implemented in the lab. The explainer documents the cost shape (more common compute, deeper specialization) so the conceptual map matches what real frontier models do.

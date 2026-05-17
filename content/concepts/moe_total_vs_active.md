---
id: moe_total_vs_active
level: concept
title: Total vs Active Parameters (MoE)
short: MoE decouples model capacity from per-token compute.
---

For a dense FFN, every parameter participates in every forward pass:

`active_params_per_token == total_params`

For an MoE FFN with `n_experts` experts and top-k routing, a token flows through only `k` experts plus the router. The active count drops:

`active_per_token ≈ k · expert_size + router_size + (everything outside the FFN block)`

Total params still include all `n_experts` experts. Mixtral 8x7B has 47B total, 13B active per token (top-2 of 8). DeepSeek-V3 has 671B total, 37B active per token (top-8 of 256 routed plus 1 shared).

The pitch: capacity (total params, which sets the upper bound on what the model can represent) grows independently of per-token compute (active params, which sets the per-token latency and FLOPs). The catch: experts need enough data to specialize, the router can collapse to a single expert if poorly trained, and load-balance loss or bias-update tricks are needed to keep expert utilization roughly equal.

At small data scale (this lab's 1.1 MB corpus, for instance), the router collapses and the cost shape is accurate but the quality benefit does not materialize.

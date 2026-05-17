---
id: attention.mqa
level: component
title: Multi-Query Attention (MQA)
short: All Q heads share a single K and a single V projection.
related_concepts: [kv_cache, memory_bound_inference]
---

GQA pushed to the limit: `n_kv_heads = 1`. Every Q head reads from the same K and V projection.

The [KV cache](concept:kv_cache) is the smallest it can be while still being per-layer, per-position. For `n_heads = 8` this is exactly 8x smaller than MHA at the same context. The cost is some modelling capacity, because all heads now project into the same key and value subspace.

**Why it matters:** MQA shines when the KV cache dominates everything else. Very long contexts, batched serving, smaller GPUs. PaLM and Falcon shipped MQA at scale and it worked well enough, though most newer models prefer GQA's middle ground. In the cost panel here, "KV cache @ full ctx" drops by the full `n_heads` factor when you pick MQA. That drop is the point.

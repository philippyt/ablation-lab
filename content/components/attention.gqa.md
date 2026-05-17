---
id: attention.gqa
level: component
title: Grouped-Query Attention (GQA)
short: Query heads grouped. Each group shares one K and one V projection.
related_concepts: [kv_cache, memory_bound_inference]
---

A compromise between MHA (no sharing) and MQA (full sharing). There are still `n_heads` query heads, but only `n_kv_heads` K and V heads, with `n_heads / n_kv_heads` queries sharing each K, V pair. At forward time the K and V tensors are repeated to match the Q heads via `repeat_interleave`.

[KV cache](concept:kv_cache) bytes scale with `n_kv_heads`, not `n_heads`:

`kv_cache_bytes = 2 * n_layers * n_kv_heads * head_dim * ctx * dtype_bytes`

With `n_heads = 8, n_kv_heads = 2`, the KV cache is exactly 4x smaller than MHA at the same context. Modelling capacity stays close to MHA in practice.

**Why it matters:** GQA is the default for most modern open weights from 7B upward (Llama 2 70B, Llama 3 family, Mistral, Mixtral, Qwen 2, Gemma 2). Swap MHA to GQA in this lab and the cost panel's "KV cache" row drops by the head-share ratio. That drop is the entire reason GQA exists.

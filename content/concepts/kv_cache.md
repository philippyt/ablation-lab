---
id: kv_cache
level: concept
title: KV Cache
short: What gets stored per past position so the next decode step does not redo work.
---

At every layer, the attention sublayer computes a Query, Key, and Value for each token. To score the current query against all previous tokens, the past K and V tensors must be available. Recomputing them on every step is wasteful; storing them once and reusing them turns each decode step into O(T) per layer instead of O(T²).

The stored object is the **KV cache**. Its size is deterministic:

`kv_cache_bytes = 2 · n_layers · n_kv_heads · head_dim · ctx · dtype_bytes`

The `2` is for K and V; one entry per layer, per head, per past position, per dtype byte. The cache is appended to on each decode step, evicted only by sequence end or context-window rollover.

**Why it matters for architecture choices.** Three of the most common attention-shape decisions exist almost entirely to shrink this number: GQA reduces `n_kv_heads` directly, MQA pushes it to 1, MLA stores a low-rank latent instead of full K and V. At long context the KV cache routinely exceeds the weight memory itself, so its growth rate dominates serving cost.

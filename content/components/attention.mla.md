---
id: attention.mla
level: component
title: Multi-head Latent Attention (MLA)
short: Compress K and V into a low-rank latent, decompress only when needed.
related_concepts: [kv_cache, memory_bound_inference]
---

Instead of caching full K and V tensors per head, MLA caches a single shared low-rank latent `c_KV ∈ R^{r}` per token, with `r` much smaller than `n_heads * head_dim`. K and V are reconstructed on the fly via two up-projection matrices:

`K = W_UK · c_KV,  V = W_UV · c_KV`

DeepSeek-V2 reports the KV cache becoming around 93% smaller than MHA at matched modelling capacity. The trick that makes it work without a quality hit is decoupling the RoPE-rotated part of Q and K from the compressed latent. A small set of "rope" head dimensions sit outside the latent compression so the relative-position property is preserved.

Parameter count: more matrices than standard attention (one down-project plus two up-projects), but the cached state per token is small.

**Why it matters:** MLA is the most aggressive KV-cache compression shipping in a frontier-class open model. DeepSeek-V2 and V3 use it. Kimi K2 picked it up too. It is harder to implement than GQA, but the KV-cache shrink is dramatic. In this lab MLA is implemented in simplified form. The cost panel's KV row drops substantially compared to MHA and stays smaller than GQA, in exchange for two extra matmuls per token to decompress the latent into per-head K and V.

---
id: attention.mha
level: component
title: Multi-Head Attention (MHA)
short: Each head holds its own Q, K, V projection. No sharing.
related_concepts: [kv_cache]
---

Standard transformer attention. With `n_heads = 8` and `head_dim = 16`, every head holds its own learned Q, K, V projection. That gives 8 separate K tensors and 8 separate V tensors per token, per layer.

The [KV cache](concept:kv_cache) stores K and V for each past position so future tokens can attend without recomputing them. Its size grows linearly in `n_heads`:

`kv_cache_bytes = 2 * n_layers * n_heads * head_dim * ctx * dtype_bytes`

At long context on a real model, this number routinely exceeds the weight memory.

**Why it matters:** MHA matches the original transformer paper and is what GPT-2, GPT-3, and Llama 1 use at all sizes. Its cost is the KV cache, which scales with `n_heads`. Modern open weights (Llama 2 70B onward, Mistral, Qwen 2, Gemma) switched to GQA because MHA's KV cache became the serving bottleneck at long context.

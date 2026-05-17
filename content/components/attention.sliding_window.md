---
id: attention.sliding_window
level: component
title: Sliding-Window Attention (SWA)
short: Each token attends only to the last W tokens, not the full history.
related_concepts: [kv_cache, memory_bound_inference]
sources:
  - "Beltagy et al., Longformer (2020)"
  - "Mistral 7B (Jiang et al., 2023), Mistral SWA = 4096"
  - "Gemma 2 alternates SWA and full-attention layers"
  - "Phi-3-small uses blocksparse + SWA"
---

Replace the full causal mask with a local one of width `W`: token at position `i` attends only to positions in `[i - W + 1, i]`. The attention pattern slides as you decode.

KV-cache size is bounded by `W`, not by `ctx`. For long generation `ctx >> W`, this is a large win:

`kv_cache_bytes = 2 * n_layers * n_kv_heads * head_dim * min(ctx, W) * dtype_bytes`

Stacking many sliding-window layers gives an effective receptive field of `n_layers * W`. Information can still flow long-range through depth, just not within a single layer.

**Why it matters:** Mistral 7B shipped pure SWA with `W = 4096`. Gemma 2 alternates SWA and full attention every other layer. The "global" layers carry long-range information and the "local" layers are cheap. At long context, SWA layers are the cheapest attention shape that still has any local-context model. In this lab SWA is not yet implemented. Conceptually its KV row would saturate at `W` instead of growing with `ctx`.

---
id: memory_bound_inference
level: concept
title: Why Decode Is Memory Bound
short: At batch=1 each step reads the full model and writes one token. Bandwidth is the limit, not FLOPs.
sources:
  - "Pope et al., Efficiently Scaling Transformer Inference (2022)"
  - "Roofline model (Williams, Waterman, Patterson, 2009)"
---

A single decode step does ~`2 · total_params` floating-point operations: one MAC per parameter, doubled for the multiply and the add. It reads `total_params · dtype_bytes` from HBM (the full weight tensor) plus `kv_cache_bytes_so_far` (cached K and V across all layers). It writes one token's worth of activations.

Arithmetic intensity is FLOPs per byte loaded:

`AI ≈ (2 · total_params) / (total_params · dtype_bytes + kv_cache_bytes)`

For bf16 weights with negligible cache, AI is approximately 1 FLOP / byte. A modern GPU (e.g., H100) has ~67 TFLOPs/sec bf16 and ~3.3 TB/sec HBM bandwidth, giving a balance ratio of about 20 FLOPs/byte. A workload at AI≈1 sits 20x below the balance point; the GPU is reading memory, not computing.

The consequences are direct:
- Bigger weights run slower in linear proportion (more bytes to read).
- Smaller KV cache (GQA, MQA, MLA) makes the cache term negligible at long context.
- Quantization (int8 weights, int4 weights) cuts the byte cost in half or quarter while leaving FLOPs unchanged, raising AI toward the compute-bound regime.
- Batching decode steps across requests reuses each weight load across many tokens, also raising AI.

Training does not have this problem: a forward + backward pass with batch B processes B · T tokens against one weight load, so AI scales with batch.

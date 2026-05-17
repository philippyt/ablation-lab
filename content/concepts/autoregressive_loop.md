---
id: autoregressive_loop
level: concept
title: Autoregressive Decoding
short: One forward pass per generated token, with the KV cache carrying the prefix.
---

The model defines `p(token_t | token_0, ..., token_{t-1})`. To sample a sequence of length N from a prompt of length P, the loop runs `(N - P)` forward passes:

1. Prefill: process the prompt as one forward pass. Token embeddings flow through every layer; attention produces K and V for each prompt position, which get stored in the KV cache.
2. Decode: for each new token, run one forward pass on just that token. The attention sublayer reads the cached K and V from all previous positions; only the new token's Q, K, V are computed fresh. Sample the next token from the output logits. Append.

Total compute is O(P + (N - P)) forward passes, each scaling with `n_layers · d_model²` for the linear projections plus the attention reduction. The prefill is compute-bound (you process P tokens in parallel); decode is memory-bound (you process 1 token per step but load the whole weight tensor plus the KV cache).

This asymmetry is why serving systems separate prefill from decode, batch decode steps across requests, and aggressively reduce KV cache size: decode is where wall-clock time accumulates.

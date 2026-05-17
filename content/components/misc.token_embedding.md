---
id: misc.token_embedding
level: component
title: Token Embedding
short: A lookup table that turns each token ID into a learned d_model vector.
related_concepts: []
---

The first thing the model does to your input is turn each token ID into a vector. The token embedding is the table that does that:

`tok_embed = nn.Embedding(vocab_size, d_model)`

It is a `vocab_size × d_model` matrix of learned parameters. For an input ID `i`, the model reads row `i` of the table and emits a `d_model`-dimensional vector. That vector is the model's input representation for that token.

In this lab `vocab_size = 72` (one row per character in the TinyStories corpus, since we use a char-level tokenizer) and `d_model = 128`. The table is 72 × 128 = 9,216 learned parameters.

Two things the token embedding does **not** do:

1. It does not add positional information. Position is handled by RoPE inside attention, or by a separate learned absolute table, depending on `pos_embed`.
2. It does not depend on context. Every occurrence of the same token starts from the same row. Context-aware representations are what the rest of the network produces.

**Why it matters:** The embedding rows are where the model stores per-token meaning. Tokens that behave similarly in the data end up with similar rows. When `tie_embeddings` is on, the same matrix is reused (transposed) as the LM head, so the input lookup and the output projection share weights. When `tie_embeddings` is off, the LM head is a separate matrix and the token embedding rows are only used at the input.

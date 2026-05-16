---
id: misc.tied_embeddings
level: component
title: Tied vs Untied Embeddings
short: Reuse the token-embedding matrix as the LM-head weights, or use a separate one.
related_concepts: []
sources:
  - "Press, Wolf, Using the Output Embedding to Improve Language Models (2017)"
  - "Tied: Llama 1, Gemma; Untied: GPT-2/3, Llama 3.1 405B"
---

The LM head is a linear projection from `d_model` back to `vocab_size`. There are two choices:

1. **Tied**: `lm_head.weight = tok_embed.weight`. The same matrix that turns token IDs into hidden vectors is reused (transposed) to turn hidden vectors back into logits.
2. **Untied**: a separate `nn.Linear(d_model, vocab_size, bias=False)`. Twice the embedding parameter cost.

Tied embeddings save `vocab_size * d_model` parameters and act as a mild regularizer. At small scales this is meaningful: with `vocab = 50,000, d = 4096`, untied costs 205M extra params.

**Why it matters:** Smaller models (Llama 1, Gemma) tie embeddings to save params and improve sample efficiency. Frontier models (GPT-3, Llama 3.1 405B) untie because (a) the extra capacity helps at the head, and (b) at 405B the saved 205M is negligible. In this lab, tied vs untied changes total params by exactly `vocab_size * d_model`. With `vocab = 72, d = 128` that is 9,216 params, small but visible in the cost panel.

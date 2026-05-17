---
id: misc.qk_norm
level: component
title: QK-Norm
short: Apply LayerNorm or RMSNorm to Q and K just before the attention dot product.
related_concepts: []
---

Before computing `Q · K^T / sqrt(d_h)`, normalize Q and K along the head dimension:

`Q ← norm(Q),  K ← norm(K)`

The dot products then live in a bounded range. This stabilizes attention logits at large model widths and prevents the softmax from saturating to one-hot early in training.

Parameter count is small. With RMSNorm-style QK-norm, each layer adds `(n_heads + n_kv_heads) * head_dim` learned scales. A LayerNorm-style version doubles that to include a bias term.

**Why it matters:** ViT-22B introduced QK-norm to make wide vision transformers trainable without aggressive learning-rate warmup. OLMo 2 brought it into language models for the same reason. Softmax saturation at scale hurts training stability. Most current 7B to 13B open weights do not bother because they train fine without it. As models get wider and run longer, QK-norm becomes a useful stability lever. In this lab it adds a small number of params per attention block and the trained model uses it when you flip the switch on.

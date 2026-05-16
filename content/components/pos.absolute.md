---
id: pos.absolute
level: component
title: Absolute Learned Position Embeddings
short: Learn one vector per position, add it to the token embedding.
related_concepts: []
sources:
  - "BERT (Devlin et al., 2019)"
  - "GPT-2 / GPT-3 (Radford et al., 2019/2020)"
  - "Original tabular transformer encoders"
---

Allocate a learned embedding matrix of shape `(ctx, d_model)` and add `pos_embed[m]` to the token embedding at position `m`. The model figures out for itself what "being at position 7" should add to a token's representation.

Parameter count: `ctx * d_model`. In this lab with `ctx = 128, d_model = 128` that is 16,384 extra params; in GPT-3 with `ctx = 2048, d_model = 12288` it is 25.2M params dedicated solely to positions.

Two limits:
1. Hard ceiling at `ctx`: positions beyond the training window have no embedding and behavior is undefined.
2. No relative property: extending the context window requires retraining or interpolation.

**Why it matters:** GPT-2 and GPT-3 used absolute learned positions and it worked, at GPT-3 cost. Modern open weights replaced absolute with RoPE because zero params and natural length extrapolation are strictly better at scale. In this lab the cost panel grows by `ctx * d_model` parameters when you pick absolute. That is the only architectural choice in the pos_embed dropdown that adds learned weights.

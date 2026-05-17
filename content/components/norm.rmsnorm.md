---
id: norm.rmsnorm
level: component
title: RMSNorm
short: LayerNorm without the mean-subtraction or bias. Just a rescale.
related_concepts: []
---

Strips the mean-subtraction and the bias from LayerNorm. The formula becomes:

`y = γ * x / sqrt(mean(x²) + ε)`

It learns one parameter per channel (`γ`) instead of two. Parameter count per norm is `d_model`.

Empirically, the mean-subtraction step in LayerNorm contributes very little. What matters at scale is the rescaling. RMSNorm preserves the rescaling and drops the rest, leaving a simpler op that has one fewer reduction over the channel dimension and one fewer learned vector.

**Why it matters:** Llama, Mistral, Qwen, and Gemma all use RMSNorm because the simplification compounds. Half the parameters per norm, one fewer reduction per token, and no measured quality loss at scale. Swapping LayerNorm to RMSNorm in this lab drops total params slightly. The bigger real-world effect is faster compute per token, which does not show in the cost panel but matters a lot in production serving.

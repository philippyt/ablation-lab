---
id: pos.nope
level: component
title: NoPE (No Positional Encoding)
short: Use no explicit position information. The causal mask carries it.
related_concepts: []
---

Decoder-only transformers can learn to encode position implicitly through the causal attention mask. A token at position 0 attends only to itself. Position 1 attends to two tokens. Position 2 attends to three, and so on. The varying attention pattern is enough signal for the model to figure out where each token sits.

Parameter count: 0. No position embeddings of any kind.

In some recent length-generalization studies, NoPE extrapolates to longer contexts at least as well as RoPE for small decoder-only models. Part of the reason is that there is no learned or formula-baked assumption about position that could break as positions grow beyond what was seen during training.

**Why it matters:** NoPE is the simplest possible answer to "how do transformers know position?". The answer "because of the causal mask" turned out to be more powerful than people assumed for years. Encoder-only models do need positional information because they have no causal mask. Decoder-only language models can sometimes do without. In this lab, NoPE adds zero params and shows that the position story is more nuanced than "you must pick RoPE or ALiBi".

---
id: pos.nope
level: component
title: NoPE (No Positional Encoding)
short: Use no explicit position information. The causal mask carries it.
related_concepts: []
sources:
  - "Haviv et al., Transformer Language Models without Positional Encodings Still Learn Positional Information (2022)"
  - "Kazemnejad et al., The Impact of Positional Encoding on Length Generalization (2023)"
  - "Some Phi-3 small-model experiments, ablations in OLMo and similar"
---

Decoder-only transformers can learn to encode position implicitly through the causal attention mask. A token at position 0 attends only to itself; position 1 attends to two tokens; and so on. The varying attention pattern is enough signal for the model to figure out where it is.

Parameter count: 0. No position embeddings of any kind.

In some recent length-generalization studies, NoPE extrapolates to longer contexts at least as well as RoPE for small decoder-only models, partly because there is no learned or formula-baked assumption about position to break.

**Why it matters:** NoPE is the simplest possible answer to "how do transformers know position?", and the answer "because of the causal mask" is more powerful than people assumed for years. Encoder-only models do need positional information (no causal mask), but decoder-only LMs can sometimes do without. In this lab, NoPE adds zero params and shows that the position story is more nuanced than "you must pick RoPE or ALiBi".

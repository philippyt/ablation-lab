---
title: Ablation Lab
emoji: 🧪
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Compose a tiny transformer, ablate a part, watch it break.
---

# Ablation Lab
Playground for breaking transformers on purpose.

You compose an architecture (norm, attention paradigm, FFN shape, activation, positional scheme, ablation toggles) and the tool does two things in parallel. It computes the exact parameter, KV cache, memory, and decode cost of the architecture from formulas. It also resolves the architecture to a real trained checkpoint, runs inference, and streams characters back. When the architecture matches a trained shape the output is real text from those trained weights. When you toggle inference-time interventions (turn attention off, swap RoPE for ALiBi, zero the absolute position table) the output degrades in a way that maps directly to what was disabled. The degraded output is the whole point.

The trained model is intentionally tiny: d_model=128, n_layers=4, n_heads=8, head_dim=16, ctx=256, char-level vocab of 72, trained on roughly 1MB of TinyStories. The architectural question is what changes when you swap a component. The scale would only obscure it.

## Findings

Single-variable comparison: Llama-3-shape and Qwen3-shape are identical on every architectural axis (norm, attention type, KV-head count, FFN, activation, positional scheme, dimensions) except QK-Norm. Both resolve to their own exact trained checkpoints, so the comparison is clean.

- Under greedy decoding at short lengths, both produce coherent TinyStories text.
- Past ~300 tokens they diverge. QK-Norm-off falls into a hard repetition loop and runs to the token cap. QK-Norm-on reaches a clean end of story and stops on its own. Reproducible.
- Both were trained from scratch with different initializations, so this is a difference between two trained models that differ on one axis, not proof QK-Norm itself causes loop avoidance.
- Plausible mechanism: QK-Norm bounds attention logit magnitude, keeping the next-token distribution from collapsing into an attractor under deterministic decoding. One controlled pair cannot establish this.
- Not a language-modeling-quality claim. QK-Norm-off had the slightly lower validation loss (0.933 vs 0.956). The effect is specifically decode-time degeneration.
- Other QK-Norm-off models in the lab (Mixtral-shape, Llama-1-shape) also do not loop, but each differs from the looping config on more than the QK-Norm axis (Mixtral on FFN/MoE, Llama-1-shape on KV-head count) and resolves to a different checkpoint.

## The Kind-A / Kind-B split

The central design call was figuring out which knobs need a separate trained checkpoint and which can be applied live to an existing one. Treating every knob as needing its own training run gives a cross-product in the thousands of configs. Treating every knob as live gives wrong output, because the trained weights expected one component and would be evaluated under another.

The compromise the tool runs on:

| Knob | Kind | Reason |
|---|---|---|
| `norm` (layernorm, rmsnorm) | A | Different parameter structure. LayerNorm has weight and bias per channel, RMSNorm has weight only. |
| `attention_type` (standard, mla) | A | MLA has a compressed-latent projection layout that doesn't exist in standard attention. |
| `n_kv_heads` (8, 2, 1) | A | The K and V projections are sized by `n_kv_heads`. Changing it changes weight shapes. |
| `ffn` (dense, moe) | A | Completely different module. Dense has one gated FFN, MoE has N expert FFNs plus a router. |
| `n_experts`, `top_k` (when MoE) | A | Change expert count and the per-block weight tensor shape changes. |
| `activation` (silu, gelu, relu2) | A | Weights are tuned to a specific elementwise nonlinearity. Substituting another at inference corrupts the FFN. |
| `qk_norm` (on, off) | A | Adds per-head Q and K normalization scales. Different parameter count. |
| `pos_embed` learned-absolute vs param-free family | A | Absolute adds a `ctx * d_model` learned table. The param-free family (rope, alibi, sinusoidal, nope) has no learned positional params, so the canonical training run picks one (rope) and the rest become Kind-B swaps on top. |
| `enable_attention` | B | Inference-only intervention. Skip the attention sublayer in the forward pass. Residual passes through. |
| `enable_ffn` | B | Same idea, FFN side. |
| `enable_norms` | B | Replace norm forward with identity at inference. |
| `pos_embed` swap inside {rope, alibi, sinusoidal, nope} | B | Parameter-free schemes substituted at inference. Q and K were trained expecting one of them, the others produce degraded output. |
| `tie_embeddings` flip | B | If trained tied, untying reads the embedding matrix as the LM head (mathematically identical). If trained untied, tying discards the trained LM head, which is the real ablation. |

Kind-A combinations are trained from scratch, one checkpoint per combination. The pruned cross-product is 2 (norm) * 4 (attention) * 2 (ffn) * 3 (activation) * 2 (qk_norm) * 2 (pos class) = 192, minus 24 for `MLA + absolute` (architecturally incoherent, no published model does it), plus 1 for a small-MoE bonus shape. That gives 169 configs, around 12 hours of training on a single H100-class GPU at this corpus size. The total time is dominated by MoE configs (~7.5 min each) versus dense (~75 sec each).

Kind-B interventions are applied live in the forward pass via a `runtime` dict threaded through the model. Toggling them never invalidates the checkpoint lookup. The Run panel surfaces them in the state pill as `Kind-B: pos_embed=alibi`, etc.

The honest version of "ablation" is the Kind-B version. Training a model with no attention does not show you what attention was doing, because the network learns to route information through whatever sublayers remain. Training normally and then disabling attention at inference is the experiment that actually answers the question.

## Architecture hash and checkpoint resolution

Every architecture maps to a 12-character hash, the prefix of `sha256(canonical_json(kind_a_fields))`. The hash function lives in both `ablation/config.py` (Python) and `frontend/src/lib/hash.ts` (TypeScript). Both implementations consume `tests/fixtures/hash_examples.json` so CI catches drift.

Only Kind-A fields are hashed. Kind-B flags do not change the hash, which is correct: flipping them does not change which trained weights we should load. The `pos_embed` field is normalized to two buckets for hashing purposes (`absolute` or `rope`), since the four param-free schemes share one canonical training run.

The resolver at `/api/resolve`:

| State | Trigger | Output |
|---|---|---|
| green | Exact hash match in manifest | Real inference. Cost panel exact. |
| green + Kind-B live | Exact hash match, plus Kind-B knob set | Real at the architecture level. The degraded output is the lesson, not a crash. |
| amber | No exact match. Nearest entry within Hamming distance 2 over Kind-A axes | Approximate inference with the nearest checkpoint. State pill lists which axes differ. Cost panel still exact. |
| red | No entry within Hamming distance 2 | No fake sample. Cost panel still exact. |

Hamming distance is over a fixed list of Kind-A axes. MoE shape (`n_experts`, `top_k`) only counts when both old and new are MoE. `n_kv_heads` only counts when neither is MLA.

## Cost panel math

Every number on the cost panel is formula-derived from the current `ArchConfig`, not measured. The formulas match `sum(p.numel())` on the instantiated model to within zero error, enforced by `tests/test_costs_match_model.py` over the entire trained set.

Key counts at the lab's toy scale (d_model=128, n_layers=4, ffn_mult=4, ctx=256):

| Quantity | Formula | Example (LayerNorm, MHA, dense SwiGLU) |
|---|---|---|
| FFN hidden | `round_up_64((2/3) * ffn_mult * d_model)` | 384 |
| Per-block attention params | `(n_heads * head_dim + 2 * n_kv_heads * head_dim + n_heads * head_dim) * d_model` | 65,536 |
| Per-block FFN params (dense) | `3 * d_model * hidden` | 147,456 |
| Per-block FFN params (MoE) | `n_experts * (3 * d_model * hidden) + d_model * n_experts` | 1,180,672 (8 experts) |
| Per-block norm params | `2 * d_model * (2 if layernorm else 1)` | 512 |
| Embedding params | `vocab_size * d_model` | 9,216 |
| LM head params | `0` if tied, else `vocab_size * d_model` | 0 |
| Absolute pos params | `ctx * d_model` if `pos_embed == "absolute"`, else `0` | 0 or 32,768 |
| QK-norm params | `n_layers * (n_heads + n_kv_heads) * head_dim` if on | 1,024 |
| Total params (baseline) |  | 863,488 |

The active-per-token count differs from total only when `ffn == "moe"`: routed experts contribute `top_k * (3 * d_model * hidden)` per block instead of the full `n_experts * (3 * d_model * hidden)`. The router weights are always active because they run for every token.

## KV cache, the central memory story

KV cache size dominates serving cost at long context on real models. The cost panel makes the relationship between architecture and KV cache visible, even at toy scale.

For standard attention (MHA, GQA, MQA), bytes per token across all layers:

```
2 * n_layers * n_kv_heads * head_dim * dtype_bytes
```

The factor of 2 is for K and V. Full-context cache is that times `ctx`. The `2 * n_kv_heads` is why GQA and MQA exist: GQA shares K and V across query head groups, MQA collapses to a single K and V across all query heads. The KV cache shrinks linearly in `n_kv_heads` while modeling capacity (governed by `n_heads`) stays the same.

For MLA, the layout is different. The cached state per token is the compressed latent plus a small rope-positional vector:

```
n_layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
```

K and V are reconstructed from the latent at attention time. The trade is two extra matmuls per token (decompress) in exchange for a much smaller cache. The lab uses `kv_lora_rank = max(8, d_model // 8)` and `qk_rope_head_dim = head_dim // 2`. At baseline this puts MLA's KV cache below MQA's.

Approximate full memory at the lab scale:

| Architecture | Total params | KV per token | KV @ ctx=256 | Total memory (weights + KV + activations) |
|---|---|---|---|---|
| Baseline (MHA, dense, LayerNorm, SiLU, RoPE) | 863K | 2,048 B | 512 KB | ~3 MB |
| Swap MHA for GQA (n_kv=2) | 765K | 512 B | 128 KB | ~2 MB |
| Swap MHA for MQA (n_kv=1) | 749K | 256 B | 64 KB | ~2 MB |
| Swap standard for MLA | 757K | 192 B | 48 KB | ~2 MB |
| Swap dense for MoE 8 experts top-2 | 5.00M | 2,048 B | 512 KB | ~11 MB |

The relative changes match real-scale results. MHA to MQA cuts KV by 8x. GQA at `n_kv=2` cuts it by 4x. MLA cuts it further still. MoE leaves KV unchanged because MoE only restructures the FFN.

## Decode bottleneck

The VRAM-tier selector in the cost panel is not a capacity check. At toy scale every model fits everywhere. The selector sets a ridge point, which is the ratio of peak compute to peak bandwidth for hardware at that VRAM size.

Bytes moved per decode token is `weights + KV @ full ctx` (every weight is read once, the full KV cache is read once). FLOPs per token is roughly `2 * active_params`. Their ratio is arithmetic intensity. Below the ridge, the workload is memory-bound and the chip is waiting on memory. Above, it is compute-bound.

Typical bandwidth and FP16 TFLOPs per VRAM tier:

| VRAM tier | Typical FP16 dense | Typical bandwidth | Ridge (FLOPs/byte) |
|---|---|---|---|
| 16 GB | 65 TFLOPs | 320 GB/s | 203 |
| 24 GB | 165 TFLOPs | 1.0 TB/s | 165 |
| 32 GB | 209 TFLOPs | 1.8 TB/s | 117 |
| 48 GB | 362 TFLOPs | 864 GB/s | 419 |
| 80 GB | 989 TFLOPs | 3.35 TB/s | 295 |
| 141 GB | 989 TFLOPs | 4.8 TB/s | 206 |
| 192 GB | 1800 TFLOPs | 6.8 TB/s | 265 |

The model is memory-bound on every tier here. Arithmetic intensity is far below every tier's ridge point, so the conclusion holds regardless of the exact hardware. The per-tier TFLOPs and bandwidth figures are representative values. The specific compute-to-bandwidth balance varies by card, which is why the ridge column is non-monotonic in VRAM size.

The absolute tok/s at lab scale is unreasonably high because the model is unreasonably small. The interesting number is how it shifts when you swap MHA for MQA (less KV per token) or push context length up (more KV at full ctx).

## Sliding-window generation

The trained context is 256 tokens. The user can request up to 5000 max_tokens. Without intervention, a rope-trained model would either crash (out-of-range rope cache) or quietly truncate at the trained ctx.

The lab handles this with a sliding KV cache. When the cache fills past `ctx`, the oldest entries drop. RoPE positions are computed on the fly from the absolute token position, not from a fixed pre-allocated buffer, so generation can continue past the trained ctx. The model only sees the most recent `ctx` tokens worth of cached K and V at any time.

This works for rope-trained checkpoints because RoPE encodes relative position into the Q-dot-K phase: the rotations of cached K entries at their original absolute positions, combined with the new Q at its own absolute position, produce the correct relative-position phase regardless of when the K was inserted into the cache. Past the trained ctx the model is operating slightly out of distribution (it never saw positions beyond 256 during training) but the math is consistent.

Absolute-positional checkpoints cannot do this. Their learned position table is `ctx` rows long and has no entries beyond. For those, the generate loop honors the ctx cap and the done event reports `stop_reason: "ctx_limit"` instead of pretending it was the max_tokens cap.

## Stop reasons

The end-of-stream event distinguishes three terminations:

| Reason | Meaning |
|---|---|
| `eos` | The model produced a sentence-ending character followed by a blank line, the natural TinyStories boundary. Generation halts early. The common case for a healthy model. |
| `max_tokens` | The user's safety cap was hit. The model rambled without producing a clean ending. Diagnostic: usually means the architecture is degraded enough that the model can't close a story. |
| `ctx_limit` | The model's context window was reached and it cannot extend (absolute-positional case). Honest report instead of mislabeling the cap. |

Each is recorded in the manifest entry and surfaced in the Run panel status line.

## Training infrastructure

The trainer is a single ~150-line file at `training/train.py`. It loads a JSON config, builds the model, trains with AdamW plus a cosine learning-rate schedule, evaluates on a validation split every 250 steps, and saves to `checkpoints/{hash}.pt` plus a manifest entry. Idempotent on hash: re-running on a config whose hash is already in the manifest is a no-op unless `--force` is passed. The manifest is written atomically (write to `.tmp` then `os.replace`) under an fcntl file lock, so concurrent trainings don't corrupt it.

`scripts/train_all.sh` iterates every config file, calls `train.py` per config, and skips configs that already have a checkpoint. Mid-batch crashes resume cleanly by rerunning without `--force`. After the training loop the script automatically runs the qk_norm divergence check.

## qk_norm divergence check

QK-norm is the smallest Kind-A axis. It adds a tiny number of parameters per attention block. At the lab's scale the qk_norm=on and qk_norm=off twins can train to nearly identical outputs from the same seed, which silently erases the lesson the tool is meant to teach (that QK-norm changes behavior).

The check, in `scripts/qk_norm_divergence_check.py`:

1. Group manifest entries by their "twin key" (every Kind-A axis except qk_norm).
2. For each (qkn=off, qkn=on) pair, greedy-decode 100 characters from a fixed prompt.
3. Hamming-compare the two outputs character by character.
4. If the ratio of differing characters is below 0.10, mark the pair as `lesson_collapsed`.
5. For collapsed pairs, queue the qkn=on config for from-scratch retraining with a different seed.
6. Cap at 2 retry rounds. Remaining fails are recorded in the manifest with `qk_norm_divergence: "fail"` and surfaced in the divergence report.

This runs automatically at the end of `train_all.sh`. It is not a manual morning script.

## MoE on a tiny corpus

The lab supports MoE with `n_experts` parallel gated FFNs and a top-k router. At baseline the canonical MoE shape is 8 experts top-2 (Mixtral-shape). A bonus small-MoE shape (4 experts top-1) is also trained.

The truthful caveat: MoE specializes when experts see enough data to learn different functions. On a 1MB corpus the router collapses to one expert. The model behaves like a dense FFN with extra unused parameters. The cost panel shows the correct shape (total much greater than active) but the quality story is not visible at this scale. The MoE explainer documents this honestly.

The lab does not implement a load-balance loss. DeepSeek-V3-style bias-update routing would help router stability but adds complexity for limited educational gain at this scale.

## Layout summary

```
ablation/ - shared model code (used by training and inference)
training/ - trainer + 169 config JSON files
backend/ - FastAPI app, SSE streaming, resolver
frontend/ - Preact + signals + Vite, all UI
content/ - markdown explainers + consequence rules
scripts/ - train_all.sh, generate_configs.py, qk_norm_divergence_check.py, build_content_index.py, demo_kind_b.py
tests/ - hash parity, cost parity
checkpoints/ - trained .pt files + manifest.json
data/ - tinystories.txt
```

The shared model module is the bit that makes the whole tool trustworthy. The inference server and the trainer instantiate the same `TinyTransformer` class with the same forward pass, so what you see in the browser is literally what was trained, not a re-implementation that approximates it.

## Credits

TinyStories corpus from [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759).
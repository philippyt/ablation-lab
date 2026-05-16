import type { ArchConfig } from "../types";

export function ffnLabel(cfg: ArchConfig): string {
  const variant = cfg.ffn === "moe" ? "MoE" : "Dense";
  const gate = cfg.activation === "silu" ? "SwiGLU"
    : cfg.activation === "gelu" ? "GeGLU"
    : "ReGLU²";
  if (cfg.ffn === "moe") {
    return `MoE · ${gate} experts · top-${cfg.top_k}/${cfg.n_experts}`;
  }
  return `${variant} ${gate}`;
}

export function diagramFfnLabel(cfg: ArchConfig): { label: string; sub?: string } {
  const gate = cfg.activation === "silu" ? "SwiGLU"
    : cfg.activation === "gelu" ? "GeGLU"
    : "ReGLU²";
  if (cfg.ffn === "moe") {
    return { label: "MoE", sub: `${gate} ${cfg.top_k}/${cfg.n_experts}` };
  }
  return { label: gate };
}

export function activationDisplay(cfg: ArchConfig): string {
  if (cfg.activation === "silu") return "SiLU";
  if (cfg.activation === "gelu") return "GELU";
  return "ReLU²";
}

export const HINTS: Record<string, string> = {
  name: "Just a label for your model. No effect on the architecture.",
  attention: "The shape of self-attention. MHA keeps a full K/V per head; GQA shares K/V across head groups; MQA shares one K/V across all heads. Smaller K/V count means smaller KV cache.",
  position: "How the model knows the order of tokens. RoPE rotates Q and K by position; ALiBi adds a distance bias; absolute learns a vector per position; NoPE skips it entirely.",
  norm: "Normalization applied before each sublayer. LayerNorm centers and rescales with learned scale + shift; RMSNorm drops the centering and the shift.",
  qk_norm: "Normalize Q and K just before the dot product. Stabilizes attention logits at large widths. Used in OLMo 2 and ViT-22B.",
  ffn: "The feed-forward block inside each transformer layer. Dense runs every parameter for every token; MoE routes each token to top-k of N expert FFNs.",
  activation: "Nonlinearity inside the FFN. SiLU pairs with SwiGLU's gating; GELU was the de facto choice in BERT and GPT-2/3; ReLU² (squared ReLU) is sparser and cheaper.",
  tied: "Whether the LM-head weights are the transposed token embedding (tied) or a separate matrix (untied). Tied saves vocab×d params.",
  layers: "Number of transformer blocks stacked. Each block is one norm-attention-norm-FFN unit.",
  d_model: "The hidden dimension of the residual stream. All projections inside a block flow through this width.",
  ctx: "Maximum sequence length the model supports at inference. KV cache and absolute position embeddings scale with this.",
  enable_attention: "Ablate attention entirely. Each block becomes Norm → FFN with a residual; the model loses any cross-token information mixing.",
  enable_ffn: "Ablate the feed-forward sublayer. Each block is just attention plus residual. The model loses the per-token MLP capacity.",
  enable_norms: "Ablate every normalization layer. Trains poorly at non-trivial depth; useful to see what norms cost (params) and to confirm they are not free.",
  prompt: "Text fed to the model. Each character becomes one token (this is a char-level model).",
  max_tokens: "Upper bound on how many new tokens to generate. Generation may stop earlier when the KV cache fills.",
  temperature: "Softmax temperature applied to the next-token logits. 0 is greedy; higher values sample more diversely; values above ~1.5 typically produce noise.",
};

import type { ArchConfig } from "../types";

export function ffnLabel(cfg: ArchConfig): string {
  if (cfg.ffn === "moe") {
    return `MoE · top-${cfg.top_k}/${cfg.n_experts}`;
  }
  return "Dense";
}

export function diagramFfnLabel(cfg: ArchConfig): { label: string; sub?: string } {
  if (cfg.ffn === "moe") {
    return { label: "MoE", sub: `top-${cfg.top_k}/${cfg.n_experts}` };
  }
  return { label: "Dense" };
}

export function activationDisplay(cfg: ArchConfig): string {
  if (cfg.activation === "silu") return "SiLU";
  if (cfg.activation === "gelu") return "GELU";
  return "ReLU²";
}

export const HINTS: Record<string, string> = {
  name: "Just a label for your model. Doesn't change the architecture.",
  attention: "How the model shares Q, K, and V across attention heads. MHA gives every query head its own K and V. GQA groups several query heads to share one K and V pair. MQA shares a single K and V across all query heads. Fewer K and V means a smaller KV cache and faster decode.",
  position: "How the model encodes word order. RoPE rotates Q and K based on each token's position. ALiBi adds a fixed distance penalty to attention scores. Absolute learns one vector per position. NoPE skips positional info entirely, which breaks word order.",
  norm: "What normalization runs before each sublayer. LayerNorm subtracts the mean, divides by the standard deviation, and learns a scale and shift. RMSNorm skips the centering and the shift, keeping just the rescale.",
  qk_norm: "Run a normalization on Q and K right before the attention dot product. Stabilizes attention scores at large model widths.",
  ffn: "The feed-forward block in each layer. Dense runs every weight for every token. MoE has N expert feed-forwards plus a router, and each token only goes through its top-k experts.",
  activation: "The nonlinearity inside the gated feed-forward. SiLU is the modern default and is paired with SwiGLU. GELU was the BERT and GPT-2/3 choice. ReLU squared is sparser and a bit cheaper.",
  tied: "Whether the output projection reuses the token embedding matrix (tied) or has its own weights (untied). Tied saves vocab times d_model parameters.",
  layers: "How many transformer blocks are stacked. One block is norm, attention, norm, then feed-forward.",
  d_model: "Width of the residual stream. Every projection inside a block flows through this dimension.",
  ctx: "Longest sequence the model can handle. KV cache size and the absolute position table both scale with this.",
  enable_attention: "Turn off the attention sublayer at inference. Each block becomes just norm and feed-forward with a residual. The model loses the ability to mix information across tokens, and output collapses fast.",
  enable_ffn: "Turn off the feed-forward sublayer at inference. Each block becomes just attention with a residual. The model loses its per-token MLP capacity.",
  enable_norms: "Turn off every normalization layer at inference. Activations propagate unscaled and output becomes noisy. Useful for seeing what norms actually do.",
  prompt: "The text you feed the model. Each character is one token (this is a char-level model on TinyStories).",
  max_tokens: "Safety cap on how many tokens the model can generate. Generation usually stops earlier when the model produces a story ending. Hitting the cap means the model rambled without finishing, which is its own diagnostic.",
  temperature: "How random the sampling is. 0 picks the most likely token every time (good for reproducibility when testing). Higher values sample more freely. Above 1.5 the output usually becomes noise.",
};

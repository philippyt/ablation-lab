import type { ArchConfig } from "../types";

export function ffnLabel(cfg: ArchConfig): string {
  if (cfg.ffn === "moe") {
    return `MoE top ${cfg.top_k} of ${cfg.n_experts}`;
  }
  return "Dense";
}

export function diagramFfnLabel(cfg: ArchConfig): { label: string; sub?: string } {
  if (cfg.ffn === "moe") {
    return { label: "MoE", sub: `top ${cfg.top_k} of ${cfg.n_experts}` };
  }
  return { label: "Dense" };
}

export function activationDisplay(cfg: ArchConfig): string {
  if (cfg.activation === "silu") return "SiLU";
  if (cfg.activation === "gelu") return "GELU";
  return "ReLU²";
}

export const HINTS: Record<string, string> = {
  name: "A label for your model. Changing it does not alter the architecture.",
  attention: "How the model shares Q, K, and V across attention heads. MHA gives every query head its own K and V. GQA groups several query heads to share one K and V pair. MQA shares a single K and V across all query heads. Fewer K and V heads mean a smaller KV cache and faster decode.",
  position: "How the model encodes token order. RoPE rotates Q and K based on position. ALiBi adds a fixed distance penalty to attention scores. Absolute learns one vector per position. NoPE skips positional info completely, which breaks word order.",
  norm: "What normalization runs before each sublayer. LayerNorm subtracts the mean, divides by standard deviation, and learns a scale and shift. RMSNorm skips centering and shift, keeping only the rescale.",
  qk_norm: "Applies normalization to Q and K right before the attention dot product. Bounds attention logit magnitude and prevents decode loops.",
  ffn: "The feed forward block in each layer. Dense runs every weight for every token. MoE has N expert blocks plus a router, so each token only visits its top k experts.",
  activation: "The nonlinearity inside the gated feed forward layer. SiLU is paired with SwiGLU. GELU was the BERT and GPT choice. ReLU squared is sparser and slightly cheaper.",
  tied: "Whether the output projection reuses the token embedding matrix or keeps separate weights. Tied saves vocab times d_model parameters.",
  layers: "How many transformer blocks are stacked. One block contains norm, attention, norm, and feed forward.",
  d_model: "Width of the residual stream. Every projection inside a block flows through this dimension.",
  ctx: "Longest sequence the model can hold in context. KV cache size and absolute position tables scale with this.",
  enable_attention: "Turn off the attention sublayer at inference. Each block becomes norm and feed forward with a residual. The model loses cross token mixing and output collapses quickly.",
  enable_ffn: "Turn off the feed forward sublayer at inference. Each block becomes attention with a residual and loses per token MLP capacity.",
  enable_norms: "Turn off every normalization layer at inference. Activations propagate unscaled and output turns noisy.",
  prompt: "The text you feed the model. Each character is one token in this char level TinyStories model.",
  max_tokens: "Safety cap on how many characters the model can generate. Healthy models usually stop early at a natural story ending.",
  temperature: "Sampling randomness. 0 picks the most likely character every time for exact reproducibility. Higher values sample more freely.",
};

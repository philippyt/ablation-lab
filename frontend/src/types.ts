export type AttentionKind = "mha" | "gqa" | "mqa" | "mla";
export type AttentionType = "standard" | "mla";
export type NormKind = "layernorm" | "rmsnorm";
export type FFNKind = "dense" | "moe";
export type PosEmbedKind = "rope" | "alibi" | "absolute" | "sinusoidal" | "nope";
export type ActivationKind = "silu" | "gelu" | "relu2";

export interface ArchConfig {
  vocab_size: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  head_dim: number;
  ctx: number;
  norm: NormKind;
  ffn: FFNKind;
  ffn_mult: number;
  n_experts: number;
  top_k: number;
  tie_embeddings: boolean;
  pos_embed: PosEmbedKind;
  activation: ActivationKind;
  qk_norm: boolean;
  attention_type: AttentionType;
  enable_attention: boolean;
  enable_ffn: boolean;
  enable_norms: boolean;
}

export interface CostPanel {
  total_params: number;
  active_params_per_token: number;
  kv_cache_per_token: number;
  kv_cache_full_ctx: number;
  approx_memory_bytes: number;
  swiglu_hidden: number;
  dtype_bytes: number;
}

export interface ManifestEntry {
  hash: string;
  name: string | null;
  attention_kind: AttentionKind;
  val_loss: number;
  arch_config: ArchConfig;
  trained_at: number;
  path: string;
}

export interface GenerateRequest {
  config: ArchConfig;
  prompt: string;
  max_tokens: number;
  temperature: number;
  top_k?: number;
  instrument?: boolean;
}

export interface Instrumentation {
  moe_active: boolean;
  attention_active: boolean;
  attention_instrumented: boolean;
  norms_active: boolean;
  n_layers: number;
  n_experts: number;
  top_k: number;
  routing_counts: number[][];
  routing_totals: number[];
  attn_entropy_mean: (number | null)[];
  attn_max_entropy: number[];
  residual_norm_mean: (number | null)[];
  token_surprises: number[];
  token_strings: string[];
}

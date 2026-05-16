import type { ArchConfig, CostPanel } from "../types";

function roundToMultiple(x: number, m: number): number {
  return Math.floor((Math.floor(x) + m - 1) / m) * m;
}

export function swigluHidden(cfg: ArchConfig): number {
  return roundToMultiple((2 / 3) * cfg.ffn_mult * cfg.d_model, 64);
}

function normParams(cfg: ArchConfig): number {
  return cfg.d_model * (cfg.norm === "layernorm" ? 2 : 1);
}

export function attentionParams(cfg: ArchConfig): number {
  if (!cfg.enable_attention) return 0;
  const base = cfg.n_layers * attnParams(cfg);
  return base + (cfg.qk_norm ? cfg.n_layers * ((cfg.n_heads + cfg.n_kv_heads) * cfg.head_dim) : 0);
}

export function feedForwardParams(cfg: ArchConfig): number {
  if (!cfg.enable_ffn) return 0;
  return cfg.n_layers * (cfg.ffn === "dense" ? ffnParamsDense(cfg) : ffnParamsMoE(cfg));
}

export function normParamsTotal(cfg: ArchConfig): number {
  if (!cfg.enable_norms) return 0;
  return cfg.n_layers * 2 * normParams(cfg) + normParams(cfg);
}

export function embeddingParams(cfg: ArchConfig): number {
  const embed = cfg.vocab_size * cfg.d_model;
  return embed + (cfg.tie_embeddings ? 0 : embed) + (cfg.pos_embed === "absolute" ? cfg.ctx * cfg.d_model : 0);
}

export function flopsPerToken(cfg: ArchConfig): number {
  return 2 * activeParamsPerToken(cfg);
}

function attnParams(cfg: ArchConfig): number {
  const q = cfg.d_model * (cfg.n_heads * cfg.head_dim);
  const k = cfg.d_model * (cfg.n_kv_heads * cfg.head_dim);
  const v = cfg.d_model * (cfg.n_kv_heads * cfg.head_dim);
  const o = cfg.n_heads * cfg.head_dim * cfg.d_model;
  return q + k + v + o;
}

function ffnParamsDense(cfg: ArchConfig): number {
  return 3 * cfg.d_model * swigluHidden(cfg);
}

function ffnParamsMoE(cfg: ArchConfig): number {
  const h = swigluHidden(cfg);
  return cfg.n_experts * (3 * cfg.d_model * h) + cfg.d_model * cfg.n_experts;
}

function ffnActiveMoE(cfg: ArchConfig): number {
  const h = swigluHidden(cfg);
  return cfg.top_k * (3 * cfg.d_model * h) + cfg.d_model * cfg.n_experts;
}

function blockParams(cfg: ArchConfig): number {
  const norms = cfg.enable_norms ? 2 * normParams(cfg) : 0;
  const attn = cfg.enable_attention ? attnParams(cfg) : 0;
  const ffn = cfg.enable_ffn ? (cfg.ffn === "dense" ? ffnParamsDense(cfg) : ffnParamsMoE(cfg)) : 0;
  return norms + attn + ffn;
}

function blockActiveParams(cfg: ArchConfig): number {
  const norms = cfg.enable_norms ? 2 * normParams(cfg) : 0;
  const attn = cfg.enable_attention ? attnParams(cfg) : 0;
  const ffn = cfg.enable_ffn ? (cfg.ffn === "dense" ? ffnParamsDense(cfg) : ffnActiveMoE(cfg)) : 0;
  return norms + attn + ffn;
}

function posParams(cfg: ArchConfig): number {
  return cfg.pos_embed === "absolute" ? cfg.ctx * cfg.d_model : 0;
}

function qkNormParams(cfg: ArchConfig): number {
  if (!cfg.qk_norm || !cfg.enable_attention) return 0;
  return cfg.n_layers * ((cfg.n_heads * cfg.head_dim) + (cfg.n_kv_heads * cfg.head_dim));
}

export function totalParams(cfg: ArchConfig): number {
  const embed = cfg.vocab_size * cfg.d_model;
  const head = cfg.tie_embeddings ? 0 : cfg.vocab_size * cfg.d_model;
  const blocks = cfg.n_layers * blockParams(cfg);
  const fnorm = cfg.enable_norms ? normParams(cfg) : 0;
  return embed + head + blocks + fnorm + posParams(cfg) + qkNormParams(cfg);
}

export function activeParamsPerToken(cfg: ArchConfig): number {
  const embed = cfg.vocab_size * cfg.d_model;
  const head = cfg.tie_embeddings ? 0 : cfg.vocab_size * cfg.d_model;
  const blocks = cfg.n_layers * blockActiveParams(cfg);
  const fnorm = cfg.enable_norms ? normParams(cfg) : 0;
  return embed + head + blocks + fnorm + posParams(cfg) + qkNormParams(cfg);
}

function mlaLatentDim(cfg: ArchConfig): number {
  return Math.max(8, Math.floor(cfg.d_model / 8));
}

function mlaRopeDim(cfg: ArchConfig): number {
  return Math.floor(cfg.head_dim / 2);
}

export function kvCachePerToken(cfg: ArchConfig, dtypeBytes = 2): number {
  if (!cfg.enable_attention) return 0;
  if (cfg.attention_type === "mla") {
    return cfg.n_layers * (mlaLatentDim(cfg) + mlaRopeDim(cfg)) * dtypeBytes;
  }
  return 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * dtypeBytes;
}

export function kvCacheFullCtx(cfg: ArchConfig, dtypeBytes = 2): number {
  return kvCachePerToken(cfg, dtypeBytes) * cfg.ctx;
}

export function approxMemoryBytes(cfg: ArchConfig, dtypeBytes = 2, batch = 1): number {
  const weights = totalParams(cfg) * dtypeBytes;
  const kv = kvCacheFullCtx(cfg, dtypeBytes);
  const acts = batch * cfg.ctx * cfg.d_model * dtypeBytes * 4;
  return weights + kv + acts;
}

export function attentionKind(cfg: ArchConfig): "mha" | "gqa" | "mqa" | "mla" {
  if (cfg.attention_type === "mla") return "mla";
  if (cfg.n_kv_heads === cfg.n_heads) return "mha";
  if (cfg.n_kv_heads === 1) return "mqa";
  return "gqa";
}

export function costPanel(cfg: ArchConfig, dtypeBytes = 2, batch = 1): CostPanel {
  return {
    total_params: totalParams(cfg),
    active_params_per_token: activeParamsPerToken(cfg),
    kv_cache_per_token: kvCachePerToken(cfg, dtypeBytes),
    kv_cache_full_ctx: kvCacheFullCtx(cfg, dtypeBytes),
    approx_memory_bytes: approxMemoryBytes(cfg, dtypeBytes, batch),
    swiglu_hidden: swigluHidden(cfg),
    dtype_bytes: dtypeBytes,
  };
}

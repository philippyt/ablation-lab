import type { ArchConfig, CostPanel } from "../types";

export interface GpuSpec {
  id: string;
  name: string;
  fp16_tflops: number;
  bandwidth_gbs: number;
  memory_gb: number;
}

export const GPUS: GpuSpec[] = [
  { id: "vram_16", name: "16 GB", fp16_tflops: 65, bandwidth_gbs: 320, memory_gb: 16 },
  { id: "vram_24", name: "24 GB", fp16_tflops: 165, bandwidth_gbs: 1008, memory_gb: 24 },
  { id: "vram_32", name: "32 GB", fp16_tflops: 209, bandwidth_gbs: 1792, memory_gb: 32 },
  { id: "vram_48", name: "48 GB", fp16_tflops: 362, bandwidth_gbs: 864, memory_gb: 48 },
  { id: "vram_80", name: "80 GB", fp16_tflops: 989, bandwidth_gbs: 3350, memory_gb: 80 },
  { id: "vram_141", name: "141 GB", fp16_tflops: 989, bandwidth_gbs: 4800, memory_gb: 141 },
  { id: "vram_192", name: "192 GB", fp16_tflops: 1800, bandwidth_gbs: 6800, memory_gb: 192 },
];

export function decodeMsPerToken(cfg: ArchConfig, gpu: GpuSpec, dtypeBytes = 2): number {
  const weightsBytes = totalParams(cfg) * dtypeBytes;
  const kvBytes = kvCacheFullCtx(cfg, dtypeBytes);
  return ((weightsBytes + kvBytes) / (gpu.bandwidth_gbs * 1e9)) * 1000;
}

export function decodeTokPerSec(cfg: ArchConfig, gpu: GpuSpec, dtypeBytes = 2): number {
  const ms = decodeMsPerToken(cfg, gpu, dtypeBytes);
  return ms > 0 ? 1000 / ms : 0;
}

export function arithmeticIntensityDecode(cfg: ArchConfig, dtypeBytes = 2): number {
  const flops = 2 * activeParamsPerToken(cfg);
  const bytes = totalParams(cfg) * dtypeBytes + kvCacheFullCtx(cfg, dtypeBytes);
  return bytes > 0 ? flops / bytes : 0;
}

export function ridgePointFlopsPerByte(gpu: GpuSpec): number {
  return (gpu.fp16_tflops * 1e12) / (gpu.bandwidth_gbs * 1e9);
}

export function minGpuThatFits(cfg: ArchConfig, dtypeBytes = 2): GpuSpec | null {
  const mem = approxMemoryBytes(cfg, dtypeBytes);
  const sorted = [...GPUS].sort((a, b) => a.memory_gb - b.memory_gb);
  for (const g of sorted) {
    if (mem <= g.memory_gb * 1e9) return g;
  }
  return null;
}

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
  if (cfg.attention_type === "mla") return attnParamsMla(cfg);
  const q = cfg.d_model * (cfg.n_heads * cfg.head_dim);
  const k = cfg.d_model * (cfg.n_kv_heads * cfg.head_dim);
  const v = cfg.d_model * (cfg.n_kv_heads * cfg.head_dim);
  const o = cfg.n_heads * cfg.head_dim * cfg.d_model;
  return q + k + v + o;
}

function attnParamsMla(cfg: ArchConfig): number {
  const latent = mlaLatentDim(cfg);
  const rope = mlaRopeDim(cfg);
  const nope = cfg.head_dim - rope;
  const q = cfg.d_model * (cfg.n_heads * cfg.head_dim);
  const kvDown = cfg.d_model * latent;
  const kRope = cfg.d_model * rope;
  const kUp = latent * (cfg.n_heads * nope);
  const vUp = latent * (cfg.n_heads * cfg.head_dim);
  const o = (cfg.n_heads * cfg.head_dim) * cfg.d_model;
  return q + kvDown + kRope + kUp + vUp + o;
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
  const perLayer = cfg.attention_type === "mla"
    ? 2 * cfg.n_heads * cfg.head_dim
    : (cfg.n_heads * cfg.head_dim) + (cfg.n_kv_heads * cfg.head_dim);
  return cfg.n_layers * perLayer;
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

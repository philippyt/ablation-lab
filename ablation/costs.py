from __future__ import annotations
from typing import TypedDict
from .config import ArchConfig

def _round_to_multiple(x: float, m: int) -> int:
    return int((int(x) + m - 1) // m * m)

def swiglu_hidden(cfg: ArchConfig) -> int:
    return _round_to_multiple((2 / 3) * cfg.ffn_mult * cfg.d_model, 64)

def _norm_params(cfg: ArchConfig) -> int:
    return cfg.d_model * (2 if cfg.norm == "layernorm" else 1)

def _attn_params(cfg: ArchConfig) -> int:
    if cfg.attention_type == "mla":
        return _attn_params_mla(cfg)
    q = cfg.d_model * (cfg.n_heads * cfg.head_dim)
    k = cfg.d_model * (cfg.n_kv_heads * cfg.head_dim)
    v = cfg.d_model * (cfg.n_kv_heads * cfg.head_dim)
    o = (cfg.n_heads * cfg.head_dim) * cfg.d_model
    return q + k + v + o

def _attn_params_mla(cfg: ArchConfig) -> int:
    latent = _mla_latent_dim(cfg)
    rope = _mla_rope_dim(cfg)
    nope = cfg.head_dim - rope
    q = cfg.d_model * (cfg.n_heads * cfg.head_dim)
    kv_down = cfg.d_model * latent
    k_rope = cfg.d_model * rope
    k_up = latent * (cfg.n_heads * nope)
    v_up = latent * (cfg.n_heads * cfg.head_dim)
    o = (cfg.n_heads * cfg.head_dim) * cfg.d_model
    return q + kv_down + k_rope + k_up + v_up + o

def _ffn_params_dense(cfg: ArchConfig) -> int:
    h = swiglu_hidden(cfg)
    return 3 * cfg.d_model * h

def _ffn_params_moe(cfg: ArchConfig) -> int:
    h = swiglu_hidden(cfg)
    experts = cfg.n_experts * (3 * cfg.d_model * h)
    router = cfg.d_model * cfg.n_experts
    return experts + router

def _ffn_active_moe(cfg: ArchConfig) -> int:
    h = swiglu_hidden(cfg)
    return cfg.top_k * (3 * cfg.d_model * h) + cfg.d_model * cfg.n_experts

def _block_params(cfg: ArchConfig) -> int:
    norms = (2 * _norm_params(cfg)) if cfg.enable_norms else 0
    attn = _attn_params(cfg) if cfg.enable_attention else 0
    ffn = (_ffn_params_dense(cfg) if cfg.ffn == "dense" else _ffn_params_moe(cfg)) if cfg.enable_ffn else 0
    return norms + attn + ffn

def _block_active_params(cfg: ArchConfig) -> int:
    norms = (2 * _norm_params(cfg)) if cfg.enable_norms else 0
    attn = _attn_params(cfg) if cfg.enable_attention else 0
    ffn = (_ffn_params_dense(cfg) if cfg.ffn == "dense" else _ffn_active_moe(cfg)) if cfg.enable_ffn else 0
    return norms + attn + ffn

def _pos_params(cfg: ArchConfig) -> int:
    if cfg.pos_embed in ("absolute", "sinusoidal"):
        return cfg.ctx * cfg.d_model if cfg.pos_embed == "absolute" else 0
    return 0

def _qk_norm_params(cfg: ArchConfig) -> int:
    if not cfg.qk_norm:
        return 0
    if cfg.attention_type == "mla":
        per_layer = 2 * cfg.n_heads * cfg.head_dim
    else:
        per_layer = (cfg.n_heads * cfg.head_dim) + (cfg.n_kv_heads * cfg.head_dim)
    return cfg.n_layers * per_layer

def total_params(cfg: ArchConfig) -> int:
    embed = cfg.vocab_size * cfg.d_model
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model
    blocks = cfg.n_layers * _block_params(cfg)
    final_norm = _norm_params(cfg) if cfg.enable_norms else 0
    qkn = _qk_norm_params(cfg) if cfg.enable_attention else 0
    return embed + head + blocks + final_norm + _pos_params(cfg) + qkn

def active_params_per_token(cfg: ArchConfig) -> int:
    embed = cfg.vocab_size * cfg.d_model
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model
    blocks = cfg.n_layers * _block_active_params(cfg)
    final_norm = _norm_params(cfg) if cfg.enable_norms else 0
    qkn = _qk_norm_params(cfg) if cfg.enable_attention else 0
    return embed + head + blocks + final_norm + _pos_params(cfg) + qkn

def _mla_latent_dim(cfg: ArchConfig) -> int:
    return max(8, cfg.d_model // 8)


def _mla_rope_dim(cfg: ArchConfig) -> int:
    return cfg.head_dim // 2


def kv_cache_per_token(cfg: ArchConfig, dtype_bytes: int = 2) -> int:
    if not cfg.enable_attention:
        return 0
    if cfg.attention_type == "mla":
        return cfg.n_layers * (_mla_latent_dim(cfg) + _mla_rope_dim(cfg)) * dtype_bytes
    return 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * dtype_bytes

def kv_cache_full_ctx(cfg: ArchConfig, dtype_bytes: int = 2) -> int:
    return kv_cache_per_token(cfg, dtype_bytes) * cfg.ctx

def approx_memory_bytes(cfg: ArchConfig, dtype_bytes: int = 2, batch_size: int = 1) -> int:
    weights = total_params(cfg) * dtype_bytes
    kv = kv_cache_full_ctx(cfg, dtype_bytes)
    activations = batch_size * cfg.ctx * cfg.d_model * dtype_bytes * 4
    return weights + kv + activations

class CostPanel(TypedDict):
    total_params: int
    active_params_per_token: int
    kv_cache_per_token: int
    kv_cache_full_ctx: int
    approx_memory_bytes: int
    swiglu_hidden: int
    dtype_bytes: int

def cost_panel(cfg: ArchConfig, dtype_bytes: int = 2, batch_size: int = 1) -> CostPanel:
    return CostPanel(
        total_params=total_params(cfg),
        active_params_per_token=active_params_per_token(cfg),
        kv_cache_per_token=kv_cache_per_token(cfg, dtype_bytes),
        kv_cache_full_ctx=kv_cache_full_ctx(cfg, dtype_bytes),
        approx_memory_bytes=approx_memory_bytes(cfg, dtype_bytes, batch_size),
        swiglu_hidden=swiglu_hidden(cfg),
        dtype_bytes=dtype_bytes,
    )
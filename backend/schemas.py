from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

class ArchConfigIn(BaseModel):
    vocab_size: int = Field(..., ge=1)
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 8
    head_dim: int = 16
    ctx: int = 256
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    ffn: Literal["dense", "moe"] = "dense"
    ffn_mult: int = 4
    n_experts: int = 4
    top_k: int = 1
    tie_embeddings: bool = True
    pos_embed: Literal["rope", "alibi", "absolute", "sinusoidal", "nope"] = "rope"
    activation: Literal["silu", "gelu", "relu2"] = "silu"
    qk_norm: bool = False
    attention_type: Literal["standard", "mla"] = "standard"
    enable_attention: bool = True
    enable_ffn: bool = True
    enable_norms: bool = True

class CostPanelOut(BaseModel):
    total_params: int
    active_params_per_token: int
    kv_cache_per_token: int
    kv_cache_full_ctx: int
    approx_memory_bytes: int
    swiglu_hidden: int
    dtype_bytes: int

class ManifestEntryOut(BaseModel):
    hash: str
    name: Optional[str] = None
    attention_kind: str
    val_loss: float
    arch_config: dict
    trained_at: int

class SurgeryIn(BaseModel):
    disabled_heads: dict[int, list[int]] = Field(default_factory=dict)
    disabled_attn_layers: list[int] = Field(default_factory=list)
    disabled_ffn_layers: list[int] = Field(default_factory=list)
    disabled_norm_layers: list[int] = Field(default_factory=list)

class KVDegradationIn(BaseModel):
    quant: Literal["none", "int8", "int4", "int2"] = "none"
    window: Optional[int] = Field(default=None, ge=4, le=512)
    sink_mode: Literal["normal", "keep_sinks", "drop_sinks"] = "normal"
    sink_tokens: int = Field(default=4, ge=0, le=16)
    drop_prob: float = Field(default=0.0, ge=0.0, le=0.9)

class GenerateIn(BaseModel):
    config: ArchConfigIn
    prompt: str
    max_tokens: int = Field(default=200, ge=1, le=5000)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=1024)
    instrument: bool = False
    surgery: Optional[SurgeryIn] = None
    kv: Optional[KVDegradationIn] = None

class RaceIn(BaseModel):
    config_a: ArchConfigIn
    config_b: ArchConfigIn
    surgery_a: Optional[SurgeryIn] = None
    surgery_b: Optional[SurgeryIn] = None
    kv_a: Optional[KVDegradationIn] = None
    kv_b: Optional[KVDegradationIn] = None
    prompt: str
    max_tokens: int = Field(default=400, ge=1, le=2000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=1024)

class InspectIn(BaseModel):
    config: ArchConfigIn
    text: str
    surgery: Optional[SurgeryIn] = None
    kv: Optional[KVDegradationIn] = None
    include_attn: bool = True
    include_logit_lens: bool = True
    max_attn_tokens: int = Field(default=32, ge=2, le=64)

class CostRequest(BaseModel):
    config: ArchConfigIn
    dtype_bytes: int = 2
    batch_size: int = 1
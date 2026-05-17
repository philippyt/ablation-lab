from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, asdict, fields
from typing import Literal, Any

AttentionKind = Literal["mha", "gqa", "mqa"]
AttentionType = Literal["standard", "mla"]
NormKind = Literal["layernorm", "rmsnorm"]
FFNKind = Literal["dense", "moe"]
PosEmbedKind = Literal["rope", "alibi", "absolute", "sinusoidal", "nope"]
ActivationKind = Literal["silu", "gelu", "relu2"]

@dataclass(frozen=True)
class ArchConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 8
    head_dim: int = 16
    ctx: int = 128
    norm: NormKind = "layernorm"
    ffn: FFNKind = "dense"
    ffn_mult: int = 4
    n_experts: int = 4
    top_k: int = 1
    tie_embeddings: bool = True
    pos_embed: PosEmbedKind = "rope"
    activation: ActivationKind = "silu"
    qk_norm: bool = False
    attention_type: AttentionType = "standard"
    enable_attention: bool = True
    enable_ffn: bool = True
    enable_norms: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}")
        if self.head_dim * self.n_heads != self.d_model:
            raise ValueError(f"head_dim*n_heads ({self.head_dim*self.n_heads}) != d_model ({self.d_model})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads={self.n_heads} not divisible by n_kv_heads={self.n_kv_heads}")
        if self.ffn == "moe" and self.top_k > self.n_experts:
            raise ValueError(f"top_k={self.top_k} > n_experts={self.n_experts}")

    @property
    def attention_kind(self) -> AttentionKind:
        if self.n_kv_heads == self.n_heads:
            return "mha"
        if self.n_kv_heads == 1:
            return "mqa"
        return "gqa"

@dataclass
class TrainConfig:
    seed: int = 1337
    lr: float = 3e-4
    batch_size: int = 64
    steps: int = 3000
    warmup_steps: int = 100
    eval_interval: int = 200
    eval_iters: int = 50
    dropout: float = 0.0
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dataset_path: str = "data/tinystories.txt"
    device: str = "auto"

ARCH_FIELDS: tuple[str, ...] = (
    "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim",
    "ctx", "norm", "ffn", "ffn_mult", "n_experts", "top_k", "tie_embeddings",
    "pos_embed", "activation", "qk_norm", "attention_type",
    "enable_attention", "enable_ffn", "enable_norms",
)

KIND_A_FIELDS: tuple[str, ...] = (
    "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim",
    "ctx", "norm", "ffn", "ffn_mult", "n_experts", "top_k",
    "pos_embed_kind_a", "activation", "qk_norm", "attention_type",
)

KIND_B_FIELDS: tuple[str, ...] = (
    "enable_attention", "enable_ffn", "enable_norms",
    "tie_embeddings", "pos_embed",
)

def _pos_embed_kind_a(pos_embed: str) -> str:
    return "absolute" if pos_embed == "absolute" else "rope"

def _hashable_dict(cfg: ArchConfig) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k in KIND_A_FIELDS:
        if k == "pos_embed_kind_a":
            d[k] = _pos_embed_kind_a(cfg.pos_embed)
        else:
            d[k] = getattr(cfg, k)
    if d["ffn"] != "moe":
        d.pop("n_experts")
        d.pop("top_k")
    if d["attention_type"] == "mla":
        d.pop("n_kv_heads")
    return d

def arch_hash(cfg: ArchConfig) -> str:
    blob = json.dumps(_hashable_dict(cfg), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]

def arch_config_from_dict(d: dict[str, Any]) -> ArchConfig:
    known = {f.name for f in fields(ArchConfig)}
    return ArchConfig(**{k: v for k, v in d.items() if k in known})

def train_config_from_dict(d: dict[str, Any]) -> TrainConfig:
    known = {f.name for f in fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in d.items() if k in known})

def arch_config_to_dict(cfg: ArchConfig) -> dict[str, Any]:
    return asdict(cfg)
from __future__ import annotations
import pytest
from ablation.config import ArchConfig
from ablation.costs import total_params, active_params_per_token
from ablation.model import TinyTransformer

VOCAB = 64

def _make_cfg(**overrides) -> ArchConfig:
    base = dict(
        vocab_size=VOCAB,
        d_model=128,
        n_layers=4,
        n_heads=8,
        n_kv_heads=8,
        head_dim=16,
        ctx=128,
        norm="layernorm",
        ffn="dense",
        ffn_mult=4,
        n_experts=4,
        top_k=1,
        tie_embeddings=True,
    )
    base.update(overrides)
    return ArchConfig(**base)

CONFIGS = {
    "baseline_mha_ln_dense": _make_cfg(),
    "gqa": _make_cfg(n_kv_heads=2),
    "mqa": _make_cfg(n_kv_heads=1),
    "rmsnorm": _make_cfg(norm="rmsnorm"),
    "moe": _make_cfg(ffn="moe"),
    "gqa_rmsnorm_moe": _make_cfg(n_kv_heads=2, norm="rmsnorm", ffn="moe"),
    "untied_lm_head": _make_cfg(tie_embeddings=False),
    "pos_rope": _make_cfg(pos_embed="rope"),
    "activation_silu": _make_cfg(activation="silu"),
    "attention_standard": _make_cfg(attention_type="standard"),
    "enable_default": _make_cfg(enable_attention=True, enable_ffn=True, enable_norms=True),
}

@pytest.mark.parametrize("name", list(CONFIGS.keys()))
def test_total_params_matches_instantiated_model(name: str) -> None:
    cfg = CONFIGS[name]
    model = TinyTransformer(cfg)
    real = sum(p.numel() for p in model.parameters())
    predicted = total_params(cfg)
    assert real == predicted, f"[{name}] real {real} != predicted {predicted}"

def test_moe_active_smaller_than_total() -> None:
    cfg = CONFIGS["moe"]
    assert active_params_per_token(cfg) < total_params(cfg)

def test_dense_active_equals_total() -> None:
    cfg = CONFIGS["baseline_mha_ln_dense"]
    assert active_params_per_token(cfg) == total_params(cfg)
from __future__ import annotations
import torch
import pytest
from ablation.config import ArchConfig
from ablation.model import TinyTransformer, _quantize_kv


def _make_model(attention_type: str = "standard", n_kv_heads: int = 2) -> TinyTransformer:
    torch.manual_seed(42)
    cfg = ArchConfig(
        vocab_size=72,
        d_model=64,
        n_layers=3,
        n_heads=4,
        n_kv_heads=n_kv_heads if attention_type == "standard" else 4,
        head_dim=16,
        ctx=32,
        norm="rmsnorm",
        attention_type=attention_type,
        qk_norm=False,
    )
    m = TinyTransformer(cfg)
    m.eval()
    return m


def test_quantize_kv_levels():
    x = torch.randn(1, 2, 8, 16)
    q2 = _quantize_kv(x, 2)
    # Each 16-element vector should have at most 4 unique values
    for t in range(8):
        assert q2[0, 0, t].unique().numel() <= 4
    q4 = _quantize_kv(x, 4)
    assert (q4 - x).abs().mean() < (q2 - x).abs().mean()


@pytest.mark.parametrize("attention_type", ["standard", "mla"])
def test_surgical_head_ablation_matches_layer_disable(attention_type: str):
    m = _make_model(attention_type=attention_type)
    idx = torch.randint(0, 72, (1, 10))
    with torch.no_grad():
        base, _ = m(idx)
        all_heads_off, _ = m(idx, runtime={"surgery": {"disabled_heads": {1: [0, 1, 2, 3]}}})
        layer_off, _ = m(idx, runtime={"surgery": {"disabled_attn_layers": [1]}})
        one_head_off, _ = m(idx, runtime={"surgery": {"disabled_heads": {1: [0]}}})
    assert torch.allclose(all_heads_off, layer_off, atol=1e-5)
    assert not torch.allclose(base, one_head_off, atol=1e-5)


@pytest.mark.parametrize("attention_type", ["standard", "mla"])
def test_kv_degradation_modes_and_window(attention_type: str):
    m = _make_model(attention_type=attention_type)
    idx = torch.randint(0, 72, (1, 14))
    for kv in (
        {"quant": "int4"},
        {"quant": "int2"},
        {"window": 8, "sink_mode": "normal"},
        {"window": 8, "sink_mode": "keep_sinks", "sink_tokens": 2},
        {"window": 12, "sink_mode": "drop_sinks", "sink_tokens": 2},
        {"drop_prob": 0.4},
    ):
        caches = [{} for _ in range(m.cfg.n_layers)]
        with torch.no_grad():
            logits, _ = m(idx[:, :10], kv_caches=caches, runtime={"kv": kv})
            logits2, _ = m(idx[:, 10:11], kv_caches=caches, runtime={"kv": kv})
        assert torch.isfinite(logits).all()
        assert torch.isfinite(logits2).all()
        if "window" in kv:
            assert caches[0]["len"] <= kv["window"]


@pytest.mark.parametrize("attention_type", ["standard", "mla"])
def test_step_metrics_collected(attention_type: str):
    m = _make_model(attention_type=attention_type)
    idx = torch.randint(0, 72, (1, 8))
    metrics: dict = {}
    with torch.no_grad():
        m(idx, runtime={"step_metrics": metrics})
    assert len(metrics["attn_logit_max"]) == m.cfg.n_layers
    assert len(metrics["attn_entropy"]) == m.cfg.n_layers
    assert len(metrics["resid_norm"]) == m.cfg.n_layers
    assert len(metrics["head_norms"]) == m.cfg.n_layers
    assert len(metrics["head_norms"][0]) == m.cfg.n_heads

from __future__ import annotations
import json
import torch
from fastapi.testclient import TestClient
from ablation.config import ArchConfig, arch_config_to_dict, arch_hash
from ablation.model import TinyTransformer
from ablation.tokenizer import CharTokenizer
import ablation.checkpoints as ckpts_mod
import backend.inference as inf_mod
from backend.app import app


def _save_dummy_ckpt(tmp_dir, cfg: ArchConfig, seed: int) -> str:
    torch.manual_seed(seed)
    tok = CharTokenizer.build_from_text("Once upon a time there was a little cat named Sam. \n!?\"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    cfg = ArchConfig(**{**arch_config_to_dict(cfg), "vocab_size": tok.vocab_size})
    h = arch_hash(cfg)
    model = TinyTransformer(cfg)
    path = tmp_dir / f"{h}.pt"
    torch.save(
        {
            "arch_config": arch_config_to_dict(cfg),
            "tokenizer_state": tok.state(),
            "state_dict": model.state_dict(),
            "final_val_loss": 0.95,
        },
        path,
    )
    return h, cfg


def test_generate_and_race_sse_e2e(tmp_path, monkeypatch):
    monkeypatch.setattr(ckpts_mod, "CHECKPOINTS_DIR", tmp_path)
    monkeypatch.setattr(ckpts_mod, "MANIFEST_PATH", tmp_path / "manifest.json")
    inf_mod._lru.clear()

    base_a = ArchConfig(vocab_size=72, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, ctx=64, norm="rmsnorm", qk_norm=False, tie_embeddings=False)
    base_b = ArchConfig(vocab_size=72, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, ctx=64, norm="rmsnorm", qk_norm=True, tie_embeddings=False)
    ha, cfg_a = _save_dummy_ckpt(tmp_path, base_a, seed=1)
    hb, cfg_b = _save_dummy_ckpt(tmp_path, base_b, seed=2)

    manifest = {
        "entries": {
            ha: {"hash": ha, "name": "A", "attention_kind": "gqa", "val_loss": 0.94, "arch_config": arch_config_to_dict(cfg_a), "trained_at": 1, "path": f"{ha}.pt"},
            hb: {"hash": hb, "name": "B", "attention_kind": "gqa", "val_loss": 0.95, "arch_config": arch_config_to_dict(cfg_b), "trained_at": 2, "path": f"{hb}.pt"},
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    client = TestClient(app)

    # 1. Single generate with surgery + kv degradation + instrumentation
    res = client.post(
        "/api/generate",
        json={
            "config": arch_config_to_dict(cfg_a),
            "prompt": "Once upon a time",
            "max_tokens": 12,
            "temperature": 0.0,
            "instrument": True,
            "surgery": {"disabled_heads": {"0": [1, 3]}, "disabled_ffn_layers": [1]},
            "kv": {"quant": "int4", "window": 32, "sink_mode": "keep_sinks", "sink_tokens": 2, "drop_prob": 0.1},
        },
    )
    assert res.status_code == 200
    body = res.text
    assert "event: start" in body
    assert "event: token" in body
    assert '"lm":' in body
    assert "event: instr" in body
    assert '"head_activity":' in body
    assert "event: done" in body

    # 2. Twin race A vs B
    race = client.post(
        "/api/race",
        json={
            "config_a": arch_config_to_dict(cfg_a),
            "config_b": arch_config_to_dict(cfg_b),
            "prompt": "Once upon a time",
            "max_tokens": 12,
            "temperature": 0.0,
            "kv_b": {"quant": "int2"},
        },
    )
    assert race.status_code == 200
    rbody = race.text
    assert "event: race_start" in rbody
    assert "event: race_step" in rbody
    assert '"top_a":' in rbody
    assert "event: race_done" in rbody

    # 3. Inspect endpoint: real attention weights + logit lens
    insp = client.post(
        "/api/inspect",
        json={
            "config": arch_config_to_dict(cfg_a),
            "text": "Once upon a time",
            "surgery": {"disabled_heads": {"0": [1]}},
            "include_attn": True,
            "include_logit_lens": True,
            "max_attn_tokens": 16,
        },
    )
    assert insp.status_code == 200
    data = insp.json()
    assert len(data["tokens"]) == 16
    assert len(data["attn_weights"]) == cfg_a.n_layers
    assert len(data["attn_weights"][0]) == cfg_a.n_heads
    # Disabled head 1 in layer 0 should be all zeros
    assert all(v == 0.0 for row in data["attn_weights"][0][1] for v in row)
    # Logit lens has Embed + each layer
    assert len(data["logit_lens"]) == cfg_a.n_layers + 1
    assert data["logit_lens"][0]["stage"] == "Embed"
    assert len(data["logit_lens"][-1]["top"]) == 5

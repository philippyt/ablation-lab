from __future__ import annotations
import itertools
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "training" / "configs"

BASE_ARCH = {
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 8,
    "head_dim": 16,
    "ctx": 256,
    "ffn_mult": 4,
    "tie_embeddings": True,
}

BASE_TRAIN = {
    "seed": 1337,
    "lr": 3e-4,
    "batch_size": 64,
    "steps": 3000,
    "warmup_steps": 100,
    "eval_interval": 250,
    "eval_iters": 40,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "dataset_path": "data/tinystories.txt",
    "device": "auto",
}

NORMS = ["layernorm", "rmsnorm"]
ATTENTIONS = [
    ("mha", {"n_kv_heads": 8, "attention_type": "standard"}),
    ("gqa", {"n_kv_heads": 2, "attention_type": "standard"}),
    ("mqa", {"n_kv_heads": 1, "attention_type": "standard"}),
    ("mla", {"n_kv_heads": 1, "attention_type": "mla"}),
]
FFNS = [
    ("dense", {"ffn": "dense"}),
    ("moe", {"ffn": "moe", "n_experts": 8, "top_k": 2}),
]
ACTIVATIONS = ["silu", "gelu", "relu2"]
QK_NORMS = [False, True]
POS_EMBEDS = ["rope", "absolute"]

def short_norm(n: str) -> str:
    return "ln" if n == "layernorm" else "rms"

def short_act(a: str) -> str:
    return {"silu": "silu", "gelu": "gelu", "relu2": "rl2"}[a]

def short_pos(p: str) -> str:
    return "rope" if p == "rope" else "abs"

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in OUT_DIR.glob("*.json")}
    written = 0
    pruned = 0
    for norm, (attn_name, attn_patch), (ffn_name, ffn_patch), act, qkn, pos in itertools.product(
        NORMS, ATTENTIONS, FFNS, ACTIVATIONS, QK_NORMS, POS_EMBEDS
    ):
        # PRUNE: MLA × absolute pos is architecturally incoherent
        if attn_name == "mla" and pos == "absolute":
            pruned += 1
            continue

        arch = dict(BASE_ARCH)
        arch.update(attn_patch)
        arch.update(ffn_patch)
        arch["norm"] = norm
        arch["activation"] = act
        arch["qk_norm"] = qkn
        arch["pos_embed"] = pos

        cfg_id = f"{short_norm(norm)}_{attn_name}_{ffn_name}_{short_act(act)}_qkn{'1' if qkn else '0'}_{short_pos(pos)}"
        name = f"{norm}+{attn_name}+{ffn_name}+{act}+qkn={'on' if qkn else 'off'}+{pos}"
        note = "auto-generated Kind-A config"
        cfg = {"name": name, "_note": note, "arch": arch, "train": dict(BASE_TRAIN)}
        out_path = OUT_DIR / f"{cfg_id}.json"
        with out_path.open("w") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        written += 1

    # bonus: small MoE (4 experts, top-1) on rms+gqa+silu+qkn=off+rope baseline
    arch = dict(BASE_ARCH)
    arch.update({"n_kv_heads": 2, "attention_type": "standard"})
    arch.update({"ffn": "moe", "n_experts": 4, "top_k": 1})
    arch["norm"] = "rmsnorm"
    arch["activation"] = "silu"
    arch["qk_norm"] = False
    arch["pos_embed"] = "rope"
    cfg = {
        "name": "rmsnorm+gqa+moe_small(4,1)+silu+qkn=off+rope",
        "_note": "bonus: small-MoE (4 experts, top-1) to demonstrate total/active gap shrinking toward dense",
        "arch": arch,
        "train": dict(BASE_TRAIN),
    }
    out_path = OUT_DIR / "rms_gqa_moe-small_silu_qkn0_rope.json"
    with out_path.open("w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    written += 1

    print(f"wrote {written} configs to {OUT_DIR}  (pruned {pruned} MLA×absolute combos)")

if __name__ == "__main__":
    main()

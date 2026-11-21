from __future__ import annotations
try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
import torch
from .config import ArchConfig, arch_hash, _pos_embed_kind_a

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = Path(os.environ.get("ABLATION_CHECKPOINTS_DIR", str(REPO_ROOT / "checkpoints")))
MANIFEST_PATH = CHECKPOINTS_DIR / "manifest.json"
LOCK_PATH = CHECKPOINTS_DIR / ".lock"

def _ensure_dirs() -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

def checkpoint_path(h: str) -> Path:
    return CHECKPOINTS_DIR / f"{h}.pt"

def save_checkpoint(
    *,
    arch_config: ArchConfig,
    state_dict: dict,
    tokenizer_state: dict,
    final_val_loss: float,
    training_log: list[dict] | None = None,
    name: str | None = None,
    source_config: str | None = None,
    seed: int | None = None,
) -> str:
    _ensure_dirs()
    h = arch_hash(arch_config)
    payload = {
        "arch_hash": h,
        "arch_config": asdict(arch_config),
        "state_dict": state_dict,
        "tokenizer_state": tokenizer_state,
        "final_val_loss": final_val_loss,
        "training_log": training_log or [],
    }
    target = checkpoint_path(h)
    tmp = target.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)
    _update_manifest(h, arch_config, final_val_loss, name, source_config=source_config, seed=seed)
    return h

def update_entry_field(h: str, key: str, value: Any) -> None:
    _ensure_dirs()
    LOCK_PATH.touch(exist_ok=True)
    with LOCK_PATH.open("w") as lf:
        if fcntl is not None:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            data = read_manifest()
            entries = data.get("entries", {})
            if h in entries:
                entries[h][key] = value
                data["entries"] = entries
                data["updated_at"] = int(time.time())
                tmp = MANIFEST_PATH.with_suffix(".json.tmp")
                with tmp.open("w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                os.replace(tmp, MANIFEST_PATH)
        finally:
            if fcntl is not None:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

def load_checkpoint(h: str, map_location: str | torch.device = "cpu") -> dict:
    path = checkpoint_path(h)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint for hash {h} at {path}")
    return torch.load(path, map_location=map_location, weights_only=False)

def read_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"entries": {}, "updated_at": None}
    with MANIFEST_PATH.open() as f:
        return json.load(f)

def _update_manifest(h: str, cfg: ArchConfig, val_loss: float, name: str | None, source_config: str | None = None, seed: int | None = None) -> None:
    _ensure_dirs()
    LOCK_PATH.touch(exist_ok=True)
    with LOCK_PATH.open("w") as lf:
        if fcntl is not None:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            data = read_manifest()
            entries = data.get("entries", {})
            prev = entries.get(h, {})
            entries[h] = {
                "hash": h,
                "name": name,
                "arch_config": asdict(cfg),
                "attention_kind": cfg.attention_kind,
                "val_loss": float(val_loss),
                "path": f"checkpoints/{h}.pt",
                "trained_at": int(time.time()),
                "source_config": source_config or prev.get("source_config"),
                "seed": seed if seed is not None else prev.get("seed"),
                "qk_norm_divergence": prev.get("qk_norm_divergence", "unchecked"),
            }
            data["entries"] = entries
            data["updated_at"] = int(time.time())
            tmp = MANIFEST_PATH.with_suffix(".json.tmp")
            with tmp.open("w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp, MANIFEST_PATH)
        finally:
            if fcntl is not None:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

def find_entry_for_config(cfg: ArchConfig) -> dict | None:
    h = arch_hash(cfg)
    return read_manifest().get("entries", {}).get(h)

def list_entries() -> list[dict]:
    return list(read_manifest().get("entries", {}).values())

KIND_A_STRUCT_FIELDS: tuple[str, ...] = (
    "vocab_size", "d_model", "n_layers", "n_heads", "head_dim", "ctx", "ffn_mult",
)

def _kind_a_diff(req_cfg: ArchConfig, entry_arch: dict) -> list[str]:
    diffs: list[str] = []
    for f in KIND_A_STRUCT_FIELDS:
        if getattr(req_cfg, f) != entry_arch.get(f):
            diffs.append(f)
    if req_cfg.norm != entry_arch.get("norm"):
        diffs.append("norm")
    if req_cfg.attention_type != entry_arch.get("attention_type"):
        diffs.append("attention_type")
    elif req_cfg.attention_type != "mla":
        if req_cfg.n_kv_heads != entry_arch.get("n_kv_heads"):
            diffs.append("n_kv_heads")
    if req_cfg.ffn != entry_arch.get("ffn"):
        diffs.append("ffn")
    elif req_cfg.ffn == "moe":
        if req_cfg.n_experts != entry_arch.get("n_experts"):
            diffs.append("n_experts")
        if req_cfg.top_k != entry_arch.get("top_k"):
            diffs.append("top_k")
    if req_cfg.activation != entry_arch.get("activation"):
        diffs.append("activation")
    if bool(req_cfg.qk_norm) != bool(entry_arch.get("qk_norm")):
        diffs.append("qk_norm")
    if _pos_embed_kind_a(req_cfg.pos_embed) != _pos_embed_kind_a(entry_arch.get("pos_embed", "rope")):
        diffs.append("pos_embed")
    return diffs

def resolve_config(cfg: ArchConfig, hamming_ceiling: int = 2) -> dict:
    h = arch_hash(cfg)
    entries = read_manifest().get("entries", {})
    exact = entries.get(h)
    if exact is not None:
        return {
            "state": "green",
            "hash": h,
            "entry": exact,
            "differing_fields": [],
            "distance": 0,
        }
    best_entry = None
    best_diffs: list[str] = []
    best_dist = None
    for e in entries.values():
        diffs = _kind_a_diff(cfg, e["arch_config"])
        d = len(diffs)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_entry = e
            best_diffs = diffs
    if best_entry is None or best_dist is None or best_dist > hamming_ceiling:
        return {
            "state": "red",
            "hash": None,
            "entry": None,
            "differing_fields": best_diffs,
            "distance": -1 if best_dist is None else best_dist,
        }
    return {
        "state": "amber",
        "hash": best_entry["hash"],
        "entry": best_entry,
        "differing_fields": best_diffs,
        "distance": best_dist,
    }
from __future__ import annotations
import fcntl
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
import torch
from .config import ArchConfig, arch_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
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
    _update_manifest(h, arch_config, final_val_loss, name)
    return h

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

def _update_manifest(h: str, cfg: ArchConfig, val_loss: float, name: str | None) -> None:
    _ensure_dirs()
    LOCK_PATH.touch(exist_ok=True)
    with LOCK_PATH.open("w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            data = read_manifest()
            entries = data.get("entries", {})
            entries[h] = {
                "hash": h,
                "name": name,
                "arch_config": asdict(cfg),
                "attention_kind": cfg.attention_kind,
                "val_loss": float(val_loss),
                "path": f"checkpoints/{h}.pt",
                "trained_at": int(time.time()),
            }
            data["entries"] = entries
            data["updated_at"] = int(time.time())
            tmp = MANIFEST_PATH.with_suffix(".json.tmp")
            with tmp.open("w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp, MANIFEST_PATH)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

def find_entry_for_config(cfg: ArchConfig) -> dict | None:
    h = arch_hash(cfg)
    return read_manifest().get("entries", {}).get(h)

def list_entries() -> list[dict]:
    return list(read_manifest().get("entries", {}).values())
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ablation.checkpoints import resolve_config
from ablation.config import ArchConfig
from backend.schemas import ArchConfigIn

router = APIRouter()


@router.post("/api/resolve")
async def post_resolve(cfg_in: ArchConfigIn) -> dict:
    try:
        cfg = ArchConfig(**cfg_in.model_dump())
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    res = resolve_config(cfg)
    return {
        "state": res["state"],
        "resolved_hash": res["hash"],
        "differing_fields": res["differing_fields"],
        "distance": res["distance"],
        "resolved_entry": (
            {
                "hash": res["entry"]["hash"],
                "name": res["entry"].get("name"),
                "val_loss": res["entry"].get("val_loss"),
            }
            if res["entry"] is not None
            else None
        ),
    }

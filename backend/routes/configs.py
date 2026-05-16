from __future__ import annotations

from fastapi import APIRouter

from ablation.checkpoints import list_entries

router = APIRouter()


@router.get("/api/configs")
def get_configs() -> dict:
    return {"entries": list_entries()}

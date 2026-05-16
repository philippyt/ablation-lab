from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ablation.config import ArchConfig
from ablation.costs import cost_panel
from backend.schemas import CostRequest, CostPanelOut

router = APIRouter()


@router.post("/api/cost", response_model=CostPanelOut)
def post_cost(req: CostRequest) -> CostPanelOut:
    try:
        cfg = ArchConfig(**req.config.model_dump())
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    panel = cost_panel(cfg, dtype_bytes=req.dtype_bytes, batch_size=req.batch_size)
    return CostPanelOut(**panel)

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from ablation.checkpoints import resolve_config
from ablation.config import ArchConfig
from backend.inference import stream_generate
from backend.schemas import GenerateIn

router = APIRouter()


@router.post("/api/generate")
async def post_generate(req: GenerateIn) -> EventSourceResponse:
    try:
        cfg = ArchConfig(**req.config.model_dump())
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    resolution = resolve_config(cfg)
    if resolution["state"] == "red":
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_faithful_checkpoint",
                "message": "No trained checkpoint within Hamming ≤ 2 on Kind-A axes. Cost panel still shown.",
                "differing_fields": resolution["differing_fields"],
                "distance": resolution["distance"],
            },
        )

    async def event_iter():
        async for ev in stream_generate(
            requested_config=cfg,
            resolution=resolution,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        ):
            event = ev.pop("event")
            yield {"event": event, "data": json.dumps(ev)}

    return EventSourceResponse(
        event_iter(),
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

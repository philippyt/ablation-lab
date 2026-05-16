from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from ablation.checkpoints import find_entry_for_config
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

    if find_entry_for_config(cfg) is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_checkpoint",
                "message": "No trained checkpoint exists for this architecture. Pick a different combination of components.",
            },
        )

    async def event_iter():
        async for ev in stream_generate(
            config=cfg,
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

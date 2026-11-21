from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from ablation.checkpoints import resolve_config
from ablation.config import ArchConfig
from backend.inference import stream_generate, stream_race
from backend.schemas import GenerateIn, RaceIn

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
                "error": "no_matching_checkpoint",
                "message": "No trained checkpoint within 2 axes. Cost panel is still exact.",
                "differing_fields": resolution["differing_fields"],
                "distance": resolution["distance"],
            },
        )

    surgery_dict = req.surgery.model_dump() if req.surgery else None
    kv_dict = req.kv.model_dump() if req.kv else None

    async def event_iter():
        async for ev in stream_generate(
            requested_config=cfg,
            resolution=resolution,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            instrument=req.instrument,
            surgery=surgery_dict,
            kv=kv_dict,
        ):
            event = ev.pop("event")
            yield {"event": event, "data": json.dumps(ev)}

    return EventSourceResponse(
        event_iter(),
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@router.post("/api/race")
async def post_race(req: RaceIn) -> EventSourceResponse:
    try:
        cfg_a = ArchConfig(**req.config_a.model_dump())
        cfg_b = ArchConfig(**req.config_b.model_dump())
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    res_a = resolve_config(cfg_a)
    res_b = resolve_config(cfg_b)
    if res_a["state"] == "red" or res_b["state"] == "red":
        which = "A" if res_a["state"] == "red" else "B"
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_matching_checkpoint",
                "message": f"Model {which} has no trained checkpoint within 2 axes.",
            },
        )

    surgery_a = req.surgery_a.model_dump() if req.surgery_a else None
    surgery_b = req.surgery_b.model_dump() if req.surgery_b else None
    kv_a = req.kv_a.model_dump() if req.kv_a else None
    kv_b = req.kv_b.model_dump() if req.kv_b else None

    async def event_iter():
        async for ev in stream_race(
            config_a=cfg_a,
            resolution_a=res_a,
            config_b=cfg_b,
            resolution_b=res_b,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            surgery_a=surgery_a,
            surgery_b=surgery_b,
            kv_a=kv_a,
            kv_b=kv_b,
        ):
            event = ev.pop("event")
            yield {"event": event, "data": json.dumps(ev)}

    return EventSourceResponse(
        event_iter(),
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

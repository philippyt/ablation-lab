from __future__ import annotations
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONTENT_INDEX = REPO_ROOT / "content" / "index.json"

router = APIRouter()


@router.get("/api/content")
def get_content() -> dict:
    if not CONTENT_INDEX.exists():
        raise HTTPException(
            status_code=503,
            detail="content index not built; run python scripts/build_content_index.py",
        )
    return json.loads(CONTENT_INDEX.read_text(encoding="utf-8"))

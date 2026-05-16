from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from backend.routes import configs as configs_route
from backend.routes import content as content_route
from backend.routes import cost as cost_route
from backend.routes import generate as generate_route
REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
app = FastAPI(title="Ablation Lab")
app.include_router(configs_route.router)
app.include_router(cost_route.router)
app.include_router(generate_route.router)
app.include_router(content_route.router)

@app.get("/")
def root():
    index = DIST_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse(
        "<h1>Frontend not built</h1><p>Run <code>cd frontend && npm install && npm run build</code>.</p>",
        status_code=503,
    )

if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")
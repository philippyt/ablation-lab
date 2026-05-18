FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf_cache \
    HF_XET_HIGH_PERFORMANCE=1 \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<2.6"
RUN pip install \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27" \
    "pydantic>=2.6" \
    "jinja2>=3.1" \
    "sse-starlette>=2.0" \
    "huggingface-hub>=1.15.0" \
    "hf-transfer>=0.1.9"

COPY ablation/ ./ablation/
COPY backend/ ./backend/
COPY content/ ./content/
COPY training/ ./training/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY pyproject.toml ./
COPY --from=frontend /build/dist ./frontend/dist

RUN python scripts/build_content_index.py
RUN cp scripts/space_start.sh ./space_start.sh && chmod +x ./space_start.sh

EXPOSE 7860
CMD ["./space_start.sh"]

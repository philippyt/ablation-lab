#!/bin/sh
set -e

echo "[space_start] fetching checkpoints from philippyt/ablation-lab-checkpoints..."
SNAP=$(python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="philippyt/ablation-lab-checkpoints",
    repo_type="model",
    allow_patterns=["*.pt", "manifest.json"],
)
print(path)
PY
)

if [ -z "$SNAP" ] || [ ! -d "$SNAP" ]; then
  echo "[space_start] snapshot_download failed (got: '$SNAP')" >&2
  exit 1
fi

export ABLATION_CHECKPOINTS_DIR="$SNAP"
echo "[space_start] checkpoints at $ABLATION_CHECKPOINTS_DIR"
echo "[space_start] starting uvicorn on :${PORT:-7860}"
exec uvicorn backend.app:app --host 0.0.0.0 --port "${PORT:-7860}"

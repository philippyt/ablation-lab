#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FORCE=""
STEPS_ARG=""
EVAL_ARG=""
PATTERN="${PATTERN:-*.json}"
SKIP_DIVERGENCE_CHECK=""
MAX_RETRIES=2

usage() {
  cat <<EOF
usage: scripts/train_all.sh [--force] [--steps N] [--eval-interval N] [--pattern GLOB] [--skip-divergence-check]

  --force                    retrain configs whose hash is already in the manifest
  --steps N                  override step count for every config (smoke test)
  --eval-interval N          override eval interval
  --pattern GLOB             only train configs matching glob (relative to training/configs/)
  --skip-divergence-check    do NOT run qk_norm divergence check after training (use only for smoke tests)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE="--force"; shift ;;
    --steps) STEPS_ARG="--steps $2"; shift 2 ;;
    --eval-interval) EVAL_ARG="--eval-interval $2"; shift 2 ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    --skip-divergence-check) SKIP_DIVERGENCE_CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

shopt -s nullglob
configs=(training/configs/$PATTERN)
shopt -u nullglob

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "no configs match pattern: training/configs/$PATTERN" >&2
  exit 1
fi

echo "[train_all] ${#configs[@]} config(s) matching '$PATTERN'"
for cfg in "${configs[@]}"; do
  echo "[train_all] === $cfg ==="
  python -m training.train --config "$cfg" $FORCE $STEPS_ARG $EVAL_ARG
done

if [[ -n "$SKIP_DIVERGENCE_CHECK" ]]; then
  echo "[train_all] skipping qk_norm divergence check (--skip-divergence-check)"
  echo "[train_all] done."
  exit 0
fi

echo ""
echo "[train_all] ============================================="
echo "[train_all] qk_norm divergence check (automated gate)"
echo "[train_all] ============================================="

for attempt in $(seq 0 $MAX_RETRIES); do
  echo "[train_all] divergence check (pass $((attempt + 1)) of $((MAX_RETRIES + 1)))..."
  fails_file="$(mktemp)"
  set +e
  python -m scripts.qk_norm_divergence_check --emit-fails > "$fails_file" 2>&1
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    echo "[train_all] all qk_norm pairs differentiated ✓"
    rm -f "$fails_file"
    break
  fi
  fail_count=$(wc -l < "$fails_file" | tr -d ' ')
  echo "[train_all] $fail_count qk_norm pair(s) collapsed."
  if [[ $attempt -eq $MAX_RETRIES ]]; then
    echo "[train_all] retry budget exhausted — marking remaining fails in manifest. Configs left collapsed:"
    cat "$fails_file"
    rm -f "$fails_file"
    break
  fi
  new_seed=$((1338 + attempt))
  echo "[train_all] retrying collapsed configs from scratch with seed=$new_seed..."
  while IFS= read -r failed_cfg; do
    if [[ -n "$failed_cfg" && -f "$failed_cfg" ]]; then
      echo "[train_all] retrying: $failed_cfg (seed=$new_seed)"
      python -m training.train --config "$failed_cfg" --force --seed $new_seed
    fi
  done < "$fails_file"
  rm -f "$fails_file"
done

echo "[train_all] done."

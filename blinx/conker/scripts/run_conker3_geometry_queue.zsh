#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%Y-%m-%d)"
LOG_PATH="$OUT_DIR/conker3_geometry_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

JSON_PATH="$OUT_DIR/conker3_window4_scale_5p0_steps1000_geometry_audit_seed42_${STAMP}.json"

echo "starting conker-3 geometry queue" | tee "$LOG_PATH"

if [[ -f "$JSON_PATH" ]]; then
  echo "skip landed geometry audit" | tee -a "$LOG_PATH"
  exit 0
fi

python3 conker/scripts/run_conker3_geometry_audit.py \
  --data-root "$DATA_ROOT" \
  --profile pilot \
  --variant window4 \
  --scale 5.0 \
  --steps 1000 \
  --seed 42 \
  --eval-batches 8 \
  --top-k 8 \
  --bits 3 4 6 \
  --json "$JSON_PATH" 2>&1 | tee -a "$LOG_PATH"

echo "conker-3 geometry queue complete" | tee -a "$LOG_PATH"

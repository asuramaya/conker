#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%Y-%m-%d)"
LOG_PATH="$OUT_DIR/conker3_mixed_quant_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

JSON_PATH="$OUT_DIR/conker3_window4_scale_5p0_steps1500_mixed_quant_audit_seed42_${STAMP}.json"

echo "starting conker-3 mixed quant queue" | tee "$LOG_PATH"

if [[ -f "$JSON_PATH" ]]; then
  echo "skip landed mixed quant audit" | tee -a "$LOG_PATH"
  exit 0
fi

python3 conker/scripts/run_conker3_mixed_quant_audit.py \
  --data-root "$DATA_ROOT" \
  --profile pilot \
  --scale 5.0 \
  --steps 1500 \
  --seed 42 \
  --eval-batches 8 \
  --json "$JSON_PATH" 2>&1 | tee -a "$LOG_PATH"

echo "conker-3 mixed quant queue complete" | tee -a "$LOG_PATH"

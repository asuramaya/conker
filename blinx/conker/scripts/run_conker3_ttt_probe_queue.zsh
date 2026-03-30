#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%Y-%m-%d)"
LOG_PATH="$OUT_DIR/conker3_ttt_probe_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

JSON_PATH="$OUT_DIR/conker3_window4_scale_10p0_steps1500_half_life_16_ttt_probe_seed42_${STAMP}.json"

if [[ -f "$JSON_PATH" ]]; then
  echo "skip landed conker-3 ttt probe" | tee "$LOG_PATH"
  exit 0
fi

echo "starting conker-3 ttt probe queue" | tee "$LOG_PATH"
python3 conker/scripts/run_conker3_ttt_probe.py \
  --data-root "$DATA_ROOT" \
  --profile pilot \
  --scale 10.0 \
  --steps 1500 \
  --seed 42 \
  --linear-half-life-max 16 \
  --chunk-len 64 \
  --eval-chunks 32 \
  --group-count 16 \
  --gate-span 0.5 \
  --gate-l2 1e-3 \
  --quant-bits 6 \
  --json "$JSON_PATH" 2>&1 | tee -a "$LOG_PATH"
echo "conker-3 ttt probe queue complete" | tee -a "$LOG_PATH"

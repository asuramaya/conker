#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%Y-%m-%d)"
LOG_PATH="$OUT_DIR/conker3_window4_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

run_cell() {
  local scale="$1"
  local steps="$2"
  local seed="$3"

  local scale_tag="${scale//./p}"
  local json_path="$OUT_DIR/conker3_window4_scale_${scale_tag}_steps${steps}_golf_bridge_seed${seed}_${STAMP}.json"

  if [[ -f "$json_path" ]]; then
    echo "skip landed window4 scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
    return
  fi

  echo "run conker3 window4 scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
  python3 conker/scripts/run_conker3_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --profile pilot \
    --variant window4 \
    --scale "$scale" \
    --steps "$steps" \
    --seed "$seed" \
    --quant-bits 6 \
    --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  echo "done conker3 window4 scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
}

echo "starting conker-3 window4 queue" | tee "$LOG_PATH"

run_cell 5.0 1000 43
run_cell 5.0 1000 44
run_cell 5.0 1500 42

echo "conker-3 window4 queue complete" | tee -a "$LOG_PATH"

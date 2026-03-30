#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%Y-%m-%d)"
LOG_PATH="$OUT_DIR/conker3_oscillatory_repl_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

run_probe() {
  local seed="$1"
  local json_path="$OUT_DIR/conker3_window4_scale_10p0_steps1500_half_life_16_osc_0p50_q46_golf_bridge_seed${seed}_${STAMP}.json"

  if [[ -f "$json_path" ]]; then
    echo "skip landed oscillatory 0.50 seed=${seed}" | tee -a "$LOG_PATH"
    return
  fi

  echo "run oscillatory 0.50 seed=${seed}" | tee -a "$LOG_PATH"
  python3 conker/scripts/run_conker3_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --profile pilot \
    --variant window4 \
    --scale 10.0 \
    --steps 1500 \
    --seed "$seed" \
    --linear-half-life-max 16 \
    --oscillatory-frac 0.50 \
    --oscillatory-period-min 4 \
    --oscillatory-period-max 64 \
    --quant-bits 4 \
    --quant-bits 6 \
    --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  echo "done oscillatory 0.50 seed=${seed}" | tee -a "$LOG_PATH"
}

echo "starting conker-3 oscillatory replication queue" | tee "$LOG_PATH"

run_probe 43
run_probe 44

echo "conker-3 oscillatory replication queue complete" | tee -a "$LOG_PATH"

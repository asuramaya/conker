#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%Y-%m-%d)"
LOG_PATH="$OUT_DIR/conker3_combo_pilot_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

run_bridge() {
  local scale="$1"
  local steps="$2"
  local seed="$3"
  local suffix="$4"
  shift 4
  local json_path="$OUT_DIR/conker3_window4_scale_${scale//./p}_steps${steps}_half_life_16_osc_0p875${suffix}_q46_golf_bridge_seed${seed}_${STAMP}.json"
  if [[ -f "$json_path" ]]; then
    echo "skip combo scale=${scale} steps=${steps} seed=${seed}${suffix}" | tee -a "$LOG_PATH"
    return
  fi
  echo "run combo scale=${scale} steps=${steps} seed=${seed}${suffix}" | tee -a "$LOG_PATH"
  python3 conker/scripts/run_conker3_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --profile pilot \
    --variant window4 \
    --scale "$scale" \
    --steps "$steps" \
    --seed "$seed" \
    --linear-half-life-max 16 \
    --oscillatory-frac 0.875 \
    --oscillatory-period-min 4 \
    --oscillatory-period-max 64 \
    --static-bank-gate \
    --quant-bits 4 \
    --quant-bits 6 \
    "$@" \
    --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  echo "done combo scale=${scale} steps=${steps} seed=${seed}${suffix}" | tee -a "$LOG_PATH"
}

echo "starting conker-3 combo pilot queue" | tee "$LOG_PATH"
run_bridge 16.0 2200 42 "_staticgate"
run_bridge 17.0 1800 42 "_staticgate"
run_bridge 17.0 2200 42 "_staticgate"
echo "conker-3 combo pilot queue complete" | tee -a "$LOG_PATH"

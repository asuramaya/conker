#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%F)"
LOG_FILE="$OUT_DIR/conker4_probe_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

run_conker3_ref() {
  local seed="$1"
  local json="$OUT_DIR/conker3_window4_scale_10p0_steps1000_half_life_16_osc_0p875_staticgate_q46_golf_bridge_seed${seed}_${STAMP}.json"
  if [[ -f "$json" ]]; then
    echo "skip conker3 ref seed=${seed}" | tee -a "$LOG_FILE"
    return
  fi
  echo "run conker3 ref seed=${seed}" | tee -a "$LOG_FILE"
  python3 conker/scripts/run_conker3_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --variant window4 \
    --scale 10 \
    --steps 1000 \
    --seed "$seed" \
    --linear-half-life-max 16 \
    --oscillatory-frac 0.875 \
    --oscillatory-period-min 4 \
    --oscillatory-period-max 64 \
    --static-bank-gate \
    --quant-bits 4 \
    --quant-bits 6 \
    --json "$json" | tee -a "$LOG_FILE"
}

run_conker4_full() {
  local seed="$1"
  local json="$OUT_DIR/conker4_support_frozen_scale_10p0_steps1000_half_life_16_osc_0p875_q46_seed${seed}_${STAMP}.json"
  if [[ -f "$json" ]]; then
    echo "skip conker4 full seed=${seed}" | tee -a "$LOG_FILE"
    return
  fi
  echo "run conker4 full seed=${seed}" | tee -a "$LOG_FILE"
  python3 conker/scripts/run_conker4_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --variant window4 \
    --scale 10 \
    --steps 1000 \
    --seed "$seed" \
    --linear-half-life-max 16 \
    --oscillatory-frac 0.875 \
    --oscillatory-period-min 4 \
    --oscillatory-period-max 64 \
    --static-bank-gate \
    --mixer-mode support \
    --quant-bits 4 \
    --quant-bits 6 \
    --json "$json" | tee -a "$LOG_FILE"
}

run_conker4_no_exact2() {
  local seed="$1"
  local json="$OUT_DIR/conker4_no_exact2_support_frozen_scale_10p0_steps1000_half_life_16_osc_0p875_q46_seed${seed}_${STAMP}.json"
  if [[ -f "$json" ]]; then
    echo "skip conker4 no_exact2 seed=${seed}" | tee -a "$LOG_FILE"
    return
  fi
  echo "run conker4 no_exact2 seed=${seed}" | tee -a "$LOG_FILE"
  python3 conker/scripts/run_conker4_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --variant window4 \
    --scale 10 \
    --steps 1000 \
    --seed "$seed" \
    --linear-half-life-max 16 \
    --oscillatory-frac 0.875 \
    --oscillatory-period-min 4 \
    --oscillatory-period-max 64 \
    --static-bank-gate \
    --mixer-mode support \
    --disable-exact2 \
    --quant-bits 4 \
    --quant-bits 6 \
    --json "$json" | tee -a "$LOG_FILE"
}

run_conker3_ref 42
run_conker4_full 42
run_conker4_no_exact2 42

echo "conker4 probe queue complete" | tee -a "$LOG_FILE"

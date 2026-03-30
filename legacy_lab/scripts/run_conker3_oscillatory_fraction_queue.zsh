#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%Y-%m-%d)"
LOG_PATH="$OUT_DIR/conker3_oscillatory_fraction_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

run_probe() {
  local osc_frac="$1"
  local seed="$2"

  local frac_tag="${osc_frac//./p}"
  local json_path="$OUT_DIR/conker3_window4_scale_10p0_steps1500_half_life_16_osc_${frac_tag}_q46_golf_bridge_seed${seed}_${STAMP}.json"

  if [[ -f "$json_path" ]]; then
    echo "skip landed oscillatory frac=${osc_frac} seed=${seed}" | tee -a "$LOG_PATH"
    return
  fi

  echo "run oscillatory frac=${osc_frac} seed=${seed}" | tee -a "$LOG_PATH"
  python3 conker/scripts/run_conker3_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --profile pilot \
    --variant window4 \
    --scale 10.0 \
    --steps 1500 \
    --seed "$seed" \
    --linear-half-life-max 16 \
    --oscillatory-frac "$osc_frac" \
    --oscillatory-period-min 4 \
    --oscillatory-period-max 64 \
    --quant-bits 4 \
    --quant-bits 6 \
    --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  echo "done oscillatory frac=${osc_frac} seed=${seed}" | tee -a "$LOG_PATH"
}

echo "starting conker-3 oscillatory fraction queue" | tee "$LOG_PATH"

run_probe 0.625 42
run_probe 0.75 42
run_probe 0.875 42

echo "conker-3 oscillatory fraction queue complete" | tee -a "$LOG_PATH"

#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%F)"
LOG_FILE="$OUT_DIR/conker4b_attr_entity_probe_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --variant window4
  --scale 10
  --steps 1000
  --seed 42
  --linear-half-life-max 16
  --oscillatory-frac 0.875
  --oscillatory-period-min 4
  --oscillatory-period-max 64
  --static-bank-gate
  --enable-exact3
  --enable-delim2
  --disable-recency
  --no-exact1-opens-mask
  --no-delim2-opens-mask
  --quant-bits 4
  --quant-bits 6
)

run_cell() {
  local label="$1"
  local json="$2"
  shift 2
  if [[ -f "$json" ]]; then
    echo "skip ${label}" | tee -a "$LOG_FILE"
    return
  fi
  echo "run ${label}" | tee -a "$LOG_FILE"
  python3 conker/scripts/run_conker4b_golf_bridge.py \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --json "$json" | tee -a "$LOG_FILE"
}

run_cell \
  "conker4b exact123+delim2+attr2 support" \
  "$OUT_DIR/conker4b_exact123_delim2_attr2_support_scale_10p0_steps1000_half_life_16_osc_0p875_q46_seed42_${STAMP}.json" \
  --enable-attr2

run_cell \
  "conker4b exact123+delim2+entity2 support" \
  "$OUT_DIR/conker4b_exact123_delim2_entity2_support_scale_10p0_steps1000_half_life_16_osc_0p875_q46_seed42_${STAMP}.json" \
  --enable-entity2

run_cell \
  "conker4b exact123+delim2+special2+number2+markup2+attr2 support" \
  "$OUT_DIR/conker4b_exact123_delim2_special2_number2_markup2_attr2_support_scale_10p0_steps1000_half_life_16_osc_0p875_q46_seed42_${STAMP}.json" \
  --enable-special2 \
  --enable-number2 \
  --enable-markup2 \
  --enable-attr2

run_cell \
  "conker4b exact123+delim2+special2+number2+markup2+entity2 support" \
  "$OUT_DIR/conker4b_exact123_delim2_special2_number2_markup2_entity2_support_scale_10p0_steps1000_half_life_16_osc_0p875_q46_seed42_${STAMP}.json" \
  --enable-special2 \
  --enable-number2 \
  --enable-markup2 \
  --enable-entity2

echo "conker4b attr/entity probe queue complete" | tee -a "$LOG_FILE"

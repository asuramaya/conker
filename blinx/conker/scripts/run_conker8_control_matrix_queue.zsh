#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="${STAMP:-$(date +%F)}"
LOG_FILE="$OUT_DIR/conker8_control_matrix_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

COMMON=(
  --data-root "$DATA_ROOT"
  --profile pilot
  --variant window4
  --scale 10
  --seed 42
  --seq-len 256
  --batch-size 16
  --steps 1000
  --learning-rate 5e-4
  --linear-half-life-max 16
  --oscillatory-frac 0.875
  --oscillatory-period-min 4
  --oscillatory-period-max 64
  --static-bank-gate
  --gate-only-mode
  --no-exact1-opens-mask
  --no-delim2-opens-mask
  --no-freeze-base
  --dynamic-support-gates
  --enable-exact3
  --enable-special2
  --enable-number2
  --enable-markup2
  --enable-attr2
  --enable-delim2
)

run_row() {
  local label="$1"
  shift
  local json_path="$OUT_DIR/${label}_${STAMP}.json"
  echo "conker8 ${label}" | tee -a "$LOG_FILE"
  python3 conker/scripts/run_conker8_golf_bridge.py \
    "${COMMON[@]}" \
    "$@" \
    --json "$json_path" | tee -a "$LOG_FILE"
}

run_row "conker8_hispan_full" \
  --disable-recency \
  --lag-profile-span 2.0 \
  --support-mask-span 2.0

run_row "conker8_hispan_lag_only" \
  --disable-recency \
  --lag-profile-span 2.0 \
  --support-mask-span 2.0 \
  --disable-learn-delimiter-mask \
  --disable-learn-number-mask \
  --disable-learn-special-mask \
  --disable-learn-urlpath-mask \
  --disable-learn-markup-mask \
  --disable-learn-attr-mask \
  --disable-learn-entity-mask

run_row "conker8_hispan_mask_only" \
  --disable-recency \
  --lag-profile-span 2.0 \
  --support-mask-span 2.0 \
  --disable-learn-lag-profile

run_row "conker8_hispan_no_dynamic_gates" \
  --disable-recency \
  --lag-profile-span 2.0 \
  --support-mask-span 2.0 \
  --no-dynamic-support-gates

run_row "conker8_hispan_with_recency" \
  --lag-profile-span 2.0 \
  --support-mask-span 2.0 \
  --recency-half-life 8.0

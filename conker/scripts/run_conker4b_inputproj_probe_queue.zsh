#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%F)"
LOG_FILE="$OUT_DIR/conker4b_inputproj_probe_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

run_probe() {
  local suffix="$1"
  local scheme="$2"
  local json="$OUT_DIR/conker4b_inputproj_${suffix}_seed42_${STAMP}.json"
  if [[ -f "$json" ]]; then
    echo "skip conker4b inputproj ${suffix}" | tee -a "$LOG_FILE"
    return
  fi
  echo "run conker4b inputproj ${suffix}" | tee -a "$LOG_FILE"
  python3 conker/scripts/run_conker4b_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --variant window4 \
    --scale 10 \
    --steps 1000 \
    --seed 42 \
    --linear-half-life-max 16 \
    --oscillatory-frac 0.875 \
    --oscillatory-period-min 4 \
    --oscillatory-period-max 64 \
    --input-proj-scheme "$scheme" \
    --static-bank-gate \
    --dynamic-support-gates \
    --gate-only-mode \
    --support-gate-mode independent \
    --enable-exact3 \
    --enable-delim2 \
    --enable-special2 \
    --enable-number2 \
    --enable-markup2 \
    --enable-attr2 \
    --disable-recency \
    --no-exact1-opens-mask \
    --no-delim2-opens-mask \
    --quant-bits 4 \
    --quant-bits 6 \
    --json "$json" | tee -a "$LOG_FILE"
}

run_probe random random
run_probe orthogonal orthogonal_rows
run_probe energy kernel_energy
run_probe split split_banks

echo "conker4b inputproj probe queue complete" | tee -a "$LOG_FILE"

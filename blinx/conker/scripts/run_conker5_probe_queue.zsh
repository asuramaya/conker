#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%F)"
LOG_FILE="$OUT_DIR/conker5_probe_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

run_probe() {
  local suffix="$1"
  shift
  local json="$OUT_DIR/conker5_${suffix}_seed42_${STAMP}.json"
  if [[ -f "$json" ]]; then
    echo "skip conker5 ${suffix}" | tee -a "$LOG_FILE"
    return
  fi
  echo "run conker5 ${suffix}" | tee -a "$LOG_FILE"
  python3 conker/scripts/run_conker5_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --variant window4 \
    --scale 10 \
    --steps 1000 \
    --seed 42 \
    --linear-half-life-max 16 \
    --oscillatory-frac 0.875 \
    --oscillatory-period-min 4 \
    --oscillatory-period-max 64 \
    --static-bank-gate \
    --quant-bits 4 \
    --quant-bits 6 \
    --json "$json" \
    "$@" | tee -a "$LOG_FILE"
}

run_probe h8_r8_cap2_lr5e4 --learning-rate 5e-4 --num-heads 8 --head-rank 8 --residual-cap 2.0
run_probe h12_r8_cap2_lr5e4 --learning-rate 5e-4 --num-heads 12 --head-rank 8 --residual-cap 2.0

echo "conker5 probe queue complete" | tee -a "$LOG_FILE"

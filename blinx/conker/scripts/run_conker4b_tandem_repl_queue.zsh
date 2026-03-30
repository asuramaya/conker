#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%F)"
LOG_FILE="$OUT_DIR/conker4b_tandem_repl_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

run_seed() {
  local suffix="$1"
  local seed="$2"
  shift 2
  local json="$OUT_DIR/conker4b_${suffix}_seed${seed}_${STAMP}.json"
  if [[ -f "$json" ]]; then
    echo "skip conker4b ${suffix} seed=${seed}" | tee -a "$LOG_FILE"
    return
  fi
  echo "run conker4b ${suffix} seed=${seed}" | tee -a "$LOG_FILE"
  python3 conker/scripts/run_conker4b_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --variant window4 \
    --scale 10 \
    --seed "$seed" \
    --linear-half-life-max 16 \
    --oscillatory-frac 0.875 \
    --oscillatory-period-min 4 \
    --oscillatory-period-max 64 \
    --static-bank-gate \
    --dynamic-support-gates \
    --gate-only-mode \
    --enable-exact3 \
    --enable-delim2 \
    --enable-special2 \
    --enable-number2 \
    --enable-markup2 \
    --enable-attr2 \
    --disable-recency \
    --no-exact1-opens-mask \
    --no-delim2-opens-mask \
    --no-freeze-base \
    --learning-rate 5e-4 \
    --quant-bits 4 \
    --quant-bits 6 \
    --json "$json" \
    "$@" | tee -a "$LOG_FILE"
}

for seed in 42 43 44; do
  run_seed tandem_seq256_steps1000_lr5e4 \
    "$seed" \
    --seq-len 256 \
    --batch-size 16 \
    --steps 1000
done

for seed in 42 43 44; do
  run_seed tandem_seq512_steps1000_bs8_lr5e4 \
    "$seed" \
    --seq-len 512 \
    --batch-size 8 \
    --steps 1000
done

echo "conker4b tandem replication queue complete" | tee -a "$LOG_FILE"

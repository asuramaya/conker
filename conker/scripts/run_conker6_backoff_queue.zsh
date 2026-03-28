#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%F)"
LOG_FILE="$OUT_DIR/conker6_backoff_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

run_cell() {
  local label="$1"
  shift
  local summary_json="$OUT_DIR/conker6_${label}_${STAMP}.json"
  if [[ -f "$summary_json" ]]; then
    echo "skip ${label}" | tee -a "$LOG_FILE"
    return
  fi

  echo "run ${label}" | tee -a "$LOG_FILE"
  python3 conker/scripts/run_conker6_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --profile pilot \
    --variant window4 \
    --scale 10 \
    --seed 42 \
    --seq-len 256 \
    --batch-size 16 \
    --steps 1000 \
    --learning-rate 5e-4 \
    --linear-half-life-max 16 \
    --oscillatory-frac 0.875 \
    --oscillatory-period-min 4 \
    --oscillatory-period-max 64 \
    --static-bank-gate \
    --freeze-base \
    --quant-bits 6 \
    --quant-bits 4 \
    --full-eval \
    "$@" \
    --json "$summary_json" | tee -a "$LOG_FILE"
}

run_cell wittenbell_seq256_steps1000 \
  --blend-mode witten_bell

run_cell absdisc_seq256_steps1000 \
  --blend-mode absolute_discount \
  --unigram-discount 0.25 \
  --exact1-discount 0.75 \
  --exact2-discount 0.75 \
  --exact3-discount 0.75

run_cell wittenbell_seq256_steps1000_exactspan512 \
  --blend-mode witten_bell \
  --exact-context-span 512

run_cell absdisc_seq256_steps1000_exactspan512 \
  --blend-mode absolute_discount \
  --exact-context-span 512 \
  --unigram-discount 0.25 \
  --exact1-discount 0.75 \
  --exact2-discount 0.75 \
  --exact3-discount 0.75

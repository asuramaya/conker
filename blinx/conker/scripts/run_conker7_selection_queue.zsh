#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="conker/out"
mkdir -p "$OUT_DIR"
STAMP="${STAMP:-$(date +%F)}"
LOG_FILE="$OUT_DIR/conker7_selection_queue_${STAMP}.log"

COMMON_TRAIN=(
  --data-root conker/data/datasets/fineweb10B_sp1024
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
  --dynamic-support-gates
  --gate-only-mode
  --enable-exact3
  --enable-delim2
  --enable-special2
  --enable-number2
  --enable-markup2
  --enable-attr2
  --disable-recency
  --no-exact1-opens-mask
  --no-delim2-opens-mask
  --load-state conker/out/conker4b_tandem_seq256_steps1500_lr5e4_seed42_2026-03-27.npz
  --teacher-mask-mode bidirectional
  --teacher-enable-exact2
  --teacher-enable-exact3
  --teacher-disable-special2
  --teacher-disable-number2
  --teacher-disable-markup2
  --teacher-disable-attr2
  --teacher-disable-delim2
  --quant-bits 6
  --quant-bits 4
)

COMMON_EVAL=(
  --data-root conker/data/datasets/fineweb10B_sp1024
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
  --dynamic-support-gates
  --gate-only-mode
  --enable-exact3
  --enable-delim2
  --enable-special2
  --enable-number2
  --enable-markup2
  --enable-attr2
  --disable-recency
  --no-exact1-opens-mask
  --no-delim2-opens-mask
  --teacher-mask-mode bidirectional
  --teacher-enable-exact2
  --teacher-enable-exact3
  --teacher-disable-special2
  --teacher-disable-number2
  --teacher-disable-markup2
  --teacher-disable-attr2
  --teacher-disable-delim2
)

run_train() {
  local label="$1"
  shift
  echo "train ${label}" | tee -a "$LOG_FILE"
  python3 conker/scripts/run_conker7_golf_bridge.py "$@" | tee -a "$LOG_FILE"
}

run_eval() {
  local label="$1"
  shift
  echo "eval ${label}" | tee -a "$LOG_FILE"
  python3 conker/scripts/run_conker7_checkpoint_eval.py "$@" | tee -a "$LOG_FILE"
}

run_row() {
  local tag="$1"
  shift
  local state_npz="$OUT_DIR/${tag}_${STAMP}.npz"
  local train_json="$OUT_DIR/${tag}_${STAMP}.json"
  local full_json="$OUT_DIR/${tag}_fullval_test_none_${STAMP}.json"
  local int6_json="$OUT_DIR/${tag}_fullval_test_int6_${STAMP}.json"
  local int6_art="$OUT_DIR/${tag}.int6.ptz"

  run_train "${tag}" \
    "${COMMON_TRAIN[@]}" \
    "$@" \
    --save-state "$state_npz" \
    --json "$train_json"

  run_eval "${tag} fullval fp16" \
    "${COMMON_EVAL[@]}" \
    "$@" \
    --state-npz "$state_npz" \
    --full-split \
    --split test \
    --transform none \
    --output-json "$full_json"

  run_eval "${tag} fullval int6" \
    "${COMMON_EVAL[@]}" \
    "$@" \
    --state-npz "$state_npz" \
    --full-split \
    --split test \
    --transform none \
    --quant-bits 6 \
    --artifact-out "$int6_art" \
    --output-json "$int6_json"
}

run_row "conker7_bidirectional_exact23_tw0p05_warmstart_tandem1500_seq256_steps1000_seed42" \
  --teacher-weight 0.05

run_row "conker7_bidirectional_exact23_tw01_start500_warmstart_tandem1500_seq256_steps1000_seed42" \
  --teacher-weight 0.10 \
  --teacher-start-step 500

run_row "conker7_bidirectional_exact23_tw0p05_start500_warmstart_tandem1500_seq256_steps1000_seed42" \
  --teacher-weight 0.05 \
  --teacher-start-step 500

echo "done: $LOG_FILE" | tee -a "$LOG_FILE"

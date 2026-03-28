#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="conker/out"
mkdir -p "$OUT_DIR"
STAMP="${STAMP:-$(date +%F)}"
LOG_FILE="$OUT_DIR/conker7_followup_queue_${STAMP}.log"

COMMON_TRAIN=(
  --data-root conker/data/datasets/fineweb10B_sp1024
  --profile pilot
  --variant window4
  --scale 10
  --seq-len 256
  --batch-size 16
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
  --teacher-weight 0.10
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
  --seq-len 256
  --batch-size 16
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
  --teacher-weight 0.10
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

SEED42_JSON="$OUT_DIR/conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42_save_${STAMP}.json"
SEED42_STATE="$OUT_DIR/conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42_save_${STAMP}.npz"
run_train "seed42 warmstart save-state" \
  "${COMMON_TRAIN[@]}" \
  --seed 42 \
  --steps 1000 \
  --save-state "$SEED42_STATE" \
  --json "$SEED42_JSON"

run_eval "seed42 warmstart fullval fp16" \
  "${COMMON_EVAL[@]}" \
  --seed 42 \
  --steps 1000 \
  --state-npz "$SEED42_STATE" \
  --full-split \
  --split test \
  --transform none \
  --output-json "$OUT_DIR/conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42_fullval_test_none_${STAMP}.json"

run_eval "seed42 warmstart fullval int6" \
  "${COMMON_EVAL[@]}" \
  --seed 42 \
  --steps 1000 \
  --state-npz "$SEED42_STATE" \
  --full-split \
  --split test \
  --transform none \
  --quant-bits 6 \
  --artifact-out "$OUT_DIR/conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42.int6.ptz" \
  --output-json "$OUT_DIR/conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42_fullval_test_int6_${STAMP}.json"

for seed in 43 44; do
  run_train "seed${seed} warmstart replicate" \
    "${COMMON_TRAIN[@]}" \
    --seed "$seed" \
    --steps 1000 \
    --json "$OUT_DIR/conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed${seed}_${STAMP}.json"
done

for tw in 0.05 0.15; do
  tag="${tw/./p}"
  run_train "teacher_weight=${tw}" \
    "${COMMON_TRAIN[@]}" \
    --seed 42 \
    --steps 1000 \
    --teacher-weight "$tw" \
    --json "$OUT_DIR/conker7_bidirectional_exact23_tw${tag}_warmstart_tandem1500_seq256_steps1000_seed42_${STAMP}.json"
done

for start_step in 250 500; do
  run_train "teacher_start=${start_step}" \
    "${COMMON_TRAIN[@]}" \
    --seed 42 \
    --steps 1000 \
    --teacher-start-step "$start_step" \
    --json "$OUT_DIR/conker7_bidirectional_exact23_tw01_start${start_step}_warmstart_tandem1500_seq256_steps1000_seed42_${STAMP}.json"
done

echo "done: $LOG_FILE" | tee -a "$LOG_FILE"

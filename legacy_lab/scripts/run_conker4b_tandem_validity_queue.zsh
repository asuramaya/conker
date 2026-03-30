#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%F)"
LOG_FILE="$OUT_DIR/conker4b_tandem_validity_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

train_and_full_eval() {
  local label="$1"
  local seed="$2"
  local seq_len="$3"
  local batch_size="$4"
  local steps="$5"
  local lr="$6"

  local summary_json="$OUT_DIR/conker4b_${label}_seed${seed}_saveeval_${STAMP}.json"
  local state_npz="$OUT_DIR/conker4b_${label}_seed${seed}_${STAMP}.npz"
  local fullval_json="$OUT_DIR/conker4b_${label}_seed${seed}_fullval_test_none_${STAMP}.json"
  local int6_json="$OUT_DIR/conker4b_${label}_seed${seed}_fullval_test_int6_${STAMP}.json"
  local int6_artifact="$OUT_DIR/conker5_${label}_seed${seed}_int6_model.ptz"

  if [[ ! -f "$summary_json" || ! -f "$state_npz" ]]; then
    echo "train conker4b tandem ${label} seed=${seed}" | tee -a "$LOG_FILE"
    python3 conker/scripts/run_conker4b_golf_bridge.py \
      --data-root "$DATA_ROOT" \
      --profile pilot \
      --variant window4 \
      --scale 10 \
      --seed "$seed" \
      --seq-len "$seq_len" \
      --batch-size "$batch_size" \
      --steps "$steps" \
      --learning-rate "$lr" \
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
      --save-state "$state_npz" \
      --json "$summary_json" | tee -a "$LOG_FILE"
  else
    echo "skip train conker4b tandem ${label} seed=${seed}" | tee -a "$LOG_FILE"
  fi

  if [[ ! -f "$fullval_json" ]]; then
    echo "fullval fp16 conker4b tandem ${label} seed=${seed}" | tee -a "$LOG_FILE"
    python3 conker/scripts/run_conker4b_checkpoint_eval.py \
      --summary-json "$summary_json" \
      --state-npz "$state_npz" \
      --data-root "$DATA_ROOT" \
      --split test \
      --transform none \
      --full-split \
      --output-json "$fullval_json" | tee -a "$LOG_FILE"
  else
    echo "skip fullval fp16 conker4b tandem ${label} seed=${seed}" | tee -a "$LOG_FILE"
  fi

  if [[ ! -f "$int6_json" ]]; then
    echo "fullval int6 conker4b tandem ${label} seed=${seed}" | tee -a "$LOG_FILE"
    python3 conker/scripts/run_conker4b_checkpoint_eval.py \
      --summary-json "$summary_json" \
      --state-npz "$state_npz" \
      --data-root "$DATA_ROOT" \
      --split test \
      --transform none \
      --full-split \
      --quant-bits 6 \
      --artifact-out "$int6_artifact" \
      --output-json "$int6_json" | tee -a "$LOG_FILE"
  else
    echo "skip fullval int6 conker4b tandem ${label} seed=${seed}" | tee -a "$LOG_FILE"
  fi
}

echo "starting conker4b tandem validity queue" | tee "$LOG_FILE"

train_and_full_eval tandem_seq256_steps1000_lr5e4 42 256 16 1000 5e-4
train_and_full_eval tandem_seq256_steps1000_lr5e4 44 256 16 1000 5e-4

train_and_full_eval tandem_seq512_steps1000_bs8_lr5e4 43 512 8 1000 5e-4
train_and_full_eval tandem_seq512_steps1000_bs8_lr5e4 44 512 8 1000 5e-4

train_and_full_eval tandem_seq256_steps1200_lr5e4 42 256 16 1200 5e-4
train_and_full_eval tandem_seq256_steps1500_lr5e4 42 256 16 1500 5e-4

echo "conker4b tandem validity queue complete" | tee -a "$LOG_FILE"

#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="${STAMP:-$(date +%F)}"
LOG_FILE="$OUT_DIR/conker4b_strict_recovery_queue_${STAMP}.log"

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
  --disable-recency
  --no-exact1-opens-mask
  --no-delim2-opens-mask
  --no-freeze-base
)

run_bridge() {
  local label="$1"
  shift
  local json_path="$OUT_DIR/${label}_${STAMP}.json"
  echo "bridge ${label}" | tee -a "$LOG_FILE"
  python3 conker/scripts/run_conker4b_golf_bridge.py \
    "${COMMON[@]}" \
    "$@" \
    --json "$json_path" | tee -a "$LOG_FILE"
}

run_full_eval() {
  local label="$1"
  local summary_json="$2"
  local state_npz="$3"
  local fp16_json="$OUT_DIR/${label}_fullval_test_none_${STAMP}.json"
  local int6_json="$OUT_DIR/${label}_fullval_test_int6_${STAMP}.json"
  local int6_artifact="$OUT_DIR/${label}.int6.ptz"

  echo "fullval fp16 ${label}" | tee -a "$LOG_FILE"
  python3 conker/scripts/run_conker4b_checkpoint_eval.py \
    --summary-json "$summary_json" \
    --state-npz "$state_npz" \
    --data-root "$DATA_ROOT" \
    --split test \
    --transform none \
    --full-split \
    --output-json "$fp16_json" | tee -a "$LOG_FILE"

  echo "fullval int6 ${label}" | tee -a "$LOG_FILE"
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
}

BASE_LABEL="conker4b_strict_tandem_seq256_steps1000_lr5e4_seed42"
BASE_JSON="$OUT_DIR/${BASE_LABEL}_${STAMP}.json"
BASE_STATE="$OUT_DIR/${BASE_LABEL}_${STAMP}.npz"

echo "starting strict recovery queue" | tee "$LOG_FILE"

echo "bridge ${BASE_LABEL}" | tee -a "$LOG_FILE"
python3 conker/scripts/run_conker4b_golf_bridge.py \
  "${COMMON[@]}" \
  --dynamic-support-gates \
  --enable-exact3 \
  --enable-delim2 \
  --enable-special2 \
  --enable-number2 \
  --enable-markup2 \
  --enable-attr2 \
  --save-state "$BASE_STATE" \
  --json "$BASE_JSON" | tee -a "$LOG_FILE"

run_full_eval "$BASE_LABEL" "$BASE_JSON" "$BASE_STATE"

run_bridge "conker4b_strict_ablate_no_exact3_seq256_steps1000_lr5e4_seed42" \
  --dynamic-support-gates \
  --enable-delim2 \
  --enable-special2 \
  --enable-number2 \
  --enable-markup2 \
  --enable-attr2

run_bridge "conker4b_strict_ablate_exact_only_seq256_steps1000_lr5e4_seed42" \
  --dynamic-support-gates \
  --enable-exact3

run_bridge "conker4b_strict_ablate_no_dynamic_gates_seq256_steps1000_lr5e4_seed42" \
  --enable-exact3 \
  --enable-delim2 \
  --enable-special2 \
  --enable-number2 \
  --enable-markup2 \
  --enable-attr2

echo "done: $LOG_FILE" | tee -a "$LOG_FILE"

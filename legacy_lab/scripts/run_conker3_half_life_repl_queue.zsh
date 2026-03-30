#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%Y-%m-%d)"
LOG_PATH="$OUT_DIR/conker3_half_life_repl_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

run_bridge() {
  local half_life_max="$1"
  local scale="$2"
  local steps="$3"
  local seed="$4"

  local half_tag="${half_life_max//./p}"
  local scale_tag="${scale//./p}"
  local json_path="$OUT_DIR/conker3_window4_scale_${scale_tag}_steps${steps}_half_life_${half_tag}_golf_bridge_seed${seed}_${STAMP}.json"

  if [[ -f "$json_path" ]]; then
    echo "skip landed half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
    return
  fi

  echo "run half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
  python3 conker/scripts/run_conker3_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --profile pilot \
    --variant window4 \
    --scale "$scale" \
    --steps "$steps" \
    --seed "$seed" \
    --linear-half-life-max "$half_life_max" \
    --quant-bits 6 \
    --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  echo "done half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
}

run_mixed_quant() {
  local half_life_max="$1"
  local scale="$2"
  local steps="$3"
  local seed="$4"

  local half_tag="${half_life_max//./p}"
  local scale_tag="${scale//./p}"
  local json_path="$OUT_DIR/conker3_window4_scale_${scale_tag}_steps${steps}_half_life_${half_tag}_mixed_quant_audit_seed${seed}_${STAMP}.json"

  if [[ -f "$json_path" ]]; then
    echo "skip landed mixed quant half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
    return
  fi

  echo "run mixed quant half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
  python3 conker/scripts/run_conker3_mixed_quant_audit.py \
    --data-root "$DATA_ROOT" \
    --profile pilot \
    --scale "$scale" \
    --steps "$steps" \
    --seed "$seed" \
    --linear-half-life-max "$half_life_max" \
    --eval-batches 8 \
    --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  echo "done mixed quant half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
}

echo "starting conker-3 half-life replication queue" | tee "$LOG_PATH"

run_bridge 8 8.0 1500 43
run_bridge 8 8.0 1500 44
run_bridge 16 8.0 1500 43
run_bridge 16 8.0 1500 44
run_mixed_quant 8 8.0 1500 42
run_bridge 8 10.0 1500 42

echo "conker-3 half-life replication queue complete" | tee -a "$LOG_PATH"

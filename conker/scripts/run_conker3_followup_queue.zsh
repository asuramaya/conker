#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%Y-%m-%d)"
LOG_PATH="$OUT_DIR/conker3_followup_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

run_cell() {
  local variant="$1"
  local scale="$2"
  local steps="$3"
  local seed="$4"

  local scale_tag="${scale//./p}"
  local json_path="$OUT_DIR/conker3_${variant}_scale_${scale_tag}_steps${steps}_golf_bridge_seed${seed}_${STAMP}.json"

  if [[ -f "$json_path" ]]; then
    echo "skip landed ${variant} scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
    return
  fi

  echo "run conker3 variant=${variant} scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
  python3 conker/scripts/run_conker3_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --profile pilot \
    --variant "$variant" \
    --scale "$scale" \
    --steps "$steps" \
    --seed "$seed" \
    --quant-bits 6 \
    --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  echo "done conker3 variant=${variant} scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
}

echo "starting conker-3 followup queue" | tee "$LOG_PATH"

run_cell linear_only 3.0 1000 42
run_cell base 3.0 1000 42
run_cell gated 3.0 1000 42
run_cell window4 3.0 1000 42
run_cell shared_embedding 3.0 1000 42
run_cell linear_only 5.0 1000 42
run_cell base 5.0 1000 42
run_cell gated 5.0 1000 42
run_cell window4 5.0 1000 42

echo "conker-3 followup queue complete" | tee -a "$LOG_PATH"

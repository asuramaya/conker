#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%Y-%m-%d)"
LOG_PATH="$OUT_DIR/conker3_decay_bank_queue_${STAMP}.log"
MATCHED_JSON="$OUT_DIR/conker3_matched_decay_bank_scale8_${STAMP}.json"

mkdir -p "$OUT_DIR"

echo "starting conker-3 decay-bank queue" | tee "$LOG_PATH"

if [[ ! -f "$MATCHED_JSON" ]]; then
  echo "build matched decay bank" | tee -a "$LOG_PATH"
  python3 conker/scripts/build_conker3_decay_bank.py \
    --data-root "$DATA_ROOT" \
    --sample-tokens 524288 \
    --projection-dim 8 \
    --max-lag 512 \
    --modes 2048 \
    --json "$MATCHED_JSON" 2>&1 | tee -a "$LOG_PATH"
fi

run_cell() {
  local bank="$1"
  local json_path="$OUT_DIR/conker3_window4_scale_8p0_steps1500_decay_${bank}_golf_bridge_seed42_${STAMP}.json"

  if [[ -f "$json_path" ]]; then
    echo "skip landed decay_bank=${bank}" | tee -a "$LOG_PATH"
    return
  fi

  echo "run decay_bank=${bank}" | tee -a "$LOG_PATH"
  if [[ "$bank" == "matched" ]]; then
    python3 conker/scripts/run_conker3_golf_bridge.py \
      --data-root "$DATA_ROOT" \
      --profile pilot \
      --variant window4 \
      --scale 8.0 \
      --steps 1500 \
      --seed 42 \
      --decay-bank custom \
      --decays-json "$MATCHED_JSON" \
      --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  else
    python3 conker/scripts/run_conker3_golf_bridge.py \
      --data-root "$DATA_ROOT" \
      --profile pilot \
      --variant window4 \
      --scale 8.0 \
      --steps 1500 \
      --seed 42 \
      --decay-bank "$bank" \
      --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  fi
  echo "done decay_bank=${bank}" | tee -a "$LOG_PATH"
}

run_cell logspace
run_cell matched
run_cell narrow

echo "conker-3 decay-bank queue complete" | tee -a "$LOG_PATH"

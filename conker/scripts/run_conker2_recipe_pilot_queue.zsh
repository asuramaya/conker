#!/bin/zsh
set -euo pipefail

ROOT=/Users/asuramaya/Code/carving_machine_v3
CONKER="$ROOT/conker"
PYTHON="/Users/asuramaya/Code/codex/.venv-mlx/bin/python"
DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$CONKER/data/datasets/fineweb10B_sp1024}"
LOG="$CONKER/out/conker2_recipe_pilot_queue_2026-03-25.log"

mkdir -p "$CONKER/out"
echo "starting conker-2 recipe pilot queue" | tee "$LOG"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "missing data root: $DATA_ROOT" | tee -a "$LOG"
  echo "run conker/scripts/link_parameter_golf_data.zsh or see conker/data/PARAMETER_GOLF_DATA.md" | tee -a "$LOG"
  exit 1
fi

run_cell() {
  local lr_tag="$1"
  local lr="$2"
  local steps="$3"
  local json="$CONKER/out/conker2_untied_base_scale_12p0_${lr_tag}_steps${steps}_golf_bridge_seed42_2026-03-25.json"

  if [[ -f "$json" ]]; then
    echo "skip $(basename "$json")" | tee -a "$LOG"
    return
  fi

  echo "run conker2 12x recipe lr=$lr steps=$steps seed=42" | tee -a "$LOG"
  cd "$ROOT"
  "$PYTHON" "$CONKER/scripts/run_conker2_golf_bridge.py" \
    --variant untied_base \
    --scale 12.0 \
    --learning-rate "$lr" \
    --quant-bits 6 \
    --data-root "$DATA_ROOT" \
    --seed 42 \
    --steps "$steps" \
    --seq-len 256 \
    --batch-size 16 \
    --profile pilot \
    --json "$json" | tee -a "$LOG"
  echo "done conker2 12x recipe lr=$lr steps=$steps seed=42" | tee -a "$LOG"
}

run_cell lr5em4 5e-4 1000
run_cell lr3em4 3e-4 1000
run_cell lr5em4 5e-4 1500

echo "conker-2 recipe pilot queue complete" | tee -a "$LOG"

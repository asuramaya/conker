#!/bin/zsh
set -euo pipefail

ROOT=/Users/asuramaya/Code/carving_machine_v3
CONKER="$ROOT/conker"
PYTHON="/Users/asuramaya/Code/codex/.venv-mlx/bin/python"
DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$CONKER/data/datasets/fineweb10B_sp1024}"
LOG="$CONKER/out/conker2_undercap_recipe_repl_queue_2026-03-26.log"

mkdir -p "$CONKER/out"
echo "starting conker-2 under-cap recipe replication queue" | tee "$LOG"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "missing data root: $DATA_ROOT" | tee -a "$LOG"
  echo "run conker/scripts/link_parameter_golf_data.zsh or see conker/data/PARAMETER_GOLF_DATA.md" | tee -a "$LOG"
  exit 1
fi

run_cell() {
  local seed="$1"
  local json="$CONKER/out/conker2_untied_base_scale_11p5_lr5em4_steps1800_golf_bridge_seed${seed}_2026-03-26.json"

  if [[ -f "$json" ]]; then
    echo "skip $(basename "$json")" | tee -a "$LOG"
    return
  fi

  echo "run conker2 under-cap repl scale=11.5 lr=5e-4 steps=1800 seed=$seed" | tee -a "$LOG"
  cd "$ROOT"
  "$PYTHON" "$CONKER/scripts/run_conker2_golf_bridge.py" \
    --variant untied_base \
    --scale 11.5 \
    --learning-rate 5e-4 \
    --quant-bits 6 \
    --data-root "$DATA_ROOT" \
    --seed "$seed" \
    --steps 1800 \
    --seq-len 256 \
    --batch-size 16 \
    --profile pilot \
    --json "$json" | tee -a "$LOG"
  echo "done conker2 under-cap repl scale=11.5 lr=5e-4 steps=1800 seed=$seed" | tee -a "$LOG"
}

for seed in 43 44; do
  run_cell "$seed"
done

echo "conker-2 under-cap recipe replication queue complete" | tee -a "$LOG"

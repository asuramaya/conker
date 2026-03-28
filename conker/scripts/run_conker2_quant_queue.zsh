#!/bin/zsh
set -euo pipefail

ROOT=/Users/asuramaya/Code/carving_machine_v3
CONKER="$ROOT/conker"
PYTHON="/Users/asuramaya/Code/codex/.venv-mlx/bin/python"
DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$CONKER/data/datasets/fineweb10B_sp1024}"
LOG="$CONKER/out/conker2_quant_queue_2026-03-25.log"

mkdir -p "$CONKER/out"
echo "starting conker-2 quant queue" | tee "$LOG"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "missing data root: $DATA_ROOT" | tee -a "$LOG"
  echo "run conker/scripts/link_parameter_golf_data.zsh or see conker/data/PARAMETER_GOLF_DATA.md" | tee -a "$LOG"
  exit 1
fi

run_cell() {
  local seed="$1"
  local json="$CONKER/out/conker2_quant_ladder_seed${seed}_2026-03-25.json"

  if [[ -f "$json" ]]; then
    echo "skip $(basename "$json")" | tee -a "$LOG"
    return
  fi

  echo "run conker2 quant ladder seed=$seed" | tee -a "$LOG"
  cd "$ROOT"
  "$PYTHON" "$CONKER/scripts/run_conker2_quant_ladder.py" \
    --data-root "$DATA_ROOT" \
    --seed "$seed" \
    --steps 1000 \
    --seq-len 256 \
    --batch-size 16 \
    --profile pilot \
    --linear-modes 256 \
    --json "$json" | tee -a "$LOG"
  echo "done conker2 quant ladder seed=$seed" | tee -a "$LOG"
}

for seed in 42 43 44; do
  run_cell "$seed"
done

echo "conker-2 quant queue complete" | tee -a "$LOG"

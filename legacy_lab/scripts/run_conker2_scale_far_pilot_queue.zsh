#!/bin/zsh
set -euo pipefail

ROOT=/Users/asuramaya/Code/carving_machine_v3
CONKER="$ROOT/conker"
PYTHON="/Users/asuramaya/Code/codex/.venv-mlx/bin/python"
DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$CONKER/data/datasets/fineweb10B_sp1024}"
LOG="$CONKER/out/conker2_scale_far_pilot_queue_2026-03-25.log"

mkdir -p "$CONKER/out"
echo "starting conker-2 far-scale pilot queue" | tee "$LOG"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "missing data root: $DATA_ROOT" | tee -a "$LOG"
  echo "run conker/scripts/link_parameter_golf_data.zsh or see conker/data/PARAMETER_GOLF_DATA.md" | tee -a "$LOG"
  exit 1
fi

run_cell() {
  local scale="$1"
  local seed="$2"
  local scale_tag="${scale//./p}"
  local json="$CONKER/out/conker2_untied_base_scale_${scale_tag}_golf_bridge_seed${seed}_2026-03-25.json"

  if [[ -f "$json" ]]; then
    echo "skip $(basename "$json")" | tee -a "$LOG"
    return
  fi

  echo "run conker2 untied_base scale=$scale seed=$seed" | tee -a "$LOG"
  cd "$ROOT"
  "$PYTHON" "$CONKER/scripts/run_conker2_golf_bridge.py" \
    --variant untied_base \
    --scale "$scale" \
    --quant-bits 6 \
    --data-root "$DATA_ROOT" \
    --seed "$seed" \
    --steps 1000 \
    --seq-len 256 \
    --batch-size 16 \
    --profile pilot \
    --json "$json" | tee -a "$LOG"
  echo "done conker2 untied_base scale=$scale seed=$seed" | tee -a "$LOG"
}

for scale in 4.0 5.0; do
  run_cell "$scale" 42
done

echo "conker-2 far-scale pilot queue complete" | tee -a "$LOG"

#!/bin/zsh
set -euo pipefail

ROOT=/Users/asuramaya/Code/carving_machine_v3
CONKER="$ROOT/conker"
PYTHON="/Users/asuramaya/Code/codex/.venv-mlx/bin/python"
DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$CONKER/data/datasets/fineweb10B_sp1024}"
LOG="$CONKER/out/conker3_probe_queue_2026-03-26.log"

mkdir -p "$CONKER/out"
echo "starting conker-3 probe queue" | tee "$LOG"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "missing data root: $DATA_ROOT" | tee -a "$LOG"
  echo "run conker/scripts/link_parameter_golf_data.zsh or see conker/data/PARAMETER_GOLF_DATA.md" | tee -a "$LOG"
  exit 1
fi

run_cell() {
  local variant="$1"
  local json="$CONKER/out/conker3_${variant}_scale_3p0_golf_bridge_seed42_2026-03-26.json"

  if [[ -f "$json" ]]; then
    echo "skip $(basename "$json")" | tee -a "$LOG"
    return
  fi

  echo "run conker3 variant=$variant" | tee -a "$LOG"
  cd "$ROOT"
  "$PYTHON" "$CONKER/scripts/run_conker3_golf_bridge.py" \
    --variant "$variant" \
    --scale 3.0 \
    --learning-rate 5e-4 \
    --quant-bits 6 \
    --data-root "$DATA_ROOT" \
    --seed 42 \
    --steps 600 \
    --seq-len 256 \
    --batch-size 16 \
    --profile pilot \
    --json "$json" | tee -a "$LOG"
  echo "done conker3 variant=$variant" | tee -a "$LOG"
}

for variant in linear_only base gated window4 window16 local_only; do
  run_cell "$variant"
done

echo "conker-3 probe queue complete" | tee -a "$LOG"

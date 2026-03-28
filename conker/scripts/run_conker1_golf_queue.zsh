#!/bin/zsh
set -euo pipefail

ROOT=/Users/asuramaya/Code/carving_machine_v3
CONKER="$ROOT/conker"
PYTHON="/Users/asuramaya/Code/codex/.venv-mlx/bin/python"
DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$CONKER/data/datasets/fineweb10B_sp1024}"
LOG="$CONKER/out/conker1_golf_queue_2026-03-25.log"

mkdir -p "$CONKER/out"
echo "starting conker-1 golf queue" | tee "$LOG"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "missing data root: $DATA_ROOT" | tee -a "$LOG"
  echo "see $CONKER/data/PARAMETER_GOLF_DATA.md for setup" | tee -a "$LOG"
  exit 1
fi

run_cell() {
  local seed="$1"
  local json="$2"

  if [[ -f "$json" ]]; then
    echo "skip $(basename "$json")" | tee -a "$LOG"
    return
  fi

  echo "run conker1 seed $seed" | tee -a "$LOG"
  cd "$ROOT"
  "$PYTHON" "$CONKER/scripts/run_conker1_golf_bridge.py" \
    --data-root "$DATA_ROOT" \
    --seed "$seed" \
    --steps 1000 \
    --seq-len 256 \
    --batch-size 16 \
    --profile pilot \
    --json "$json" | tee -a "$LOG"
  echo "done conker1 seed $seed" | tee -a "$LOG"
}

for seed in 42 43 44; do
  run_cell "$seed" "$CONKER/out/conker1_golf_bridge_seed${seed}_2026-03-25.json"
done

echo "conker-1 golf queue complete" | tee -a "$LOG"

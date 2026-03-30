#!/bin/zsh
set -euo pipefail

ROOT=/Users/asuramaya/Code/carving_machine_v3
CONKER="$ROOT/conker"
PYTHON="/Users/asuramaya/Code/codex/.venv-mlx/bin/python"
DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$CONKER/data/datasets/fineweb10B_sp1024}"
LOG="$CONKER/out/conker2_fairness_queue_2026-03-25.log"

mkdir -p "$CONKER/out"
echo "starting conker-2 fairness queue" | tee "$LOG"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "missing data root: $DATA_ROOT" | tee -a "$LOG"
  echo "run conker/scripts/link_parameter_golf_data.zsh or see conker/data/PARAMETER_GOLF_DATA.md" | tee -a "$LOG"
  exit 1
fi

run_cell() {
  local preset="$1"
  local scale="$2"
  local seed="$3"
  local scale_tag="${scale//./p}"
  local json="$CONKER/out/${preset}_scale_${scale_tag}_golf_bridge_seed${seed}_2026-03-25.json"

  if [[ -f "$json" ]]; then
    echo "skip $(basename "$json")" | tee -a "$LOG"
    return
  fi

  echo "run $preset scale=$scale seed=$seed" | tee -a "$LOG"
  cd "$ROOT"
  "$PYTHON" "$CONKER/scripts/run_golf_scaled_bridge.py" \
    --preset "$preset" \
    --scale "$scale" \
    --data-root "$DATA_ROOT" \
    --seed "$seed" \
    --steps 1000 \
    --seq-len 256 \
    --batch-size 16 \
    --profile pilot \
    --json "$json" | tee -a "$LOG"
  echo "done $preset scale=$scale seed=$seed" | tee -a "$LOG"
}

for preset in hierarchical_v6_fast_mid_delay hierarchical_v6_silenced; do
  for scale in 0.68 0.29; do
    for seed in 42 43 44; do
      run_cell "$preset" "$scale" "$seed"
    done
  done
done

echo "conker-2 fairness queue complete" | tee -a "$LOG"

#!/bin/zsh
set -euo pipefail

ROOT=/Users/asuramaya/Code/carving_machine_v3
CONKER="$ROOT/conker"
LOG="$CONKER/out/bpe_budget_bridge_queue_2026-03-25.log"
PYTHON="/Users/asuramaya/Code/codex/.venv-mlx/bin/python"

mkdir -p "$CONKER/out"
echo "starting conker bpe budget bridge queue" | tee "$LOG"

run_cell() {
  local preset="$1"
  local scale="$2"
  local seed="$3"
  local scale_tag="${scale//./p}"
  local json="$CONKER/out/${preset}_scale_${scale_tag}_bpe_bridge_seed${seed}_2026-03-25.json"

  if [[ -f "$json" ]]; then
    echo "skip $(basename "$json")" | tee -a "$LOG"
    return
  fi

  echo "run $preset scale $scale seed $seed" | tee -a "$LOG"
  cd "$ROOT"
  "$PYTHON" "$CONKER/scripts/run_bpe_budget_bridge.py" \
    --preset "$preset" \
    --scale "$scale" \
    --seed "$seed" \
    --steps 1000 \
    --profile pilot \
    --data "$ROOT/data/text8" \
    --bpe-cache "$ROOT/data/bpe_1024.json" \
    --json "$json" | tee -a "$LOG"
  echo "done $preset scale $scale seed $seed" | tee -a "$LOG"
}

for preset in hierarchical_v6_silenced hierarchical_v6_fast_mid_delay; do
  for scale in 0.375 0.5 0.75 1.0; do
    for seed in 42 43 44; do
      run_cell "$preset" "$scale" "$seed"
    done
  done
done

echo "conker bpe budget bridge queue complete" | tee -a "$LOG"

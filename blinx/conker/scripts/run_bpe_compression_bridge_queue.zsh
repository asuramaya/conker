#!/bin/zsh
set -euo pipefail

ROOT=/Users/asuramaya/Code/carving_machine_v3
CONKER="$ROOT/conker"
LOG="$CONKER/out/bpe_compression_bridge_queue_2026-03-24.log"
PYTHON="/Users/asuramaya/Code/codex/.venv-mlx/bin/python"

mkdir -p "$CONKER/out"
echo "starting conker bpe compression bridge queue" | tee "$LOG"

run_cell() {
  local preset="$1"
  local seed="$2"
  local json="$3"

  if [[ -f "$json" ]]; then
    echo "skip $(basename "$json")" | tee -a "$LOG"
    return
  fi

  echo "run $preset seed $seed" | tee -a "$LOG"
  cd "$ROOT"
  "$PYTHON" "$CONKER/scripts/run_bpe_compression_bridge.py" \
    --preset "$preset" \
    --seed "$seed" \
    --steps 1000 \
    --profile pilot \
    --data "$ROOT/data/text8" \
    --bpe-cache "$ROOT/data/bpe_1024.json" \
    --json "$json" | tee -a "$LOG"
  echo "done $preset seed $seed" | tee -a "$LOG"
}

for preset in hierarchical_v6 hierarchical_v6_silenced hierarchical_v6_fast_mid_delay gru_opt; do
  for seed in 42 43 44; do
    run_cell "$preset" "$seed" "$CONKER/out/${preset}_bpe_bridge_seed${seed}_2026-03-24.json"
  done
done

echo "conker bpe compression bridge queue complete" | tee -a "$LOG"

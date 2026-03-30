#!/bin/zsh
set -euo pipefail

ROOT=/Users/asuramaya/Code/carving_machine_v3
CONKER="$ROOT/conker"
PYTHON="/Users/asuramaya/Code/codex/.venv-mlx/bin/python"
DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$CONKER/data/datasets/fineweb10B_sp1024}"
LOG="$CONKER/out/conker_frontier_golf_queue_2026-03-25.log"

mkdir -p "$CONKER/out"
echo "starting conker frontier golf queue" | tee "$LOG"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "missing data root: $DATA_ROOT" | tee -a "$LOG"
  echo "run conker/scripts/link_parameter_golf_data.zsh or see conker/data/PARAMETER_GOLF_DATA.md" | tee -a "$LOG"
  exit 1
fi

run_single() {
  local preset="$1"
  local seed="$2"
  local json="$3"

  if [[ -f "$json" ]]; then
    echo "skip $(basename "$json")" | tee -a "$LOG"
    return
  fi

  echo "run $preset seed $seed" | tee -a "$LOG"
  cd "$ROOT"
  "$PYTHON" "$CONKER/scripts/run_golf_single_bridge.py" \
    --preset "$preset" \
    --data-root "$DATA_ROOT" \
    --seed "$seed" \
    --steps 1000 \
    --seq-len 256 \
    --batch-size 16 \
    --profile pilot \
    --json "$json" | tee -a "$LOG"
  echo "done $preset seed $seed" | tee -a "$LOG"
}

run_conker1() {
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

for preset in hierarchical_v6_fast_mid_delay hierarchical_v6_silenced; do
  for seed in 42 43 44; do
    run_single "$preset" "$seed" "$CONKER/out/${preset}_golf_bridge_seed${seed}_2026-03-25.json"
  done
done

for seed in 42 43 44; do
  run_conker1 "$seed" "$CONKER/out/conker1_golf_bridge_seed${seed}_2026-03-25.json"
done

echo "conker frontier golf queue complete" | tee -a "$LOG"

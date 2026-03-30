#!/bin/zsh
set -euo pipefail

ROOT=/Users/asuramaya/Code/carving_machine_v3
CONKER="$ROOT/conker"
PYTHON="/Users/asuramaya/Code/codex/.venv-mlx/bin/python"
DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$CONKER/data/datasets/fineweb10B_sp1024}"
LOG="$CONKER/out/conker2_muon_pilot_queue_2026-03-26.log"

mkdir -p "$CONKER/out"
echo "starting conker-2 muon pilot queue" | tee "$LOG"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "missing data root: $DATA_ROOT" | tee -a "$LOG"
  echo "run conker/scripts/link_parameter_golf_data.zsh or see conker/data/PARAMETER_GOLF_DATA.md" | tee -a "$LOG"
  exit 1
fi

wait_for_idle() {
  while ps -A -o command= | rg -q 'run_conker2_golf_bridge.py|run_conker3_golf_bridge.py'; do
    echo "waiting for active conker mlx jobs to finish" | tee -a "$LOG"
    sleep 60
  done
}

run_cell() {
  local tag="$1"
  local momentum="$2"
  local warm_start="$3"
  local warm_steps="$4"
  local json="$CONKER/out/conker2_untied_base_scale_11p5_lr5em4_steps1800_muon_${tag}_golf_bridge_seed42_2026-03-26.json"

  if [[ -f "$json" ]]; then
    echo "skip $(basename "$json")" | tee -a "$LOG"
    return
  fi

  wait_for_idle
  echo "run conker2 muon tag=$tag seed=42" | tee -a "$LOG"
  cd "$ROOT"
  "$PYTHON" "$CONKER/scripts/run_conker2_muon_bridge.py" \
    --variant untied_base \
    --scale 11.5 \
    --learning-rate 5e-4 \
    --muon-momentum "$momentum" \
    --muon-backend-steps 5 \
    --muon-momentum-warmup-start "$warm_start" \
    --muon-momentum-warmup-steps "$warm_steps" \
    --quant-bits 6 \
    --data-root "$DATA_ROOT" \
    --seed 42 \
    --steps 1800 \
    --seq-len 256 \
    --batch-size 16 \
    --profile pilot \
    --json "$json" | tee -a "$LOG"
  echo "done conker2 muon tag=$tag seed=42" | tee -a "$LOG"
}

run_cell mom95_warm500 0.95 0.85 500
run_cell mom99_warm1500 0.99 0.92 1500

echo "conker-2 muon pilot queue complete" | tee -a "$LOG"

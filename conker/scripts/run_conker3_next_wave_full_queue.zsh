#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%Y-%m-%d)"
LOG_PATH="$OUT_DIR/conker3_next_wave_full_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

run_bridge_q46() {
  local half_life_max="$1"
  local scale="$2"
  local steps="$3"
  local seed="$4"
  local suffix="${5:-}"
  local local_hidden_mult="${6:-}"
  local local_scale_override="${7:-}"

  local half_tag="${half_life_max//./p}"
  local scale_tag="${scale//./p}"
  local extra_tag=""
  if [[ -n "$suffix" ]]; then
    extra_tag="_${suffix}"
  fi
  local json_path="$OUT_DIR/conker3_window4_scale_${scale_tag}_steps${steps}_half_life_${half_tag}${extra_tag}_q46_golf_bridge_seed${seed}_${STAMP}.json"

  if [[ -f "$json_path" ]]; then
    echo "skip landed bridge half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed}${extra_tag}" | tee -a "$LOG_PATH"
    return
  fi

  local cmd=(
    python3 conker/scripts/run_conker3_golf_bridge.py
    --data-root "$DATA_ROOT"
    --profile pilot
    --variant window4
    --scale "$scale"
    --steps "$steps"
    --seed "$seed"
    --linear-half-life-max "$half_life_max"
    --quant-bits 4
    --quant-bits 6
    --json "$json_path"
  )
  if [[ -n "$local_hidden_mult" ]]; then
    cmd+=(--local-hidden-mult "$local_hidden_mult")
  fi
  if [[ -n "$local_scale_override" ]]; then
    cmd+=(--local-scale-override "$local_scale_override")
  fi

  echo "run bridge half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed}${extra_tag}" | tee -a "$LOG_PATH"
  "${cmd[@]}" 2>&1 | tee -a "$LOG_PATH"
  echo "done bridge half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed}${extra_tag}" | tee -a "$LOG_PATH"
}

run_mixed_quant() {
  local half_life_max="$1"
  local scale="$2"
  local steps="$3"
  local seed="$4"

  local half_tag="${half_life_max//./p}"
  local scale_tag="${scale//./p}"
  local json_path="$OUT_DIR/conker3_window4_scale_${scale_tag}_steps${steps}_half_life_${half_tag}_mixed_quant_audit_seed${seed}_${STAMP}.json"

  if [[ -f "$json_path" ]]; then
    echo "skip landed mixed quant half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
    return
  fi

  echo "run mixed quant half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
  python3 conker/scripts/run_conker3_mixed_quant_audit.py \
    --data-root "$DATA_ROOT" \
    --profile pilot \
    --scale "$scale" \
    --steps "$steps" \
    --seed "$seed" \
    --linear-half-life-max "$half_life_max" \
    --eval-batches 8 \
    --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  echo "done mixed quant half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed}" | tee -a "$LOG_PATH"
}

run_ttt_probe() {
  local half_life_max="$1"
  local scale="$2"
  local steps="$3"
  local seed="$4"
  local group_count="$5"
  local gate_l2="$6"
  local suffix="$7"

  local half_tag="${half_life_max//./p}"
  local scale_tag="${scale//./p}"
  local json_path="$OUT_DIR/conker3_window4_scale_${scale_tag}_steps${steps}_half_life_${half_tag}_ttt_${suffix}_seed${seed}_${STAMP}.json"

  if [[ -f "$json_path" ]]; then
    echo "skip landed ttt half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed} ${suffix}" | tee -a "$LOG_PATH"
    return
  fi

  echo "run ttt half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed} ${suffix}" | tee -a "$LOG_PATH"
  python3 conker/scripts/run_conker3_ttt_probe.py \
    --data-root "$DATA_ROOT" \
    --profile pilot \
    --scale "$scale" \
    --steps "$steps" \
    --seed "$seed" \
    --linear-half-life-max "$half_life_max" \
    --chunk-len 64 \
    --eval-chunks 32 \
    --group-count "$group_count" \
    --gate-l2 "$gate_l2" \
    --quant-bits 6 \
    --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  echo "done ttt half-life=${half_life_max} scale=${scale} steps=${steps} seed=${seed} ${suffix}" | tee -a "$LOG_PATH"
}

echo "starting conker-3 next wave full queue" | tee "$LOG_PATH"

# Trust or kill the current packed frontier.
run_bridge_q46 16 16.0 1500 43
run_bridge_q46 16 16.0 1500 44

# Push farther on the packed scale law.
run_bridge_q46 16 18.0 1500 42
run_bridge_q46 16 20.0 1500 42

# Local-path shrink row under packed scoring.
run_bridge_q46 16 16.0 1500 42 "shrink75_s20" 0.75 0.20
run_bridge_q46 16 16.0 1500 42 "shrink50_s20" 0.50 0.20

# One more packed-focused TTT follow-up.
run_ttt_probe 16 16.0 1500 42 8 1e-2 "g8_l2e2"

# Protected-int4 audit on the packed winner scale.
run_mixed_quant 16 16.0 1500 42

echo "conker-3 next wave full queue complete" | tee -a "$LOG_PATH"

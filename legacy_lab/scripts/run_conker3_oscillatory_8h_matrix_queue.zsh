#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%Y-%m-%d)"
LOG_PATH="$OUT_DIR/conker3_oscillatory_8h_matrix_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

run_bridge() {
  local scale="$1"
  local osc_frac="$2"
  local seed="$3"
  local local_hidden_mult="${4:-}"
  local local_scale_override="${5:-}"

  local scale_tag="${scale//./p}"
  local frac_tag="${osc_frac//./p}"
  local suffix=""
  local extra_args=()

  if [[ -n "$local_hidden_mult" ]]; then
    local lhm_tag="${local_hidden_mult//./p}"
    suffix="${suffix}_lhm_${lhm_tag}"
    extra_args+=(--local-hidden-mult "$local_hidden_mult")
  fi
  if [[ -n "$local_scale_override" ]]; then
    local ls_tag="${local_scale_override//./p}"
    suffix="${suffix}_lscale_${ls_tag}"
    extra_args+=(--local-scale-override "$local_scale_override")
  fi

  local json_path="$OUT_DIR/conker3_window4_scale_${scale_tag}_steps1500_half_life_16_osc_${frac_tag}${suffix}_q46_golf_bridge_seed${seed}_${STAMP}.json"

  if [[ -f "$json_path" ]]; then
    echo "skip scale=${scale} osc=${osc_frac} seed=${seed}${suffix}" | tee -a "$LOG_PATH"
    return
  fi

  echo "run scale=${scale} osc=${osc_frac} seed=${seed}${suffix}" | tee -a "$LOG_PATH"
  python3 conker/scripts/run_conker3_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --profile pilot \
    --variant window4 \
    --scale "$scale" \
    --steps 1500 \
    --seed "$seed" \
    --linear-half-life-max 16 \
    --oscillatory-frac "$osc_frac" \
    --oscillatory-period-min 4 \
    --oscillatory-period-max 64 \
    --quant-bits 4 \
    --quant-bits 6 \
    "${extra_args[@]}" \
    --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  echo "done scale=${scale} osc=${osc_frac} seed=${seed}${suffix}" | tee -a "$LOG_PATH"
}

echo "starting conker-3 oscillatory 8h matrix queue" | tee "$LOG_PATH"
echo "q1 packed scaling pilots" | tee -a "$LOG_PATH"
run_bridge 12.0 0.75 42
run_bridge 12.0 0.875 42
run_bridge 16.0 0.75 42
run_bridge 16.0 0.875 42
run_bridge 18.0 0.75 42
run_bridge 18.0 0.875 42

echo "q2 local-path byte allocation on 18x / 0.875" | tee -a "$LOG_PATH"
run_bridge 18.0 0.875 42 0.75
run_bridge 18.0 0.875 42 0.50
run_bridge 18.0 0.875 42 "" 0.20

echo "q3 replication on near-cap rows" | tee -a "$LOG_PATH"
run_bridge 16.0 0.75 43
run_bridge 16.0 0.75 44
run_bridge 16.0 0.875 43
run_bridge 16.0 0.875 44
run_bridge 18.0 0.75 43
run_bridge 18.0 0.75 44
run_bridge 18.0 0.875 43
run_bridge 18.0 0.875 44

echo "conker-3 oscillatory 8h matrix queue complete" | tee -a "$LOG_PATH"

#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${CONKER_GOLF_DATA_ROOT:-$ROOT_DIR/conker/data/datasets/fineweb10B_sp1024}"
OUT_DIR="$ROOT_DIR/conker/out"
STAMP="$(date +%Y-%m-%d)"
LOG_PATH="$OUT_DIR/conker3_packed_next_wave_queue_${STAMP}.log"

mkdir -p "$OUT_DIR"

run_bridge() {
  local scale="$1"
  local steps="$2"
  local seed="$3"
  local extra_tag="$4"
  shift 4
  local json_path="$OUT_DIR/conker3_window4_scale_${scale//./p}_steps${steps}_half_life_16_osc_0p875${extra_tag}_q46_golf_bridge_seed${seed}_${STAMP}.json"
  if [[ -f "$json_path" ]]; then
    echo "skip bridge scale=${scale} steps=${steps} seed=${seed}${extra_tag}" | tee -a "$LOG_PATH"
    return
  fi
  echo "run bridge scale=${scale} steps=${steps} seed=${seed}${extra_tag}" | tee -a "$LOG_PATH"
  python3 conker/scripts/run_conker3_golf_bridge.py \
    --data-root "$DATA_ROOT" \
    --profile pilot \
    --variant window4 \
    --scale "$scale" \
    --steps "$steps" \
    --seed "$seed" \
    --linear-half-life-max 16 \
    --oscillatory-frac 0.875 \
    --oscillatory-period-min 4 \
    --oscillatory-period-max 64 \
    --quant-bits 4 \
    --quant-bits 6 \
    "$@" \
    --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  echo "done bridge scale=${scale} steps=${steps} seed=${seed}${extra_tag}" | tee -a "$LOG_PATH"
}

run_pack_train() {
  local scale="$1"
  local steps="$2"
  local seed="$3"
  local extra_tag="$4"
  shift 4
  local json_path="$OUT_DIR/conker3_window4_scale_${scale//./p}_steps${steps}_half_life_16_osc_0p875${extra_tag}_packtrain_q46_seed${seed}_${STAMP}.json"
  if [[ -f "$json_path" ]]; then
    echo "skip packtrain scale=${scale} steps=${steps} seed=${seed}${extra_tag}" | tee -a "$LOG_PATH"
    return
  fi
  echo "run packtrain scale=${scale} steps=${steps} seed=${seed}${extra_tag}" | tee -a "$LOG_PATH"
  python3 conker/scripts/run_conker3_pack_train_bridge.py \
    --data-root "$DATA_ROOT" \
    --profile pilot \
    --scale "$scale" \
    --steps "$steps" \
    --seed "$seed" \
    --linear-half-life-max 16 \
    --oscillatory-frac 0.875 \
    --oscillatory-period-min 4 \
    --oscillatory-period-max 64 \
    --train-quant-bits 6 \
    --quant-bits 4 \
    --quant-bits 6 \
    "$@" \
    --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  echo "done packtrain scale=${scale} steps=${steps} seed=${seed}${extra_tag}" | tee -a "$LOG_PATH"
}

run_subsystem_quant() {
  local json_path="$OUT_DIR/conker3_window4_scale_16p0_steps1500_half_life_16_osc_0p875_subsystem_quant_seed42_${STAMP}.json"
  if [[ -f "$json_path" ]]; then
    echo "skip subsystem quant probe" | tee -a "$LOG_PATH"
    return
  fi
  echo "run subsystem quant probe" | tee -a "$LOG_PATH"
  python3 conker/scripts/run_conker3_subsystem_quant_probe.py \
    --data-root "$DATA_ROOT" \
    --profile pilot \
    --scale 16.0 \
    --steps 1500 \
    --seed 42 \
    --linear-half-life-max 16 \
    --oscillatory-frac 0.875 \
    --oscillatory-period-min 4 \
    --oscillatory-period-max 64 \
    --json "$json_path" 2>&1 | tee -a "$LOG_PATH"
  echo "done subsystem quant probe" | tee -a "$LOG_PATH"
}

echo "starting conker-3 packed next wave queue" | tee "$LOG_PATH"
echo "q1 cap-fit sweep" | tee -a "$LOG_PATH"
run_bridge 16.5 1500 42 ""
run_bridge 17.0 1500 42 ""
run_bridge 17.5 1500 42 ""

echo "q2 longer-train on current under-cap winner" | tee -a "$LOG_PATH"
run_bridge 16.0 1800 42 ""
run_bridge 16.0 2200 42 ""

echo "q3 pack-trained bridge" | tee -a "$LOG_PATH"
run_pack_train 16.0 1800 42 ""

echo "q4 subsystem mixed precision" | tee -a "$LOG_PATH"
run_subsystem_quant

echo "q5 static bank gate pilot" | tee -a "$LOG_PATH"
run_bridge 16.0 1500 42 "_staticgate" --static-bank-gate
run_bridge 16.0 1800 42 "_staticgate" --static-bank-gate

echo "conker-3 packed next wave queue complete" | tee -a "$LOG_PATH"

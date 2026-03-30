# Parallel Exploration

As of March 29, 2026.

This page turns the current `Conker` frontier story into runnable parallel lanes.

## Goal

Measure the active post-reset causal branches without blurring roles:

- keep `Conker-10` as the public bridge baseline
- re-measure `Conker-11` as the strongest current controller line
- push `Conker-13` into a real pilot as the cleaner direct-substrate mutation

## Lane A: Conker-13 Pilot

Purpose:

- test whether the cleaner five-axis controller over the `Conker-3` substrate becomes a real measured branch rather than just a smoke result

Runner:

```bash
python3 conker/scripts/run_conker13_golf_bridge.py \
  --data-root conker/data/datasets/fineweb10B_sp1024 \
  --seed 42 \
  --steps 1000 \
  --seq-len 256 \
  --batch-size 16 \
  --json conker/out/conker13_seed42_parallel.json
```

Success read:

- beats the current `Conker-10` restart result
- shows less uniform controller behavior than the smoke run

## Lane B: Conker-11 Re-measure

Purpose:

- keep the strongest current controller branch on a tracked recipe
- verify whether bounded lag control still holds up once rerun cleanly

Runner:

```bash
python3 conker/scripts/run_conker11_golf_bridge.py \
  --data-root conker/data/datasets/fineweb10B_sp1024 \
  --seed 42 \
  --steps 1000 \
  --seq-len 256 \
  --batch-size 16 \
  --global-lag-cap 0.5 \
  --json conker/out/conker11_seed42_parallel.json
```

Success read:

- stays ahead of the honest `~2.06` floor
- bounded lag mass remains active instead of collapsing back to unrestricted past

## Lane C: Conker-10 Bridge Baseline

Purpose:

- keep a live bridge baseline for `giddy-up`-driven causal proxy work
- compare new controller lines against the strongest current bridge pair

Runner:

```bash
python3 conker/scripts/run_conker10_golf_bridge.py \
  --data-root conker/data/datasets/fineweb10B_sp1024 \
  --seed 42 \
  --steps 120 \
  --seq-len 64 \
  --batch-size 8 \
  --structure-proxy-peak \
  --structure-proxy-candidate4 \
  --json conker/out/conker10_peak_candidate4_parallel.json
```

Success read:

- reproduces the `peak + candidate4` bridge ordering
- remains the comparison point for any new bridge feature or controller overlay

## Coordination Rules

- treat `Conker-10` as the bridge baseline, not the main frontier target
- compare `Conker-11` and `Conker-13` against the clean anchor and against each other
- do not promote a branch into public frontier docs until it has a tracked artifact and a stable writeup
- leave `Conker-12` on hold unless a hard specialization mutation exists first

## Companion Lanes

- `BLINX`: [parallel exploration](../../blinx/docs/PARALLEL_EXPLORATION.md)
- `giddy-up`: [parallel exploration](../../giddy-up/docs/parallel_exploration.md)

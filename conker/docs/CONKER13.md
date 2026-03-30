# Conker-13

**March 29, 2026**

`Conker-13` is the cleaner mutation after `Conker-11/12`.

Instead of hanging a controller on top of the `Conker-9` residual-source bank, it
goes back to the `Conker-3` substrate itself:

- frozen linear multiscale substrate
- parallel local residual coder
- one higher-order controller tensor

The point is to realize the "mask as program" idea without relying on the
`exact1/exact2/delim2/recency` source bank at all.

## Core Idea

`Conker-13` uses a single five-axis learned tensor over:

- lag bucket
- linear mode group
- local window offset
- latent program slot
- effect channel

The effect channels then drive three direct controls:

- linear group gates over the frozen multiscale bank
- local offset gates over the local residual coder input
- a global local-scale multiplier for the final linear/local blend

So the controller is no longer deciding "which handcrafted residual source
opens the mask." It is deciding:

- which parts of the linear memory scaffold matter now
- which offsets inside the local coder matter now
- how hard the local residual should speak at all

## Why It Exists

`Conker-12` proved the higher-order controller path can be real, but it was still
parasitic on the older `Conker-9/11` source stack.

`Conker-13` tests the cleaner thesis:

- maybe the recursive controller should steer the multiscale substrate and local
  coder directly
- maybe the right higher-order object is not "route over residual sources"
- maybe it is "co-program horizon groups and local offsets together"

That makes this branch much closer to the mutation you asked for:

- frozen linear multiscale substrate
- parallel local residual coder
- higher-order `n x n x n x n x n`-style controller alone

## Implementation

- model:
  [conker13.py](/Users/asuramaya/Code/carving_machine_v3/conker/conker/src/conker13.py)
- runner:
  [run_conker13_golf_bridge.py](/Users/asuramaya/Code/carving_machine_v3/conker/conker/scripts/run_conker13_golf_bridge.py)

## Status

- scaffold live
- syntax/import checks passed
- no training score claim yet

## Suggested Smoke Run

```bash
python3 conker/scripts/run_conker13_golf_bridge.py \
  --data-root chronohorn/data/roots/fineweb10B_sp1024 \
  --seed 42 \
  --steps 50 \
  --seq-len 64 \
  --batch-size 8 \
  --profile pilot \
  --variant window4 \
  --scale 10.0 \
  --lag-lookbacks 2,4,8,16,32,64,128,0 \
  --mode-groups 8 \
  --program-slots 4 \
  --program-temperature 1.0 \
  --linear-gate-span 1.0 \
  --local-gate-span 1.0 \
  --local-scale-span 0.5 \
  --json conker/out/conker13_smoke.json
```

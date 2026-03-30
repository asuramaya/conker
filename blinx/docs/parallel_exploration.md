# Parallel Exploration

As of March 29, 2026.

This page turns the active BLINX line into runnable parallel lanes.

## Goal

Keep one repo-text baseline, one payload/dictionary comparison, and one latest-branch probe running at the same time so the next mutation is driven by evidence instead of guesswork.

## Lane A: BLINX-5 Variant Ablation

Purpose:

- keep the dense vs sparse vs adaptive payload result live
- verify that mask-format tweaks are still not the main bottleneck on ordinary text

Runner:

```bash
python3 conker/scripts/run_blinx5_variant_ablation.py \
  conker/docs/BLINX7.md \
  --json-out conker/out/blinx5_variant_parallel.json
```

## Lane B: BLINX-6 Dictionary Factoring Probe

Purpose:

- measure whether shared dictionary packing wins anywhere on the same representative surface
- keep the factoring branch honest against the payload baseline

Runner:

```bash
python3 conker/scripts/run_blinx6_lossless_probe.py \
  conker/docs/BLINX7.md \
  --json-out conker/out/blinx6_parallel.json
```

## Lane C: BLINX-7 Typed Pruning Probe

Purpose:

- keep the latest branch measured against the same input
- confirm whether typed pruning changes anything before opening a new mutation line

Runner:

```bash
python3 conker/scripts/run_blinx7_lossless_probe.py \
  conker/docs/BLINX7.md \
  --json-out conker/out/blinx7_parallel.json
```

## Success Read

- repo-text lanes should tell us whether current active branches still stall at `0` profitable rounds
- if all three lanes stay negative, the next branch should target candidate generation, not another small packing tweak
- if `BLINX-6/7` starts to beat the baseline on a cleaner layered surface, preserve that surface as a tracked regression case

## Coordination Rules

- do not treat repo-text failure as evidence that the structure signal is fake
- do treat repeated `0`-round results as evidence that the bottleneck is upstream of packing polish
- use `BLINX-7` only as the latest pruning ablation, not as proof that pruning is the right long-term direction

## Companion Lanes

- `Conker`: [parallel exploration](../../conker/docs/parallel_exploration.md)
- `giddy-up`: [parallel exploration](../../giddy-up/docs/parallel_exploration.md)

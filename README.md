# Conker

`Conker` is the causal compression and Parameter Golf line extracted from `carving_machine_v3`.

This repo reflects the March 28, 2026 reset. The old pre-fix `Conker-5/7` frontier is historical but invalidated; the clean surviving anchor is patched strict `Conker-4b`, and the active post-reset work is the `Conker-10` through `Conker-13` restart line.

This repo keeps the causal runtime, training code, branch notes, and submission-facing utilities. Closely related tooling now lives in sibling repos:

- [`blinx`](https://github.com/asuramaya/blinx): noncausal lossless-compression and oracle-discovery line
- [`conker-detect`](https://github.com/asuramaya/conker-detect): structural, artifact-boundary, and legality auditing
- [`conker-ledger`](https://github.com/asuramaya/conker-ledger): backlog, lineage, survival, and public validity bundles
- [`giddy-up`](https://github.com/asuramaya/giddy-up): oracle/bridge layer between BLINX discovery and causal Conker features

## Current State

Clean anchor:

- patched strict `Conker-4b`
- full held-out fp16 `2.0971 bpb`
- full held-out int6 `2.1055 bpb`
- int6 artifact `3,730,410` bytes

Best measured restart result:

- `Conker-10` memory-first pilot
- bridge fp16 `2.2397 bpb`
- `int6 2.2608`
- packed memory bytes `12,599,296`

Active research line:

- `Conker-10`: memory-first restart
- `Conker-11`: recursive causal router
- `Conker-12`: higher-order four-axis program tensor
- `Conker-13`: direct five-axis controller over the `Conker-3` substrate

Historical but invalidated:

- old tandem / teacher rows in the `0.55 -> 0.53` range
- invalidated by accidentally trainable structural buffers in `Conker-4b`

Artifact-boundary lesson:

- an old tandem packed artifact inflated to `11.87 MB` because it incorrectly serialized regenerated deterministic substrate
- corrected packing brought the same invalid branch down to about `3.72 MB`

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Link an existing Parameter Golf data export:

```bash
zsh conker/scripts/link_parameter_golf_data.zsh /path/to/parameter-golf/data
```

Run the current memory-first pilot:

```bash
python3 conker/scripts/run_conker10_golf_bridge.py \
  --data-root conker/data/datasets/fineweb10B_sp1024 \
  --seed 42 \
  --steps 500 \
  --seq-len 256 \
  --batch-size 16 \
  --profile pilot \
  --variant window4 \
  --scale 10.0 \
  --json conker/out/example_conker10.json
```

Fresh-process full eval on a saved checkpoint:

```bash
python3 conker/scripts/run_conker7_checkpoint_eval.py \
  --state path/to/checkpoint.npz \
  --data-root conker/data/datasets/fineweb10B_sp1024 \
  --split test \
  --full-split \
  --json conker/out/example_full_eval.json
```

## Repo Layout

- `conker/src/`: branch models and quantization utilities
- `conker/scripts/`: bridge runners, eval helpers, queue scripts, and packers
- `conker/data/`: expected dataset and tokenizer layout
- `conker/docs/`: detailed branch notes and copied lab-era records
- `docs/`: public navigation layer for current frontier, validity, and history
- `conker/submissions/`: packaged candidate artifacts and manifests
- `carving_machine/`: vendored runtime core still used by this repo

## Branch Map

- `Conker-3`: reservoir shaping and oscillatory-bank work
- `Conker-4b`: residual exact experts; now also the strict clean anchor after the reset
- `Conker-5`: old tandem frontier, now invalidated by the `Conker-4b` freeze bug
- `Conker-6`: illegal causal-mask hologram
- `Conker-7`: future-teacher branch, historically strong but inherited the same invalid structural surface
- `Conker-8`: explicit legal weighted-structure rebuild, inert
- `Conker-9`: legal lag-controller rebuild, inert
- `Conker-10`: first memory-first restart
- `Conker-11`: renamed recursive causal router
- `Conker-12`: higher-order four-axis program tensor
- `Conker-13`: direct five-axis controller over linear mode groups and local offsets

## Docs

- [Docs Index](./docs/README.md)
- [Presentation](./docs/presentation.md)
- [Current Frontier](./docs/current_frontier.md)
- [Validity](./docs/validity.md)
- [Negative Results](./docs/negative_results.md)
- [Rescue](./docs/rescue.md)
- [History](./HISTORY.md)
- [Roadmap](./ROADMAP.md)

Archival branch notes:

- [Conker-4b](./conker/docs/CONKER4B.md)
- [Conker-5](./docs/branches/CONKER5.md)
- [Conker-6](./docs/branches/CONKER6.md)
- [Conker-7](./docs/branches/CONKER7.md)
- [Conker-8](./conker/docs/CONKER8.md)
- [Conker-9](./conker/docs/CONKER9.md)
- [Conker-10](./conker/docs/CONKER10.md)
- [Conker-11](./conker/docs/CONKER11.md)
- [Conker-12](./conker/docs/CONKER12.md)
- [Conker-13](./conker/docs/CONKER13.md)

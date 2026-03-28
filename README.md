# Conker

Standalone extraction of the `Conker` compression research line from `carving_machine_v3`.

This repo keeps the runtime and branch history for the tiny causal compression models that survived enough scrutiny to matter. It is deliberately narrower than the original lab tree:

- training/runtime code lives here
- branch docs and submission history live here
- audit, backlog-analysis, and side-channel tooling live in [`conker-detect`](https://github.com/asuramaya/conker-detect)

## Current State

Current best legal `Conker-7` full-holdout row:
- fp16 `0.5283 bpb`
- int6 `0.5315 bpb`
- int6 artifact `4,153,894` bytes

Current clearly legal tandem baseline:
- `Conker-5` replicated full-holdout fp16 `~0.5503 bpb`
- `Conker-5` replicated full-holdout int6 `~0.5540 bpb`

Important negative result:
- `Conker-6` reached much lower scores only by learning a non-causal mask side channel
- that branch is kept here as an invalid but useful lesson, not as a live submission line

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Link an existing Parameter Golf data export:

```bash
zsh conker/scripts/link_parameter_golf_data.zsh /path/to/parameter-golf/data
```

Run the active `Conker-7` bridge:

```bash
python3 conker/scripts/run_conker7_golf_bridge.py \
  --data-root conker/data/datasets/fineweb10B_sp1024 \
  --seed 42 \
  --steps 1000 \
  --seq-len 256 \
  --batch-size 16 \
  --profile pilot \
  --variant window4 \
  --scale 10.0 \
  --json conker/out/example_conker7.json
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

- `carving_machine/`: vendored runtime core that `Conker` still depends on
- `conker/src/`: branch models and quantization utilities
- `conker/scripts/`: bridge runners, eval helpers, and queue scripts
- `conker/data/`: expected dataset and tokenizer layout
- `conker/docs/`: original branch notes carried over from the lab tree
- `docs/`: cleaned standalone navigation layer for current frontier, validity, and history
- `conker/submissions/`: packaged candidate artifacts and manifests

## Branch Map

- `Conker-3`: reservoir shaping and oscillatory-bank work
- `Conker-4b`: residual exact experts over a frozen base
- `Conker-5`: legal tandem frontier
- `Conker-6`: invalid but important causal-mask failure
- `Conker-7`: future-aware teacher, causal student

## Docs

- [Current Frontier](./docs/current_frontier.md)
- [Validity](./docs/validity.md)
- [Negative Results](./docs/negative_results.md)
- [History](./HISTORY.md)
- [Roadmap](./ROADMAP.md)

Branch docs:
- [Conker-5](./docs/branches/CONKER5.md)
- [Conker-6](./docs/branches/CONKER6.md)
- [Conker-7](./docs/branches/CONKER7.md)

## Related Repos

- [`conker-detect`](https://github.com/asuramaya/conker-detect): audit and backlog-analysis tooling extracted from the `Conker-6` failure line

# Giddy-Up

`Giddy-Up` is the bridge repo between `BLINX` oracle/discovery work and causal `Conker` runtime features.

This tree is now absorbed into the main [`conker`](../README.md) repo. It is
preserved here as a subtree, not as a separate canonical repo.

It exists to keep oracle analysis, leave-one-out attack logic, and causal bridge features in one place so the noncausal discovery surface and the legal causal runtime surface stay explicitly separated.

What lives here:

- `giddy_up/oracle.py`: bidirectional context-oracle analysis
- `giddy_up/attack.py`: self-inclusion, future-context, and rulebook-cost attacks on oracle claims
- `giddy_up/features.py`: strictly causal proxy features for `Conker`
- `giddy_up/conker10_adapter.py`: optional replay adapter for saved `Conker-10` bridge checkpoints

Core boundary:

- `BLINX` may generate oracle analyses and labels here
- `Conker` may only consume causal bridge features or exported artifacts from here
- `Conker` must not depend on live bidirectional oracle scoring at evaluation time

Repo layout:

- `giddy_up/`: package code
- `scripts/`: oracle analysis/export CLIs
- `docs/`: boundary and workflow notes

Quick start:

```bash
python3 scripts/run_oracle_analysis.py --json-out out/oracle.json
python3 scripts/run_oracle_attack.py --json-out out/oracle_attack.json
python3 scripts/run_oracle_export.py --radius 2 --jsonl-out out/oracle_labels.jsonl
```

Repo relationships inside this tree:

- [Conker Root](../README.md): canonical private repo
- [Absorbed BLINX](../blinx/README.md): oracle producer subtree

See also:

- [Docs Index](./docs/README.md)
- [Status](./STATUS.md)
- [Salvage Matrix](./SALVAGE_MATRIX.md)
- [Presentation](./docs/presentation.md)
- [Parallel Exploration](./docs/parallel_exploration.md)
- [Rescue](./docs/rescue.md)
- [Architecture](./docs/ARCHITECTURE.md)

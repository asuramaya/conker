# Conker Core For Blinx

Inherited runtime surface from `Conker`, now used as the private base for `BLINX`.

Purpose:

- keep compression-first model code, runners, artifacts, and submission packaging in one subtree
- separate the active `conker` line from the broader archival `carving_machine` lab
- keep the public story reproducible after the March 28, 2026 reset

Layout:

- `src/`: model code
- `scripts/`: bridge runners, queues, eval helpers, and packaging
- `data/`: tokenizer caches, manifests, and setup notes
- `out/`: JSON artifacts, logs, checkpoints, and audits
- `docs/`: branch notes, frontier summaries, and reset documents
- `submissions/`: packaged candidate artifacts

Current inherited state:

- the old pre-fix `Conker-5/7` frontier is invalidated
- root cause: `Conker-4b` structural buffers were accidentally trainable because `freeze(keys=...)` was called incorrectly
- strict retrain collapses the old `~0.55 / 0.53` story to about `2.10 bpb`
- `Conker-8` and `Conker-9` are honest strict rebuilds and both sit near the same strict floor
- `Conker-10` is the first memory-first rebuild; its first pilot is real but weak
- `BLINX-1` is the first noncausal lossless branch layered on top
- `BLINX-4` is the control-first successor to `BLINX-3`
- `BLINX-5` is the payload-first successor to `BLINX-4`
- `BLINX-5a/b/c` are explicit payload ablations for that branch
- `BLINX-6` is the dictionary-factoring successor to `BLINX-5`
- `BLINX-7` is the latest typed-pruning successor to `BLINX-6`
- `giddy_up` is now treated as a sibling bridge repo; local `conker/src/giddy_up` is a producer-side mirror, not the canonical home

Start here:

- [CURRENT_FRONTIER.md](./docs/CURRENT_FRONTIER.md)
- [BLINX7.md](./docs/BLINX7.md)
- [BLINX_ORACLE.md](./docs/BLINX_ORACLE.md)
- [BLINX_LITERATURE.md](./docs/BLINX_LITERATURE.md)
- [VALIDITY_WORKFLOW.md](./docs/VALIDITY_WORKFLOW.md)
- [Conker Legacy Handoff](https://github.com/asuramaya/conker/blob/main/legacy_lab/HANDOFF.md)

Companion public tools:

- [`giddy-up`](https://github.com/asuramaya/giddy-up): oracle / bridge layer between BLINX and causal Conker
- [`conker`](https://github.com/asuramaya/conker): causal sibling repo
- [`conker-detect`](https://github.com/asuramaya/conker-detect): structural / legality auditor
- [`conker-ledger`](https://github.com/asuramaya/conker-ledger): backlog / lineage / survival analyzer

Unified workflow:

- use `blinx` as the repo entrypoint and treat this inherited `conker/` subtree as BLINX internals
- keep audit logic in `conker-detect`
- keep bundle/report logic in `conker-ledger`
- see [VALIDITY_WORKFLOW.md](./docs/VALIDITY_WORKFLOW.md)

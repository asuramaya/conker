# Conker

Submission-facing compression branch for the Parameter Golf line.

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

Current repo state:

- the old pre-fix `Conker-5/7` frontier is invalidated
- root cause: `Conker-4b` structural buffers were accidentally trainable because `freeze(keys=...)` was called incorrectly
- strict retrain collapses the old `~0.55 / 0.53` story to about `2.10 bpb`
- `Conker-8` and `Conker-9` are honest strict rebuilds and both sit near the same strict floor
- `Conker-10` is the first memory-first rebuild; its first pilot is real but weak
- `Conker-11`: renamed recursive causal router
- `Conker-12`: higher-order four-axis program tensor
- `Conker-13`: direct five-axis controller over the `Conker-3` substrate
- `giddy_up` is now treated as a sibling bridge repo; local `conker/src/giddy_up` is a consumer-side mirror, not the canonical home

Start here:

- [CURRENT_FRONTIER.md](./docs/CURRENT_FRONTIER.md)
- [COMPRESSION_MATRIX.md](./docs/COMPRESSION_MATRIX.md)
- [VALIDITY_WORKFLOW.md](./docs/VALIDITY_WORKFLOW.md)
- [HANDOFF.md](../legacy_lab/HANDOFF.md)
- [CONKER11.md](./docs/CONKER11.md)
- [CONKER12.md](./docs/CONKER12.md)
- [CONKER13.md](./docs/CONKER13.md)

Companion public tools:

- [`giddy-up`](https://github.com/asuramaya/giddy-up): oracle / bridge layer between BLINX and causal Conker
- [`blinx`](https://github.com/asuramaya/blinx): noncausal oracle and lossless-compression sibling
- [`conker-detect`](https://github.com/asuramaya/conker-detect): structural / legality auditor
- [`conker-ledger`](https://github.com/asuramaya/conker-ledger): backlog / lineage / survival analyzer

Unified workflow:

- use `conker` as the umbrella entrypoint
- keep audit logic in `conker-detect`
- keep bundle/report logic in `conker-ledger`
- keep bridge logic canonical in `giddy-up`
- see [VALIDITY_WORKFLOW.md](./docs/VALIDITY_WORKFLOW.md)

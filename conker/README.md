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

Start here:

- [CURRENT_FRONTIER.md](/Users/asuramaya/Code/carving_machine_v3/conker/docs/CURRENT_FRONTIER.md)
- [COMPRESSION_MATRIX.md](/Users/asuramaya/Code/carving_machine_v3/conker/docs/COMPRESSION_MATRIX.md)
- [HANDOFF.md](/Users/asuramaya/Code/carving_machine_v3/conker/HANDOFF.md)

Companion public tools:

- `conker-detect`: structural / legality auditor
- `conker-ledger`: backlog / lineage / survival analyzer

Unified workflow:

- use `conker` as the umbrella entrypoint
- keep audit logic in `conker-detect`
- keep bundle/report logic in `conker-ledger`
- see [VALIDITY_WORKFLOW.md](/Users/asuramaya/Code/carving_machine_v3/conker/docs/VALIDITY_WORKFLOW.md)

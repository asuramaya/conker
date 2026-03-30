# Scripts

Queue runners, eval scripts, packaging helpers, and submission prep live here.

Good contents:

- one-cell-at-a-time experiment queues
- Parameter Golf data setup helpers
- FineWeb / token-loss and bpb evaluation wrappers
- artifact packing and manifest generation
- reproducibility scripts for submission candidates
- umbrella helpers that hand detector outputs to the ledger packager
- official-data baselines and `Conker-1` runners

Current entrypoints:

- `link_parameter_golf_data.zsh`: symlink an existing official export into `conker/data`
- `run_golf_single_bridge.py`: official-data single-expert bridge
- `run_conker_frontier_golf_queue.zsh`: baseline-vs-`Conker-1` frontier queue
- `run_conker2_golf_bridge.py`: official-data bridge for the linear-plus-correction branch
- `run_conker2_golf_queue.zsh`: `Conker-2` seed queue
- `run_blinx1_lossless_probe.py`: noncausal lossless-compression probe for the `BLINX-1` branch
- `run_blinx4_lossless_probe.py`: controlled lossless-compression probe for the `BLINX-4` branch
- `run_blinx5_lossless_probe.py`: payload-aware lossless-compression probe for the `BLINX-5` branch
- `run_blinx5_variant_probe.py`: single-variant probe for `BLINX-5a/b/c`
- `run_blinx5_variant_ablation.py`: ablation matrix for `BLINX-5a/b/c`
- `run_blinx6_lossless_probe.py`: direct-vs-shared dictionary lossless-compression probe for the `BLINX-6` branch
- `run_blinx7_lossless_probe.py`: typed-dictionary lossless-compression probe for the `BLINX-7` branch
- `run_blinx_oracle_analysis.py`: analysis-only bidirectional uniqueness scan over local BLINX files
- `run_blinx_oracle_export.py`: per-position BLINX oracle label export as JSONL for Chronohorn supervision
- `run_validity_bundle.py`: generate starter manifests and assemble validity bundles through the sibling `conker-ledger` repo

`run_blinx_oracle_export.py` emits a versioned per-position JSONL stream plus derived supervision fields that are directly supported by the corpus scan:

- `schema_version = 2` is written into every row and the command summary so Chronohorn can reject shape drift explicitly

- `candidate_set_leq_4` and `candidate_set_leq_8` for the bidirectional inclusive support set
- `required_radius`, computed as the smallest radius up to `--required-radius-max` that makes the current center byte deterministic in the scanned corpus
- `future_uplift` and `self_inclusion_uplift` as per-position deterministic deltas between the relevant support regimes
- `clean_bridge_score`, `memory_trust`, and `bridge_confidence` as support-size-derived confidence signals
- `teacher_candidate_tokens` and `teacher_candidate_counts` as a ranked leave-out candidate teacher, sorted by descending support then token id

The JSONL rows keep the earlier label fields intact for backward compatibility and add `schema_version` as the first field. `required_radius` is `null` when no deterministic radius is found up to `--required-radius-max`.

Use `--required-radius-max` to widen the bounded search for `required_radius` when you want more than the export radius itself.

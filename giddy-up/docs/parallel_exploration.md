# Parallel Exploration

As of March 29, 2026.

This page turns the bridge layer into parallel operational lanes without blurring the legality boundary.

## Goal

Refresh the oracle side, refresh the attack side, and export causalizable offline labels in parallel so `Conker` always consumes a current bridge surface rather than stale intuition.

## Lane A: Oracle Analysis Refresh

Purpose:

- keep the raw bidirectional structure probe current on the target tree

Runner:

```bash
python3 scripts/run_oracle_analysis.py \
  --json-out out/oracle_parallel.json
```

## Lane B: Oracle Attack Refresh

Purpose:

- quantify self-inclusion and future-context uplift before any bridge feature is promoted

Runner:

```bash
python3 scripts/run_oracle_attack.py \
  --json-out out/oracle_attack_parallel.json
```

## Lane C: Offline Label Export

Purpose:

- export causalizable labels for downstream bridge experiments without importing live oracle scoring into `Conker`

Runner:

```bash
python3 scripts/run_oracle_export.py \
  --radius 2 \
  --jsonl-out out/oracle_labels_parallel.jsonl
```

## Success Read

- the `candidate4` family remains strong enough to justify further shaping work
- attack results make the self-inclusion and future-context uplift explicit
- exported labels are ready for `Conker` bridge experiments without moving the legality boundary

## Coordination Rules

- keep live oracle scoring on the BLINX side only
- treat exported labels and causal bridge features as the only legal handoff into `Conker`
- do not promote `agreement_mass` over `candidate4` unless a new bridge run beats the current baseline cleanly

## Companion Lanes

- `Conker`: [parallel exploration](../../conker/docs/parallel_exploration.md)
- `BLINX`: [parallel exploration](../../blinx/docs/PARALLEL_EXPLORATION.md)

# Asura

**March 29, 2026**

`Asura` is a first-pass legal recursive-routing prototype built after the `Conker-8/9` result that horizon choice alone was too weak.

Goal:

- keep the strict causal `Conker-4b` legality surface
- keep `Conker-9`'s fixed legal lag buckets
- add a second controller layer that decides:
  - which residual source owns the current step
  - which sources are allowed to open the candidate mask
  - when the residual path should abstain

This is the minimal honest version of the earlier “mask as program” idea:

- no trainable full `n x n` mask
- no diagonal / upper-triangle freedom
- no hidden structural buffers pretending to be fixed
- recursive control is expressed only through:
  - legal lag-bucket mixing
  - source routing
  - opener routing
  - residual-strength control

Implementation:

- model:
  [asura.py](/Users/asuramaya/Code/carving_machine_v3/conker/src/asura.py)
- runner:
  [run_asura_golf_bridge.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/run_asura_golf_bridge.py)

Architecture:

1. `Conker-3` base produces causal base features.
2. A lag controller mixes fixed legal lookback buckets, as in `Conker-9`.
3. Residual count sources are recomputed under that lag mix.
4. A source-ownership controller emits a softmax over active sources plus abstain.
5. An opener controller decides how strongly each source opens the candidate mask.
6. A residual-strength controller scales the whole residual path.

So `Asura` is recursive in the narrow sense that:

- lag routing changes the source features
- source routing changes which residual channels matter
- opener routing changes where the residual is even allowed to act

but every stage stays past-only and auditable.

Current patch note:

- the unrestricted `0` lag bucket is no longer part of the same free softmax as
  the finite lag buckets
- instead it is a separately capped gate, which keeps the controller from
  collapsing into "always use full history" and forces some mass to remain on
  bounded legal horizons

Current status:

- scaffold live
- no score claim yet
- meant to answer whether recursive causal routing can recover any of the old hidden-control gain without reintroducing illegal learned structure

Suggested first run:

```bash
python3 conker/scripts/run_asura_golf_bridge.py \
  --data-root conker/data/datasets/fineweb10B_sp1024 \
  --seed 42 \
  --steps 1000 \
  --seq-len 256 \
  --batch-size 16 \
  --profile pilot \
  --variant window4 \
  --scale 10.0 \
  --enable-exact3 \
  --enable-special2 \
  --enable-number2 \
  --enable-markup2 \
  --enable-attr2 \
  --enable-delim2 \
  --source-controller-temperature 1.0 \
  --opener-controller-temperature 1.0 \
  --residual-controller-temperature 1.0 \
  --global-lag-cap 0.5 \
  --json conker/out/asura_seed42.json
```

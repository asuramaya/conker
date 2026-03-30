# Asura-Tetra

**March 29, 2026**

`Asura-Tetra` is a tractable attempt at the “higher-order mask program” idea.

It is not a literal dense `n x n x n x n` tensor over sequence positions. That
would be too large, too brittle, and impossible to justify inside this branch.

Instead, it builds a compact four-axis controller tensor over:

- lag bucket
- routed source
- latent program slot
- effect channel

The effect channels currently modulate:

- source-route logits
- opener logits

So the model now has one explicit higher-order interaction surface:

- lag choice can change which latent program slot is active
- the program slot can change which source gets route mass
- the same program can change which source opens the candidate mask

That is the first honest attempt in this tree to realize the “mask as
recursive program” idea with an actual four-axis learned object while keeping
the legality surface fixed and causal.

Implementation:

- model:
  [asura_tetra.py](/Users/asuramaya/Code/carving_machine_v3/conker/src/asura_tetra.py)
- runner:
  [run_asura_tetra_golf_bridge.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/run_asura_tetra_golf_bridge.py)

Suggested smoke run:

```bash
python3 conker/scripts/run_asura_tetra_golf_bridge.py \
  --data-root chronohorn/data/roots/fineweb10B_sp1024 \
  --seed 42 \
  --steps 50 \
  --seq-len 64 \
  --batch-size 8 \
  --profile pilot \
  --variant window4 \
  --scale 10.0 \
  --enable-exact3 \
  --enable-special2 \
  --enable-number2 \
  --enable-markup2 \
  --enable-attr2 \
  --enable-delim2 \
  --global-lag-cap 0.5 \
  --program-slots 4 \
  --program-temperature 1.0 \
  --program-route-span 1.0 \
  --program-opener-span 1.0 \
  --json conker/out/asura_tetra_smoke.json
```

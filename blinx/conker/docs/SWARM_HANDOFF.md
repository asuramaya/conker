# Swarm Handoff

Rescue status:

- this is preserved operator-facing handoff material from the inherited Conker research surface
- active BLINX priorities may differ from the tasks framed below
- use [Presentation](../../docs/presentation.md), [Rescue](../../docs/rescue.md), and [Current Frontier](./CURRENT_FRONTIER.md) before treating this file as current direction

## Mission

Help push `conker/` toward a real Parameter Golf submission.

Treat this as a compression problem, not an architecture-aesthetics problem:

- score = lower `bpb`
- official data matters more than archive lore
- weirdness is allowed only if it buys bits

## Ground Rules

- Work in `conker/`, not the archival `carving_machine/` branch.
- Use official golf data as truth.
- Do not treat `text8` or old char-level results as sacred.
- MLX on the Mac is still the research ground truth.
- Optimize for:
  - exact official-style `bpb`
  - learned payload bytes
  - train time

## Current Confirmed Frontier

- Speed frontier:
  - `Conker-2 linear_only` widened
- Score frontier, replicated:
  - `Conker-2 untied_base scale 10.0`
  - `2.2650 bpb`
  - `11.779 MB` learned payload at post-train `int6`
- `12.0x` with the default recipe bent badly
- `12.0x` salvage on seed `42` worked:
  - `lr=5e-4`, `steps=1500`
  - `2.1993 bpb`
  - not replicated yet

## Current Live Question

Is the salvaged `12.0x` row real across seeds, or was seed `42` lucky?

That is the highest-value unresolved question right now.

## Best Leads

1. Optimizer/recipe scaling
- Large-width `Conker-2` is recipe-sensitive.
- Muon-like hidden-matrix optimization is now a live idea.
- Simple stabilizers may matter as much as optimizer choice.

2. Packaging realism
- We are now close enough to the payload regime that packing/quantization is no longer hypothetical.
- `int6` is currently the best low-bit tradeoff.

3. Hybrid mechanism
- The correction path still beats widened `linear_only` on score.
- We still do not have a sharp positive account of what it buys.

## Good Agent Tasks

1. Replicate the best salvaged `12.0x` recipe.
2. Test Muon-on-2D-hidden-weights with AdamW on embeddings/biases/head.
3. Find a minimal stabilizer for large-width training.
4. Audit exact packed artifact bytes for the current `10.0x` and salvaged `12.0x` branches.
5. Propose a contest-shaped training recipe once the `12.0x` replication lands.

## Bad Agent Tasks

- Reviving char-level stories from the archive
- New decorative mechanisms without a compression argument
- More width sweeps before the `12.0x` salvage row is resolved
- Treating `linear_only` speed wins as score wins

## Important Files

- [conker/HANDOFF.md](https://github.com/asuramaya/conker/blob/main/legacy_lab/HANDOFF.md)
- [conker/docs/COMPRESSION_MATRIX.md](https://github.com/asuramaya/conker/blob/main/conker/docs/COMPRESSION_MATRIX.md)
- [conker/docs/CONKER2.md](https://github.com/asuramaya/conker/blob/main/conker/docs/CONKER2.md)
- [conker/scripts/run_conker2_golf_bridge.py](https://github.com/asuramaya/conker/blob/main/conker/scripts/run_conker2_golf_bridge.py)

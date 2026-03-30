# BLINX-4

**March 29, 2026**

`BLINX-4` is the first attempt to apply the post-reset `Conker-10..13` lesson directly to the current `BLINX-3` codec surface.

The lesson is not "copy the causal model."
The lesson is:

- identify the real compression substrate
- put control on the true decision points
- keep accounting tied to packed size, not to removal fraction alone

For `BLINX-3`, the true substrate is already compact:

- choose a context radius
- build a unique bidirectional context dictionary
- greedily punch removable centers
- decide whether the round is worth keeping
- choose how much pair grammar to spend on the final rulebook

`BLINX-4` adds a small discrete controller over those choices instead of making the dictionary logic itself more elaborate.

## Mutation

Code:

- [blinx4.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/src/blinx4.py)
- [run_blinx4_lossless_probe.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/scripts/run_blinx4_lossless_probe.py)

The controller is phase-based.
Each round it picks among a small library of strategies:

- `probe`
- `harvest`
- `consolidate`
- `refine`

Each phase controls:

- which radii are allowed
- minimum occurrence threshold
- minimum removed-byte threshold
- dictionary cap
- candidate pair-rule budgets
- whether the local search favors net profit or discovery among non-lossmaking candidates

This is the `BLINX` analogue of the newer `Conker` work:

- direct control over the real substrate
- no fake "higher-order magic" on a side surface
- explicit pressure against rulebook bloat

## Expected Read

What should improve if this direction is correct:

- fewer rounds that remove bytes but lose after side-data accounting
- smaller, more selective dictionaries
- better choice of when to spend pair-grammar budget

What would count as failure:

- the controller just reproduces `BLINX-3` behavior with extra complexity
- packed size does not improve over the same source file
- the early `probe` phase keeps accepting rounds that do not pay for themselves

## Status

Current status: branch scaffold live, local probe pending frontier judgment.

This should be judged against `BLINX-3` on the same file surface by:

- exact roundtrip after serialize/deserialize
- delta vs source `zlib`
- phase usage by round
- whether dictionary caps and pair-budget search actually bind

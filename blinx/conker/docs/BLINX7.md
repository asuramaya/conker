# BLINX-7

Date: March 29, 2026

`BLINX-7` is a `BLINX-6` successor that keeps the same wire format and shared-dictionary packing, but changes how each round chooses its dictionary.

## Mutation

`BLINX-6` still built one flat candidate dictionary per round, then asked whether `direct` or `shared` packing was cheaper. That worked once the synthetic test had genuinely reusable round dictionaries, but it still mixed all context families into one global top-count pool.

`BLINX-7` adds a typed pruning surface:

- `global`: old top-count pruning
- `typed`: stratified pruning over simple byte-class buckets
- `adaptive`: evaluate both and keep the cheaper round

The byte classes are intentionally cheap:

- whitespace
- digits
- uppercase
- lowercase
- brackets
- quotes
- connector punctuation like `._-/:\,;|`
- other printable ASCII
- non-ASCII / fallback

The type bucket for a candidate is:

- context radius
- left-edge byte class
- center-byte class
- right-edge byte class

The important constraint is that `BLINX-7` does **not** relax global decodability. The typed policy is only a pruning strategy over already globally unique context keys. The final dictionary still maps raw context bytes to one byte value.

## First Read

The first pilot is negative but useful.

- On the same repo surfaces where `BLINX-6` stayed at `0` rounds, `BLINX-7` also stays at `0` rounds.
- On the clean disjoint-layer synthetic where `BLINX-6` finally wins, `BLINX-7` matches it exactly rather than beating it.
- For the deepest tested disjoint case, both `global` and `typed` land at `5043` packed bytes versus source `zlib = 6823`, with `3` accepted rounds and `shared` dictionary packing.

So the current evidence is that typed pruning alone is not the unlock. It is a fair ablation, but the bottleneck still looks like candidate generation and reusable structure, not just dictionary pollution.

## Why This Exists

The `BLINX-6` disjoint-layer falsification showed the codec starts to work once rounds produce reusable shared entries across clean context layers. That suggests the next mutation should preserve those layers during candidate selection, instead of letting a few high-frequency buckets monopolize the round dictionary.

So `BLINX-7` is testing a narrower claim:

- maybe the branch does not need a richer payload format
- maybe it just needs a less destructive dictionary-selection policy

## Wire Format

`BLINX-7` deliberately reuses the `BLINX-6` payload format for now.

That keeps the ablation honest:

- same mask encoding options
- same `direct` vs `shared` dictionary packing
- only dictionary selection changes

The stats payload reports `wire_format = "blx6-compatible"` to make that explicit.

## Expected Read

If `BLINX-7` helps:

- repo surfaces should still mostly look hard
- synthetic layered corpora should prefer `typed` or `adaptive`
- the gain should show up before any new packing trick is introduced

If it does not help:

- the problem is probably not dictionary pollution alone
- the next branch should mutate candidate generation itself, not just pruning

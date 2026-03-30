# BLINX-1

**March 28, 2026**

`BLINX-1` is a separate branch from the causal `Conker` line.

Goal:

- lossless compression
- no left-to-right scoring constraint
- treat the document as a field that can be simplified by iterative removal
- make the decoder run the same reconstruction steps in reverse

This is the first concrete scaffold for the “computation through removal” idea:

- frozen field: the current byte sequence
- learned or discovered rule: which bytes are predictable from surrounding context
- reconstruction: put the removed bytes back from a transmitted context dictionary

## Current Prototype

Code:

- [blinx1.py](/Users/asuramaya/Code/carving_machine_v3/conker/src/blinx1.py)
- [run_blinx1_lossless_probe.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/run_blinx1_lossless_probe.py)

The first implementation is intentionally simple and fully reversible:

1. scan the current byte stream for contexts of the form `(left_byte, right_byte)`
2. keep only contexts that always imply the same center byte
3. greedily remove non-adjacent center bytes that match those unique contexts
4. transmit:
   - the round mask
   - the context dictionary
   - the remaining survivor bytes
5. decode by replaying the rounds in reverse

This is not a good compressor yet. It is a clean lossless probe for:

- how much of text can be removed under cheap simultaneous context rules
- how expensive the side data is
- whether iterative “punching” can beat just storing the field directly

## Why It Exists

The post-reset `Conker` story says:

- causal left-to-right modeling is the competition line
- but several discoveries were really about control, removal, and structural simplification
- a separate branch should test that directly instead of forcing it back into `bpb`

So `BLINX-1` is not a new causal frontier model.
It is the first noncausal lossless-compression branch.

## Success Criteria

The first milestone is modest:

- exact roundtrip on arbitrary files
- transparent accounting of:
  - removed bytes
  - remaining survivor bytes
  - side-dictionary cost
  - packed size after a standard compressor like `zlib`

The second milestone, if the removal fraction is real, is to replace the hand-built context rule with:

- richer noncausal context
- learned punch policies
- iterative reconstruction schedules
- eventually masked/discrete denoising rather than fixed byte-context lookup

## Status

Current status: scaffold live, no frontier claim.

This branch should be judged by:

- lossless roundtrip correctness
- whether the removal protocol buys anything over direct compression
- what it teaches about “denoise everywhere at once” for text as a compression process

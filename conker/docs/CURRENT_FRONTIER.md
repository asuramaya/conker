# Current Frontier

**March 28, 2026**

This document is the repo-level reset after the `Conker-4b` freeze bug was found and fixed.

## Summary

- The old pre-fix `Conker-5/7` frontier is invalidated.
- Root cause: structural buffers that were meant to be fixed were accidentally trainable.
- The strongest historical contaminated row was `Conker-7` at `0.5283 fp16 / 0.5315 int6` on full held-out eval.
- After the fix, the strict retrain collapses to about `2.10 bpb`.
- `Conker-8` and `Conker-9` are honest rebuilds around that strict floor and do not recover the lost gain.
- `Conker-10` is the first memory-first rebuild. Its first pilot is real, but worse than the strict floor and mostly routes back to the neural base.

## What Broke

In `Conker-4b`, these structural tensors were intended to be frozen:

- `causal_mask`
- `delimiter_mask`
- `number_mask`
- `special_mask`
- `markup_mask`
- `attr_mask`
- related lookup / class buffers

Instead, they stayed trainable because `freeze(keys=...)` was called incorrectly. The result was:

- the saved `causal_mask` stopped being strictly lower-triangular
- class masks stopped being fixed binary supports
- exact-count features became weighted soft counts
- future positions inside the chunk could influence current predictions through the learned structure

This invalidates the old `Conker-5/7` frontier as a clean causal submission line.

There was a second, separate artifact-boundary bug in the same historical line:

- old packed tandem artifacts accidentally serialized regenerated deterministic substrate
- that inflated one tandem payload to about `11.87 MB`
- corrected packing dropped the same branch to about `3.72 MB`

So the repo reset has two different lessons:

- model illegality: structural buffers learned things they were not supposed to learn
- artifact-boundary illegality: the old packer counted code-generated substrate as stored payload

## Historical But Invalidated

These numbers are still useful as research artifacts, but they are not the live legal frontier:

- `Conker-5` tandem: about `0.55 bpb`
- `Conker-7` warm-start teacher branch: `0.5283 fp16 / 0.5315 int6`

Interpretation:

- those rows taught us that learned structure and control mattered far more than we expected
- but they were achieved through accidentally trainable structural buffers, not the declared strict architecture

## Current Honest Frontier

Patched strict retrain of `Conker-4b`, seed `42`, `window4 / 10x / seq256 / batch16 / 1000 / lr5e-4`:

- bridge fp16: `2.0589 bpb`
- full held-out fp16: `2.0971 bpb`
- full held-out int6: `2.1055 bpb`
- int6 artifact: `3,730,410` bytes

This is the last clean anchor inside the current code line.

## Rebuilds After The Fix

### Conker-8

Goal:

- make the hidden structural learning explicit and legal
- learn past-only lag profiles and within-support token weights

Result:

- first strict pilot: `2.0600 bpb`
- ablations removing learned lag or learned support weights: still `~2.0598`
- stronger support-mask amplitudes go `NaN`

Read:

- legal weighted-mask rebuild is effectively inert
- stronger mask learning is unstable

### Conker-9

Goal:

- replace weighted masks with a legal causal controller over fixed lag buckets

Result:

- first pilot: `2.0600 bpb`

Read:

- the controller learns real nonzero parameters
- but legal lag selection alone does not move the strict floor

### Conker-10

Goal:

- switch to memory-first:
  - packed training unigram
  - packed training bigram
  - hashed trigram buckets
  - normalized backoff
  - learned controller mixing memory with the base

First pilot:

- seed `42`
- `500` steps
- `1,000,000` packed train tokens
- `2,048` trigram buckets
- bridge fp16: `2.2397 bpb`
- `int6`: `2.2608`
- packed memory bytes: `12,599,296`

Checkpoint readout:

- controller mean source weights on a held-out batch:
  - base: `0.9914`
  - unigram: `0.0056`
  - bigram: `0.0019`
  - trigram: `0.0011`

Read:

- the memory tables are populated and available
- the controller learned real parameters
- but it still routed almost everything back to the base

## What We Learned

1. The main hidden power in the invalidated line was structural control, not the reservoir itself.
2. Legal weighted masks do not recover that gain.
3. Legal lag selection alone does not recover that gain.
4. First-pass packed memory also does not help unless the model actually trusts it.
5. From here on, every new branch must be audited before its numbers matter.

## Live Next Steps

No scaling yet. The next honest experiments are:

1. `Conker-10` memory-only / cache-only baseline with the same tables.
2. Fixed interpolation over `[base, unigram, bigram, trigram]`.
3. Packed prior + online score-first cache.
4. Controller features that explicitly expose cache confidence / agreement.

## Related Files

- [CONKER4B.md](/Users/asuramaya/Code/carving_machine_v3/conker/docs/CONKER4B.md)
- [CONKER8.md](/Users/asuramaya/Code/carving_machine_v3/conker/docs/CONKER8.md)
- [CONKER9.md](/Users/asuramaya/Code/carving_machine_v3/conker/docs/CONKER9.md)
- [CONKER10.md](/Users/asuramaya/Code/carving_machine_v3/conker/docs/CONKER10.md)
- [COMPRESSION_MATRIX.md](/Users/asuramaya/Code/carving_machine_v3/conker/docs/COMPRESSION_MATRIX.md)
- [HANDOFF.md](/Users/asuramaya/Code/carving_machine_v3/conker/HANDOFF.md)

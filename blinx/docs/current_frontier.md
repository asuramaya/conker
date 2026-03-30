# Current Frontier

## Reset

This standalone repo now follows the post-reset `Conker` line.

The old pre-fix `Conker-5/7` frontier is invalidated for two separate reasons:

- structural buffers in `Conker-4b` were accidentally trainable, so old tandem and teacher rows learned illegal structure
- an old tandem packer also serialized regenerated deterministic substrate, inflating one historical artifact to about `11.87 MB`

Those are distinct failures:

- model illegality
- artifact-boundary illegality

## Clean Anchor

Current clean anchor is patched strict `Conker-4b`:

- seed `42`
- `window4 / 10x / seq256 / batch16 / 1000 / lr=5e-4`
- bridge fp16 `2.0589 bpb`
- full held-out fp16 `2.0971 bpb`
- full held-out int6 `2.1055 bpb`
- int6 artifact `3,730,410` bytes

This is the last score in this line that survives:

- strict structural freezing
- fresh-process full eval
- packed-artifact accounting

## Historical But Invalidated

These numbers remain useful as research artifacts, but they are not the live legal frontier:

- `Conker-5` tandem around `0.55 bpb`
- `Conker-7` warm-start teacher row at `0.5283 fp16 / 0.5315 int6`

They taught the right architectural lesson:

- learned structure and control mattered much more than expected

But they were achieved through accidentally trainable structural buffers, not the declared strict architecture.

## Honest Rebuilds

### Conker-8

- explicit legal lag profile + support-mask weighting
- first strict pilot `2.0600 bpb`
- ablations removing lag or support learning stay `~2.0598`

Read:

- legal weighted-mask rebuild is effectively inert

### Conker-9

- legal lag-bucket controller over fixed past horizons
- first pilot `2.0600 bpb`

Read:

- the controller learns a real lag policy
- horizon selection alone does not move the strict floor

### Conker-10

- first memory-first restart:
  - packed unigram
  - packed bigram
  - hashed trigram buckets
  - normalized posterior backoff
  - learned mixer with the neural base

First pilot:

- bridge fp16 `2.2397 bpb`
- `int6 2.2608`
- packed memory bytes `12,599,296`

Direct falsifications on the same tables:

- memory-only `6.0892 bpb`
- fixed heavy memory blend `2.8436 bpb`

Read:

- the first packed-memory construction is weak
- the problem is not just that the learned mixer was too conservative

## Active Direction

The live branch is now memory-first, not teacher-first:

- better packed memory construction
- packed prior + score-first online cache
- only then causal control on top

## Related

- [Validity](./validity.md)
- [Negative Results](./negative_results.md)
- [Conker-4b archival note](../conker/docs/CONKER4B.md)
- [Conker-10 archival note](../conker/docs/CONKER10.md)

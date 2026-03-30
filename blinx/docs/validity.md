# Validity

`Conker` validity is now defined by two separate boundaries:

1. model legality
2. artifact-boundary legality

The old tandem / teacher frontier failed both in different ways.

## Current Standard

Any serious `Conker` claim should survive all of these:

1. full held-out evaluation
- slice evals are for search only
- claim numbers should come from fresh-process full validation

2. packed-artifact evaluation
- if the submission-shaped object is `int6`, the real score is the packed `int6` score
- fp16 checkpoint numbers are not enough

3. actual artifact bytes on disk
- use serialized artifact size, not helper estimates
- raw replay checkpoints and packed artifacts are different objects

4. trained-checkpoint audits, not init audits
- legality checks must run on what training actually produced
- the dead `Conker-6` and old tandem line both looked cleaner before training than after

5. strict structural freezing where intended
- if a tensor is supposed to be fixed, verify it is not in `trainable_parameters()`
- this is where the old `Conker-4b` line broke

6. no accidental packing of regenerated substrate
- deterministic code-generated tensors should not cross the artifact boundary as stored payload
- this is where the old tandem packer broke

## The Two Historical Failures

### Model illegality

In the old `Conker-5/7` line, structural buffers that were meant to be frozen stayed trainable:

- `causal_mask`
- token-class masks
- support masks
- related lookup buffers

That let the model learn illegal structure and dominate the score.

### Artifact-boundary illegality

Separately, an old tandem packer serialized regenerated deterministic substrate such as:

- `base.linear_kernel`
- `base.linear_in_proj`
- `base.linear_decays`

That inflated one historical tandem artifact to about `11.87 MB`.
Corrected packing brought the same invalid branch down to about `3.72 MB`.

## Practical Ladder

Current conservative ladder for any future branch:

1. bridge metric for search only
2. saved checkpoint replay in a fresh process
3. full held-out fp16 eval
4. full held-out packed eval
5. real packed bytes
6. structural and artifact-boundary audit

If a result dies anywhere on that ladder, it is not the live frontier.

## Current Honest Anchor

The last score in this repo that survives the full ladder is patched strict `Conker-4b`:

- full held-out fp16 `2.0971 bpb`
- full held-out int6 `2.1055 bpb`
- int6 artifact `3,730,410` bytes

Everything below that in historical branch notes is either:

- invalidated
- exploratory
- or both

For external auditing and public validity bundles, use:

- [`conker-detect`](https://github.com/asuramaya/conker-detect)
- [`conker-ledger`](https://github.com/asuramaya/conker-ledger)

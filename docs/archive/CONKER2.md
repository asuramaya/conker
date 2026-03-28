# Conker-2

`Conker-2` is the speed pivot branch.

It keeps the frozen-dynamics thesis but changes the main path:

- main path: frozen linear multiscale substrate, computed in parallel across the sequence
- correction path: much smaller frozen nonlinear expert
- output: learned mixer over the two logit streams plus a small residual bias

## Why It Exists

`Conker-1` won the first official-data row, but it paid for that with two sequential experts.

`Conker-2` asks a sharper question:

- can most of the compression signal come from a parallel frozen linear substrate?
- does only a small nonlinear correction remain necessary?

If yes, this is the first branch that points toward both better wall-clock use and a more contest-shaped compressor.

## Pass Condition

`Conker-2` survives only if it does both:

- stays near or above `Conker-1` on official-data token loss / bits-per-token
- materially improves throughput or train time relative to `Conker-1`

## Current Read

After the cleanup patch and the full official 3-seed mechanism row:

- tying the duplicated embeddings reduced trainable params to `438,242`
- the tied variants came back:
  - `base`: `2.6530` bpb
  - `linear_only`: `2.6228` bpb
  - `equal_logit`: `2.6106` bpb
  - `correction_only`: `3.0142` bpb

So the current evidence is:

- the frozen linear substrate is doing the real work
- the nonlinear correction path has not yet earned its cost
- the tied cleanup hurt relative to the original untied `Conker-2` score of `2.5767` bpb, so embedding sharing must now be treated as a direct ablation target

That ablation is now resolved:

- `untied_base` replicated at `2.5742` bpb over seeds `42/43/44`
- `untied_equal_logit` replicated at `2.5872` bpb

So the current winner in the family is the untied hybrid, and the next question is not “should embeddings be tied?” anymore. It is “does `untied_base` still win after matched-budget fairness, and then after an FFT/scan linear path?”

Those rows are now closed:

- matched-budget fairness did not kill `untied_base`
- the FFT path is slightly faster but slightly worse on score

The current next question is narrower:

- does widening `untied_base` buy enough bits to justify its extra learned budget once low-bit quantization is used as the payload lever?

The quant row says yes to low-bit compression, but only up to a point:

- `fp16`: `2.5828` bpb at `1.797 MB`
- `uniform int6`: `2.5969` bpb at `0.590 MB`
- `uniform int4`: `2.6636` bpb at `0.527 MB`

So `int6` is the live scale-up setting and `int4` is already the wrong side of the cliff.

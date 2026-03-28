# 3-Bit PolarQuant

`3-bit PolarQuant` is a promising compression idea, but for `conker` it is still a research translation problem, not an immediate build.

## What The Published Work Actually Shows

The strongest current sources are about **KV cache** quantization, not packed model weights:

- Google Research's `PolarQuant` uses random preconditioning plus polar transformation and avoids per-block normalization overhead
- a separate `PolarQuant` line for key-cache quantization reports strong `3-bit` results versus naive int3 baselines

From the published comparisons:

- Google emphasizes that avoiding normalization can save roughly `1 to 2 bits` of overhead per quantized value in normalization-based schemes
- the openreview `PolarQuant33` comparisons show that `3-bit` polar quantization can stay much closer to bf16 than naive `Int-3` on key caches
- the same appendix notes `QJL` reaches about `3.13` effective bits for a `3-bit` key-cache schema

Sources:

- https://research.google/pubs/polarquant-quantizing-kv-caches-with-polar-transformation/
- https://openreview.net/pdf/f871dc73d430806b70e1ed32e0172b4f8e785c4e.pdf

## Why This Does Not Transfer Cleanly Yet

`Conker` currently cares about **packed trainable weights**, not KV caches.

That means:

- the published distortion results do not directly apply
- the memory-overhead story is related, but not identical
- we would need a new packer and dequant path for weight matrices

## Why It Still Matters

Our current `uniform int3` story is bad.

So a structured `3-bit` scheme is one of the few realistic ways to test whether the current `int3` cliff is about:

- low precision itself
- or the wrong quantization geometry

## The Right `Conker` Translation

If we build this, the first version should be:

- large trainable matrices only
- rowwise or blockwise random preconditioning
- pairwise polar packing on the preconditioned vectors
- no immediate full inference-kernel rewrite
- first test as **offline pack -> dequant -> eval** against current uniform `int3`

## Decision Rule

Do **not** build this before the Muon pilot is resolved.

If Muon improves the under-cap frontier, optimizer work has a clearer claim on time.

If Muon does not help much, `3-bit PolarQuant` becomes a better next gamble because it attacks the current `int3` cliff directly.

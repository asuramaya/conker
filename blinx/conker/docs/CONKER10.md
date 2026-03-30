# Conker-10

**March 28, 2026**

`Conker-10` is the first memory-first rebuild after the strict collapse of the old `Conker-5/7` line.

Design:

- fixed packed training-memory priors built from `1,000,000` train tokens
- exact unigram prior
- exact bigram counts
- hashed trigram counts with `2,048` buckets
- proper normalized posterior backoff:
  - `p1 = (c1 + alpha_bigram * p_uni) / (n1 + alpha_bigram)`
  - `p2 = (c2 + alpha_trigram * p1) / (n2 + alpha_trigram)`
- learned causal controller mixes:
  - base predictor
  - unigram prior
  - bigram posterior
  - trigram posterior

Implementation:

- model:
  [conker10.py](/Users/asuramaya/Code/carving_machine_v3/conker/src/conker10.py)
- runner:
  [run_conker10_golf_bridge.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/run_conker10_golf_bridge.py)

First pilot:

- seed `42`
- `window4 / 10x / seq256 / batch16 / 500 / lr5e-4`
- packed tokens: `1,000,000`
- trigram buckets: `2,048`
- `alpha_bigram=4.0`
- `alpha_trigram=2.0`

Result:

- bridge fp16 `2.2397 bpb`
- `int6 2.2608`
- `int4 2.6028`
- packed memory bytes: `12,599,296`
- artifact:
  [conker10_packedmix_b2048_seq256_steps500_lr5e4_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker10_packedmix_b2048_seq256_steps500_lr5e4_seed42_2026-03-28.json)

Checkpoint readout on a held-out batch:

- mean controller source weights:
  - base: `0.9914`
  - unigram: `0.0056`
  - bigram: `0.0019`
  - trigram: `0.0011`
- controller parameters are not zero:
  - hidden weight mean abs `0.2080`
  - output weight mean abs `0.1816`
- average bigram row mass on that batch: `4945.0`
- average trigram bucket mass on that batch: `643.6`

Read:

- the memory sources are present and well populated
- the controller did learn real parameters
- but it still routed almost all probability mass back to the neural base
- in this first form, packed training memory is not helping enough to displace the base

Interpretation:

- the memory-first direction is still the right thing to test after PR `#1030`
- but this exact `Conker-10` instantiation is weak
- the next useful variants should not just add more table bytes
- they should test:
  - stronger fixed interpolation baselines
  - direct cache-only / memory-only baselines
  - online score-first cache on top of packed priors
  - controller features that explicitly expose cache confidence / agreement

## Conker-10a falsifications

**March 28, 2026**

Two direct follow-ups on the same packed tables (`1,000,000` train tokens, `2,048` trigram buckets):

1. Memory-only baseline

- mode: `memory_only`
- no training (`--skip-train`)
- bridge fp16 `6.0892 bpb`
- artifact:
  [conker10a_memoryonly_b2048_seq256_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker10a_memoryonly_b2048_seq256_seed42_2026-03-28.json)

2. Fixed interpolation baseline

- mode: `fixed_interp`
- fixed weights:
  - base `0.20`
  - unigram `0.05`
  - bigram `0.25`
  - trigram `0.50`
- seed `42`, `500` steps
- bridge fp16 `2.8436 bpb`
- `int6 2.8753`
- artifact:
  [conker10a_fixedblend_b2048_seq256_steps500_lr5e4_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker10a_fixedblend_b2048_seq256_steps500_lr5e4_seed42_2026-03-28.json)

Read:

- the packed memory tables by themselves are very weak in this form
- forcing high memory weight is much worse than the learned mixer
- so the first `Conker-10` failure is not just “controller too timid”
- the tables themselves, as currently constructed, are not strong enough to compete with the base

That narrows the next honest branch to:

- better packed memory construction
- or packed prior + online score-first cache

and rules out the simple story that the learned mixer merely refused to use already-good memory.

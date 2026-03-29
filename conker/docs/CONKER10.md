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

## Structure-Proxy Pilot

**March 28, 2026**

A tiny causal proxy experiment was added to the controller:

- feature set:
  - trigram entropy
  - trigram peak probability
  - base-vs-trigram top-token agreement
- switch:
  - `--structure-proxy`

Run conditions:

- synthetic code-like shard corpus under `/tmp/conker_blinx_proxy_data`
- seed `42`
- `window4 / 1x / seq64 / batch8 / 80 / lr1e-3`
- packed tokens: `8,000`
- trigram buckets: `256`

Result:

- baseline `Conker-10` test bits/token: `0.1435`
- proxy `Conker-10` test bits/token: `0.1336`
- the proxy improved the short synthetic pilot while staying causal

Interpretation:

- the controller responds to an explicit structure-confidence proxy
- this does not repair the original FineWeb frontier yet
- but it is a minimal causal signal worth keeping for the next memory/controller branch

## Giddy-Up Bridge

**March 28, 2026**

The structure-proxy work is now split into a bridge layer called `giddy_up`:

- BLINX side:
  - offline oracle targets from bidirectional context analysis
- Conker side:
  - strictly causal proxy features computed from prefix-only memory/base distributions

Current Conker bridge implementation lives in:

- [giddy_up/features.py](/Users/asuramaya/Code/carving_machine_v3/conker-standalone/conker/src/giddy_up/features.py)
- [run_conker10_structure_proxy_matrix.py](/Users/asuramaya/Code/carving_machine_v3/conker-standalone/conker/scripts/run_conker10_structure_proxy_matrix.py)
- [giddy_up/conker10_adapter.py](/Users/asuramaya/Code/carving_machine_v3/conker-standalone/conker/src/giddy_up/conker10_adapter.py)

Current read on the bounded synthetic pilot:

- raw `candidate4` proxy was too sharp and could catastrophically blow up
- softening it to `top4_mass * (1 - normalized_entropy)` fixed that instability
- the best stable pair in the small synthetic runs is:
  - `peak + soft candidate4`

This remains a direction-finding result only. It has not yet been run as an official FineWeb submission path.

## Sampled Legality Audit

**March 28, 2026**

The current saved `Conker-10` bridge checkpoint was replayed through `conker-detect` using a runtime adapter and a sampled legality pass.

Audit artifact:

- [conker10_giddyup_probe_legality_vocab1024_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker-detect/out/conker10_giddyup_probe_legality_vocab1024_2026-03-28.json)

Sampled checks passed:

- normalization over the explicit `1024`-token alphabet
- repeatability
- future-suffix invariance
- answer-mask invariance

Interpretation:

- the current bridge features look provisionally legal under the repo's stricter causal standard
- this is still a sampled behavioral audit, not a full proof
- BLINX itself remains illegal as an eval-time scorer and is only valid as an offline oracle

## FineWeb Giddy-Up Pilot

**March 28, 2026**

First real FineWeb-side `Giddy-Up` comparison on the official dataset surface:

- data root:
  - `conker/data/datasets/fineweb10B_sp1024`
- recipe:
  - `window4 / 1x / seq64 / batch8 / 120 / lr1e-3`
  - packed tokens `200,000`
  - trigram buckets `2,048`
  - seed `42`

Results:

- baseline:
  - [json](/Users/asuramaya/Code/carving_machine_v3/conker-standalone/conker/out/conker10_giddyup_fineweb_baseline_seed42_2026-03-28.json)
  - test bits/token `8.3499`
- `peak`:
  - [json](/Users/asuramaya/Code/carving_machine_v3/conker-standalone/conker/out/conker10_giddyup_fineweb_peak_seed42_2026-03-28.json)
  - test bits/token `8.0457`
- `peak + soft candidate4`:
  - [json](/Users/asuramaya/Code/carving_machine_v3/conker-standalone/conker/out/conker10_giddyup_fineweb_peak_candidate4_seed42_2026-03-28.json)
  - [state](/Users/asuramaya/Code/carving_machine_v3/conker-standalone/conker/out/conker10_giddyup_fineweb_peak_candidate4_seed42_2026-03-28.npz)
  - test bits/token `8.0348`

Legality audit on the saved FineWeb bridge checkpoint:

- [conker10_giddyup_fineweb_peak_candidate4_legality_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker-detect/out/conker10_giddyup_fineweb_peak_candidate4_legality_2026-03-28.json)

Sampled checks passed:

- normalization
- repeatability
- future-suffix invariance
- answer-mask invariance

Read:

- the same bridge pair that survived the synthetic softening step also helps on the real FineWeb path
- the gain is modest but directional
- the current saved FineWeb bridge checkpoint also passes the sampled behavioral legality audit
- this is still a small pilot, not a frontier claim

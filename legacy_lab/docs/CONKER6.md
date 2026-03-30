# Conker-6

`Conker-6` is the legal cache branch:

- normalized causal cache
- `Conker-3` base demoted to optional smoother/gate
- no two-pass rescoring
- no non-normalized blend tricks

## First Prototype

Initial implementation:

- [conker6.py](/Users/asuramaya/Code/carving_machine_v3/conker/src/conker6.py)
- [run_conker6_golf_bridge.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/run_conker6_golf_bridge.py)
- legality attack script:
  [audit_conker6_legality.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/audit_conker6_legality.py)

The first branch used:

- order-1/2/3 exact-history cache
- normalized probability output
- three blend modes:
  - `cache_only`
  - `fixed_blend`
  - `learned_gate`

## Main Results

Full held-out `cache_only`, `seq_len=256 batch=16 steps=1000`, seed `42`:

- fp16 `0.0721 bpb`
- `int4 0.0714`
- `int6 0.0740`
- payload `~0.254 MB`
- artifact:
  [conker6_cacheonly_seq256_steps1000_full_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_cacheonly_seq256_steps1000_full_2026-03-28.json)

This was originally believed to be numerically normalized and causally clean, based on:

- [conker6_legality_cacheonly_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_legality_cacheonly_2026-03-28.json)

That audit was performed on a fresh untrained mask and should **not** be trusted as evidence about the trained winner.

## What The Tiny Trainables Actually Are

The reported `66,560` trainables are only:

- `causal_mask`: `(256, 256)` = `65,536`
- `vocab_axis`: `(1024,)` = `1,024`

Critical ablations:

`fixed_vocabulary + fixed_causal_mask`, full held-out:

- fp16 `5.7521 bpb`
- `int4 5.7612`
- `int6 5.7430`
- artifact:
  [conker6_cacheonly_seq256_steps1000_fixedbuffers_full_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_cacheonly_seq256_steps1000_fixedbuffers_full_2026-03-28.json)

`fixed_vocabulary + learnable_causal_mask`, full held-out:

- fp16 `0.0721 bpb`
- `int4 0.0714`
- `int6 0.0740`
- artifact:
  [conker6_cacheonly_seq256_steps1000_maskonly_full_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_cacheonly_seq256_steps1000_maskonly_full_2026-03-28.json)

So the current `Conker-6` win is overwhelmingly the learned `causal_mask`. `vocab_axis` is irrelevant in practice.

## What Failed

`fixed_blend`, same recipe:

- full held-out `0.4004 bpb`
- much worse than raw `cache_only`
- artifact:
  [conker6_wittenbell_seq256_steps1000_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_wittenbell_seq256_steps1000_2026-03-28.json)

`learned_gate`, same recipe:

- started in the right region
- went `NaN` late in training
- artifact:
  [conker6_learnedgate_seq256_steps1000_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_learnedgate_seq256_steps1000_2026-03-28.json)

Flattened `exact_context_span=512`:

- raw `cache_only` row was already poor on the bridge slice
- legality audit says it is still normalized and causal in flat-stream order
- but it fails row-wise future invariance because it crosses batch-row boundaries
- audit:
  [conker6_legality_cacheonly_exactspan512_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_legality_cacheonly_exactspan512_2026-03-28.json)

So `exact_context_span>0` should be treated cautiously. It is not the main line.

## Paper-Driven Backoff Tests

Following Jurafsky & Martin Chapter 3:

- [Chapter 3 PDF](https://web.stanford.edu/~jurafsky/slp3/3.pdf)

I tested two normalized smoothing families:

`witten_bell`, full held-out:

- `0.4004 bpb`

`absolute_discount`, full held-out:

- `1.3097 bpb`

Artifacts:

- [conker6_wittenbell_seq256_steps1000_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_wittenbell_seq256_steps1000_2026-03-28.json)
- [conker6_absdisc_seq256_steps1000_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_absdisc_seq256_steps1000_2026-03-28.json)

So textbook smoothing did **not** improve this branch. Raw hard exact backoff is currently much stronger on this task.

## Two Sharp Probes

`cache_only`, but only `1` training step:

- full held-out `0.3333 bpb`
- artifact:
  [conker6_cacheonly_seq256_steps1_full_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_cacheonly_seq256_steps1_full_2026-03-28.json)

This means the branch is **not** just a no-training inference cache. The small trainable subset matters.

`cache_only`, but `exact3` disabled:

- full held-out `0.0983 bpb`
- artifact:
  [conker6_cacheonly_seq256_steps1000_noexact3_full_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_cacheonly_seq256_steps1000_noexact3_full_2026-03-28.json)

So `exact3` buys about `0.026 bpb` relative to the legal `0.0721` winner. The branch is genuinely higher-order exact context, not just lower-order cacheing.

## Learned Mask Geometry

Post-hoc mask ablation on the trained legal winner:

- script:
  [run_conker6_mask_ablation.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/run_conker6_mask_ablation.py)
- artifact:
  [conker6_mask_ablation_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_ablation_2026-03-28.json)

Key results:

- baseline: `0.0721 bpb`
- nonnegative clamp: `0.0714`
- magnitude prune `90%`: `4.4732`
- magnitude prune `95%`: `4.3152`
- magnitude prune `98%`: `4.1847`
- row-top-k `16`: `4.5754`
- row-top-k `8`: `4.3589`

Deviation read:

- `mag_0.90`: cosine `0.3291`, L2 deviation `0.9055`
- `mag_0.95`: cosine `0.2409`, L2 deviation `0.9307`
- `mag_0.98`: cosine `0.1492`, L2 deviation `0.9482`
- `rowtopk_16`: cosine `0.3596`, L2 deviation `0.8948`
- `rowtopk_8`: cosine `0.2585`, L2 deviation `0.9263`

Interpretation:

- the learned causal mask is **not** naively sparse
- post-hoc pruning destroys the score even when keeping `10-12%` of entries
- row-wise top-k pruning is no better than global magnitude pruning
- within the active lower-triangular support, the learned mask is already effectively nonnegative; the tiny clamp gain should be treated as eval noise or irrelevant off-support cleanup

So the next structure attack is not magnitude sparsity. It is sign, normalization, and smoother low-rank or banded parameterizations of the dense causal weighting.

## Matrix Dump And Structure Attack

I dumped the trained legal `256x256` causal mask as raw data and visuals:

- raw matrix:
  [conker6_mask_geometry_2026-03-28.mask.npy](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.mask.npy)
- csv:
  [conker6_mask_geometry_2026-03-28.mask.csv](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.mask.csv)
- lag profile:
  [conker6_mask_geometry_2026-03-28.lag_profile.csv](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.lag_profile.csv)
- singular values:
  [conker6_mask_geometry_2026-03-28.singular_values.csv](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.singular_values.csv)
- heatmaps:
  [mask](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.mask.png)
  [lag-mean](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.lag_mean_mask.png)
  [lag-mean diff](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.lag_mean_diff.png)
- summary:
  [conker6_mask_geometry_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.json)

Raw geometry:

- active entries live in a very narrow range: about `0.887` to `1.057`
- mean active weight: `0.9587`
- lag means live in about `0.9459 .. 1.0298`
- mean lag std: `0.0187`
- top singular-value energy:
  - top `8`: `53.5%`
  - top `16`: `62.7%`
  - top `32`: `72.1%`

So visually the matrix looks almost trivial:

- near-uniform positive lower triangle
- close to lag-only / Toeplitz
- fairly compressible in SVD energy

But the structure attacks say the opposite functionally:

- `toeplitz_mean`: `5.7521 bpb`
- `toeplitz_band_32`: `4.9290`
- `toeplitz_band_64`: `5.3167`
- `lowrank_8_masked`: `5.7522`
- `lowrank_16_masked`: `5.7521`
- `row_normalized`: `5.7521`

The most important shock is `toeplitz_mean`:

- cosine to the learned mask: `0.9998`
- L2 deviation: `0.0188`
- but score collapses from `0.0721` to `5.7521`

So the mask is not using coarse geometry alone. Tiny structured perturbations destroy it. The branch is acting like a dense causal weighting with extremely sharp functional dependence on exact per-entry values.

## Cross-Seed Residual Test

I ran the critic's distinguishing experiment directly:

- script:
  [run_conker6_seed_residual_compare.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/run_conker6_seed_residual_compare.py)
- summary:
  [conker6_seed_residual_compare_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_seed_residual_compare_2026-03-28.json)
- per-seed dumps:
  [seed42](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_seed_residual_compare_2026-03-28_seed42)
  [seed43](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_seed_residual_compare_2026-03-28_seed43)
  [seed44](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_seed_residual_compare_2026-03-28_seed44)

Full held-out scores stay the same across seeds:

- seed `42`: `0.07209 bpb`
- seed `43`: `0.07141`
- seed `44`: `0.071999`

The stable part is the coarse mask geometry:

- raw mask cosine across seeds: about `0.9996`
- Toeplitz-mean cosine across seeds: about `0.999997 .. 1.000001`
- Toeplitz-mean Pearson across seeds: about `0.964 .. 0.968`

But after subtracting each seed's Toeplitz mean, the residuals are almost orthogonal:

- residual cosine:
  - `42/43`: `0.0909`
  - `42/44`: `0.0999`
  - `43/44`: `0.0857`
- residual sign agreement:
  - `42/43`: `0.518`
  - `42/44`: `0.517`
  - `43/44`: `0.510`

So the critic's distinction mostly lands:

- the data-driven object that is stable across seeds is the lag profile / Toeplitz mean
- the tiny non-Toeplitz residual that actually makes the branch work is **not** stable across seeds

That means the current `Conker-6` win should not be read as a robust discovered residual pattern in the data. It looks more like:

- a very stable coarse causal geometry
- plus tiny seed-specific microstructure that each lands in a different equally good basin

This also explains the earlier shock:

- `toeplitz_mean` stays extremely close to the trained mask
- but removing the seed-specific residual collapses the score

So the next attack should be on the mechanism of that microstructure, not on finding a single canonical residual pattern.

## Retraction: Trained-Mask Legality Failure

The saved trained mask changed the story completely.

Direct trained-mask causality / normalization check:

- patched saved seed-42 mask into a fresh `Conker-6` model
- row-wise future-token attack:
  - `max_abs_logit_diff = 18.4207`
  - `mean_abs_logit_diff = 0.000296`
- flat-stream future attack:
  - `max_abs_logit_diff = 0.0`
- normalization:
  - `sum_min = 1.0`
  - `sum_max = 1.4260`
  - `max_abs_sum_error = 0.4260`

So the trained winner is **not** a legal normalized row-wise causal model.

Residual-substitution attack:

- artifact:
  [conker6_residual_substitution_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_residual_substitution_2026-03-28.json)
- `baseline_seed42_full`: `0.0720933 bpb`
- `baseline_seed42_strictlower`: `5.7521`
- `toeplitz_lower_plus_upperdiag_seed42`: `0.0721075`
- `toeplitz + uniform noise`: about `0.072106`
- `toeplitz + lag-matched noise`: `0.072106`
- `toeplitz + shuffled residual`: `0.0721066`
- `toeplitz + sign-randomized residual`: `0.0721071`
- `toeplitz42 + residual43`: `0.0720930`
- `toeplitz42 + residual44`: `0.0720924`

Interpretation:

- the lower-triangle residual does **not** carry the magic
- random or swapped lower residuals work just as well
- the whole `0.072` effect survives as long as the learned diagonal and upper-triangle part is preserved
- zeroing that non-causal part collapses the score to `5.7521`

So the current `Conker-6` branch is not a legal frontier. It is a useful falsification:

- the apparent win came from non-row-causal, non-normalized mask geometry
- the lower residual microstructure is mostly irrelevant
- the upper/diagonal learned mask entries are the real source of the score

## Current Read

`Conker-6` in its current form is invalid as a legal submission branch.

The next work should focus on:

- explicitly causal row-constrained mask parameterizations
- normalized probability checks on the **trained** model, not fresh init
- legal cache mechanics that do not depend on upper-triangle or diagonal mask entries

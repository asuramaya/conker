`Conker-4b` is the residual-corrector redesign of `Conker-4`.

Structure:
- frozen `Conker-3` base expert
- sparse residual calibration over exact-count features
- no direct probability-space mixing

Residual sources tested:
- `exact1`: exact 1-token backoff counts
- `exact2`: exact 2-token backoff counts
- `exact3`: exact 3-token backoff counts
- `special2`: exact-2 continuation restricted to rare / identifier-like tokens
- `number2`: exact-2 continuation restricted to number-like tokens
- `recency`: decayed token-history counts

Seed-42 results on official golf data, `window4 / 10x / half_life=16 / osc=87.5%`:

- reference `Conker-3 1000`: `2.0865 bpb`
- `Conker-4b full 500`: `1.9547 bpb`
- `Conker-4b full 550`: unstable (`NaN`)
- `Conker-4b full 1000 lr=1e-4`: poor (`3.5480 bpb`)
- `Conker-4b no_exact2 500`: `2.1429 bpb`
- `Conker-4b no_recency 500`: `2.0079 bpb`
- `Conker-4b no_recency 1000`: `1.8818 bpb`
- `Conker-4b exact2 only 1000`: `2.0364 bpb`
- `Conker-4b exact1 only 1000`: `2.0413 bpb`
- `Conker-4b exact2 + exact3 1000`: `1.8811 bpb`
- `Conker-4b exact1 + exact2 + exact3 1000`: `1.8374 bpb`
- `Conker-4b exact1 + exact2 1000`, but `exact1` support-only: `1.8807 bpb`
- `Conker-4b exact1 + exact2 + exact3 1000`, with `exact1` support-only: `1.8389 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 1000`, with `exact1` and `delim2` support-only: `1.8187 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 1000`, support-only: `1.8102 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + number2 1000`, support-only: `1.8095 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 1000`, support-only: `1.8061 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + urlpath2 1000`, support-only: `1.8178 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + urlpath2 1000`, support-only: `1.8047 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + markup2 1000`, support-only: `1.8098 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 1000`, support-only: `1.8034 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + attr2 1000`, support-only: `1.8103 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, support-only: `1.8018 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, support-only + dynamic reservoir-conditioned support gates: `1.7985 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, gate-only learned expert selection over fixed source maps: `1.7974 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, ownership softmax + abstain: unstable (`NaN`)
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, ownership top-1: `2.0049 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, ownership top-2: `2.0369 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, gate-only + overlap penalty `1e-3`: `1.7979 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, gate-only + overlap penalty `3e-3`: `1.7987 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, gate-only + overlap penalty `1e-3` + gate temperature `0.75`: `1.7975 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, gate-only + input projection `random`: `1.7982 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, gate-only + input projection `orthogonal_rows`: `1.7988 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, gate-only + input projection `kernel_energy`: `1.7982 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, gate-only + input projection `split_banks`: `1.7980 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, gate-only + input projection `split_banks`, 3-seed means: fp16 `1.7990`, `int6 1.7990`, `int4 1.7987`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1500`, gate-only: `1.7970 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 2000`, gate-only: `1.7980 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, gate-only, `seq_len=512 batch=8`: `1.7975 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1500`, gate-only, `seq_len=512 batch=8`: `1.7878 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1500`, gate-only, `seq_len=512 batch=8`, 3-seed means: fp16 `1.7871`, `int6 1.7868`, `int4 1.7851`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, gate-only, `seq_len=1024 batch=4`: `1.7908 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1500`, gate-only, `seq_len=1024 batch=4`: `1.7883 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, gate-only, `seq_len=256 batch=16`, `exact_context_span=512`: unstable (`NaN`)
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, gate-only, `seq_len=256 batch=16`, `exact_context_span=1024`: unstable (`NaN`)
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, tandem (`freeze_base=False`), `seq_len=256 batch=16`, `lr=5e-4`: `0.5624 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, tandem (`freeze_base=False`), `seq_len=256 batch=16`, `lr=3e-4`: unstable (`NaN`)
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, tandem (`freeze_base=False`), `seq_len=512 batch=8`, `lr=5e-4`: `0.5717 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, tandem (`freeze_base=False`), `seq_len=256 batch=16`, `lr=5e-4`, 3-seed means: fp16 `0.5615`, `int6 0.5745`, `int4 0.7412`
- fresh-process checkpoint re-eval of the saved seed-43 tandem state: `test_bpb 0.5648`
- sampled train/val exact-window overlap audit (`20k` train vs `5k` val windows): `0/5000` overlaps at lengths `32/64/128/256`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + entity2 1000`, support-only: unstable (`NaN`)
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + entity2 1000`, support-only: unstable (`NaN`)
- `Conker-4b exact1 + exact2 + exact3 + delim2 + stack2 1000`, support-only: `1.8182 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + stack2 1000`, support-only: `1.8032 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + delimsub2 1000`, support-only: unstable (`NaN`)
- `Conker-4b exact1 + exact2 + exact3 + wordclass2 1000`, support-only: unstable (`NaN`)
- `Conker-4b exact1 + exact2 + exact3 + delim2 + wordclass2 1000`, all broad experts support-only: unstable (`NaN`)
- `Conker-4b exact1 + exact2 + exact3 + wordclass2 500`, support-only, `lr=5e-4`, `cap=2.0`: `2.9659 bpb`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + wordclass2 500`, support-only, `lr=5e-4`, `cap=0.5`: `3.8151 bpb`

Replicated winners:

- `Conker-4b no_recency 1000`, 3-seed means:
  - fp16 `1.8823 bpb`
  - `int6 1.8837`
  - `int4 1.8836`
  - train time `47.2s`
  - payload `~0.252 MB`
- `Conker-4b exact1 + exact2 + exact3 + delim2 1000`, with `exact1` and `delim2` support-only, 3-seed means:
  - fp16 `1.8179 bpb`
  - `int6 1.8183`
  - `int4 1.8181`
  - train time `67.8s`
  - payload `~0.256 MB`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 1000`, support-only, 3-seed means:
  - fp16 `1.8056 bpb`
  - `int6 1.8058`
  - `int4 1.8054`
  - train time `43.6s`
  - payload `~0.293 MB`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 1000`, support-only, 3-seed means:
  - fp16 `1.8033 bpb`
  - `int6 1.8035`
  - `int4 1.8032`
  - train time `46.9s`
  - payload `~0.297 MB`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, support-only, 3-seed means:
  - fp16 `1.8020 bpb`
  - `int6 1.8022`
  - `int4 1.8018`
  - train time `54.7s`
  - payload `~0.313 MB`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, support-only + dynamic reservoir-conditioned support gates, 3-seed means:
  - fp16 `1.7987 bpb`
  - `int6 1.7988`
  - `int4 1.7984`
  - train time `92.5s`
  - payload `~0.313 MB`
- `Conker-4b exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2 1000`, gate-only learned expert selection over fixed source maps, 3-seed means:
  - fp16 `1.7973 bpb`
  - `int6 1.7974`
  - `int4 1.7970`
  - train time `63.5s`
  - payload `~0.313 MB`

Read:

- the residual formulation is real
- `exact2` is essential
- `exact3` is also real and improves the exact-context stack
- `exact1` helps most as support, not as a candidate opener
- `recency` helps the short early peak, but is also the likely source of the later instability
- the stable survivor is the exact-context residual without recency

Current architectural read:

- `exact2` is the spine
- `exact1` broadens/supports but should not open candidates by itself
- `exact3` sharpens the continuation enough to materially improve the branch
- the delimiter expert is useful only as support; letting it open candidates destabilizes the branch
- `special2` is real; rare / identifier-like continuation is another orthogonal residual source
- `number2` is also real; number-like continuation helps slightly more than `special2` alone
- `special2 + number2` stack cleanly
- `urlpath2` is real but weaker; it overlaps with `special2` and `number2`, then adds a small extra gain on top
- `markup2` is real and stronger than `urlpath2`; markup / HTML-like continuation looks more orthogonal to the live stack
- `attr2` is real but secondary; it helps, especially on top of `markup2`, but less than `markup2` itself
- learned dynamic support gating is real; a tiny reservoir-conditioned gate over the existing support experts beats another hand-written expert slice
- gate-only learned expert selection is also real, and slightly better; once the sparse source maps are present, learning only which sources to trust beats learning extra per-source magnitudes
- explicit ownership competition is not live in its first form; support experts seem to need overlapping co-support rather than hard or near-hard exclusion
- softer overlap control did not beat the gate-only baseline; mild penalties and sharper temperatures are at best near-ties, not a new branch
- input projection geometry is not a major live lever in its first tested forms; orthogonalization hurt, kernel-energy scaling tied baseline, and the apparent `split_banks` near-tie did not replicate
- longer training alone is not monotone; `1500` helped slightly, `2000` gave it back
- longer exact memory is now the strongest live lever; `seq_len=512 batch=8 steps=1500` replicated to `1.7871 bpb`, clearly beating the `256/1000` gate-only frontier
- but the gain does not keep improving with arbitrary base-model horizon; `seq_len=1024` was slightly worse than the `512/1500` row
- naive exact-only lookback widening is unstable; simply widening the exact-match support band at `seq_len=256` blew up immediately
- the much larger missed lever was tandem training; unfreezing the inherited `Conker-3` base and training it jointly with the residual stack dropped seed-42 score to `0.5624 bpb` at `256/1000` and `0.5717 bpb` at `512/1000`
- that tandem branch is not yet stable-by-default; `lr=3e-4` on the same `256/1000` row blew up late, so the new question is optimizer-region refinement, not whether the sign is real
- the tandem `256/1000 lr=5e-4` row now holds across seeds and survives a fresh-process checkpoint re-eval, so the main easy leakage explanations have been weakened substantially
- hostile sequence-destruction checks on the saved seed-43 tandem checkpoint behave like a real language model should:
  - fresh-process `test / none`: `0.5648 bpb` over `204,800` tokens
  - fresh-process `test / reverse`: `2.2799 bpb` over `204,800` tokens
  - fresh-process `test / shuffle`: `2.3015 bpb` over `204,800` tokens
  - fresh-process `train / none`: `1.3306 bits/token` over `204,800` tokens, versus `1.3759 bits/token` on validation
  So the tandem branch is using sequential structure, and train-vs-val is close enough that the easy memorization story remains weak.
- full held-out validation sweep of the same saved seed-43 tandem checkpoint (official-style contiguous `fineweb_val_*` scan) landed at:
  - `eval_loss 0.9651`
  - `eval_bits_per_token 1.3924`
  - `eval_bpb 0.5716`
  - `eval_tokens 62,021,632`
  This is slightly worse than the `204,800`-token checkpoint slice (`0.5648`), but it preserves the same basic story: the tandem branch stays in the high-`0.5x` regime on the full held-out split.
- `entity2` is not a live branch in its current form; it destabilizes immediately
- `stack2` is real but weak; bracket-obligation support only gives a tiny gain and is mostly redundant with the current exact-support stack
- delimiter subtype continuation is not a live branch in its current form; it destabilizes immediately
- the coarse `wordclass2` expert is not a live branch; in its current form it is either unstable or stably bad
- the best current replicated frozen-base row is `exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2`, with gate-only learned expert selection over fixed source maps and `seq_len=512 batch=8 steps=1500`, at `1.7871 bpb`
- the new live tandem frontier is the same expert stack with `freeze_base=False`, `seq_len=256 batch=16 steps=1000 lr=5e-4`, replicated to `0.5615 bpb`

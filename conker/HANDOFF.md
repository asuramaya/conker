# Conker Handoff

This subtree is the clean restart point for the compression-first line.

## Current Live Read

The current `Conker-4b` frontier is no longer the `256`-token gate-only row.

Replicated `Conker-4b exact123 + delim2 + special2 + number2 + markup2 + attr2`, with gate-only learned expert selection over fixed source maps:

- `seq_len=512 batch=8 steps=1500`: fp16 `1.7871 bpb`, `int6 1.7868`, `int4 1.7851`, `130.0s`

This beat the previous replicated gate-only frontier:

- `seq_len=256 batch=16 steps=1000`: fp16 `1.7973 bpb`, `int6 1.7974`, `int4 1.7970`

So the current strongest live lever is longer exact memory, not another micro-expert or gate regularizer.

Follow-up memory probes sharpened that:

- `seq_len=1024 batch=4 steps=1000`: fp16 `1.7908`
- `seq_len=1024 batch=4 steps=1500`: fp16 `1.7883`

So the live sweet spot is not simply “more base-model horizon.” `1024` is slightly worse than the replicated `512/1500` row.

Naive exact-only lookback widening also failed:

- `seq_len=256 batch=16 steps=1000 exact_context_span=512`: unstable (`NaN`)
- `seq_len=256 batch=16 steps=1000 exact_context_span=1024`: unstable (`NaN`)

So the next memory/copy step needs a better interface than simply widening the exact-match support band.

The much bigger correction is architectural:

- `Conker-4b` had been treated as inheriting a frozen `Conker-3` base
- tandem training that inherited base changes the branch completely

Seed `42` tandem pilots:

- `seq_len=256 batch=16 steps=1000 lr=5e-4 freeze_base=False`: fp16 `0.5624`, `int6 0.5752`, `int4 0.7367`
- `seq_len=512 batch=8 steps=1000 lr=5e-4 freeze_base=False`: fp16 `0.5717`, `int6 0.5786`, `int4 0.7399`
- `seq_len=256 batch=16 steps=1000 lr=3e-4 freeze_base=False`: unstable (`NaN`)

So the frozen-import assumption was wrong. The live question is now tandem replication and optimizer stability, not whether joint training is worth trying.

That replication is now effectively in:

- `seq_len=256 batch=16 steps=1000 lr=5e-4 freeze_base=False`, 3-seed means: fp16 `0.5615`, `int6 0.5745`, `int4 0.7412`

Hostile checks also landed:

- fresh-process checkpoint re-eval of a saved seed-43 tandem state: `test_bpb 0.5648`
- fresh-process transformed eval of that same saved seed-43 tandem state:
  - `test / reverse`: `2.2799 bpb` over `204,800` tokens
  - `test / shuffle`: `2.3015 bpb` over `204,800` tokens
  - `train / none`: `1.3306 bits/token` over `204,800` tokens
- full held-out validation sweep of that same saved seed-43 tandem state:
  - `test / none / full_split`: `0.5716 bpb` over `62,021,632` tokens
- sampled train/val exact-window overlap audit (`20k` train vs `5k` val windows): `0/5000` overlaps at lengths `32/64/128/256`

So the main easy leakage explanations have been weakened. The saved tandem checkpoint collapses under reversed and shuffled validation, the full held-out sweep stays in the same high-`0.5x` regime as the small checkpoint slice, and train-vs-val remains close enough that the easy memorization story is still weak. The live work is now optimizer-region refinement on the tandem branch, not proving that tandem matters at all.

## Current Compression Framing

Treat the project as a search for a tiny text compressor:

- the model's job is to reduce surprise
- state is useful only if it buys bits
- every mechanism should be judged by whether it survives clock change, shift, and baseline comparison
- the official Parameter Golf dataset is now the primary truth source
- `text8` remains useful only as a cheap smoke bridge

## Surviving Findings

Char clock:

- mixed memory mattered
- random third channels looked strong
- many prettier stories died

BPE clock:

- the char-level `random_third` frontier collapsed
- plain `v6` beat all BPE random-third width variants on fixed-bank hard-shift rollout
- BPE `v6_silenced` then beat plain BPE `v6` on hard-shift rollout, so the learned second channel is no longer the winner under the coarser clock

Contest-shaped residual prototype:

- tiny dense trunk + frozen residual branch is interesting
- first pilot is weaker than BPE `v6` and BPE `fast+mid-delay`
- do not treat it as the frontier yet

## Live Frontier

At the moment:

- archive BPE survivors still matter as ingredients:
  - `v6_silenced` for stability
  - `fast+mid-delay` for ordinary compression
- official golf data has now changed the ranking:
  - `fast+mid-delay`: mean `6.6892` bits/token, about `2.7461` bpb
  - `v6_silenced`: mean `6.7287` bits/token, about `2.7623` bpb
  - `Conker-1`: mean `6.3793` bits/token, about `2.6188` bpb
  - `Conker-2`: mean `6.2766` bits/token, about `2.5767` bpb
- `Conker-2` is now the live frontier

After the cleanup patch:

- `Conker-2` now computes exact official-style `bpb` internally
- tying the duplicate embeddings shrank the trainable budget from `471,010` to `438,242`
- across seeds `42/43/44`, the patched tied variants came back:
  - `base`: `2.6530` bpb
  - `linear_only`: `2.6228` bpb
  - `equal_logit`: `2.6106` bpb
  - `correction_only`: `3.0142` bpb
  - `no_bias`: `2.6686` bpb
  - `probability_mix`: `2.6867` bpb
- `equal_logit` is the best tied variant
- `linear_only` is almost as good and dramatically faster
- but the original untied `Conker-2` artifact is still better at `2.5767` bpb

That untied result has now been replicated in the new runner:

- `untied_base`: `2.5742` bpb over seeds `42/43/44`
- `untied_equal_logit`: `2.5872` bpb over seeds `42/43/44`

So the embedding tie was the real regression. The live winner is now `untied_base`.

Matched-budget fairness is also now closed:

- `untied_base`: `2.5742` bpb at `471,010` params
- `linear_only`: `2.6228` bpb at `201,856` params
- closest scaled single expert:
  - `fast+mid-delay`, scale `0.29`: `2.6746` bpb at `458,914` params

So the current win is not just budget shape. `untied_base` survives the fairness attack.

The FFT/scan pivot is also now closed:

- kernel `untied_base`: `2.5742` bpb, `128.9s`
- `untied_base_fft`: `2.5817` bpb, `122.7s`
- kernel `linear_only`: `2.6228` bpb, `4.0s`
- `linear_only_fft`: `2.6262` bpb, `3.1s`

So the FFT path is real and slightly faster, but it is not the score frontier yet.

The low-bit quantization ladder is also now closed on the current winner:

- `fp16`: `2.5828` bpb at `1.797 MB` learned payload
- `uniform int8`: `2.6097` bpb at `0.652 MB`
- `uniform int6`: `2.5969` bpb at `0.590 MB`
- `uniform int4`: `2.6636` bpb at `0.527 MB`
- `uniform int3`: `2.9158` bpb at `0.496 MB`

So the quantization story is now:

- `int6` is the current sweet spot
- `int4` is already a meaningful cliff
- the low-bit surface is not monotone
- quantization should be used to widen the learned part, not to shrink the current branch for its own sake

The first payload-aware widening pilot is also now in for seed `42`:

- scale `1.25`: `2.5277` bpb, `626,906` params, `119s`
- scale `1.5`: `2.5126` bpb, `799,186` params, `121s`
- scale `1.75`: `2.4838` bpb, `987,850` params, `127s`
- scale `2.0`: `2.4467` bpb, `1,192,898` params, `132s`

And with post-train `int6`:

- scale `1.25`: `2.5617` bpb at `0.809 MB`
- scale `1.5`: `2.5471` bpb at `0.873 MB`
- scale `1.75`: `2.5164` bpb at `0.984 MB`
- scale `2.0`: `2.4806` bpb at `1.187 MB`

So the widening pilot is a real pass. The next row is not “scale forever.” It is replication of `1.75` and `2.0`.

That replication is now also closed:

- scale `1.75`: `2.4860` bpb, `987,850` params, `136.7s`
- scale `2.0`: `2.4504` bpb, `1,192,898` params, `139.2s`

With post-train `int6`:

- scale `1.75`: `2.5196` bpb at `0.984 MB`
- scale `2.0`: `2.4837` bpb at `1.187 MB`

So the widened branch is real. `untied_base` scale `2.0` is now the live score frontier, and it still beats the old fp16 baseline even after post-train `int6`.

The widened `linear_only` challenge is also now closed:

- scale `1.75`: `2.5336` bpb, `400,864` params, `4.6s`
- scale `2.0`: `2.5153` bpb, `476,416` params, `4.9s`

With post-train `int6`:

- scale `1.75`: `2.5754` bpb at `0.359 MB`
- scale `2.0`: `2.5579` bpb at `0.423 MB`

So the branch split is now clean:

- widened `linear_only` is the speed frontier
- widened `untied_base` is still the score frontier
- the correction path is still earning its cost at the current best widened scale

The higher-scale hybrid pilot is now also in for seed `42`:

- scale `2.5`: `2.4273` bpb, `1,652,146` params, `158.8s`
- scale `3.0`: `2.4014` bpb, `2,176,930` params, `186.4s`

With post-train `int6`:

- scale `2.5`: `2.4648` bpb at `1.209 MB`
- scale `3.0`: `2.4385` bpb at `1.586 MB`

So the score curve has not bent yet. The next honest row is replication of `2.5` and `3.0`, not more speculation.

That replication is now also closed:

- scale `2.5`: `2.4231` bpb, `1,652,146` params, `166.5s`
- scale `3.0`: `2.3964` bpb, `2,176,930` params, `193.7s`

With post-train `int6`:

- scale `2.5`: `2.4610` bpb at `1.209 MB`
- scale `3.0`: `2.4341` bpb at `1.586 MB`

So the live score frontier is now `untied_base` scale `3.0`. The gain over the replicated `2.0x` row is real, and the payload is still nowhere near the challenge cap.

The next far-scale pilot is also now in for seed `42`:

- scale `4.0`: `2.3565` bpb, `3,423,106` params, `232.2s`
- scale `5.0`: `2.3280` bpb, `4,931,426` params, `308.6s`

With post-train `int6`:

- scale `4.0`: `2.3989` bpb at `2.479 MB`
- scale `5.0`: `2.3689` bpb at `3.560 MB`

So the curve still has not bent. The next informative row is not `6x` or `7x`; it is a jump toward the actual contest-sized payload neighborhood.

That cap-neighborhood pilot is now also in for seed `42`:

- scale `8.0`: `2.2826` bpb, `11,029,250` params, `555.5s`
- scale `10.0`: `2.2482` bpb, `16,405,186` params, `831.2s`
- scale `12.0`: `2.3128` bpb, `22,829,698` params, `1065.6s`

With post-train `int6`:

- scale `8.0`: `2.3277` bpb at `7.929 MB`
- scale `10.0`: `2.3002` bpb at `11.779 MB`
- scale `12.0`: `2.3555` bpb at `16.379 MB`

So the first clear bend is now visible:

- `10.0x` is still improving strongly
- `12.0x` is worse and already slightly over the estimated `int6` cap

That makes the next row obvious: replicate `8.0x` and `10.0x`, and stop spending time on `12.0x`.

That replication is now also closed:

- scale `8.0`: `2.2875` bpb, `11,029,250` params, `548.0s`
- scale `10.0`: `2.2650` bpb, `16,405,186` params, `825.9s`

With post-train `int6`:

- scale `8.0`: `2.3318` bpb at `7.929 MB`
- scale `10.0`: `2.3104` bpb at `11.779 MB`

So `10.0x` is now the real cap-neighborhood frontier.

The next uncertainty is narrower:

- is the `12.0x` bend a real architectural limit?
- or did the fixed `1e-3` AdamW recipe simply stop matching the branch?

That is why the next row is a `12.0x` recipe salvage pilot, not more width.

That salvage pilot is now in for seed `42`:

- baseline `1e-3`, `1000` steps: `2.3128` bpb
- `5e-4`, `1000` steps: `2.2327` bpb
- `3e-4`, `1000` steps: `2.2623` bpb
- `5e-4`, `1500` steps: `2.1993` bpb

With post-train `int6`:

- baseline: `2.3555`
- `5e-4`, `1000` steps: `2.2850`
- `3e-4`, `1000` steps: `2.3159`
- `5e-4`, `1500` steps: `2.2546`

So the `12.0x` bend was mostly recipe, not architecture. That replication is now closed:

- `12.0x`, `lr=5e-4`, `1500` steps: `2.1867` bpb, `22,829,698` params, `1765.8s`
- post-train `int6`: `2.2413` bpb at `16.379 MB`

That means:

- salvaged `12.0x` is the best replicated score row in the branch
- it clearly beats the old `10.0x` frontier (`2.2650` bpb)
- but its current `int6` payload estimate lands slightly over the nominal `16 MB` cap

So the next row is no longer more salvage. It is cap fit: run slightly smaller scales with the salvaged recipe and see how much of the `12.0x` gain survives under the payload line.

That cap-fit pilot is now in for seed `42`:

- `11.5x`, `lr=5e-4`, `1500` steps: `2.1827` bpb, `21,125,266` params, `1769.4s`, `int6 -> 2.2381` at `15.159 MB`
- `11.75x`, `lr=5e-4`, `1500` steps: `2.1671` bpb, `21,969,290` params, `1742.6s`, `int6 -> 2.2180` at `15.763 MB`

Cap-fit replication, 3-seed means:

- `11.5x`, `lr=5e-4`, `1500` steps: `2.1831` bpb, `21,125,266` params, `1747.9s`, `int6 -> 2.2367` at `15.159 MB`
- `11.75x`, `lr=5e-4`, `1500` steps: `2.1852` bpb, `21,969,290` params, `1702.3s`, `int6 -> 2.2391` at `15.763 MB`

So the cap-fit replication says:

- both scales hold under the nominal `16 MB` line
- the seed-42 `11.75x` win did not replicate
- `11.5x` is now the best replicated under-cap frontier
- the next row is fixed-budget recipe work at `11.5x`, not more scale chasing

## New Base Engine

The `Conker-2` family is still the base engine to attack, but the center of gravity has shifted:

- main path: frozen linear multiscale substrate
- correction path: tiny frozen nonlinear expert
- current evidence says the linear path is carrying the win more than the hybrid
- the embedding tie is not yet justified by score, so it must be treated as an ablation, not a free cleanup
- the best current branch is the untied hybrid, not any tied cleanup variant

This is the first branch here that both:

- improves official-data compression over `Conker-1`
- materially improves train time over `Conker-1`

## What To Build Here

Promote only compression-first work:

- BPE or later tokenizer branches
- FineWeb-facing evaluation code
- submission packing code
- compact contest-shaped models
- direct bpb-oriented ablations

Do not copy the whole archival lab here.
Move only surviving lines or new compression-first implementations.

## Immediate Next Steps

1. Keep kernel `untied_base` as the score winner.
2. Use `int6` as the working payload lever.
3. Keep widened `linear_only` as the speed frontier.
4. Keep salvaged `12.0x`, `lr=5e-4`, `1500` steps as the best raw score frontier.
5. Promote `11.5x`, `lr=5e-4`, `1500` steps to the replicated under-cap frontier.
6. Run fixed-budget recipe pilots on `11.5x`.
7. Keep the FFT path as the speed sidecar unless a later recipe closes the score gap.

That under-cap recipe pilot is now in for seed `42`:

- baseline `11.5x`, `lr=5e-4`, `1500` steps: `2.1827` bpb, `int6 -> 2.2381`
- `11.5x`, `lr=4e-4`, `1500` steps: `2.1938` bpb, `int6 -> 2.2487`
- `11.5x`, `lr=4e-4`, `1800` steps: `2.1909` bpb, `int6 -> 2.2426`
- `11.5x`, `lr=5e-4`, `1800` steps: `2.1753` bpb, `int6 -> 2.2284`

So the fixed-budget recipe row says:

- lower LR hurts at this under-cap scale
- longer training helps
- the only recipe worth replicating is `11.5x`, `lr=5e-4`, `1800` steps

That replication is now also closed, 3-seed means:

- `11.5x`, `lr=5e-4`, `1800` steps: `2.1757` bpb, `int6 -> 2.2276`, `2465.4s`

So the live under-cap frontier is now:

- `Conker-2 untied_base`, `11.5x`, `lr=5e-4`, `1800` steps

The first optimizer-side attack is also now in:

- `Muon`, `mom95_warm500`, seed `42`: `3.1610` bpb, `int6 -> 3.5204`
- `Muon`, `mom99_warm1500`, seed `42`: `3.1449` bpb, `int6 -> 3.7174`

So the immediate Muon answer is:

- both tested schedules fail badly on the live under-cap branch
- this is not a subtle optimizer tradeoff
- keep AdamW as the live optimizer unless a much sharper Muon hypothesis appears

Queue support for that step now exists in:

- `conker/scripts/link_parameter_golf_data.zsh`
- `conker/scripts/run_golf_single_bridge.py`
- `conker/scripts/run_conker_frontier_golf_queue.zsh`
- `conker/scripts/run_conker2_golf_queue.zsh`
- `conker/scripts/run_conker2_ablation_queue.zsh`
- `conker/scripts/run_conker2_scale_queue.zsh`
- `conker/scripts/run_conker2_scale_repl_queue.zsh`
- `conker/scripts/run_conker2_linear_scale_queue.zsh`
- `conker/scripts/run_conker2_scale_pilot_queue.zsh`
- `conker/scripts/run_conker2_scale_high_repl_queue.zsh`
- `conker/scripts/run_conker2_scale_far_pilot_queue.zsh`
- `conker/scripts/run_conker2_cap_pilot_queue.zsh`
- `conker/scripts/run_conker2_cap_repl_queue.zsh`
- `conker/scripts/run_conker2_recipe_pilot_queue.zsh`
- `conker/scripts/run_conker2_recipe_repl_queue.zsh`
- `conker/scripts/run_conker2_cap_fit_pilot_queue.zsh`
- `conker/scripts/run_conker2_cap_fit_repl_queue.zsh`
- `conker/scripts/run_conker2_undercap_recipe_pilot_queue.zsh`
- `conker/scripts/run_conker2_undercap_recipe_repl_queue.zsh`

## Conker-3

`Conker-3` is now the dangerous redesign side branch.

Its new assumptions are:

- the frozen linear substrate is the real survivor
- the `Conker-2` correction branch may be buying mostly local residual repair, not a second necessary dynamics
- so the next sharp ablation is to replace the nonlinear correction expert with a fully parallel local residual coder

This branch is intentionally probe-first, not cap-first:

- official golf data
- small seed-42 probes
- `600` steps
- scale `3.0`

First queue:

- `linear_only`
- `base` (additive local residual, window `8`)
- `gated`
- `window4`
- `window16`
- `local_only`

If `base` does not beat `linear_only` quickly, `Conker-3` should be killed without spending near-cap training on it.

That first kill row is now closed, seed `42`, `600` steps, scale `3.0`:

- `linear_only`: `2.6253` bpb
- `base`: `2.5082`
- `gated`: `2.5041`
- `window4`: `2.4946`
- `window16`: `2.5155`
- `local_only`: `2.5425`

So `Conker-3` survives.

The follow-up row is also now in, seed `42`, `1000` steps:

- scale `3.0`
  - `linear_only`: `2.4847` bpb
  - `base`: `2.3434`
  - `gated`: `2.3241`
  - `window4`: `2.3245`
  - `shared_embedding`: `2.3729`
- scale `5.0`
  - `linear_only`: `2.4570` bpb
  - `base`: `2.2762`
  - `gated`: `2.2783`
  - `window4`: `2.2677`

So the current `Conker-3` read is:

- local residual repair is real
- shared embeddings hurt again
- `window4` is the clean mechanism winner at the first dangerous scale
- gating has not earned its complexity yet

That `window4` row has now been attacked further:

- `window4`, `5.0x`, `1000` steps, seed `43`: `2.2704` bpb
- `window4`, `5.0x`, `1000` steps, seed `44`: `2.2683`
- 3-seed mean for `window4`, `5.0x`, `1000` steps: `2.2688`
- `window4`, `5.0x`, `1500` steps, seed `42`: `2.2080`
- `window4`, `5.0x`, `1500` steps, seed `43`: `2.2139`
- `window4`, `5.0x`, `1500` steps, seed `44`: `2.2114`
- 3-seed mean for `window4`, `5.0x`, `1500` steps: `2.2111`
- `window4`, `8.0x`, `1000` steps, seed `42`: `2.2025`
- `window4`, `8.0x`, `1000` steps, seed `43`: `2.2087`
- `window4`, `8.0x`, `1000` steps, seed `44`: `2.2083`
- 3-seed mean for `window4`, `8.0x`, `1000` steps: `2.2065`
- `window4`, `8.0x`, `1500` steps, seed `42`: `2.1519`
- `window4`, `8.0x`, `1500` steps, seed `43`: `2.1600`
- `window4`, `8.0x`, `1500` steps, seed `44`: `2.1532`
- 3-seed mean for `window4`, `8.0x`, `1500` steps: `2.1550`
- `window4`, `10.0x`, `1000` steps, seed `42`: `2.1777`
- `window4`, `10.0x`, `1500` steps, seed `42`: `2.1352`

So `Conker-3` now has a real fast frontier candidate:

- `window4`, `5.0x`, `1000` steps is stable across seeds
- it still improves materially with more compute
- it still improves with width on the first `8.0x` pilot
- `8.0x` also survives the longer-step replication
- the `10.0x`, `1000`-step pilot is already worse than replicated `8.0x`, so the branch looks short-budget recipe-limited there
- `10.0x`, `1500` steps immediately recovers and becomes the best single-seed `Conker-3` score so far

The first offline `3-bit` geometry audit is also in on that branch:

- baseline audit eval: `2.3704` bpb
- worst single-matrix `int3` hit: `local_readout.out.weight`, `+0.2200 bpb`
- next tier:
  - `local_embedding.weight`, `+0.0510`
  - `linear_readout.layers.0.weight`, `+0.0410`
  - `linear_readout.out.weight`, `+0.0388`

So the packed-weight read is now:

- the `3-bit` cliff is concentrated, not uniform
- the local residual output head is the main fragile object
- future structured low-bit work should target the local path first

The first mixed low-bit audit is also in on `window4`, `5.0x`, `1500`:

- baseline audit eval: `2.3171` bpb
- `uniform_int3`: `2.6864`, `+0.3693`
- `uniform_int4`: `2.3703`, `+0.0532`
- `int3_keep_local_out`: `2.4592`, `+0.1421`
- `int3_keep_local_out_embed`: `2.4006`, `+0.0836`
- `int3_keep_local_path`: `2.3772`, `+0.0601`
- `int3_rest_local_int4`: `2.4074`, `+0.0921`

So the low-bit direction is now:

- naive `int3` is still bad
- `uniform_int4` is already decent on this branch
- selective `int3` only gets close when most of the local path stays higher precision
- keeping the local path at `int4` is worse than keeping it high precision

Replicated mixed low-bit audit on `window4`, `8.0x`, `1500` steps, 3-seed means:

- baseline audit eval: `2.2570` bpb
- `uniform_int3`: `2.6818`, `+0.4248`
- `uniform_int4`: `2.3142`, `+0.0572`
- `int3_keep_local_out`: `2.4047`, `+0.1477`
- `int3_keep_local_out_embed`: `2.3356`, `+0.0786`
- `int3_keep_local_path`: `2.3027`, `+0.0457`
- `int3_rest_local_int4`: `2.3457`, `+0.0887`

So the replicated low-bit answer is:

- `uniform_int4` is the clean low-bit baseline
- the best selective `int3` scheme is to keep the whole local path high precision
- keeping the local path at `int4` is not enough

The first decay-bank substrate ablation is also now in on `window4`, `8.0x`, `1500`, seed `42`:

- `logspace`: `2.1540` bpb
- autocorrelation-matched bank: `2.1670`
- narrow-bank control (`half_life_max=32`): `2.1027`

So the current substrate read is:

- offline autocorrelation matching did not beat the generic prior
- but a tighter short-horizon reservoir beat both
- the live `Conker-3` branch appears to prefer a shorter linear memory scaffold than the original log-spaced bank

That narrow-bank win is now replicated on `window4`, `8.0x`, `1500`:

- narrow bank (`half_life_max=32`), 3-seed mean: `2.1064` bpb
- original log-spaced bank, 3-seed mean: `2.1550` bpb
- mean train time: `53.5s`

So the shorter-bank improvement is real, not a seed-42 artifact.

There is also a higher-width narrow-bank pilot:

- `window4`, `10.0x`, `1500`, seed `42`: `2.0871` bpb
- `uniform_int6`: `2.1211` bpb

The first narrow-bank mixed-quant audit says the score win may come with worse packing behavior:

- baseline audit eval: `2.2185` bpb
- `uniform_int4`: `2.3438`, `+0.1253`
- `uniform_int3`: `2.8153`, `+0.5968`
- best selective `int3` in that seed: `int3_keep_local_path` at `2.3876`

So the next substrate sweep should map both:

- bridge score vs `max_half_life`
- and low-bit robustness vs `max_half_life`

That sweep is now partially in on `window4`, `8.0x`, `1500`, seed `42`:

- `half_life_max=16`: `2.1014` bpb, `uniform_int6 -> 2.1355`
- `32`: `2.1075`, `uniform_int6 -> 2.1404`
- `64`: `2.1141`, `uniform_int6 -> 2.1487`
- `128`: `2.1382`, `uniform_int6 -> 2.1699`
- previous broad log-spaced `512`: `2.1540`

So the substrate curve is monotone in the tested range:

- shorter half-life caps are better
- `16` is currently the best tested reservoir cap
- broader tails hurt score and speed on this branch

The next clean substrate row is now:

- replicate `half_life_max=16`
- compare `16` vs `32` directly across seeds
- then re-run the mixed-quant audit on the `16` winner

That changed after the low-cap follow-up on `window4`, `8.0x`, `1500`, seed `42`:

- `4`: `2.1031` bpb, `uniform_int6 -> 2.1359`
- `8`: `2.0993` bpb, `uniform_int6 -> 2.1302`
- `12`: `2.1011` bpb, `uniform_int6 -> 2.1332`
- `16`: `2.1014` bpb, `uniform_int6 -> 2.1355`

So the refined substrate read is:

- the best tested cap is now `8`
- `4` is already too short
- `8` appears to be the balance point where the reservoir still helps without collapsing into the local path

The next clean substrate row is now:

- replicate `half_life_max=8` vs `16`
- then re-run the mixed-quant audit on the `8` winner

That row is now closed.

`window4`, `8.0x`, `1500`, 3-seed means:

- `half_life_max=8`: `2.1000` bpb, `uniform_int6 -> 2.1323`, `52.7s`
- `half_life_max=16`: `2.1013` bpb, `uniform_int6 -> 2.1348`, `67.5s`

So `8` is the replicated substrate winner, but only by about `0.0013` bpb.

The first mixed-quant audit on that winner says the better substrate is also worse to pack:

- baseline audit eval: `2.2155` bpb
- `uniform_int4`: `2.3378`, `+0.1222`
- `uniform_int3`: `2.9449`, `+0.7293`
- best selective `int3` in that seed: `int3_keep_local_path` at `2.4992`

There is also a higher-width pilot with the winning bank:

- `window4`, `10.0x`, `1500`, `half_life_max=8`, 3-seed mean: `2.0849` bpb
- `uniform_int6 -> 2.1180`

So the live state is now:

- best replicated substrate at `8x`: `half_life_max=8`
- best replicated score overall: `10x`, `1500`, `half_life_max=8`
- but the quantization tradeoff got worse, not better

The direct seed-42 packing comparison is:

- `half_life_max=8`
  - `uniform_int4`: `+0.1222`
  - `uniform_int3`: `+0.7293`
  - `int3_keep_local_path`: `+0.2837`
- `half_life_max=16`
  - `uniform_int4`: `+0.1175`
  - `uniform_int3`: `+0.6159`
  - `int3_keep_local_path`: `+0.1991`

So the live frontier is now explicitly split:

- `8` wins on bridge score
- `16` wins on low-bit robustness

That split is now confirmed at `10.0x`, `1500`, 3-seed means:

- `half_life_max=8`
  - fp16: `2.0832`
  - `uniform_int6`: `2.1040`
  - `uniform_int4`: `2.2393`
- `half_life_max=16`
  - fp16: `2.0837`
  - `uniform_int6`: `2.1019`
  - `uniform_int4`: `2.2283`

So if packed score is treated as the real objective, `16` is currently the better `Conker-3` substrate.

That packed winner still has headroom.

Packed-first scaling wave on `half_life_max=16`, seed `42`:

- `12x`: fp16 `2.0709`, `int6 2.0920`, `int4 2.2140`
- `14x`: fp16 `2.0616`, `int6 2.0793`, `int4 2.1966`
- `16x`: fp16 `2.0507`, `int6 2.0726`, `int4 2.1715`

Payloads at the current frontier point:

- `16x int6`: `13.527 MB`
- `16x int4`: `9.027 MB`

So the packed curve is still improving and `16x int6` remains under the nominal cap.

The first precision-allocation audit on `10x`, `1500`, `half_life_max=16` says:

- `uniform_int4`: `+0.1215`
- `int4_keep_local_out`: `+0.1020`
- `int4_keep_local_out_embed`: `+0.0928`
- `int4_keep_local_path`: `+0.0903`

So protected `int4` works, but the improvement is modest compared with simply scaling the packed model upward.

There is now one live TTT result too.

`window4`, `10.0x`, `1500`, `half_life_max=16`, seed `42`:

- grouped reservoir-mode gate, `16` groups
- online chunk adaptation, `64`-token chunks, `32` chunks
- reservoir and readout frozen, only the gate adapts

Results:

- fp16
  - baseline: `2.2149`
  - `1` step @ `0.10`: `2.2151`
  - `3` steps @ `0.03`: `2.2136`
- `int6`
  - baseline: `2.2203`
  - `1` step @ `0.10`: `2.2190`
  - `3` steps @ `0.03`: `2.2175`

So the first TTT answer is:

- routing adaptation is not a big raw-score gain
- it is slightly more useful on the packed model than on fp16
- if pursued, it should be pursued as a packed-model repair mechanism

There is also now a real oscillatory result on the packed-winner substrate.

`window4`, `10.0x`, `1500`, `half_life_max=16`:

- `25%` oscillatory modes, seed `42`
  - fp16: `2.0751`
  - `int6`: `2.0955`
  - `int4`: `2.2348`
- `50%` oscillatory modes, 3-seed means
  - fp16: `2.0613`
  - `int6`: `2.0833`
  - `int4`: `2.2343`

Compared with the non-osc baseline:

- baseline fp16: `2.0837`
- baseline `int6`: `2.1019`
- baseline `int4`: `2.2283`

So the current interpretation is:

- damped oscillatory modes are a real architectural win
- they help the packed `int6` frontier
- they slightly hurt the `int4` frontier

High-oscillation follow-up, seed `42`, same branch:

- `62.5%`: fp16 `2.0568`, `int6 2.0801`, `int4 2.2416`, `69.6s`
- `75%`: fp16 `2.0526`, `int6 2.0740`, `int4 2.2361`, `73.6s`
- `87.5%`: fp16 `2.0517`, `int6 2.0738`, `int4 2.2449`, `105.8s`

So the live read is now:

- fp16 and `int6` still improve above `50%`
- `75%` and `87.5%` are basically tied on packed `int6`
- `87.5%` looks worse on `int4` and much worse on train time
- the next replication row should concentrate on `75%` and `87.5%`

That replication is now closed, 3-seed means:

- `75%`: fp16 `2.0531`, `int6 2.0745`, `int4 2.2404`, `73.2s`
- `87.5%`: fp16 `2.0512`, `int6 2.0729`, `int4 2.2410`, `114.8s`

So the current interpretation is:

- `87.5%` is the best raw and `int6` oscillatory fraction so far
- the win over `75%` is tiny
- `75%` remains marginally better on `int4`
- the next clean experiment is a period-range sweep at `87.5%`, not more fraction pushing

That period sweep is now closed, seed `42`:

- `4..32`: fp16 `2.0518`, `int6 2.0772`, `int4 2.2432`, `81.6s`
- `4..64`: fp16 `2.0515`, `int6 2.0725`, `int4 2.2411`, `92.8s`
- `8..64`: fp16 `2.0602`, `int6 2.0788`, `int4 2.2281`, `108.8s`
- `8..128`: fp16 `2.0596`, `int6 2.0819`, `int4 2.2238`, `121.1s`

So the live oscillatory interpretation is:

- `87.5%` / `4..64` is still the best overall fp16 + `int6` oscillatory setting
- narrowing the period range to `4..32` does not improve the packed winner
- pushing the whole band upward weakens the main packed branch, even if `int4` softens slightly

The 8-hour laptop matrix is now closed. On `window4`, `1500`, `half_life_max=16`, periods `4..64`:

- `16x`, `75%`, 3-seed means: fp16 `2.0342`, `int6 2.0565`, `int4 2.2131`, `13.527 MB`
- `16x`, `87.5%`, 3-seed means: fp16 `2.0309`, `int6 2.0542`, `int4 2.2231`, `13.527 MB`
- `18x`, `75%`, 3-seed means: fp16 `2.0282`, `int6 2.0509`, `int4 2.2127`, `16.588 MB`
- `18x`, `87.5%`, 3-seed means: fp16 `2.0271`, `int6 2.0508`, `int4 2.2205`, `16.588 MB`

Local-path byte-allocation probes on `18x`, `87.5%`, seed `42`:

- `local_hidden_mult=0.75`: fp16 `2.0315`, `int6 2.0572`, `int4 2.2310`
- `local_hidden_mult=0.50`: fp16 `2.0383`, `int6 2.0603`, `int4 2.2442`
- `local_scale_override=0.20`: fp16 `2.0291`, `int6 2.0523`, `int4 2.2342`

So the live packed read is:

- the best replicated under-cap oscillatory branch is now `16x`, `87.5%`, `4..64`
- `87.5%` wins on fp16 and `int6`, but `75%` stays better on `int4`
- `18x` keeps improving raw score but lands just over the nominal `int6` cap
- shrinking the local path is not the next lever

The packed next-wave row is now closed enough to change the branch:

- cap-fit pilots on `87.5%`, seed `42`
  - `16.5x`: fp16 `2.0313`, `int6 2.0536`, `int4 2.2292`, `14.264 MB`
  - `17.0x`: fp16 `2.0295`, `int6 2.0514`, `int4 2.2258`, `15.020 MB`
  - `17.5x`: fp16 `2.0297`, `int6 2.0515`, `int4 2.2311`, `15.794 MB`
- longer train on `16x`, seed `42`
  - `1800` steps: fp16 `2.0281`, `int6 2.0473`, `int4 2.2234`
  - `2200` steps: fp16 `1.9899`, `int6 2.0146`, `int4 2.1788`
- static bank gate on `16x`, seed `42`
  - `1500` steps: fp16 `2.0237`, `int6 2.0452`, `int4 2.1886`
  - `1800` steps: fp16 `2.0214`, `int6 2.0361`, `int4 2.1877`
- naive pack-train (`int6` every step) failed
  - `16x`, `1800`: fp16 `3.5570`, `int6 3.6240`, `int4 3.8628`

So the live interpretation is:

- longer training is now the strongest known lever on this branch
- static bank gating is the first new mechanism that clearly helps packed score beyond the oscillatory bank itself
- cap-fit scaling between `16x` and `17.5x` matters less than training the current winner longer
- naive pack-training should be dropped

Combo pilots on that same branch family, seed `42`:

- `16x`, `2200`, static gate: fp16 `1.9856`, `int6 2.0066`, `int4 2.1479`, `210.6s`
- `17x`, `1800`, static gate: fp16 `2.0186`, `int6 2.0347`, `int4 2.1850`, `193.3s`
- `17x`, `2200`, static gate: fp16 `1.9814`, `int6 2.0019`, `int4 2.1463`, `230.5s`

So the live frontier is now:

- best seed-42 raw packed branch: `17x / 2200 / staticgate`
- safer likely-under-cap branch to replicate first: `16x / 2200 / staticgate`

That replication is now closed:

- `16x / 2200 / staticgate`, 3-seed means: fp16 `1.9845`, `int6 2.0067`, `int4 2.1506`, `13.527 MB`
- `17x / 2200 / staticgate`, 3-seed means: fp16 `1.9828`, `int6 2.0042`, `int4 2.1523`, `15.020 MB`
- `18x / 2200 / staticgate`, seed `42`: fp16 `1.9790`, `int6 2.0027`, `int4 2.1415`, `16.588 MB`

So the live packed frontier is now:

- best replicated under-cap branch: `17x / 2200 / staticgate`
- best raw pilot above cap: `18x / 2200 / staticgate`

First `Conker-4` probe:

- reference branch for comparison:
  - `Conker-3 window4 10x 1000 staticgate`: fp16 `2.0865`, `int6 2.1081`, `int4 2.2601`
- `Conker-4` as a direct probability mixer over:
  - live `Conker-3`
  - exact 1-token backoff
  - exact 2-token backoff
  - recency prior
- outcomes:
  - learned MLP mixer: `NaN`
  - support-weighted mixer: `NaN`
  - support-weighted mixer with neural expert frozen out of the trainable set: `NaN`
  - zero-step static frozen-neural ensemble: fp16 `3.7207`, `int6 3.7619`, `int4 3.7724`
  - low-LR (`1e-4`) frozen-neural run: `NaN`

So the first `Conker-4` answer is negative:

- the exact experts as currently constructed are not “already solved compression” waiting for a tiny mixer
- the direct probability-space ensemble is worse than the live `Conker-3` expert even before training
- if `Conker-4` returns, it should likely return as a residual / calibration expert family rather than this raw direct mixture

`Conker-4b` is that residual / calibration redesign, and it worked immediately.

On the same official-data family (`window4 / 10x / half_life=16 / osc=87.5%`):

- reference `Conker-3 1000`: `2.0865 bpb`
- `Conker-4b full 500`: `1.9547 bpb`
- `Conker-4b no_exact2 500`: `2.1429 bpb`
- `Conker-4b no_recency 500`: `2.0079 bpb`
- `Conker-4b no_recency 1000`, 3-seed means:
  - fp16 `1.8823`
  - `int6 1.8837`
  - `int4 1.8836`
  - `47.2s`
  - payload `~0.252 MB`

So the current `Conker-4` family read is:

- direct expert probability mixing failed
- exact-context residual calibration works
- `exact2` is essential
- `recency` helps early but appears to be the instability source
- the stable survivor is `Conker-4b no_recency 1000`

Latest exact-context architecture cuts, seed `42`:

- `exact2` only, `1000`: `2.0364`
- `exact1` only, `1000`: `2.0413`
- `exact2 + exact3`, `1000`: `1.8811`
- `exact1 + exact2`, `1000`, but `exact1` support-only: `1.8807`
- `exact1 + exact2 + exact3`, `1000`: `1.8374`
- `exact1 + exact2 + exact3`, `1000`, but `exact1` support-only: `1.8389`
- `exact1 + exact2 + exact3 + delim2`, `1000`, with `exact1` and `delim2` support-only: `1.8187`
- `exact1 + exact2 + exact3 + delim2 + special2`, `1000`, support-only: `1.8102`
- `exact1 + exact2 + exact3 + delim2 + number2`, `1000`, support-only: `1.8095`
- `exact1 + exact2 + exact3 + delim2 + special2 + number2`, `1000`, support-only: `1.8061`
- `exact1 + exact2 + exact3 + delim2 + urlpath2`, `1000`, support-only: `1.8178`
- `exact1 + exact2 + exact3 + delim2 + special2 + number2 + urlpath2`, `1000`, support-only: `1.8047`
- `exact1 + exact2 + exact3 + delim2 + markup2`, `1000`, support-only: `1.8098`
- `exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2`, `1000`, support-only: `1.8034`
- `exact1 + exact2 + exact3 + delim2 + attr2`, `1000`, support-only: `1.8103`
- `exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2`, `1000`, support-only: `1.8018`
- `exact1 + exact2 + exact3 + delim2 + entity2`, `1000`, support-only: unstable (`NaN`)
- `exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + entity2`, `1000`, support-only: unstable (`NaN`)
- `exact1 + exact2 + exact3 + delim2 + stack2`, `1000`, support-only: `1.8182`
- `exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + stack2`, `1000`, support-only: `1.8032`
- `exact1 + exact2 + exact3 + delim2 + delimsub2`, `1000`, support-only: unstable (`NaN`)
- removing base-confidence coupling from `exact1 + exact2`, `1000`: `1.8822`
- `exact1 + exact2 + exact3 + wordclass2`, `1000`, support-only: unstable (`NaN`)
- `exact1 + exact2 + exact3 + delim2 + wordclass2`, `1000`, all broad experts support-only: unstable (`NaN`)
- `exact1 + exact2 + exact3 + wordclass2`, `500`, support-only, `lr=5e-4`, `cap=2.0`: `2.9659`
- `exact1 + exact2 + exact3 + delim2 + wordclass2`, `500`, support-only, `lr=5e-4`, `cap=0.5`: `3.8151`

So the architecture answer is now:

- `exact2` is the main exact-context signal, but it is not sufficient by itself
- `exact1` helps most as support, not as a candidate opener
- `exact3` is real and materially improves the branch
- the delimiter expert is also real, but only as support; letting it open candidates destabilizes the branch
- `special2` and `number2` are both real support-only residuals and they stack
- `urlpath2` is real but weaker and mostly overlapping; it only adds a small extra gain on top of `special2 + number2`
- `markup2` is real and stronger than `urlpath2`; markup / HTML-like continuation is a better next expert family
- `attr2` is real but secondary; it helps less than `markup2`
- `entity2` is not a live branch in its current form
- `stack2` is real but weak; bracket-obligation support is mostly redundant with the current exact-support stack
- delimiter subtype continuation is not a live branch in its current form
- the coarse `wordclass2` expert is not a live direction in its current form; it is either unstable or stably harmful
- base-confidence coupling is almost irrelevant
- the best current replicated branch is `exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2`, support-only

Replicated `exact123 + delim2` support-only row, 3-seed means:

- fp16 `1.8179`
- `int6 1.8183`
- `int4 1.8181`
- `67.8s`
- payload `~0.256 MB`

Replicated `exact123 + delim2 + special2 + number2` support-only row, 3-seed means:

- fp16 `1.8056`
- `int6 1.8058`
- `int4 1.8054`
- `43.6s`
- payload `~0.293 MB`

Replicated `exact123 + delim2 + special2 + number2 + markup2` support-only row, 3-seed means:

- fp16 `1.8033`
- `int6 1.8035`
- `int4 1.8032`
- `46.9s`
- payload `~0.297 MB`

Replicated `exact123 + delim2 + special2 + number2 + markup2 + attr2` support-only row, 3-seed means:

- fp16 `1.8020`
- `int6 1.8022`
- `int4 1.8018`
- `54.7s`
- payload `~0.313 MB`

Replicated `exact123 + delim2 + special2 + number2 + markup2 + attr2` support-only row with dynamic reservoir-conditioned support gates, 3-seed means:

- fp16 `1.7987`
- `int6 1.7988`
- `int4 1.7984`
- `92.5s`
- payload `~0.313 MB`

Replicated `exact123 + delim2 + special2 + number2 + markup2 + attr2` gate-only learned expert-selection row, 3-seed means:

- fp16 `1.7973`
- `int6 1.7974`
- `int4 1.7970`
- `63.5s`
- payload `~0.313 MB`

Ownership probe on the same exact stack, seed `42`:

- softmax + abstain: unstable (`NaN`)
- top-1 ownership: fp16 `2.0049`, `int6 2.0080`, `int4 2.0081`
- top-2 ownership: fp16 `2.0369`, `int6 2.0403`, `int4 2.0404`

Softer overlap-control probe on the same gate-only stack, seed `42`:

- overlap penalty `1e-3`: fp16 `1.7979`, `int6 1.7978`, `int4 1.7975`
- overlap penalty `3e-3`: fp16 `1.7987`, `int6 1.7986`, `int4 1.7983`
- overlap penalty `1e-3` + temperature `0.75`: fp16 `1.7975`, `int6 1.7977`, `int4 1.7973`

Input-projection probe on the same gate-only stack, seed `42`:

- `random`: fp16 `1.7982`, `int6 1.7983`, `int4 1.7978`
- `orthogonal_rows`: fp16 `1.7988`, `int6 1.7989`, `int4 1.7984`
- `kernel_energy`: fp16 `1.7982`, `int6 1.7982`, `int4 1.7978`
- `split_banks`: fp16 `1.7980`, `int6 1.7979`, `int4 1.7976`

Replicated `split_banks` input-projection row, 3-seed means:

- fp16 `1.7990`
- `int6 1.7990`
- `int4 1.7987`
- `57.6s`

Next-wave pilot on the gate-only winner, seed `42`:

- `steps=1500`: fp16 `1.7970`, `int6 1.7973`, `int4 1.7968`, `88.3s`
- `steps=2000`: fp16 `1.7980`, `int6 1.7979`, `int4 1.7976`, `119.3s`
- `seq_len=512 batch=8 steps=1000`: fp16 `1.7975`, `int6 1.7972`, `int4 1.7955`, `98.8s`
- `seq_len=512 batch=8 steps=1500`: fp16 `1.7878`, `int6 1.7875`, `int4 1.7857`, `150.2s`

Negative `Conker-5` pure learned-discriminator probe, seed `42`:

- frozen `Conker-3` substrate
- no hand-coded exact/support experts
- `8` learned residual heads, rank `8`
- fp16 `3.4012`
- `int6 3.4113`
- `int4 3.4111`
- `34.8s`
- payload `~0.595 MB`

## Conker-6

`Conker-6` started as the legal cache branch, but the trained-mask attacks invalidated that framing.

- normalized causal cache
- no two-pass rescoring
- no non-normalized blend tricks
- optional `Conker-3` base used only as smoother/gate

Artifacts:

- [CONKER6.md](/Users/asuramaya/Code/carving_machine_v3/conker/conker/docs/CONKER6.md)
- [conker6.py](/Users/asuramaya/Code/carving_machine_v3/conker/src/conker6.py)
- [run_conker6_golf_bridge.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/run_conker6_golf_bridge.py)
- [audit_conker6_legality.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/audit_conker6_legality.py)

Originally best apparent row, seed `42`:

- `cache_only`
- `seq_len=256 batch=16 steps=1000`
- full held-out fp16 `0.0721 bpb`
- `int4 0.0714`
- `int6 0.0740`
- payload `~0.254 MB`

Fresh-init legality attack on that row:

- normalized: probability sums stay near `1`
- row-wise future invariance: exact `0`
- flat-stream invariance: exact `0`
- audit:
  [conker6_legality_cacheonly_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_legality_cacheonly_2026-03-28.json)

Trainable-subset attack:

- only reported trainables:
  - `causal_mask` `(256,256)` = `65,536`
  - `vocab_axis` `(1024,)` = `1,024`
- `fixed_vocabulary + fixed_causal_mask`: full held-out `5.7521 bpb`
- `fixed_vocabulary + learnable_causal_mask`: full held-out `0.0721 bpb`

That fresh-init legality attack was misleading.

Direct saved trained-mask attack:

- row-wise future max logit diff: `18.4207`
- flat-stream future max diff: `0.0`
- probability sum max: `1.4260`
- max abs normalization error: `0.4260`

So the trained winner is not a legal normalized row-wise causal model.

Important failures:

- `fixed_blend`: full held-out `0.2233` slice / `0.4004` full
- `learned_gate`: late `NaN`
- `witten_bell`: full held-out `0.4004`
- `absolute_discount`: full held-out `1.3097`

So textbook smoothing has not beaten raw hard exact backoff on this task.

Two sharp probes:

- `steps=1` full held-out: `0.3333 bpb`
- `disable_exact3` full held-out: `0.0983 bpb`

Interpretation:

- the branch is not a zero-training trick; the small trainable subset matters
- the small trainable subset is overwhelmingly `causal_mask`
- `exact3` is worth about `0.026 bpb`
- the current win is genuinely higher-order exact context

Mask-geometry ablation on the trained legal winner:

- baseline: `0.0721 bpb`
- nonnegative clamp: `0.0714`
- magnitude prune `90/95/98%`: `4.4732 / 4.3152 / 4.1847`
- row-top-k `16/8`: `4.5754 / 4.3589`
- artifact:
  [conker6_mask_ablation_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_ablation_2026-03-28.json)

Read:

- naive sparsity kills the branch
- row-wise top-k pruning is no better than global magnitude pruning
- within active support, the learned mask is already effectively nonnegative
- the next search is not more pruning; it is structured dense parameterization of the causal weighting

Matrix dump + structure attack:

- raw mask:
  [npy](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.mask.npy)
  [csv](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.mask.csv)
- visuals:
  [mask](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.mask.png)
  [lag-mean](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.lag_mean_mask.png)
  [lag-mean diff](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.lag_mean_diff.png)
- summary:
  [conker6_mask_geometry_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.json)

Important read:

- the `256x256` mask looks almost trivial:
  - active weights only about `0.887 .. 1.057`
  - lag means only about `0.946 .. 1.030`
  - top-`32` singular values already capture about `72.1%` of spectral energy
- but structured approximations all fail:
  - `toeplitz_mean`: `5.7521 bpb`
  - `lowrank_16_masked`: `5.7521`
  - `row_normalized`: `5.7521`

Most surprising:

- `toeplitz_mean` stays extremely close to the learned mask:
  - cosine `0.9998`
  - L2 deviation `0.0188`
- but still destroys the score

So the next attack is not “compress the mask because it looks simple.” It is understanding why tiny structured perturbations erase the effect.

Cross-seed residual comparison:

- summary:
  [conker6_seed_residual_compare_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_seed_residual_compare_2026-03-28.json)
- per-seed dumps:
  [seed42](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_seed_residual_compare_2026-03-28_seed42)
  [seed43](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_seed_residual_compare_2026-03-28_seed43)
  [seed44](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_seed_residual_compare_2026-03-28_seed44)

Read:

- all three seeds still score about `0.0714 .. 0.0721 bpb`
- raw masks are almost identical: cosine about `0.9996`
- Toeplitz means are even closer: cosine about `1.0`, Pearson about `0.964 .. 0.968`
- but residuals after subtracting each seed's Toeplitz mean are nearly orthogonal:
  - residual cosine about `0.086 .. 0.100`
  - sign agreement about `0.510 .. 0.518`

Residual-substitution attack:

- [conker6_residual_substitution_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_residual_substitution_2026-03-28.json)
- `baseline_seed42_full`: `0.0720933 bpb`
- `baseline_seed42_strictlower`: `5.7521`
- `toeplitz_lower_plus_upperdiag_seed42`: `0.0721075`
- random or swapped lower residuals all remain about `0.0721`

So the lower residual is not the source of the apparent win. The crucial part is the learned diagonal and upper-triangle mask entries, which are also what make the branch invalid.

Current read:

- `Conker-6` in its current trained form is invalid
- the interesting remaining lesson is about mask geometry, not a legal score frontier

## Conker-7

New branch:

- [conker7.py](/Users/asuramaya/Code/carving_machine_v3/conker/src/conker7.py)
- [run_conker7_golf_bridge.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/run_conker7_golf_bridge.py)
- [CONKER7.md](/Users/asuramaya/Code/carving_machine_v3/conker/conker/docs/CONKER7.md)

Purpose:

- keep eval strictly causal
- use future-aware exact-history signal only during training
- ask whether the illegal `Conker-6` lesson can be distilled into a legal student

Seed-42 pilots on `window4 / 10x / 256 / batch16 / lr5e-4`:

Future-only teacher, `exact2 + exact3`, `500` steps:

- fp16 `0.6768 bpb`
- `int6 0.6887`
- `int4 0.9742`
- artifact:
  [conker7_future_exact23_seq256_steps500_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_future_exact23_seq256_steps500_seed42_2026-03-28.json)

Future-only teacher, `exact2 + exact3`, `1000` steps:

- fp16 `0.6168 bpb`
- `int6 0.6282`
- `int4 0.8132`
- artifact:
  [conker7_future_exact23_seq256_steps1000_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_future_exact23_seq256_steps1000_seed42_2026-03-28.json)

Future-only rich teacher (`exact2 + exact3 + special2 + number2 + markup2 + attr2 + delim2`), `500` steps:

- unstable `NaN`
- artifact:
  [conker7_future_rich_seq256_steps500_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_future_rich_seq256_steps500_seed42_2026-03-28.json)

Read:

- narrow future-aware supervision is real
- the legal student improves with more steps
- broad future-aware lexical supervision destabilizes
- still behind legal tandem `Conker-5`, so this is a live research branch, not a new submission line yet

Best next `Conker-7` probes:

1. lower `teacher_weight` to `0.10` and `0.25`
2. `exact3`-only teacher
3. `bidirectional` vs `future` teacher on the narrow exact row
4. warm-start from a legal tandem checkpoint or add late-onset teacher curriculum

Completed sequence on the same seed-`42` branch:

- future `exact2 + exact3`, `teacher_weight=0.10`, `1000` steps:
  - fp16 `0.5792`
  - `int6 0.5915`
  - `int4 0.7612`
  - artifact:
    [conker7_future_exact23_tw01_seq256_steps1000_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_future_exact23_tw01_seq256_steps1000_seed42_2026-03-28.json)
- future `exact2 + exact3`, `teacher_weight=0.25`, `1000` steps:
  - fp16 `0.5945`
  - `int6 0.6065`
  - `int4 0.7763`
- future `exact3`-only, `teacher_weight=0.10`, `1000` steps:
  - fp16 `0.5819`
  - `int6 0.5953`
  - `int4 0.7644`
- bidirectional `exact2 + exact3`, `teacher_weight=0.10`, `1000` steps:
  - fp16 `0.5776`
  - `int6 0.5905`
  - `int4 0.7607`
  - artifact:
    [conker7_bidirectional_exact23_tw01_seq256_steps1000_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_bidirectional_exact23_tw01_seq256_steps1000_seed42_2026-03-28.json)
- bidirectional `exact2 + exact3`, `teacher_weight=0.10`, `teacher_start_step=500`, `1000` steps:
  - fp16 `0.5666`
  - `int6 0.5789`
  - `int4 0.7370`
  - artifact:
    [conker7_bidirectional_exact23_tw01_start500_seq256_steps1000_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_bidirectional_exact23_tw01_start500_seq256_steps1000_seed42_2026-03-28.json)

Warm-start from the legal tandem `Conker-5` `seq256 / 1500 / lr5e-4` checkpoint is the breakthrough:

- bidirectional `exact2 + exact3`, `teacher_weight=0.10`, warm-start, `500` steps:
  - fp16 `0.5357`
  - `int6 0.5480`
  - `int4 0.6304`
  - artifact:
    [conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps500_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps500_seed42_2026-03-28.json)
- bidirectional `exact2 + exact3`, `teacher_weight=0.10`, warm-start, `1000` steps:
  - fp16 `0.5183`
  - `int6 0.5301`
  - `int4 0.6140`
  - artifact:
    [conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42_2026-03-28.json)

Current read:

- the teacher should stay narrow: `exact2 + exact3`
- the teacher should stay weak: `teacher_weight=0.10`
- bidirectional teacher beats future-only on the stable row
- delayed teacher helps from scratch
- warm-starting from the legal tandem student is the real lever

Follow-up queue landed:

- queue:
  [run_conker7_followup_queue.zsh](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/run_conker7_followup_queue.zsh)
- evaluator:
  [run_conker7_checkpoint_eval.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/run_conker7_checkpoint_eval.py)
- log:
  [conker7_followup_queue_2026-03-28.log](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_followup_queue_2026-03-28.log)

Honest eval on the saved seed-42 warm-start winner:

- fp16 full split (`62,021,632` tokens):
  - `0.5283 bpb`
  - artifact:
    [conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42_fullval_test_none_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42_fullval_test_none_2026-03-28.json)
- `int6` full split:
  - `0.5315 bpb`
  - compressed artifact `4,153,894` bytes
  - artifact:
    [conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42_fullval_test_int6_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42_fullval_test_int6_2026-03-28.json)

Warm-start sweep around the winner, seed `42`:

- `teacher_weight=0.05`:
  - fp16 `0.5126`
  - `int6 0.5249`
  - `int4 0.6090`
  - artifact:
    [conker7_bidirectional_exact23_tw0p05_warmstart_tandem1500_seq256_steps1000_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_bidirectional_exact23_tw0p05_warmstart_tandem1500_seq256_steps1000_seed42_2026-03-28.json)
- `teacher_weight=0.15`:
  - fp16 `0.5222`
  - `int6 0.5342`
  - `int4 0.6168`
  - artifact:
    [conker7_bidirectional_exact23_tw0p15_warmstart_tandem1500_seq256_steps1000_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_bidirectional_exact23_tw0p15_warmstart_tandem1500_seq256_steps1000_seed42_2026-03-28.json)
- `teacher_start=250`:
  - fp16 `0.5141`
  - `int6 0.5260`
  - `int4 0.6092`
  - artifact:
    [conker7_bidirectional_exact23_tw01_start250_warmstart_tandem1500_seq256_steps1000_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_bidirectional_exact23_tw01_start250_warmstart_tandem1500_seq256_steps1000_seed42_2026-03-28.json)
- `teacher_start=500`:
  - fp16 `0.5122`
  - `int6 0.5242`
  - `int4 0.6083`
  - artifact:
    [conker7_bidirectional_exact23_tw01_start500_warmstart_tandem1500_seq256_steps1000_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_bidirectional_exact23_tw01_start500_warmstart_tandem1500_seq256_steps1000_seed42_2026-03-28.json)

Warm-start “replication” rows:

- [seed43](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed43_2026-03-28.json)
- [seed44](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed44_2026-03-28.json)

These are numerically identical to the seed-42 bridge row, so they are not true statistical replications. The current warm-start fine-tune path is effectively deterministic under the sequential training stream.

Next honest move:

1. rerun the best warm-start rows with seeded data-order variation or randomized train-stream offsets
2. full-holdout eval the best of:
   - `teacher_weight=0.05`
   - `teacher_start=500`
3. if either holds, update the submission framing around `Conker-7`

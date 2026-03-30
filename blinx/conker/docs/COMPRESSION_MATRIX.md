# Compression Matrix

Rescue status:

- this is preserved dense matrix material inside the inherited BLINX/Conker note store
- many rows below are historical Conker-side records, not active BLINX claims
- use [Presentation](../../docs/presentation.md), [Rescue](../../docs/rescue.md), and [Current Frontier](./CURRENT_FRONTIER.md) for the current public read

This is the next experiment matrix for `conker/`.

It is deliberately narrower than the archival matrix. Every row is framed as a compression question:

- does this state organization buy bits?
- does it survive a coarser clock?
- does it survive transfer?
- does it survive budget pressure?

## Surviving Branches

Current BPE survivors:

- `v6_silenced`: best hard-shift / boundary-stability branch
- `fast+mid-delay`: best transfer-side mixed-memory branch and best frozen BPE compressor so far on held-out `text8`
- `contest_residual_bpe_pilot`: sidecar idea only, not frontier

First external-score bridge on `text8`:

- `fast+mid-delay` is the best frozen BPE compressor on held-out `text8` test bpb: `2.3717`
- plain `v6` follows at `2.3904`
- `v6_silenced` is worse on that external score at `2.4044`
- `gru_opt` is still far ahead as a direct compressor at `1.9865`

Current non-survivors under BPE:

- `random_third_w64`
- `random_third_w128`
- `random_third_w200`
- plain BPE `v6` as the hard-shift winner
- contest residual pilot as a frontier model

`text8` bridge status:

- useful for cheap smoke
- no longer treated as the primary score source for `conker`

## Official Data Row

Completed on the fixed `fineweb10B_sp1024` subset:

- `fast+mid-delay`: mean `6.6892` bits/token, about `2.7461` bpb, `695,912` params
- `v6_silenced`: mean `6.7287` bits/token, about `2.7623` bpb, `695,912` params
- `Conker-1`: mean `6.3793` bits/token, about `2.6188` bpb, `1,401,394` params
- `Conker-2`: mean `6.2766` bits/token, about `2.5767` bpb, `471,010` params

So:

- `Conker-1` beat both single experts
- `Conker-2` then beat `Conker-1`
- the speed pivot paid off instead of merely being faster

Patched `Conker-2` mechanism row, 3-seed means (`42/43/44`), after tying embeddings and removing dead state:

- `base`: `6.4625` bits/token, `2.6530` bpb, `438,242` params, `128.1s`
- `linear_only`: `6.3889` bits/token, `2.6228` bpb, `201,856` params, `4.0s`
- `correction_only`: `7.3424` bits/token, `3.0142` bpb, `259,584` params, `120.1s`
- `equal_logit`: `6.3594` bits/token, `2.6106` bpb, `437,888` params, `125.8s`
- `no_bias`: `6.5006` bits/token, `2.6686` bpb, `429,026` params, `125.2s`
- `probability_mix`: `6.5447` bits/token, `2.6867` bpb, `429,026` params, `126.6s`

So the current read is:

- the linear path carries the win
- the nonlinear correction path is weak on its own
- learned mixing is not helping yet
- probability-space mixing is worse than logit-space mixing
- `equal_logit` is the best patched mixed variant
- `linear_only` is almost as good while being about `31x` faster than the patched base
- but all patched variants are still worse than the original untied `Conker-2` result (`2.5767` bpb)

Untied embedding replication, 3-seed means:

- `untied_base`: `6.2706` bits/token, `2.5742` bpb, `471,010` params, `128.9s`
- `untied_equal_logit`: `6.3022` bits/token, `2.5872` bpb, `470,656` params, `124.0s`

So the embedding-sharing diagnosis is now clean:

- the tie was the main regression
- the old frontier survives under the new exact-`bpb` runner
- `untied_base` is the current live winner
- `untied_equal_logit` is better than all tied variants, but still worse than `untied_base`

Matched-budget fairness row, 3-seed means:

- `untied_base`: `2.5742` bpb, `471,010` params, `128.9s`
- `linear_only`: `2.6228` bpb, `201,856` params, `4.0s`
- `fast+mid-delay`, scale `0.29`: `2.6746` bpb, `458,914` params, `146.9s`
- `fast+mid-delay`, scale `0.68`: `2.7269` bpb, `589,096` params, `211.1s`
- `v6_silenced`, scale `0.29`: `3.0175` bpb, `458,914` params, `136.4s`
- `v6_silenced`, scale `0.68`: `2.8021` bpb, `589,096` params, `279.2s`

So the fairness row says:

- `untied_base` stays clearly ahead of the budget-matched frozen single experts
- `linear_only` is still the speed specialist, but loses on score
- the next move is no longer another fairness cut
- the next move is the FFT/scan pivot on the `untied_base` linear path

FFT/scan pivot, 3-seed means:

- `untied_base`: `2.5742` bpb, `128.9s`
- `untied_base_fft`: `2.5817` bpb, `122.7s`
- `linear_only`: `2.6228` bpb, `4.0s`
- `linear_only_fft`: `2.6262` bpb, `3.1s`

So the scan pivot says:

- the FFT path matches the kernel path numerically at inference, but training lands slightly worse
- `untied_base_fft` is faster by about `6.2s` on average, but loses about `0.0075 bpb`
- `linear_only_fft` is also slightly worse than kernel `linear_only`
- the live score winner stays `untied_base` with the materialized kernel

Low-bit quantization ladder on kernel `untied_base`, 3-seed means:

- `fp16`: `2.5828` bpb, learned payload `1.797 MB`
- `uniform int8`: `2.6097` bpb, learned payload `0.652 MB`
- `uniform int6`: `2.5969` bpb, learned payload `0.590 MB`
- `uniform int4`: `2.6636` bpb, learned payload `0.527 MB`
- `uniform int3`: `2.9158` bpb, learned payload `0.496 MB`

So the quant row says:

- the low-bit surface is not monotone
- `int6` is the current sweet spot
- `int4` is already a real cliff
- `int3` is not viable
- quantization should now be used as a scaling lever, not as an end in itself

Payload-aware widening pilot on kernel `untied_base`, seed `42`:

- scale `1.25`: `2.5277` bpb, `626,906` params, `119s`, `int6 -> 2.5617` at `0.809 MB`
- scale `1.5`: `2.5126` bpb, `799,186` params, `121s`, `int6 -> 2.5471` at `0.873 MB`
- scale `1.75`: `2.4838` bpb, `987,850` params, `127s`, `int6 -> 2.5164` at `0.984 MB`
- scale `2.0`: `2.4467` bpb, `1,192,898` params, `132s`, `int6 -> 2.4806` at `1.187 MB`

So the widening pilot says:

- `untied_base` improves monotonically through scale `2.0` on seed `42`
- the gain is large enough to matter, not noise-sized
- train time rose only modestly across this range
- `int6` remains a viable payload lever even after widening
- the next step is replication of scales `1.75` and `2.0`, not blind growth beyond them

Payload-aware widening replication on kernel `untied_base`, 3-seed means:

- scale `1.75`: `2.4860` bpb, `987,850` params, `136.7s`, `int6 -> 2.5196` at `0.984 MB`
- scale `2.0`: `2.4504` bpb, `1,192,898` params, `139.2s`, `int6 -> 2.4837` at `1.187 MB`

Against the old kernel `untied_base` frontier:

- baseline: `2.5742` bpb, `471,010` params, `128.9s`

So the widening replication says:

- the gain is real across seeds
- scale `2.0` is the new live score frontier
- train time only rose by about `10.3s` over the original frontier
- even post-train `int6`, widened `2.0` is still better than the old fp16 baseline
- the next attack is widened `linear_only`, not more blind widening

Widened `linear_only` challenge, 3-seed means:

- scale `1.75`: `2.5336` bpb, `400,864` params, `4.6s`, `int6 -> 2.5754` at `0.359 MB`
- scale `2.0`: `2.5153` bpb, `476,416` params, `4.9s`, `int6 -> 2.5579` at `0.423 MB`

Against widened `untied_base`:

- `untied_base` scale `1.75`: `2.4860` bpb
- `untied_base` scale `2.0`: `2.4504` bpb

So the widened linear challenge says:

- the linear path remains the speed frontier by a huge margin
- but the widened hybrid still owns the score frontier
- the correction path is still earning its keep at the current best scale
- the next honest attack is a higher-scale hybrid pilot, not deleting the correction path

Higher-scale hybrid pilot on kernel `untied_base`, seed `42`:

- scale `2.5`: `2.4273` bpb, `1,652,146` params, `158.8s`, `int6 -> 2.4648` at `1.209 MB`
- scale `3.0`: `2.4014` bpb, `2,176,930` params, `186.4s`, `int6 -> 2.4385` at `1.586 MB`

So the high-scale pilot says:

- the score curve is still improving through `3.0x`
- train time is rising, but not explosively yet
- even post-train `int6`, the `3.0x` pilot is better than the replicated `2.0x` fp16 frontier
- the next correct move is replication, not extrapolation

Higher-scale hybrid replication on kernel `untied_base`, 3-seed means:

- scale `2.5`: `2.4231` bpb, `1,652,146` params, `166.5s`, `int6 -> 2.4610` at `1.209 MB`
- scale `3.0`: `2.3964` bpb, `2,176,930` params, `193.7s`, `int6 -> 2.4341` at `1.586 MB`

Against the previous widened frontier:

- scale `2.0`: `2.4504` bpb, `1,192,898` params, `139.2s`

So the higher-scale replication says:

- the score curve is still improving through `3.0x`
- the gain over `2.0x` is real across seeds, about `0.0540 bpb`
- train time is rising, but still smoothly
- there is still no payload wall in sight
- the next honest move is one more higher-scale pilot, not declaring victory

Far-scale hybrid pilot on kernel `untied_base`, seed `42`:

- scale `4.0`: `2.3565` bpb, `3,423,106` params, `232.2s`, `int6 -> 2.3989` at `2.479 MB`
- scale `5.0`: `2.3280` bpb, `4,931,426` params, `308.6s`, `int6 -> 2.3689` at `3.560 MB`

So the far-scale pilot says:

- the score curve is still improving through `5.0x`
- train time is rising faster now, but still not catastrophically
- there is still no learned-payload wall anywhere near the current frontier
- the next informative move is a jump toward the real payload regime, not `6x` or `7x`

Cap-neighborhood pilot on kernel `untied_base`, seed `42`:

- scale `8.0`: `2.2826` bpb, `11,029,250` params, `555.5s`, `int6 -> 2.3277` at `7.929 MB`
- scale `10.0`: `2.2482` bpb, `16,405,186` params, `831.2s`, `int6 -> 2.3002` at `11.779 MB`
- scale `12.0`: `2.3128` bpb, `22,829,698` params, `1065.6s`, `int6 -> 2.3555` at `16.379 MB`

So the cap-neighborhood pilot says:

- the curve still improves strongly at `8.0x` and `10.0x`
- the first clear bend appears at `12.0x`
- `12.0x` is already slightly over the estimated `int6` cap and worse than `10.0x`
- the next honest move is replication of `8.0x` and `10.0x`, not more time on `12.0x`

Cap-neighborhood replication on kernel `untied_base`, 3-seed means:

- scale `8.0`: `2.2875` bpb, `11,029,250` params, `548.0s`, `int6 -> 2.3318` at `7.929 MB`
- scale `10.0`: `2.2650` bpb, `16,405,186` params, `825.9s`, `int6 -> 2.3104` at `11.779 MB`

So the cap-neighborhood replication says:

- `10.0x` is the real cap-neighborhood frontier
- `8.0x` is slightly worse but still strong
- the architecture still buys bits near contest-sized learned payloads
- the next uncertainty is no longer scale itself, but whether the `12.0x` bend is a recipe failure

`12.0x` recipe salvage pilot, seed `42`:

- baseline `1e-3`, `1000` steps: `2.3128` bpb, `int6 -> 2.3555`
- `5e-4`, `1000` steps: `2.2327` bpb, `int6 -> 2.2850`
- `3e-4`, `1000` steps: `2.2623` bpb, `int6 -> 2.3159`
- `5e-4`, `1500` steps: `2.1993` bpb, `int6 -> 2.2546`

`12.0x` recipe salvage replication, 3-seed means:

- `12.0x`, `lr=5e-4`, `1500` steps: `2.1867` bpb, `22,829,698` params, `1765.8s`
- post-train `int6`: `2.2413` bpb at `16.379 MB`

So the salvage replication says:

- the `12.0x` bend was mostly a bad optimizer regime, not a hard architectural wall
- salvaged `12.0x` is now the best replicated score frontier in the branch
- it clearly beats the old `10.0x` cap-neighborhood frontier on score
- but it lands slightly over the nominal `16 MB` `int6` payload line
- the next step is cap fit, not more recipe combinatorics

## Next Matrix

| Question | Why it matters for compression | Candidates | Metric | Pass condition |
|---|---|---|---|---|
| Can `untied_base` widen profitably before score or Mac time bends the wrong way? | The quant row freed payload, so the next real question is whether extra learned capacity buys bits | seed-42 pilot closed; now replicate `untied_base` scales `1.75`, `2.0` with post-train `int6` | exact official `bpb`, train time, fp16 payload, int6 payload | pilot passed; replication decides the live widened frontier |
| Does the win still belong to the hybrid once it is widened? | If widening helps only the linear path, the correction path should be demoted again | widened `untied_base` vs widened `linear_only` at scales `1.75`, `2.0` | same official metric | passed: hybrid keeps a real edge at the replicated widened frontier |
| Does widening keep paying beyond `2.0x`, or has the curve started to bend? | The branch is still tiny relative to the artifact cap, but Mac time is not free | `untied_base` pilot scales `2.5`, `3.0` with post-train `int6` | exact official `bpb`, train time, fp16 payload, int6 payload | one pilot scale beats `2.0x` cleanly enough to deserve replication |
| Where does the scale curve first bend? | The branch is still far below the submission cap, but time is not free | `untied_base` pilot scales `4.0`, `5.0` with post-train `int6` | exact official `bpb`, train time, fp16 payload, int6 payload | at least one higher scale beats `3.0x` cleanly enough to deserve replication |
| What happens near the actual payload cap? | The relevant question is no longer local curvature but whether the architecture still buys bits near contest-sized learned payloads | pilot closed; now replicate `untied_base` scales `8.0`, `10.0` with post-train `int6` | exact official `bpb`, train time, fp16 payload, int6 payload | pilot passed, `12.0x` bent, replication decides the cap-neighborhood frontier |
| Can the salvaged `12.0x` recipe be pulled back under the payload line without giving back too much score? | The best replicated branch now lives slightly over the nominal `16 MB` `int6` cap, so the next question is fit, not raw scale | `11.5x`, `11.75x` with the salvaged `12.0x` recipe (`lr=5e-4`, `1500` steps) | exact official `bpb`, train time, `int6` payload, post-train `int6` `bpb` | one scale lands under `16 MB` and stays close enough to salvaged `12.0x` to become the cap-fit frontier |
| When do we start packing? | No submission artifact exists yet | best widened `Conker-2` variant only | exact official-style `bpb`, packed bytes | only pack after the widening row is settled |

Cap-fit replication, 3-seed means:

- `11.5x`, `lr=5e-4`, `1500` steps: `2.1831` bpb, `21,125,266` params, `1747.9s`, `int6 -> 2.2367` at `15.159 MB`
- `11.75x`, `lr=5e-4`, `1500` steps: `2.1852` bpb, `21,969,290` params, `1702.3s`, `int6 -> 2.2391` at `15.763 MB`

So the cap-fit replication says:

- both scales hold under the nominal `16 MB` line
- the seed-42 `11.75x` win did not replicate
- `11.5x` is now the best replicated under-cap frontier
- the next step is fixed-budget recipe work at `11.5x`

## Immediate Execution Order

1. Keep kernel `untied_base` as the score frontier.
2. Use `int6` as the current payload sweet spot.
3. Treat widened `linear_only` as the speed frontier, not the score frontier.
4. Keep salvaged `12.0x`, `lr=5e-4`, `1500` steps as the best raw score frontier.
5. Promote `11.5x`, `lr=5e-4`, `1500` steps to the replicated under-cap frontier.
6. Run fixed-budget recipe pilots on `11.5x`.
7. Keep the FFT path as a speed sidecar unless a later training recipe closes the score gap.

Under-cap recipe pilot, seed `42`:

- baseline `11.5x`, `lr=5e-4`, `1500` steps: `2.1827` bpb, `int6 -> 2.2381`
- `11.5x`, `lr=4e-4`, `1500` steps: `2.1938` bpb, `int6 -> 2.2487`
- `11.5x`, `lr=4e-4`, `1800` steps: `2.1909` bpb, `int6 -> 2.2426`
- `11.5x`, `lr=5e-4`, `1800` steps: `2.1753` bpb, `int6 -> 2.2284`

So the fixed-budget recipe row says:

- lower LR hurts at this under-cap scale
- longer training helps
- the only recipe worth replicating is `11.5x`, `lr=5e-4`, `1800` steps

That replication is now also closed, 3-seed means:

- `11.5x`, `lr=5e-4`, `1800` steps: `2.1757` bpb, `21,125,266` params, `2465.4s`, `int6 -> 2.2276` at `15.159 MB`

So the under-cap recipe replication says:

- the longer-training gain is real across seeds
- `11.5x`, `lr=5e-4`, `1800` steps is now the best replicated under-cap `Conker-2` branch
- the next fixed-budget question is optimizer quality, not more LR guessing

Muon pilot on that live under-cap branch:

- `mom95_warm500`, seed `42`: `3.1610` bpb, `int6 -> 3.5204`, `3912.4s`
- `mom99_warm1500`, seed `42`: `3.1449` bpb, `int6 -> 3.7174`, `2352.0s`

So the first Muon answer is already clear:

- both tested Muon schedules fail catastrophically on the live `Conker-2` under-cap branch
- this is not a marginal optimizer miss
- `Conker-2` should stay on AdamW until a much more defensible Muon hypothesis exists

## Conker-3 Side Branch

`Conker-3` starts from a new assumption set:

- the frozen linear substrate is the true survivor
- the `Conker-2` nonlinear correction may be mostly local residual repair
- so the next dangerous redesign is to replace that correction expert with a fully parallel local residual coder

Initial probe regime:

- official golf data
- seed `42`
- `600` steps
- scale `3.0`

Initial variants:

- `linear_only`
- `base`
- `gated`
- `window4`
- `window16`
- `local_only`

Pass condition:

- `base` must beat `linear_only` in this cheap probe regime

Kill condition:

- if `base` does not beat `linear_only`, or if window/gating do not matter, kill `Conker-3` early and keep `Conker-2` as the live branch

Initial probe results, seed `42`, `600` steps, scale `3.0`:

- `linear_only`: `2.6253` bpb
- `base`: `2.5082`
- `gated`: `2.5041`
- `window4`: `2.4946`
- `window16`: `2.5155`
- `local_only`: `2.5425`

So the cheap kill batch says:

- `Conker-3` survives
- local residual repair is real
- shorter local context already looks better than the wider local window

Follow-up probe results, seed `42`, `1000` steps:

- scale `3.0`
  - `linear_only`: `2.4847` bpb, `7.6s`
  - `base`: `2.3434`, `14.0s`
  - `gated`: `2.3241`, `23.1s`
  - `window4`: `2.3245`, `13.6s`
  - `shared_embedding`: `2.3729`, `12.4s`
- scale `5.0`
  - `linear_only`: `2.4570` bpb, `12.9s`
  - `base`: `2.2762`, `22.4s`
  - `gated`: `2.2783`, `31.4s`
  - `window4`: `2.2677`, `21.0s`

So the follow-up row says:

- the redesign is still alive after more compute
- shared embeddings hurt again
- at `3.0x`, `gated` and `window4` are effectively tied
- at `5.0x`, `window4` becomes the clean mechanism winner
- `base` and `gated` collapse onto each other at `5.0x`, so gating currently looks like dead complexity

`window4` replication and longer-step probe:

- `window4`, `5.0x`, `1000` steps, seed `43`: `2.2704` bpb, `21.4s`
- `window4`, `5.0x`, `1000` steps, seed `44`: `2.2683` bpb, `21.0s`
- 3-seed mean for `window4`, `5.0x`, `1000` steps: `2.2688` bpb
- `window4`, `5.0x`, `1500` steps, seed `42`: `2.2080` bpb, `32.1s`
- `window4`, `5.0x`, `1500` steps, seed `43`: `2.2139` bpb, `31.9s`
- `window4`, `5.0x`, `1500` steps, seed `44`: `2.2114` bpb, `32.1s`
- 3-seed mean for `window4`, `5.0x`, `1500` steps: `2.2111` bpb
- `window4`, `8.0x`, `1000` steps, seed `42`: `2.2025` bpb, `35.2s`
- `window4`, `8.0x`, `1000` steps, seed `43`: `2.2087` bpb, `44.5s`
- `window4`, `8.0x`, `1000` steps, seed `44`: `2.2083` bpb, `46.2s`
- 3-seed mean for `window4`, `8.0x`, `1000` steps: `2.2065` bpb
- `window4`, `8.0x`, `1500` steps, seed `42`: `2.1519` bpb, `52.4s`
- `window4`, `8.0x`, `1500` steps, seed `43`: `2.1600` bpb, `51.4s`
- `window4`, `8.0x`, `1500` steps, seed `44`: `2.1532` bpb, `51.9s`
- 3-seed mean for `window4`, `8.0x`, `1500` steps: `2.1550` bpb
- `window4`, `10.0x`, `1000` steps, seed `42`: `2.1777` bpb, `55.5s`
- `window4`, `10.0x`, `1500` steps, seed `42`: `2.1352` bpb, `70.2s`

So the next `Conker-3` answer is:

- `window4 5.0x` is stable across seeds
- it still improves materially with more steps
- `Conker-3` now has a real fast frontier candidate, not just a cheap probe survivor
- the branch is still improving with width on the first `8.0x` pilot
- `8.0x` also survives the longer-step replication
- the `10.0x`, `1000`-step pilot is already worse than replicated `8.0x`, which suggests the branch is short-budget recipe-limited there
- `10.0x`, `1500` steps immediately recovers and becomes the best single-seed `Conker-3` score so far

Offline weight-geometry audit on `window4`, `5.0x`, `1000` steps, seed `42`:

- baseline audit eval: `2.3704` bpb
- worst single-matrix `int3` hit: `local_readout.out.weight`, `+0.2200 bpb`
- next tier:
  - `local_embedding.weight`, `+0.0510`
  - `linear_readout.layers.0.weight`, `+0.0410`
  - `linear_readout.out.weight`, `+0.0388`
- several large matrices are nearly harmless at `int4`

So the `3-bit` research direction is now clearer:

- the low-bit cliff is concentrated, not uniform
- the local residual output head is the main fragile object
- structured packing should target that local path first instead of treating all trainable matrices equally

Mixed low-bit audit on `window4`, `5.0x`, `1500` steps, seed `42`:

- baseline audit eval: `2.3171` bpb
- `uniform_int3`: `2.6864`, `+0.3693`
- `uniform_int4`: `2.3703`, `+0.0532`
- `int3_keep_local_out`: `2.4592`, `+0.1421`
- `int3_keep_local_out_embed`: `2.4006`, `+0.0836`
- `int3_keep_local_path`: `2.3772`, `+0.0601`
- `int3_rest_local_int4`: `2.4074`, `+0.0921`

So the packing direction is now sharper:

- naive `int3` is still bad
- `uniform_int4` is already fairly competitive on this branch
- selective `int3` only gets close when most of the local path stays higher precision
- the right next quant experiments should treat the local path as a protected island
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

Decay-bank substrate ablation on `window4`, `8.0x`, `1500` steps, seed `42`:

- `logspace`: `2.1540` bpb
- autocorrelation-matched bank: `2.1670`
- narrow-bank control (`half_life_max=32`): `2.1027`

So the current substrate answer is:

- naive autocorrelation matching did not beat the generic log-spaced bank
- but the narrow control beat both
- the live `Conker-3` branch appears to want a shorter linear memory scaffold, not a broader one

Replicated narrow-bank row on `window4`, `8.0x`, `1500` steps, 3-seed means:

- narrow bank (`half_life_max=32`): `2.1064` bpb
- mean train time: `53.5s`

Against the original log-spaced bank:

- log-spaced `8.0x`, `1500`: `2.1550` bpb
- narrow `8.0x`, `1500`: `2.1064` bpb

So the narrower substrate win replicated cleanly.

Narrow-bank `window4`, `10.0x`, `1500` steps, seed `42`:

- `2.0871` bpb
- `uniform_int6`: `2.1211` bpb

Narrow-bank mixed low-bit audit on `window4`, `8.0x`, `1500`, seed `42`:

- baseline audit eval: `2.2185` bpb
- `uniform_int3`: `2.8153`, `+0.5968`
- `uniform_int4`: `2.3438`, `+0.1253`
- `int3_keep_local_path`: `2.3876`, `+0.1691`

So the new tradeoff is explicit:

- tighter short-memory reservoirs improve bridge score
- but may worsen low-bit packing robustness
- the next substrate sweep should map both score and quantization behavior across `max_half_life`

Half-life cap sweep on `window4`, `8.0x`, `1500`, seed `42`:

- `16`: `2.1014` bpb, `uniform_int6 -> 2.1355`
- `32`: `2.1075` bpb, `uniform_int6 -> 2.1404`
- `64`: `2.1141` bpb, `uniform_int6 -> 2.1487`
- `128`: `2.1382` bpb, `uniform_int6 -> 2.1699`
- broad log-spaced `512`: `2.1540` bpb

That curve says:

- the best tested cap is now `16`
- performance degrades monotonically as the half-life tail widens
- the next replication row should be `16` vs `32`, not a return to broader banks

Low half-life follow-up on `window4`, `8.0x`, `1500`, seed `42`:

- `4`: `2.1031` bpb, `uniform_int6 -> 2.1359`
- `8`: `2.0993` bpb, `uniform_int6 -> 2.1302`
- `12`: `2.1011` bpb, `uniform_int6 -> 2.1332`
- `16`: `2.1014` bpb, `uniform_int6 -> 2.1355`

So the refined curve says:

- the best tested cap is now `8`
- `4` is already too short
- `8` appears to be the current balance point between reservoir complementarity and redundancy with the local path
- the next replication row should be `8` vs `16`

That replication row is now closed on `window4`, `8.0x`, `1500`, 3-seed means:

- `half_life_max=8`: `2.1000` bpb, `uniform_int6 -> 2.1323`, `52.7s`
- `half_life_max=16`: `2.1013` bpb, `uniform_int6 -> 2.1348`, `67.5s`

So `8` wins, but narrowly.

Mixed-quant audit on `window4`, `8.0x`, `1500`, `half_life_max=8`, seed `42`:

- baseline audit eval: `2.2155` bpb
- `uniform_int3`: `2.9449`, `+0.7293`
- `uniform_int4`: `2.3378`, `+0.1222`
- `int3_keep_local_path`: `2.4992`, `+0.2837`

So the better substrate is also less quantization-friendly.

`window4`, `10.0x`, `1500`, `half_life_max=8`, seed `42`:

- `2.0845` bpb
- `uniform_int6 -> 2.1181`

Replicated `window4`, `10.0x`, `1500`, `half_life_max=8`, 3-seed means:

- `2.0849` bpb
- `uniform_int6 -> 2.1180`
- `84.7s`

That is the current `Conker-3` score frontier.

Direct packing comparison at seed `42`:

- `half_life_max=8`: `uniform_int4 +0.1222`, `uniform_int3 +0.7293`
- `half_life_max=16`: `uniform_int4 +0.1175`, `uniform_int3 +0.6159`

So the better substrate is also the worse packer.

Packed showdown on `window4`, `10.0x`, `1500`, 3-seed means:

- `half_life_max=8`: fp16 `2.0832`, `int6 2.1040`, `int4 2.2393`
- `half_life_max=16`: fp16 `2.0837`, `int6 2.1019`, `int4 2.2283`

This is the cleanest current answer:

- fp16 still slightly favors `8`
- packed score favors `16`
- under the real compression framing, `16` is ahead

Packed-first scaling wave on `half_life_max=16`, seed `42`:

- `12x`: fp16 `2.0709`, `int6 2.0920`, `int4 2.2140`
- `14x`: fp16 `2.0616`, `int6 2.0793`, `int4 2.1966`
- `16x`: fp16 `2.0507`, `int6 2.0726`, `int4 2.1715`

So the packed curve is still improving through `16x`, and `16x int6` is still under cap.

Precision-allocation audit on `10x`, `1500`, `half_life_max=16`, seed `42`:

- `uniform_int4`: `+0.1215`
- `int4_keep_local_out`: `+0.1020`
- `int4_keep_local_out_embed`: `+0.0928`
- `int4_keep_local_path`: `+0.0903`

So protected `int4` helps, but not as much as the packed scaling wave itself.

First online TTT routing probe on `window4`, `10.0x`, `1500`, `half_life_max=16`, seed `42`:

- fp16
  - baseline: `2.2149`
  - `1` gate step @ `0.10`: `2.2151`
  - `3` gate steps @ `0.03`: `2.2136`
- `int6`
  - baseline: `2.2203`
  - `1` gate step @ `0.10`: `2.2190`
  - `3` gate steps @ `0.03`: `2.2175`

So the first routing-TTT read is:

- weak in fp16
- slightly more useful on the packed model
- worth one more packed-focused follow-up, but not yet a mainline win

Oscillatory hybrid probe on `window4`, `10.0x`, `1500`, `half_life_max=16`:

- `25%` oscillatory modes, seed `42`: fp16 `2.0751`, `int6 2.0955`, `int4 2.2348`
- `50%` oscillatory modes, 3-seed means: fp16 `2.0613`, `int6 2.0833`, `int4 2.2343`

Against the non-osc baseline:

- baseline fp16 `2.0837`
- baseline `int6 2.1019`
- baseline `int4 2.2283`

So oscillatory modes are a real win on fp16 and `int6`, but not on `int4`.

High-oscillation follow-up on the same branch, seed `42`:

- `62.5%`: fp16 `2.0568`, `int6 2.0801`, `int4 2.2416`, `69.6s`
- `75%`: fp16 `2.0526`, `int6 2.0740`, `int4 2.2361`, `73.6s`
- `87.5%`: fp16 `2.0517`, `int6 2.0738`, `int4 2.2449`, `105.8s`

So the fraction story now is:

- more oscillation still helps fp16 and `int6`
- `75%` and `87.5%` are effectively tied on packed `int6`
- `87.5%` is worse on `int4` and much slower
- replication should focus on `75%` and `87.5%`, not all higher fractions

Replication, 3-seed means:

- `75%`: fp16 `2.0531`, `int6 2.0745`, `int4 2.2404`, `73.2s`
- `87.5%`: fp16 `2.0512`, `int6 2.0729`, `int4 2.2410`, `114.8s`

So the replicated fraction read is:

- `87.5%` is the best fp16 and `int6` branch by a very small margin
- `75%` is still slightly better on `int4`
- the next question is period range, not more fraction pushing

Period-range sweep on `87.5%`, seed `42`:

- `4..32`: fp16 `2.0518`, `int6 2.0772`, `int4 2.2432`, `81.6s`
- `4..64`: fp16 `2.0515`, `int6 2.0725`, `int4 2.2411`, `92.8s`
- `8..64`: fp16 `2.0602`, `int6 2.0788`, `int4 2.2281`, `108.8s`
- `8..128`: fp16 `2.0596`, `int6 2.0819`, `int4 2.2238`, `121.1s`

So the period answer is:

- `4..64` remains the best overall oscillatory cadence range
- `4..32` is close on fp16 but worse packed
- higher period bands help `int4` slightly and hurt raw score too much

Oscillatory 8-hour packed matrix, `window4`, `1500`, `half_life_max=16`, periods `4..64`:

- `12x`, `75%`, seed `42`: fp16 `2.0436`, `int6 2.0658`, `int4 2.2297`, `90.6s`
- `12x`, `87.5%`, seed `42`: fp16 `2.0410`, `int6 2.0638`, `int4 2.2405`, `111.2s`
- `16x`, `75%`, 3-seed means: fp16 `2.0342`, `int6 2.0565`, `int4 2.2131`, `220.4s`
- `16x`, `87.5%`, 3-seed means: fp16 `2.0309`, `int6 2.0542`, `int4 2.2231`, `208.8s`
- `18x`, `75%`, 3-seed means: fp16 `2.0282`, `int6 2.0509`, `int4 2.2127`, `253.1s`
- `18x`, `87.5%`, 3-seed means: fp16 `2.0271`, `int6 2.0508`, `int4 2.2205`, `256.5s`

Local-path byte-allocation probes on `18x`, `87.5%`, seed `42`:

- base: fp16 `2.0283`, `int6 2.0510`, `int4 2.2314`
- `local_hidden_mult=0.75`: fp16 `2.0315`, `int6 2.0572`, `int4 2.2310`
- `local_hidden_mult=0.50`: fp16 `2.0383`, `int6 2.0603`, `int4 2.2442`
- `local_scale_override=0.20`: fp16 `2.0291`, `int6 2.0523`, `int4 2.2342`

So the matrix answer is:

- scaling still buys packed score through `16x`, and `18x` keeps helping on raw score even while slipping over the nominal `int6` cap
- the best replicated under-cap packed branch is `16x`, `87.5%`, `4..64`
- `87.5%` wins on fp16 and `int6`, but `75%` remains the better `int4` branch
- shrinking the local path is not the next lever

Packed next-wave row:

- `16.5x`, seed `42`: fp16 `2.0313`, `int6 2.0536`, `int4 2.2292`, `14.264 MB`
- `17.0x`, seed `42`: fp16 `2.0295`, `int6 2.0514`, `int4 2.2258`, `15.020 MB`
- `17.5x`, seed `42`: fp16 `2.0297`, `int6 2.0515`, `int4 2.2311`, `15.794 MB`
- `16x`, `1800`, seed `42`: fp16 `2.0281`, `int6 2.0473`, `int4 2.2234`
- `16x`, `2200`, seed `42`: fp16 `1.9899`, `int6 2.0146`, `int4 2.1788`
- `16x`, `1500`, static bank gate, seed `42`: fp16 `2.0237`, `int6 2.0452`, `int4 2.1886`
- `16x`, `1800`, static bank gate, seed `42`: fp16 `2.0214`, `int6 2.0361`, `int4 2.1877`
- `16x`, `1800`, naive pack-train, seed `42`: fp16 `3.5570`, `int6 3.6240`, `int4 3.8628`

So the new answer is:

- longer training is a much stronger lever than the small cap-fit scale moves
- static bank gating is a real packed-score improvement
- naive train-through-pack is not viable in its current form

Combo pilots, seed `42`:

- `16x`, `2200`, static gate: fp16 `1.9856`, `int6 2.0066`, `int4 2.1479`, `210.6s`
- `17x`, `1800`, static gate: fp16 `2.0186`, `int6 2.0347`, `int4 2.1850`, `193.3s`
- `17x`, `2200`, static gate: fp16 `1.9814`, `int6 2.0019`, `int4 2.1463`, `230.5s`

So the combined read is:

- longer training and static bank gating compound cleanly
- `17x / 2200 / staticgate` is the best seed-42 packed branch so far
- `16x / 2200 / staticgate` remains the safer under-cap version to replicate first

Static-gate frontier replication:

- `16x`, `2200`, static gate, 3-seed means: fp16 `1.9845`, `int6 2.0067`, `int4 2.1506`, `13.527 MB`
- `17x`, `2200`, static gate, 3-seed means: fp16 `1.9828`, `int6 2.0042`, `int4 2.1523`, `15.020 MB`
- `18x`, `2200`, static gate, seed `42`: fp16 `1.9790`, `int6 2.0027`, `int4 2.1415`, `16.588 MB`

So the frontier is now:

- best replicated under-cap branch: `17x / 2200 / staticgate`
- best raw pilot above cap: `18x / 2200 / staticgate`

`Conker-4` first probe:

- reference:
  - `Conker-3 window4 10x 1000 staticgate`: fp16 `2.0865`, `int6 2.1081`, `int4 2.2601`
- direct probability mixer over:
  - live `Conker-3`
  - exact 1-token backoff
  - exact 2-token backoff
  - recency prior
- results:
  - learned MLP mixer: `NaN`
  - support-weighted mixer: `NaN`
  - support-weighted mixer with neural expert actually frozen out of the trainable set: `NaN`
  - zero-step static frozen-neural ensemble: fp16 `3.7207`, `int6 3.7619`, `int4 3.7724`
  - low-LR (`1e-4`) frozen-neural run: `NaN`

So the matrix answer is:

- the first exact-expert ensemble is a real negative result
- this is not “already solved, just ensemble it” in the current raw-count formulation
- if a `Conker-4` branch returns, it should likely be residual / calibration style, not this direct probability-space mix

`Conker-4b` is that residual / calibration return, and it changed the branch quickly.

On `window4 / 10x / half_life=16 / osc=87.5%`:

- reference `Conker-3 1000`: `2.0865 bpb`
- `Conker-4b full 500`: `1.9547`
- `Conker-4b full 550`: unstable (`NaN`)
- `Conker-4b no_exact2 500`: `2.1429`
- `Conker-4b no_recency 500`: `2.0079`
- `Conker-4b no_recency 1000`, 3-seed means:
  - fp16 `1.8823`
  - `int6 1.8837`
  - `int4 1.8836`
  - `47.2s`
  - payload `~0.252 MB`

So the updated matrix answer is:

- the residual formulation is the right `Conker-4` interface
- `exact2` is essential
- recency is useful but destabilizing in the current form
- the live `Conker-4` survivor is `Conker-4b no_recency 1000`

Latest architecture cuts on that survivor family, seed `42`:

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
- `exact1 + exact2`, `1000`, no base-confidence coupling: `1.8822`
- `exact1 + exact2 + exact3 + wordclass2`, `1000`, support-only: unstable (`NaN`)
- `exact1 + exact2 + exact3 + delim2 + wordclass2`, `1000`, all broad experts support-only: unstable (`NaN`)
- `exact1 + exact2 + exact3 + wordclass2`, `500`, support-only, `lr=5e-4`, `cap=2.0`: `2.9659`
- `exact1 + exact2 + exact3 + delim2 + wordclass2`, `500`, support-only, `lr=5e-4`, `cap=0.5`: `3.8151`

So the new matrix answer is:

- `exact2` is the spine, but not sufficient by itself
- `exact1` should contribute support without opening candidates
- `exact3` is real and materially improves the branch
- the delimiter expert is also real, but only as support
- `special2` and `number2` are both real support-only residuals
- `special2 + number2` stack, so the additive-correction pattern is holding
- `urlpath2` is real but weaker and mostly overlapping; it only adds a small extra gain on top of `special2 + number2`
- `markup2` is real and stronger than `urlpath2`; markup / HTML-like continuation is a better next expert family
- `attr2` is real but secondary; it helps less than `markup2`
- `entity2` is not a live branch in its current form
- `stack2` is real but weak; bracket-obligation support is mostly redundant with the current stack of exact experts
- delimiter subtype continuation is not a live branch in its current form
- the coarse `wordclass2` expert is not orthogonal enough in its current form; it is unstable or stably harmful
- base-confidence coupling is almost irrelevant

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

Replicated longer-memory row on the same gate-only winner, 3-seed means:

- `seq_len=512 batch=8 steps=1500`: fp16 `1.7871`, `int6 1.7868`, `int4 1.7851`, `130.0s`

So the longer exact-memory row is real, and it is now the live `Conker-4b` frontier.

Longer-horizon base-model pilot on the same row, seed `42`:

- `seq_len=1024 batch=4 steps=1000`: fp16 `1.7908`, `int6 1.7902`, `int4 1.7865`, `160s`
- `seq_len=1024 batch=4 steps=1500`: fp16 `1.7883`, `int6 1.7874`, `int4 1.7842`, `365s`

So `1024` is slightly worse than the replicated `512/1500` row. The gain is not coming from arbitrarily more base-model context.

Exact-only lookback pilots on the `256` gate-only row, seed `42`:

- `seq_len=256 batch=16 steps=1000 exact_context_span=512`: unstable (`NaN`)
- `seq_len=256 batch=16 steps=1000 exact_context_span=1024`: unstable (`NaN`)

So naive wider exact-memory bands are not a free win. The next memory/copy experiments need a better interface than simply widening the exact-match support mask.

Tandem-training pilots on the same `Conker-4b` stack, seed `42`:

- `seq_len=256 batch=16 steps=1000 lr=5e-4`, `freeze_base=False`: fp16 `0.5624`, `int6 0.5752`, `int4 0.7367`, `100s`
- `seq_len=256 batch=16 steps=1000 lr=3e-4`, `freeze_base=False`: unstable (`NaN`)
- `seq_len=512 batch=8 steps=1000 lr=5e-4`, `freeze_base=False`: fp16 `0.5717`, `int6 0.5786`, `int4 0.7399`, `222s`

So the “frozen import” assumption was wrong. Tandem training the inherited `Conker-3` base with the residual stack is a step-change branch, but it appears to have a narrow stable optimizer regime.

Replicated tandem row, 3-seed means:

- `seq_len=256 batch=16 steps=1000 lr=5e-4`, `freeze_base=False`: fp16 `0.5615`, `int6 0.5745`, `int4 0.7412`

Hostile validation checks:

- fresh-process checkpoint re-eval of saved seed-43 tandem state: `test_bpb 0.5648`
- fresh-process transformed eval of the same saved seed-43 tandem state:
  - `test / reverse`: `2.2799 bpb` over `204,800` tokens
  - `test / shuffle`: `2.3015 bpb` over `204,800` tokens
  - `train / none`: `1.3306 bits/token` over `204,800` tokens
- full held-out validation sweep of the same saved seed-43 tandem state:
  - `test / none / full_split`: `0.5716 bpb` over `62,021,632` tokens
- sampled train/val exact-window overlap audit (`20k` train vs `5k` val windows): `0/5000` overlaps at lengths `32/64/128/256`

So the current tandem result is not explained by the bridge accidentally reporting train loss; the saved checkpoint collapses under reversed/shuffled validation, the full held-out sweep stays in the same high-`0.5x` regime as the small checkpoint slice, and the first sampled overlap audit does not show blatant train/val duplication.

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

First cache branch, now known to be invalid in its trained form:

- normalized causal cache
- no two-pass rescoring
- no non-normalized blend tricks

Previously best apparent row, seed `42`:

- `cache_only`
- `seq_len=256 batch=16 steps=1000`
- full held-out fp16 `0.0721 bpb`
- `int4 0.0714`
- `int6 0.0740`
- payload `~0.254 MB`
- artifact:
  [conker6_cacheonly_seq256_steps1000_full_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_cacheonly_seq256_steps1000_full_2026-03-28.json)

Fresh-init legality audit on the same recipe:

- probability sums near `1`
- row-wise future invariance: exact `0`
- flat-stream invariance: exact `0`
- artifact:
  [conker6_legality_cacheonly_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_legality_cacheonly_2026-03-28.json)

Trainable-subset attack:

- only reported trainables:
  - `causal_mask` `(256,256)` = `65,536`
  - `vocab_axis` `(1024,)` = `1,024`
- `fixed_vocabulary + fixed_causal_mask`: full held-out `5.7521 bpb`
- `fixed_vocabulary + learnable_causal_mask`: full held-out `0.0721 bpb`

Artifacts:

- [conker6_cacheonly_seq256_steps1000_fixedbuffers_full_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_cacheonly_seq256_steps1000_fixedbuffers_full_2026-03-28.json)
- [conker6_cacheonly_seq256_steps1000_maskonly_full_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_cacheonly_seq256_steps1000_maskonly_full_2026-03-28.json)

That fresh-init audit turned out to be misleading.

Trained-mask attack now says:

- patched saved seed-42 trained mask:
  - row-wise future max logit diff `18.4207`
  - flat-stream future max diff `0.0`
  - probability sum max `1.4260`
  - max abs normalization error `0.4260`

So the trained `Conker-6` winner is not a legal normalized row-wise causal model.

Discounted backoff experiments, full held-out:

- `witten_bell`: `0.4004 bpb`
- `absolute_discount`: `1.3097 bpb`

Artifacts:

- [conker6_wittenbell_seq256_steps1000_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_wittenbell_seq256_steps1000_2026-03-28.json)
- [conker6_absdisc_seq256_steps1000_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_absdisc_seq256_steps1000_2026-03-28.json)

So proper textbook smoothing has not yet beaten raw hard exact backoff here.

Two sharp probes:

- `steps=1`, full held-out: `0.3333 bpb`
- `disable_exact3`, full held-out: `0.0983 bpb`

Artifacts:

- [conker6_cacheonly_seq256_steps1_full_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_cacheonly_seq256_steps1_full_2026-03-28.json)
- [conker6_cacheonly_seq256_steps1000_noexact3_full_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_cacheonly_seq256_steps1000_noexact3_full_2026-03-28.json)

Interpretation:

- some small trained component still matters
- that component is overwhelmingly the learned `causal_mask`
- `exact3` is worth about `0.026 bpb`
- the branch is genuinely higher-order exact context, not just low-order cacheing

Mask-geometry ablation on the trained legal winner:

- baseline: `0.0721 bpb`
- nonnegative clamp: `0.0714`
- magnitude prune `90/95/98%`: `4.4732 / 4.3152 / 4.1847`
- row-top-k `16/8`: `4.5754 / 4.3589`
- artifact:
  [conker6_mask_ablation_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_ablation_2026-03-28.json)

Deviation read:

- pruning variants move far from the learned mask:
  - cosine similarity only `0.15-0.36`
  - L2 deviation about `0.89-0.95`
- within active lower-triangular support, the learned mask is already effectively nonnegative

So the `Conker-6` win is not a sparse-selector trick. It is a dense learned causal-weight geometry layered on top of the exact-history cache.

Mask dump + structure attack:

- raw `256x256` mask dump:
  [npy](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.mask.npy)
  [csv](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.mask.csv)
- visuals:
  [mask](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.mask.png)
  [lag-mean](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.lag_mean_mask.png)
  [lag-mean diff](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.lag_mean_diff.png)
- summary:
  [conker6_mask_geometry_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_mask_geometry_2026-03-28.json)

What the geometry says:

- active weights lie in a very narrow band, about `0.887 .. 1.057`
- lag means also lie in a narrow band, about `0.946 .. 1.030`
- top singular-value energy is fairly concentrated:
  - top `8`: `53.5%`
  - top `16`: `62.7%`
  - top `32`: `72.1%`

So visually the mask looks almost Toeplitz and low-rank.

But the structure attacks fail catastrophically:

- `toeplitz_mean`: `5.7521 bpb`
- `toeplitz_band_32`: `4.9290`
- `toeplitz_band_64`: `5.3167`
- `lowrank_8_masked`: `5.7522`
- `lowrank_16_masked`: `5.7521`
- `row_normalized`: `5.7521`

Most important:

- `toeplitz_mean` has cosine `0.9998` to the learned mask and only `0.0188` L2 deviation
- yet it collapses the score from `0.0721` to `5.7521`

So the current `Conker-6` mask is not just “close enough” lag geometry. It depends sharply on exact per-entry values.

Cross-seed residual test:

- summary:
  [conker6_seed_residual_compare_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_seed_residual_compare_2026-03-28.json)

Three seeds all stay in the same score regime:

- seed `42`: `0.07209 bpb`
- seed `43`: `0.07141`
- seed `44`: `0.071999`

Stable across seeds:

- raw mask cosine about `0.9996`
- Toeplitz-mean cosine about `1.0`
- Toeplitz-mean Pearson about `0.964 .. 0.968`

Not stable across seeds:

- residual cosine only about `0.086 .. 0.100`
- residual sign agreement only about `0.510 .. 0.518`

Residual-substitution attack changed that again:

- [conker6_residual_substitution_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker6_residual_substitution_2026-03-28.json)
- `baseline_seed42_full`: `0.0720933 bpb`
- `baseline_seed42_strictlower`: `5.7521`
- `toeplitz_lower_plus_upperdiag_seed42`: `0.0721075`
- random or swapped lower residuals all stay around `0.0721`

So the lower residual is not the real source of the win. The decisive part is the learned diagonal + upper-triangle mask geometry, which is exactly what makes the branch non-row-causal and non-normalized.

## What To Avoid

- importing char-level random-third intuitions into `conker`
- ranking branches by `text8` once official golf data is available
- treating `Conker-2` as final before its mixer and budget row is attacked
- packing artifacts before the mechanism row is settled

## Conker-7

`Conker-7` is the first future-aware training / causal eval branch:

- legal causal student at inference
- future-aware exact-history teacher on training chunks only
- tandem `Conker-4b` substrate underneath

Artifacts:

- [conker7.py](https://github.com/asuramaya/conker/blob/main/conker/src/conker7.py)
- [run_conker7_golf_bridge.py](https://github.com/asuramaya/conker/blob/main/conker/scripts/run_conker7_golf_bridge.py)
- [CONKER7.md](https://github.com/asuramaya/conker/blob/main/conker/docs/CONKER7.md)

Seed `42`, `window4 / 10x / 256 / batch16 / lr5e-4`:

Future-only teacher, `exact2 + exact3`, `500` steps:

- fp16 `0.6768 bpb`
- `int6 0.6887`
- `int4 0.9742`

Future-only teacher, `exact2 + exact3`, `1000` steps:

- fp16 `0.6168 bpb`
- `int6 0.6282`
- `int4 0.8132`

Future-only rich teacher (`exact2 + exact3 + special2 + number2 + markup2 + attr2 + delim2`), `500` steps:

- `NaN`

Read:

- narrow future-aware supervision is real and stable
- more steps help
- broad future-aware lexical supervision destabilizes quickly
- this is still behind the legal tandem `Conker-5` frontier, but it is the first clean evidence that future information can be distilled into a causal student without using it at eval

Sequence follow-up, same seed `42` recipe:

- future `exact2 + exact3`, `teacher_weight=0.10`, `1000` steps:
  - fp16 `0.5792`
  - `int6 0.5915`
  - `int4 0.7612`
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
- bidirectional `exact2 + exact3`, `teacher_weight=0.10`, `teacher_start_step=500`, `1000` steps:
  - fp16 `0.5666`
  - `int6 0.5789`
  - `int4 0.7370`

Warm-start from the legal tandem `Conker-5` `seq256 / 1500 / lr5e-4` checkpoint:

- bidirectional `exact2 + exact3`, `teacher_weight=0.10`, warm-start, `500` steps:
  - fp16 `0.5357`
  - `int6 0.5480`
  - `int4 0.6304`
- bidirectional `exact2 + exact3`, `teacher_weight=0.10`, warm-start, `1000` steps:
  - fp16 `0.5183`
  - `int6 0.5301`
  - `int4 0.6140`

Updated read:

- teacher should be narrow and weak
- bidirectional training signal is slightly better than future-only on the stable row
- warm-start is the decisive lever
- `Conker-7` is now below the legal tandem seed-42 baseline and is the live architecture branch to replicate next

Fresh-process full held-out eval on the saved seed-42 warm-start winner:

- fp16 full split (`62,021,632` tokens): `0.5283 bpb`
- `int6` full split: `0.5315 bpb`
- compressed `int6` artifact: `4,153,894` bytes

Warm-start fine-tune sweep, seed `42`:

- `teacher_weight=0.05`: fp16 `0.5126`, `int6 0.5249`, `int4 0.6090`
- `teacher_weight=0.15`: fp16 `0.5222`, `int6 0.5342`, `int4 0.6168`
- `teacher_start=250`: fp16 `0.5141`, `int6 0.5260`, `int4 0.6092`
- `teacher_start=500`: fp16 `0.5122`, `int6 0.5242`, `int4 0.6083`

Warm-start “replication” rows on seeds `43/44` came back numerically identical to the seed-42 bridge row. That means the current fine-tune recipe is effectively deterministic under the sequential training stream and does not count as a true statistical replication yet.

## Conker-4b Strict Recovery

**March 28, 2026**

Root cause:

- `Conker-4b` was using tuple-valued `freeze(keys=...)` calls.
- MLX stores that tuple as one composite `_no_grad` key instead of freezing each member.
- So `causal_mask`, `vocab_axis`, `token_class_ids`, and the token-class masks were all trainable in the old tandem and warm-start branches.

Patched strict retrain, seed `42`, `window4 / 10x / 256 / batch16 / 1000 / lr5e-4`:

- `exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2`, gate-only + dynamic support gates, tandem:
  - bridge fp16 `2.0589 bpb`
  - full held-out fp16 `2.0971`
  - full held-out `int6 2.1055`
  - `int6` artifact `3,730,410` bytes

Strict bridge ablations on the same patched branch:

- no `exact3`: `2.0621`
- exact-only (`exact1 + exact2 + exact3`): `2.0608`
- no dynamic support gates: `2.0606`

Interpretation:

- the pre-fix `Conker-5/7` frontier was dominated by the learned structural side channel
- after the freeze fix, the intended residual exact-expert branch is much weaker
- the next honest branch must be rebuilt on top of the patched strict model, not the old warm-start ancestry

## Conker-8

**March 28, 2026**

First strict rebuild after the `Conker-4b` freeze bug:

- explicit learned lag profile over past lags only
- explicit learned within-support token weights for delimiter / number / special / urlpath / markup / attr / entity
- patched strict `Conker-4b` residual stack underneath

Seed `42`, `window4 / 10x / seq256 / batch16 / 1000 / lr5e-4`, tandem, gate-only + dynamic support gates:

- full explicit structure:
  - bridge fp16 `2.0600 bpb`
  - `int6 2.0994`
  - `int4 2.3985`
- ablate learned support-mask weights:
  - bridge fp16 `2.0598`
- ablate learned lag profile:
  - bridge fp16 `2.0598`

Interpretation:

- the first legal structural rebuild does not recover the old contaminated frontier
- the explicit lag/profile and support-mask weights are effectively inert in this simple form
- `Conker-8` v0 is basically tied with the patched strict `Conker-4b` baseline

Control matrix, same seed-42 recipe:

- high-span full explicit structure (`lag=2.0`, `support=2.0`):
  - `NaN`
- high-span lag-only:
  - `2.0601 bpb`
- high-span mask-only:
  - `NaN`
- high-span full structure with dynamic support gates disabled:
  - `NaN`
- high-span full structure with recency enabled:
  - `2.0867 bpb`

Interpretation:

- the lag surface still does nothing
- the support-mask surface is unstable at higher amplitude
- the gate layer is not the source of that instability
- recency makes the legal branch worse
- next structure work should target routing/control, not stronger weighted masks

## Conker-9

**March 28, 2026**

First explicit causal-controller branch:

- fixed legal lag buckets `(2, 4, 8, 16, 32, 64, 128, full)`
- small controller over causal base features
- controller blends bucketed exact/count features before the strict residual stack

Seed `42`, `window4 / 10x / seq256 / batch16 / 1000 / lr5e-4`:

- bridge fp16 `2.0600 bpb`
- `int6 2.0976`
- `int4 2.3898`

The controller did learn nonzero parameters:

- `lag_gate_feature_weights` mean abs `0.1197`
- `lag_gate_bias` mean abs `0.1057`

Interpretation:

- this is not a no-op controller
- but legal horizon selection alone still does not move the strict `~2.06` floor
- next controller work should target routing/open-set decisions, not just lag selection

## Conker-10

**March 28, 2026**

First memory-first `Conker` rebuild:

- fixed packed training memory from `1,000,000` train tokens
- exact unigram + exact bigram + hashed trigram counts
- normalized posterior backoff
- controller mixes `[base, unigram, bigram, trigram]`

Seed `42`, `window4 / 10x / seq256 / batch16 / 500 / lr5e-4`:

- bridge fp16 `2.2397 bpb`
- `int6 2.2608`
- `int4 2.6028`
- packed memory bytes `12,599,296`

Checkpoint readout:

- mean source weights on a held-out batch:
  - base `0.9914`
  - unigram `0.0056`
  - bigram `0.0019`
  - trigram `0.0011`

Interpretation:

- the controller is learning, but it is almost entirely routing back to the base
- the fixed packed priors are not persuasive enough in this first form
- memory-first remains the right search direction, but this exact branch is weak
- the next honest follow-ups are:
  - cache-only and fixed-interpolation baselines
  - packed prior + online score-first cache
  - controller features that expose cache confidence more directly

Conker-10a falsifications on the same packed tables:

- memory-only baseline:
  - bridge fp16 `6.0892 bpb`
- fixed interpolation baseline with weights
  - base `0.20`
  - unigram `0.05`
  - bigram `0.25`
  - trigram `0.50`
  - bridge fp16 `2.8436 bpb`
  - `int6 2.8753`

Interpretation:

- memory-only is catastrophic
- forcing large memory mass is much worse than the learned mixer
- so the current packed tables themselves are weak, not just underused

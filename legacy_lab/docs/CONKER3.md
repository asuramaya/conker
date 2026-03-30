# Conker-3

`Conker-3` is the dangerous redesign branch.

It starts from new assumptions:

- the frozen linear substrate in `Conker-2` is the real engine
- the nonlinear correction expert may have been buying mostly local residual repair
- if that is true, a fully parallel local residual coder should replace the second dynamical expert

So `Conker-3` keeps:

- frozen linear multiscale substrate
- official golf data
- exact `bpb` bridge

And throws away:

- the frozen nonlinear correction reservoir
- the expert-mixer framing from `Conker-2`

## New Bet

`Conker-3` predicts:

- base score should still come from the frozen linear substrate
- the remaining error should be mostly local and parallel
- a causal local residual coder over recent token embeddings should beat `linear_only`
- if it does not, the correction story in `Conker-2` is more dynamical than local

## First Probe Matrix

These probes are intentionally short and cheap:

- official golf data
- seed `42`
- `600` steps
- scale `3.0`

Variants:

- `linear_only`
- `base` (additive local residual, window `8`)
- `gated`
- `window4`
- `window16`
- `local_only`

## Kill Conditions

`Conker-3` dies early if any of these are true:

- `base` does not beat `linear_only`
- `local_only` is not meaningfully worse than `base`
- window size barely matters and gating does nothing

If those happen, the new assumption was wrong and `Conker-2` remains the live branch.

## Pass Conditions

`Conker-3` earns more time only if:

- `base` beats `linear_only` on the short probe
- at least one local-window ablation shows real structure
- train time stays near the `linear_only` regime rather than the old `Conker-2` hybrid regime

## Results So Far

Initial kill row, seed `42`, `600` steps, scale `3.0`:

- `linear_only`: `2.6253` bpb
- `base`: `2.5082`
- `gated`: `2.5041`
- `window4`: `2.4946`
- `window16`: `2.5155`
- `local_only`: `2.5425`

So the redesign survives its first kill condition.

Follow-up row, seed `42`, `1000` steps:

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

Current read:

- local residual repair is real
- shared embeddings hurt here too
- `gated` and `window4` are effectively tied at `3.0x`
- `window4` is the clean winner at `5.0x`
- gating currently looks like dead complexity

Replication and cheap-compute follow-up:

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

So the next read is:

- `window4 5.0x` is stable across seeds
- `Conker-3` still has cheap compute headroom
- the redesign is no longer just alive; it has a real fast frontier candidate
- widening to `8.0x` is still buying bits on the pilot
- `8.0x` also survives the longer-step replication
- the `10.0x`, `1000`-step pilot is worse than replicated `8.0x`, which implies the branch is short-budget recipe-limited there
- but `10.0x`, `1500` steps immediately recovers and becomes the best single-seed `Conker-3` score so far

Offline geometry audit on `window4`, `5.0x`, `1000` steps, seed `42`:

- baseline audit eval: `2.3704` bpb
- most fragile matrix at `int3`: `local_readout.out.weight`, `+0.2200 bpb`
- next tier at `int3`:
  - `local_embedding.weight`, `+0.0510`
  - `linear_readout.layers.0.weight`, `+0.0410`
  - `linear_readout.out.weight`, `+0.0388`
- several large matrices are nearly harmless at `int4`

So the `3-bit` question is now sharper:

- the cliff is not uniform
- the local residual output head is the main fragile object
- structured low-bit work should target the local path first, not the whole model equally

Mixed low-bit audit on `window4`, `5.0x`, `1500` steps, seed `42`:

- baseline audit eval: `2.3171` bpb
- `uniform_int3`: `2.6864`, `+0.3693`
- `uniform_int4`: `2.3703`, `+0.0532`
- `int3_keep_local_out`: `2.4592`, `+0.1421`
- `int3_keep_local_out_embed`: `2.4006`, `+0.0836`
- `int3_keep_local_path`: `2.3772`, `+0.0601`
- `int3_rest_local_int4`: `2.4074`, `+0.0921`

So the packing read is now:

- naive `int3` is still bad
- `uniform_int4` is already surprisingly competitive on this branch
- selective `int3` only gets close when most of the local path stays high precision
- the next structured quant work should target the local path as a protected island, not just the output head alone
- keeping the local path at `int4` is worse than keeping it high precision

Replicated mixed low-bit audit on `window4`, `8.0x`, `1500` steps, 3-seed means:

- baseline audit eval: `2.2570` bpb
- `uniform_int3`: `2.6818`, `+0.4248`
- `uniform_int4`: `2.3142`, `+0.0572`
- `int3_keep_local_out`: `2.4047`, `+0.1477`
- `int3_keep_local_out_embed`: `2.3356`, `+0.0786`
- `int3_keep_local_path`: `2.3027`, `+0.0457`
- `int3_rest_local_int4`: `2.3457`, `+0.0887`

So the stronger packing read is:

- `uniform_int4` is the clean low-bit baseline
- the best selective `int3` scheme is to keep the whole local path high precision
- keeping the local path at `int4` is not enough

Decay-bank substrate ablation on `window4`, `8.0x`, `1500` steps, seed `42`:

- `logspace`: `2.1540` bpb
- autocorrelation-matched bank: `2.1670`
- narrow-bank control (`half_life_max=32`): `2.1027`

So the substrate read is now:

- offline autocorrelation matching did not beat the current generic prior
- but a shorter-band reservoir beat both
- this branch currently wants a tighter linear memory scaffold, not a broader one

Replicated narrow-bank row on `window4`, `8.0x`, `1500` steps, 3-seed means:

- narrow bank (`half_life_max=32`): `2.1064` bpb
- `uniform_int6`: `2.1401` bpb on the two landed quantized bridge seeds
- mean train time: `53.5s`

Against the original log-spaced bank:

- log-spaced `8.0x`, `1500`: `2.1550` bpb
- narrow `8.0x`, `1500`: `2.1064` bpb

So the substrate result is now replicated:

- the shorter reservoir is not a seed-42 fluke
- on this branch, long-tail linear memory was wasted or harmful
- the reservoir is acting more like a short/medium-horizon convolution scaffold than a broad all-timescale memory system

Narrow-bank `window4`, `10.0x`, `1500` steps, seed `42`:

- `2.0871` bpb
- `uniform_int6`: `2.1211` bpb
- `78.7s`

That is the best single-seed `Conker-3` score so far.

Narrow-bank mixed low-bit audit on `window4`, `8.0x`, `1500` steps, seed `42`:

- baseline audit eval: `2.2185` bpb
- `uniform_int3`: `2.8153`, `+0.5968`
- `uniform_int4`: `2.3438`, `+0.1253`
- `int3_keep_local_out`: `2.4925`, `+0.2740`
- `int3_keep_local_out_embed`: `2.4170`, `+0.1985`
- `int3_keep_local_path`: `2.3876`, `+0.1691`
- `int3_rest_local_int4`: `2.4187`, `+0.2002`

So the narrow-bank tradeoff is:

- better score
- worse low-bit robustness, at least on the first seed
- which means the next half-life sweep should be evaluated on both bridge score and packing stability

Half-life cap sweep on `window4`, `8.0x`, `1500` steps, seed `42`:

- `half_life_max=16`: `2.1014` bpb, `uniform_int6 -> 2.1355`, `51s`
- `half_life_max=32`: `2.1075` bpb, `uniform_int6 -> 2.1404`, `51s`
- `half_life_max=64`: `2.1141` bpb, `uniform_int6 -> 2.1487`, `53s`
- `half_life_max=128`: `2.1382` bpb, `uniform_int6 -> 2.1699`, `79s`
- previous broad log-spaced `512`: `2.1540` bpb

So the substrate curve is now explicit:

- shorter is better across the whole tested range
- `16` is the best cap so far
- widening the half-life tail steadily hurts both score and speed
- the current reservoir is not acting like “more timescales is better”
- it is acting like a short-horizon exponential convolution bank whose useful support ends early

Low half-life follow-up on `window4`, `8.0x`, `1500`, seed `42`:

- `half_life_max=4`: `2.1031` bpb, `uniform_int6 -> 2.1359`, `51s`
- `half_life_max=8`: `2.0993` bpb, `uniform_int6 -> 2.1302`, `55s`
- `half_life_max=12`: `2.1011` bpb, `uniform_int6 -> 2.1332`, `69s`
- `half_life_max=16`: `2.1014` bpb, `uniform_int6 -> 2.1355`, `51s`

So the full tested curve now says:

- the optimum is not “as short as possible”
- the best tested cap is `8`
- pushing below `8` starts to lose complementarity with the local path
- pushing above `8` reintroduces harmful long-tail memory

Replicated `8` vs `16` on `window4`, `8.0x`, `1500` steps, 3-seed means:

- `half_life_max=8`: `2.1000` bpb, `uniform_int6 -> 2.1323`, `52.7s`
- `half_life_max=16`: `2.1013` bpb, `uniform_int6 -> 2.1348`, `67.5s`

So the replicated answer is:

- `8` really is the current substrate winner
- but the margin over `16` is small, about `0.0013` bpb
- the speed difference is larger and also favors `8`

Mixed-quant audit on `window4`, `8.0x`, `1500`, `half_life_max=8`, seed `42`:

- baseline audit eval: `2.2155` bpb
- `uniform_int3`: `2.9449`, `+0.7293`
- `uniform_int4`: `2.3378`, `+0.1222`
- `int3_keep_local_out`: `2.6109`, `+0.3954`
- `int3_keep_local_out_embed`: `2.5327`, `+0.3172`
- `int3_keep_local_path`: `2.4992`, `+0.2837`
- `int3_rest_local_int4`: `2.5173`, `+0.3018`

So the `8`-bank tradeoff is harsher than expected:

- bridge score improves
- but quantization robustness gets worse, especially at `int3`
- `uniform_int4` remains the only clean low-bit baseline here

`window4`, `10.0x`, `1500`, `half_life_max=8`, seed `42`:

- `2.0845` bpb
- `uniform_int6 -> 2.1181`
- `111.1s`

Replicated `window4`, `10.0x`, `1500`, `half_life_max=8`, 3-seed means:

- `2.0849` bpb
- `uniform_int6 -> 2.1180`
- `84.7s`

That is now the live `Conker-3` frontier.

Direct mixed-quant comparison, seed `42`:

- `half_life_max=8`
  - baseline audit eval: `2.2155`
  - `uniform_int4`: `2.3378`, `+0.1222`
  - `uniform_int3`: `2.9449`, `+0.7293`
  - `int3_keep_local_path`: `2.4992`, `+0.2837`
- `half_life_max=16`
  - baseline audit eval: `2.2125`
  - `uniform_int4`: `2.3300`, `+0.1175`
  - `uniform_int3`: `2.8284`, `+0.6159`
  - `int3_keep_local_path`: `2.4115`, `+0.1991`

So the sharper conclusion is:

- `8` is the better substrate for score
- `16` is the better substrate for packing robustness
- the current frontier is no longer a single scalar winner; it is a score/packing tradeoff

Packed showdown on `window4`, `10.0x`, `1500`, 3-seed means:

- `half_life_max=8`
  - fp16: `2.0832` bpb
  - `uniform_int6`: `2.1040`
  - `uniform_int4`: `2.2393`
  - `87.5s`
- `half_life_max=16`
  - fp16: `2.0837` bpb
  - `uniform_int6`: `2.1019`
  - `uniform_int4`: `2.2283`
  - `103.8s`

So under the packed criterion:

- `8` wins the raw fp16 bridge by a hair
- `16` wins both `int6` and `int4`
- if packability is treated as part of the score, `16` is the better compressor

Packed-first scaling wave on `half_life_max=16`, seed `42`:

- `12x`
  - fp16: `2.0709`
  - `uniform_int6`: `2.0920`
  - `uniform_int4`: `2.2140`
  - payloads: `8.320 MB` at `int6`, `5.555 MB` at `int4`
- `14x`
  - fp16: `2.0616`
  - `uniform_int6`: `2.0793`
  - `uniform_int4`: `2.1966`
  - payloads: `10.771 MB` at `int6`, `7.189 MB` at `int4`
- `16x`
  - fp16: `2.0507`
  - `uniform_int6`: `2.0726`
  - `uniform_int4`: `2.1715`
  - payloads: `13.527 MB` at `int6`, `9.027 MB` at `int4`

So the packed scale law is still improving through `16x`, and `16x int6` is still under the nominal `16 MB` line.

Precision-allocation audit on `10x`, `1500`, `half_life_max=16`, seed `42`:

- baseline audit eval: `2.1948`
- `uniform_int4`: `2.3164`, `+0.1215`
- `int4_keep_local_out`: `2.2969`, `+0.1020`
- `int4_keep_local_out_embed`: `2.2876`, `+0.0928`
- `int4_keep_local_path`: `2.2851`, `+0.0903`

So the first byte-allocation lesson is:

- small fp16 islands on the local path do help
- protected `int4` is better than pure `int4`
- but the gain is modest relative to simply scaling the packed model up

First online TTT routing probe on `window4`, `10.0x`, `1500`, `half_life_max=16`, seed `42`:

- setup:
  - grouped reservoir-mode gate
  - `16` gate groups
  - online chunk adaptation on contiguous test chunks
  - `chunk_len=64`, `32` chunks
  - reservoir frozen, readout frozen, only the gate adapts
- fp16:
  - baseline: `2.2149` bpb
  - `1` gate step at `lr=0.10`: `2.2151`
  - `3` gate steps at `lr=0.03`: `2.2136`
- `int6`:
  - baseline: `2.2203`
  - `1` gate step at `lr=0.10`: `2.2190`
  - `3` gate steps at `lr=0.03`: `2.2175`

So the first TTT answer is:

- routing adaptation did not help much in fp16
- it helped a little more on the packed `int6` model
- the idea is alive, but only weakly so far
- if this line continues, it should be treated as a packed-model repair mechanism, not a raw-score miracle

Oscillatory-mode hybrid probe on `window4`, `10.0x`, `1500`, `half_life_max=16`:

- `25%` oscillatory modes, seed `42`
  - fp16: `2.0751`
  - `uniform_int6`: `2.0955`
  - `uniform_int4`: `2.2348`

- `50%` oscillatory modes, 3-seed means
  - fp16: `2.0613`
  - `uniform_int6`: `2.0833`
  - `uniform_int4`: `2.2343`
  - `84.8s`

Against the non-oscillatory `half_life=16` baseline at the same scale:

- baseline fp16: `2.0837`
- baseline `int6`: `2.1019`
- baseline `int4`: `2.2283`

So the oscillatory read is:

- damped oscillatory modes are clearly alive
- they improve fp16 and packed `int6` materially
- they do not help `int4`; in fact they are slightly worse there
- the likely interpretation is that phase-sensitive fixed modes buy prediction, but are a bit harder to quantize at very low precision

High-oscillation follow-up on the same branch, seed `42`:

- `62.5%` oscillatory modes
  - fp16: `2.0568`
  - `uniform_int6`: `2.0801`
  - `uniform_int4`: `2.2416`
  - `69.6s`
- `75%` oscillatory modes
  - fp16: `2.0526`
  - `uniform_int6`: `2.0740`
  - `uniform_int4`: `2.2361`
  - `73.6s`
- `87.5%` oscillatory modes
  - fp16: `2.0517`
  - `uniform_int6`: `2.0738`
  - `uniform_int4`: `2.2449`
  - `105.8s`

So the follow-up read is:

- increasing oscillatory fraction above `50%` keeps helping fp16 and `int6`
- the packed `int6` frontier is now effectively between `75%` and `87.5%`
- `int4` gets worse again by `87.5%`
- train time rises sharply at very high oscillatory fraction, so the best packed branch is not obviously the best efficiency branch

Replication on the two serious high fractions, 3-seed means:

- `75%` oscillatory modes
  - fp16: `2.0531`
  - `uniform_int6`: `2.0745`
  - `uniform_int4`: `2.2404`
  - `73.2s`
- `87.5%` oscillatory modes
  - fp16: `2.0512`
  - `uniform_int6`: `2.0729`
  - `uniform_int4`: `2.2410`
  - `114.8s`

So the current oscillatory fraction read is:

- `87.5%` is the best raw fp16 and `int6` branch so far
- the gain over `75%` is tiny
- `75%` is slightly better on `int4`
- the main cost of pushing oscillation higher is time, not just low-bit robustness

Period-range sweep on `87.5%` oscillatory modes, same branch, seed `42`:

- periods `4..32`
  - fp16: `2.0518`
  - `uniform_int6`: `2.0772`
  - `uniform_int4`: `2.2432`
  - `81.6s`
- periods `4..64`
  - fp16: `2.0515`
  - `uniform_int6`: `2.0725`
  - `uniform_int4`: `2.2411`
  - `92.8s`
- periods `8..64`
  - fp16: `2.0602`
  - `uniform_int6`: `2.0788`
  - `uniform_int4`: `2.2281`
  - `108.8s`
- periods `8..128`
  - fp16: `2.0596`
  - `uniform_int6`: `2.0819`
  - `uniform_int4`: `2.2238`
  - `121.1s`

So the period read is:

- the default `4..64` range is still the best overall oscillatory setting so far
- narrowing to `4..32` preserves raw score but weakens packed robustness
- shifting the band upward to `8..64` or `8..128` gives up too much raw score
- lower-frequency oscillation may help `int4` slightly, but not enough to beat the best overall packed branch

Oscillatory 8-hour packed matrix on `window4`, `1500`, `half_life_max=16`, periods `4..64`:

- `12x`, `75%`, seed `42`
  - fp16: `2.0436`
  - `uniform_int6`: `2.0658`
  - `uniform_int4`: `2.2297`
  - `90.6s`
- `12x`, `87.5%`, seed `42`
  - fp16: `2.0410`
  - `uniform_int6`: `2.0638`
  - `uniform_int4`: `2.2405`
  - `111.2s`

- `16x`, `75%`, 3-seed means
  - fp16: `2.0342`
  - `uniform_int6`: `2.0565`
  - `uniform_int4`: `2.2131`
  - `220.4s`
  - `int6` payload: `13.527 MB`
- `16x`, `87.5%`, 3-seed means
  - fp16: `2.0309`
  - `uniform_int6`: `2.0542`
  - `uniform_int4`: `2.2231`
  - `208.8s`
  - `int6` payload: `13.527 MB`

- `18x`, `75%`, 3-seed means
  - fp16: `2.0282`
  - `uniform_int6`: `2.0509`
  - `uniform_int4`: `2.2127`
  - `253.1s`
  - `int6` payload: `16.588 MB`
- `18x`, `87.5%`, 3-seed means
  - fp16: `2.0271`
  - `uniform_int6`: `2.0508`
  - `uniform_int4`: `2.2205`
  - `256.5s`
  - `int6` payload: `16.588 MB`

Local-path byte-allocation probes on `18x`, `87.5%`, seed `42`:

- base: fp16 `2.0283`, `int6 2.0510`, `int4 2.2314`
- `local_hidden_mult=0.75`: fp16 `2.0315`, `int6 2.0572`, `int4 2.2310`
- `local_hidden_mult=0.50`: fp16 `2.0383`, `int6 2.0603`, `int4 2.2442`
- `local_scale_override=0.20`: fp16 `2.0291`, `int6 2.0523`, `int4 2.2342`

So the matrix answer is:

- packed scaling keeps helping through `16x`, and even `18x` is still better on raw score
- the best replicated under-cap packed branch is now `16x`, `87.5%`, `4..64`
- `87.5%` wins on fp16 and `int6` at `16x`, but `75%` remains clearly better on `int4`
- shrinking the local path does not buy enough packed score to justify the loss

Packed next-wave answers:

- cap-fit pilots on `87.5%`, seed `42`
  - `16.5x`: fp16 `2.0313`, `int6 2.0536`, `int4 2.2292`, `14.264 MB`
  - `17.0x`: fp16 `2.0295`, `int6 2.0514`, `int4 2.2258`, `15.020 MB`
  - `17.5x`: fp16 `2.0297`, `int6 2.0515`, `int4 2.2311`, `15.794 MB`
- longer train on `16x`, seed `42`
  - `1800` steps: fp16 `2.0281`, `int6 2.0473`, `int4 2.2234`
  - `2200` steps: fp16 `1.9899`, `int6 2.0146`, `int4 2.1788`
- naive train-through-pack (`int6` every step) failed hard
  - `16x`, `1800`: fp16 `3.5570`, `int6 3.6240`, `int4 3.8628`
- static two-bank gating is real
  - `16x`, `1500`: fp16 `2.0237`, `int6 2.0452`, `int4 2.1886`
  - `16x`, `1800`: fp16 `2.0214`, `int6 2.0361`, `int4 2.1877`

So the current read is:

- longer training is the biggest lever so far
- static bank gating is the first new mechanism in a while that clearly helps packed score
- cap-fit between `16x` and `17.5x` helps only modestly compared with more steps
- naive pack-training should be abandoned in its current form

Combo pilots on the winning branch family, seed `42`:

- `16x`, `2200`, static bank gate
  - fp16: `1.9856`
  - `uniform_int6`: `2.0066`
  - `uniform_int4`: `2.1479`
  - `210.6s`
- `17x`, `1800`, static bank gate
  - fp16: `2.0186`
  - `uniform_int6`: `2.0347`
  - `uniform_int4`: `2.1850`
  - `193.3s`
- `17x`, `2200`, static bank gate
  - fp16: `1.9814`
  - `uniform_int6`: `2.0019`
  - `uniform_int4`: `2.1463`
  - `230.5s`

So the combo answer is:

- longer training and static bank gating do compound
- the current best seed-42 branch is now `17x / 2200 / staticgate`
- `16x / 2200 / staticgate` is almost as good while staying farther under the nominal `int6` cap

Static-gate frontier replication is now closed:

- `16x`, `2200`, static gate, 3-seed means
  - fp16: `1.9845`
  - `uniform_int6`: `2.0067`
  - `uniform_int4`: `2.1506`
  - `270.2s`
  - `int6` payload: `13.527 MB`
- `17x`, `2200`, static gate, 3-seed means
  - fp16: `1.9828`
  - `uniform_int6`: `2.0042`
  - `uniform_int4`: `2.1523`
  - `294.9s`
  - `int6` payload: `15.020 MB`
- `18x`, `2200`, static gate, seed `42`
  - fp16: `1.9790`
  - `uniform_int6`: `2.0027`
  - `uniform_int4`: `2.1415`
  - `354.3s`
  - `int6` payload: `16.588 MB`

So the live answer is:

- the best replicated under-cap branch is now `17x / 2200 / staticgate`
- `16x / 2200 / staticgate` remains the safer cheaper branch, but it is no longer the score frontier
- `18x / 2200 / staticgate` is the better raw pilot, but it is already over the nominal `int6` line

## Muons And Quants

Not in the first kill batch.

For `Conker-3`, these are second-wave ablations:

- `Muon`: only after a local-residual variant has shown real signal; otherwise optimizer work is wasted on a dead mechanism
- quantization: only after the best short-probe variant is identified; the question then becomes whether the new branch survives low-bit packing as well as `Conker-2`

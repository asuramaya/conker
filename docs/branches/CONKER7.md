# Conker-7

Historical status:

- the best saved `Conker-7` row (`0.5283 fp16 / 0.5315 int6`) is preserved here as a research artifact
- it is not the current clean frontier
- it inherited the same accidentally trainable structural buffers that invalidated the old tandem `Conker-5` line

`Conker-7` is the first explicit **future-aware training / causal eval** branch:

- legal causal student at inference time
- future-aware exact-history teacher on training chunks only
- same `Conker-4b` tandem substrate underneath
- no future access at eval

Code:

- student wrapper:
  [conker7.py](/Users/asuramaya/Code/carving_machine_v3/conker/src/conker7.py)
- bridge runner:
  [run_conker7_golf_bridge.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/run_conker7_golf_bridge.py)

## Purpose

The dead `Conker-6` mask branch showed that future information is extremely valuable, but illegal when used at eval. `Conker-7` asks a cleaner question:

- can a causal student absorb some of that future-aware signal during training?

The first teacher is narrow on purpose:

- future-only exact-2 / exact-3 matches inside the training chunk
- optional future lexical slices as a later probe

## First Pilots

`window4 / 10x / 256 / batch16 / lr5e-4`, seed `42`

Future-only teacher, `exact2 + exact3`, `500` steps:

- fp16 `0.6768 bpb`
- `int6 0.6887`
- `int4 0.9742`
- train time `52.6s`
- artifact:
  [conker7_future_exact23_seq256_steps500_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_future_exact23_seq256_steps500_seed42_2026-03-28.json)

Future-only teacher, `exact2 + exact3`, `1000` steps:

- fp16 `0.6168 bpb`
- `int6 0.6282`
- `int4 0.8132`
- train time `103.0s`
- artifact:
  [conker7_future_exact23_seq256_steps1000_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_future_exact23_seq256_steps1000_seed42_2026-03-28.json)

Future-only **rich** teacher, `exact2 + exact3 + special2 + number2 + markup2 + attr2 + delim2`, `500` steps:

- unstable `NaN`
- artifact:
  [conker7_future_rich_seq256_steps500_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_future_rich_seq256_steps500_seed42_2026-03-28.json)

## Read

This branch is alive, but not yet frontier:

- narrow future teacher (`exact2 + exact3`) is stable
- longer training improves it materially
- broad future lexical teacher is too aggressive and destabilizes training

Compared with legal tandem `Conker-5`:

- `Conker-7` is still behind the replicated `~0.57` full-holdout tandem line
- but it is the first direct evidence that future-aware supervision can move a legal causal student in the right direction

So the current interpretation is:

- future signal is useful
- but it must be narrow and high-precision
- broad future supervision behaves more like noise than guidance

## Next Best Experiments

1. Lower teacher weight:
   - `0.10`
   - `0.25`
   - `0.50`

2. Sharpen the teacher:
   - `exact3` only
   - `future` vs `bidirectional`

3. Add curriculum / late distillation:
   - warm up the causal student first
   - then turn on the future teacher

4. Warm-start from a legal tandem checkpoint instead of training the student from scratch.

The current best `Conker-7` lesson is not a new submission row. It is the training principle:

- **distill future-aware exactness into a causal student, but keep the teacher narrow.**

## Sequence Follow-Up

Narrow-teacher sweep on the same seed-`42` recipe:

Future-only `exact2 + exact3`, `1000` steps:

- `teacher_weight=0.10`: fp16 `0.5792`, `int6 0.5915`, `int4 0.7612`
- `teacher_weight=0.25`: fp16 `0.5945`, `int6 0.6065`, `int4 0.7763`
- `teacher_weight=0.50`: fp16 `0.6168`, `int6 0.6282`, `int4 0.8132`

Sharper / wider teacher variants:

- future-only `exact3`-only, `teacher_weight=0.10`: fp16 `0.5819`, `int6 0.5953`, `int4 0.7644`
- bidirectional `exact2 + exact3`, `teacher_weight=0.10`: fp16 `0.5776`, `int6 0.5905`, `int4 0.7607`
- bidirectional `exact2 + exact3`, `teacher_weight=0.10`, `teacher_start_step=500`: fp16 `0.5666`, `int6 0.5789`, `int4 0.7370`

Warm-start from legal tandem `Conker-5` (`seq256 / 1500 / lr5e-4`) changed the branch:

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

Fresh-process full held-out eval of the saved warm-start winner:

- fp16 full split (`62,021,632` tokens): `0.5283 bpb`
  - artifact:
    [conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42_fullval_test_none_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42_fullval_test_none_2026-03-28.json)
- `int6` full split: `0.5315 bpb`
  - compressed artifact: `4,153,894` bytes
  - artifact:
    [conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42_fullval_test_int6_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker7_bidirectional_exact23_tw01_warmstart_tandem1500_seq256_steps1000_seed42_fullval_test_int6_2026-03-28.json)

Warm-start sweep around the winner:

- `teacher_weight=0.05`: fp16 `0.5126`, `int6 0.5249`, `int4 0.6090`
- `teacher_weight=0.15`: fp16 `0.5222`, `int6 0.5342`, `int4 0.6168`
- `teacher_start=250`: fp16 `0.5141`, `int6 0.5260`, `int4 0.6092`
- `teacher_start=500`: fp16 `0.5122`, `int6 0.5242`, `int4 0.6083`

Warm-start “replications” on seeds `43/44` landed numerically identical to the seed-`42` bridge row:

- fp16 `0.51832082...`
- `int6 0.530149...`
- `int4 0.614024...`

That is not real statistical replication. It means the current warm-start fine-tune path is effectively deterministic under this sequential training stream, so the next true replication work needs seeded data-order variation or a shuffled-start protocol.

## Updated Read

The stable lessons are now:

- the teacher must stay narrow: `exact2 + exact3`
- the teacher must stay weak, and warm-start prefers even less than the from-scratch branch
- bidirectional training signal is slightly better than future-only on the narrow row
- delayed teacher helps from scratch
- warm-starting from the legal tandem student is the real lever

So `Conker-7` is no longer just a research curiosity. The seed-`42` warm-start branch is now:

- `0.5283 bpb` on honest full held-out fp16 eval
- `0.5315 bpb` on honest full held-out `int6`

and the local search suggests the best next legal row is probably:

- bidirectional `exact2 + exact3`
- warm-start from tandem
- `teacher_weight=0.05`
- delayed teacher around `250 .. 500`

# Conker-8

`Conker-8` is the first strict rebuild after the `Conker-4b` structural-buffer freeze bug.

Goal:

- keep the patched strict `Conker-4b` legality surface
- turn the accidental learned structure into explicit trainables
- do it under hard constraints:
  - strict lower-triangular causal geometry only
  - fixed token supports with only within-support weighting

Architecture:

- inherits the patched `Conker-4b` residual expert stack
- adds an explicit learned lag profile over past lags only
- adds explicit learned token-support weights for:
  - delimiter
  - number
  - special
  - urlpath
  - markup
  - attr
  - entity

Current status:

- implementation:
  - [conker8.py](/Users/asuramaya/Code/carving_machine_v3/conker/src/conker8.py)
  - [run_conker8_golf_bridge.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/run_conker8_golf_bridge.py)

First pilot, seed `42`, `window4 / 10x / seq256 / batch16 / steps1000 / lr5e-4`:

- explicit lag profile + explicit support-mask weights
- tandem (`freeze_base=False`)
- gate-only + dynamic support gates
- `exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2`
- bridge fp16: `2.0600 bpb`
- bridge `int6`: `2.0994`
- bridge `int4`: `2.3985`
- artifact:
  [conker8_explicit_structure_seq256_steps1000_lr5e4_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker8_explicit_structure_seq256_steps1000_lr5e4_seed42_2026-03-28.json)

Ablations on the same recipe:

- disable learned support-mask weights, keep learned lag profile:
  - `2.0598 bpb`
  - artifact:
    [conker8_ablate_mask_learning_seq256_steps1000_lr5e4_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker8_ablate_mask_learning_seq256_steps1000_lr5e4_seed42_2026-03-28.json)
- disable learned lag profile, keep learned support-mask weights:
  - `2.0598 bpb`
  - artifact:
    [conker8_ablate_lag_learning_seq256_steps1000_lr5e4_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker8_ablate_lag_learning_seq256_steps1000_lr5e4_seed42_2026-03-28.json)

Read:

- the minimal legal structural rebuild does **not** recover the old contaminated `Conker-5/7` gains
- in this first form, neither explicit lag learning nor explicit support-mask learning matters much
- the result is effectively tied with the patched strict `Conker-4b` baseline at about `2.06 bpb`

Control-matrix follow-up, seed `42`, same recipe:

- high-span full explicit structure (`lag_profile_span=2.0`, `support_mask_span=2.0`):
  - diverged to `NaN` around step `500`
  - artifact:
    [conker8_hispan_full_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker8_hispan_full_2026-03-28.json)
- high-span lag-only:
  - bridge fp16 `2.0601 bpb`
  - artifact:
    [conker8_hispan_lag_only_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker8_hispan_lag_only_2026-03-28.json)
- high-span mask-only:
  - diverged to `NaN` around step `500`
  - artifact:
    [conker8_hispan_mask_only_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker8_hispan_mask_only_2026-03-28.json)
- high-span full structure with dynamic support gates disabled:
  - diverged to `NaN` around step `500`
  - artifact:
    [conker8_hispan_no_dynamic_gates_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker8_hispan_no_dynamic_gates_2026-03-28.json)
- high-span full structure with recency enabled:
  - bridge fp16 `2.0867 bpb`
  - artifact:
    [conker8_hispan_with_recency_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker8_hispan_with_recency_2026-03-28.json)

Updated read:

- the lag-profile surface stays inert even when pushed harder
- the explicit support-mask surface is the unstable part
- dynamic support gates are not causing that instability; removing them does not save the branch
- recency does not rescue the legal structure branch and makes it worse here
- the next branch should move structure into control/routing, not stronger mask weighting

Next honest directions:

1. train explicit **structure controllers**, not just weighted masks
2. let legal structure modulate support gates / opener decisions instead of only count weights
3. move the strict replay/oracle into Rust (`conker-rs`) so the next branch can be audited from day one

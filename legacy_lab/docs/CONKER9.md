# Conker-9

`Conker-9` is the first explicit causal-controller branch after `Conker-8`.

Goal:

- stop learning weighted masks
- learn a legal controller instead
- let the model choose which **past horizon** to trust

Architecture:

- patched strict `Conker-4b` residual stack
- fixed legal lag buckets:
  - `(2, 4, 8, 16, 32, 64, 128, full)`
- a small controller over causal base features emits a softmax over lag buckets
- bucket-specific exact/count features are blended before the normal residual merge

Implementation:

- model:
  [conker9.py](/Users/asuramaya/Code/carving_machine_v3/conker/src/conker9.py)
- runner:
  [run_conker9_golf_bridge.py](/Users/asuramaya/Code/carving_machine_v3/conker/scripts/run_conker9_golf_bridge.py)

First pilot, seed `42`, `window4 / 10x / seq256 / batch16 / 1000 / lr5e-4`:

- gate-only + dynamic support gates
- `exact1 + exact2 + exact3 + delim2 + special2 + number2 + markup2 + attr2`
- lag controller over `(2, 4, 8, 16, 32, 64, 128, 0)`
- bridge fp16 `2.0600 bpb`
- `int6 2.0976`
- `int4 2.3898`
- artifact:
  [conker9_lag_controller_seq256_steps1000_lr5e4_seed42_2026-03-28.json](/Users/asuramaya/Code/carving_machine_v3/conker/out/conker9_lag_controller_seq256_steps1000_lr5e4_seed42_2026-03-28.json)

Controller state did move:

- `lag_gate_feature_weights` mean abs `0.1197`, max abs `0.2376`
- `lag_gate_bias` mean abs `0.1057`, max abs `0.1635`

Read:

- the controller learned a real lag policy
- but horizon selection alone does not beat the patched strict floor
- `Conker-9` is effectively tied with strict `Conker-4b` and `Conker-8` at about `2.06 bpb`

Meaning:

- the missing signal is not just “which past horizon to trust”
- the next controller branch should target:
  - opener/candidate decisions
  - expert activation/routing
  - abstain/fallback control

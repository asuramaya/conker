# BLINX-5 Variants

**March 29, 2026**

`BLINX-5` is the payload-aware branch family.
`BLINX-5a/b/c` are ablation variants that isolate the location-payload choice.

## Variants

- `BLINX-5a`
  - file: [blinx5a.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/src/blinx5a.py)
  - mask policy: dense packed bitset only
- `BLINX-5b`
  - file: [blinx5b.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/src/blinx5b.py)
  - mask policy: sparse delta-coded position list only
- `BLINX-5c`
  - file: [blinx5c.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/src/blinx5c.py)
  - mask policy: adaptive choice between bitset and sparse position payload

## Why These Three

This is the smallest clean ablation that answers the payload question:

- is dense always better?
- is sparse always better?
- or does adaptive choice actually matter?

`BLINX-5c` is the explicit ablation form of the current adaptive `BLINX-5` idea.

## Runners

- probe one variant:
  - [run_blinx5_variant_probe.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/scripts/run_blinx5_variant_probe.py)
- compare the family across local files:
  - [run_blinx5_variant_ablation.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/scripts/run_blinx5_variant_ablation.py)

# BLINX-5

**March 29, 2026**

`BLINX-5` is the first payload-economics mutation on top of `BLINX-4`.

`BLINX-4` already showed the right structural lesson:

- the codec should control the real round knobs directly
- removal fraction alone is not the metric
- packed size after side data is the real score

But the local probes also showed the main bottleneck:

- there were removable candidates
- the controller correctly refused to take them
- the dense round mask was often too expensive relative to the bytes removed

So `BLINX-5` keeps the `BLINX-4` phase controller and changes the payload representation.

## Mutation

Code:

- [blinx5.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/src/blinx5.py)
- [run_blinx5_lossless_probe.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/scripts/run_blinx5_lossless_probe.py)

Each accepted round now evaluates two location encodings:

- dense packed bitset
- sparse delta-coded removed-position list

The round keeps whichever encoding gives the smaller packed candidate once the full payload is serialized and `zlib`-scored.

This is a narrower claim than a new codec family.
It is a direct answer to the `BLINX-4` finding that the payload, not the policy, was dominating the loss.

## Expected Read

What should improve if this mutation matters:

- sparse rounds should become admissible more often
- packed size should drop on the same files without changing the controller
- mask-format usage should reveal whether the branch really prefers sparse position lists

What would count as failure:

- no rounds become newly profitable
- bitset wins almost everywhere anyway
- dictionary cost still dominates enough that mask savings do not matter

## Status

Current status: branch scaffold live, local probe pending comparison against `BLINX-4`.

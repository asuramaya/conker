# BLINX-6

**March 29, 2026**

`BLINX-6` is the first cross-round dictionary-factoring mutation on top of `BLINX-5`.

`BLINX-5` already showed:

- sparse position payloads help
- but they do not rescue the branch on their own
- dictionary and rulebook cost are still the larger side-data burden

So `BLINX-6` attacks the next bottleneck directly.

## Mutation

Code:

- [blinx6.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/src/blinx6.py)
- [run_blinx6_lossless_probe.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/scripts/run_blinx6_lossless_probe.py)

The branch keeps the `BLINX-5` phase controller and adaptive mask storage, but changes dictionary packing.

Each candidate payload now compares two dictionary layouts:

- `direct`
  - emit each round dictionary inline, as before
- `shared`
  - emit one global table of unique `(context key, value)` entries
  - let each round reference that table by delta-coded entry-id lists

The serialized artifact keeps whichever mode yields the smaller packed payload.

## Expected Read

What should improve if this mutation matters:

- repeated dictionary entries across rounds should stop paying full key cost each time
- the shared mode should start winning on multi-round candidates
- local probes with several accepted rounds should show `dictionary_mode = shared`

What would count as failure:

- the selected mode stays `direct` almost everywhere
- accepted-round count does not change
- current repo surfaces still have no profitable rounds because the branch is not yet reaching the multi-round regime where factoring helps

## Status

Current status: branch scaffold live, local probe pending.

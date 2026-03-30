# Status

As of March 29, 2026.

## Role

`BLINX` is the noncausal codec, oracle, and lossless-compression lab.

## Current Public State

Active branch line:

- `BLINX-4`: control-first codec mutation
- `BLINX-5`: payload-aware location encoding
- `BLINX-6`: direct-vs-shared dictionary factoring
- `BLINX-7`: typed dictionary pruning

Ordinary repo-text result:

- active branches still sit at `0` accepted profitable rounds on the usual local repo surfaces

Clean synthetic positive result:

- `BLINX-6` and `BLINX-7` both reach `3` accepted rounds on the disjoint recursive synthetic
- packed size `5043` bytes versus source `zlib = 6823`
- winning payload uses `shared` dictionary packing

## Core Finding

- side-data economics still dominate on ordinary text
- sparse mask storage helps, but it was not the unlock
- shared dictionary factoring only matters once the data produces genuinely reusable round dictionaries
- typed pruning alone does not beat that cleaner layered synthetic case

## Next Step

- change candidate generation rather than adding another small packing tweak
- preserve cleaner reusable context layers during round construction
- keep using BLINX primarily as a structure probe until the codec line wins on less synthetic text

## Read Next

- [Salvage Matrix](./SALVAGE_MATRIX.md)
- [Parallel Exploration](./docs/parallel_exploration.md)
- [Status in context](./docs/presentation.md)
- [Current Frontier](./docs/current_frontier.md)
- [Negative Results](./docs/negative_results.md)
- [Rescue](./docs/rescue.md)

## Companion Repos

- [`asuramaya/conker`](https://github.com/asuramaya/conker)
- [`asuramaya/giddy-up`](https://github.com/asuramaya/giddy-up)
- [`asuramaya/conker-detect`](https://github.com/asuramaya/conker-detect)
- [`asuramaya/conker-ledger`](https://github.com/asuramaya/conker-ledger)

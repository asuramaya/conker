# Status

As of March 29, 2026.

## Role

`Conker` is now the canonical private repo for the whole active tree:

- causal runtime and submission work
- absorbed `BLINX` noncausal codec/oracle work in [`./blinx`](./blinx/README.md)
- absorbed `Giddy-Up` bridge work in [`./giddy-up`](./giddy-up/README.md)

## Current Public State

Clean anchor:

- patched strict `Conker-4b`
- full held-out fp16 `2.0971 bpb`
- full held-out int6 `2.1055 bpb`
- int6 artifact `3,730,410` bytes

Best documented restart result:

- `Conker-10` memory-first pilot
- bridge fp16 `2.2397 bpb`
- int6 `2.2608`
- packed memory bytes `12,599,296`

Active exploratory branches:

- `Conker-10`: memory-first restart
- `Conker-11`: recursive causal router
- `Conker-12`: higher-order program tensor
- `Conker-13`: direct five-axis controller over the `Conker-3` substrate

## Core Finding

- the old `~0.55 / 0.53` frontier is historical but invalidated
- packed memory alone has been weak so far
- legal structural control is more promising than memory-only rebuilds
- the newer controller branches still need harder specialization before they become a stable public frontier

## Next Step

- push the `Conker-13` line into a real measured pilot
- force specialization instead of soft controller collapse
- promote a new branch into the public frontier only after it has tracked artifacts and a stable writeup

## Read Next

- [Salvage Matrix](./SALVAGE_MATRIX.md)
- [Parallel Exploration](./docs/parallel_exploration.md)
- [Status in context](./docs/presentation.md)
- [Current Frontier](./docs/current_frontier.md)
- [Validity](./docs/validity.md)
- [Negative Results](./docs/negative_results.md)

## Related Trees

- [Absorbed BLINX](./blinx/README.md)
- [Absorbed Giddy-Up](./giddy-up/README.md)
- [`asuramaya/conker-detect`](https://github.com/asuramaya/conker-detect)
- [`asuramaya/conker-ledger`](https://github.com/asuramaya/conker-ledger)

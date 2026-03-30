# Status

As of March 29, 2026.

## Role

`Giddy-Up` is the bridge repo between BLINX-side oracle discovery and causal Conker-side runtime features.

## Current Public State

Canonical package surface:

- `giddy_up/oracle.py`: bidirectional oracle analysis
- `giddy_up/attack.py`: leave-one-out, future-context, and self-inclusion attacks
- `giddy_up/features.py`: strictly causal bridge features
- `giddy_up/conker10_adapter.py`: optional replay adapter for saved checkpoints

Best current bridge read:

- March 29 small FineWeb refresh keeps `peak + candidate4` in front
- `7.9664` bits/token, `3.2704 bpb`
- baseline on the same recipe: `8.0239` bits/token, `3.2940 bpb`

## Core Finding

- the `candidate4` family remains the strongest causalized bridge signal in the current small regime
- raw agreement-style features are weaker and do not compose cleanly yet
- live bidirectional oracle scoring still belongs on the BLINX side, not inside causal evaluation

## Next Step

- stay near the `candidate4` family rather than promoting `agreement_mass`
- test better soft candidate shaping or a gated agreement path
- keep the bridge boundary explicit: BLINX produces probes, Conker consumes only causal outputs

## Read Next

- [Salvage Matrix](./SALVAGE_MATRIX.md)
- [Parallel Exploration](./docs/parallel_exploration.md)
- [Status in context](./docs/presentation.md)
- [Architecture](./docs/ARCHITECTURE.md)
- [Rescue](./docs/rescue.md)
- [Repo README](./README.md)

## Companion Repos

- [`asuramaya/blinx`](https://github.com/asuramaya/blinx)
- [`asuramaya/conker`](https://github.com/asuramaya/conker)

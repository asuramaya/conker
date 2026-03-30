# BLINX

`BLINX` is the noncausal lossless-compression and oracle-discovery line.

This tree is now absorbed into the main [`conker`](../README.md) repo. It is
preserved here as a subtree, not as a separate canonical repo.

This repo starts from the post-reset `Conker` surface, but it is not trying to win the causal `bpb` leaderboard. It treats a document as a field rather than a left-to-right stream: remove bytes only when surrounding context can reconstruct them exactly, transmit the punch mask and reconstruction side data, then replay the rounds in reverse.

The active line here is:

- `BLINX-4`: control-first codec mutation over real codec knobs
- `BLINX-5`: payload-aware location encoding
- `BLINX-5a/b/c`: dense, sparse, and adaptive payload ablations
- `BLINX-6`: direct-vs-shared dictionary factoring
- `BLINX-7`: typed dictionary pruning on top of the `BLX6` wire format

The current practical finding is simple: on ordinary repo text, side-data economics still dominate; BLINX only starts winning clearly on cleaner recursive synthetic structure.

## Quick Start

Run the current payload-aware probe on a local text file:

```bash
python3 conker/scripts/run_blinx5_lossless_probe.py \
  conker/docs/BLINX5.md \
  --json-out conker/out/blinx5_probe.json
```

What you get:

- exact roundtrip validation
- original byte count
- remaining survivor bytes
- side-data cost
- packed size after `zlib`
- per-round removal fractions

## Layout

- `conker/`: inherited runtime and experiment surface
- `conker/src/`: codec branches
- `conker/scripts/`: probe runners and ablations
- `conker/docs/`: detailed BLINX notes plus inherited context
- `docs/`: public navigation layer for current results and history
- `carving_machine/`: vendored runtime core still needed by the inherited surface

Related trees in this repo:

- [Absorbed Giddy-Up](../giddy-up/README.md): oracle and bridge layer
- [Conker Root](../README.md): canonical causal root and repo front door

Current runnable entrypoints:

- [run_blinx1_lossless_probe.py](./conker/scripts/run_blinx1_lossless_probe.py)
- [run_blinx4_lossless_probe.py](./conker/scripts/run_blinx4_lossless_probe.py)
- [run_blinx5_lossless_probe.py](./conker/scripts/run_blinx5_lossless_probe.py)
- [run_blinx5_variant_probe.py](./conker/scripts/run_blinx5_variant_probe.py)
- [run_blinx5_variant_ablation.py](./conker/scripts/run_blinx5_variant_ablation.py)
- [run_blinx6_lossless_probe.py](./conker/scripts/run_blinx6_lossless_probe.py)
- [run_blinx7_lossless_probe.py](./conker/scripts/run_blinx7_lossless_probe.py)

Current branch notes:

- [BLINX1.md](./conker/docs/BLINX1.md)
- [BLINX4.md](./conker/docs/BLINX4.md)
- [BLINX5.md](./conker/docs/BLINX5.md)
- [BLINX5_VARIANTS.md](./conker/docs/BLINX5_VARIANTS.md)
- [BLINX_LITERATURE.md](./conker/docs/BLINX_LITERATURE.md)
- [BLINX6.md](./conker/docs/BLINX6.md)
- [BLINX7.md](./conker/docs/BLINX7.md)

Docs and navigation:

- [Docs Index](./docs/README.md)
- [Status](./STATUS.md)
- [Salvage Matrix](./SALVAGE_MATRIX.md)
- [Presentation](./docs/presentation.md)
- [Parallel Exploration](./docs/parallel_exploration.md)
- [Current Frontier](./docs/current_frontier.md)
- [Validity](./docs/validity.md)
- [Negative Results](./docs/negative_results.md)
- [Rescue](./docs/rescue.md)

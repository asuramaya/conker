# Presentation

This page is the shortest current read on `giddy-up`.

Use it when you want the boundary story before the implementation details.

## What Giddy-Up Is

`Giddy-Up` is the bridge repo between noncausal BLINX discovery and causal Conker runtime. It is the canonical home for oracle analysis, attack logic, and causal bridge features; it is not a standalone model line.

Its job is to keep the legality boundary explicit instead of burying it inside either sibling repo.

## Core Contract

- `BLINX` may use oracle analysis and attack tooling here
- `Conker` may consume only causal bridge features or exported offline artifacts
- live bidirectional oracle scoring must not enter causal evaluation

## Where To Read

1. [Status](../STATUS.md)
2. [Salvage Matrix](../SALVAGE_MATRIX.md)
3. [Architecture](./ARCHITECTURE.md)
4. [Parallel Exploration](./parallel_exploration.md)
5. [Rescue](./rescue.md)
6. [`../README.md`](../README.md)

## Companion Repos

- [`asuramaya/blinx`](https://github.com/asuramaya/blinx)
- [`asuramaya/conker`](https://github.com/asuramaya/conker)

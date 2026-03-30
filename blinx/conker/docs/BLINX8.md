# BLINX-8

Date: March 30, 2026

`BLINX-8` revisits the tetra / orthogonal-slot idea on the codec side.

## Mutation

Instead of one flat round dictionary competing globally, `BLINX-8` adds an
`orthogonal` dictionary policy:

- split candidates into disjoint slot families
- force early dictionary budget to represent different slots
- only then spend extra budget within slots

The current slot families are intentionally cheap:

- numeric
- boundary / quote
- text-like
- fallback

## Intended Read

If `BLINX-7` failed because one global candidate pool kept collapsing onto the
same few context families, `BLINX-8` should preserve more structurally distinct
round dictionaries before packing is evaluated.

## Scope

This is still a codec-side policy mutation:

- same `BLX6`-compatible wire format
- same round selection logic
- no new learned decoder state

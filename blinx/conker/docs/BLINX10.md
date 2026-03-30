# BLINX-10

Date: March 30, 2026

`BLINX-10` attacks the removal-scheduling bottleneck directly.

## Mutation

`BLINX-9` already expanded the candidate pool by allowing small support sets.
`BLINX-10` keeps that idea, but changes how a round chooses removals:

- keep contexts with support size up to `4`
- score candidate removals by support and branch-set size
- choose a global non-overlapping schedule with weighted interval selection
- then emit the same compact branch code payload as `BLINX-9`

So this branch tries to stop wasting good local candidates because a greedy
left-to-right pass claimed a worse conflicting one earlier in the round.

## Intended Read

If the main bottleneck was “greedy local scheduling leaves value on the table,”
`BLINX-10` should:

- remove a better subset of the same candidate pool
- improve on `BLINX-9` without changing the basic wire format much
- tell us whether scheduling is a real bottleneck or only a secondary one

## Scope

This is still a direct wire-format experiment:

- not `BLX6`-compatible
- still no shared dictionary factoring
- focused on global round scheduling rather than later packing tweaks

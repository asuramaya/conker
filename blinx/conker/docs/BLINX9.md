# BLINX-9

Date: March 30, 2026

`BLINX-9` attacks the candidate-generation bottleneck directly.

## Mutation

Earlier BLINX branches only removed centers when the surrounding context mapped
to exactly one possible byte. `BLINX-9` allows small branch sets instead:

- keep contexts with support size up to `4`
- store the small option set in the round dictionary
- emit a compact branch code per removed center

So this branch spends a little side-data to unlock a much larger removable set.

## Intended Read

If the main bottleneck was “exact uniqueness is too strict,” `BLINX-9` should:

- remove more bytes per round
- reach profitable rounds on surfaces where `BLINX-7/8` stayed at `0`
- show whether small-support contexts are worth coding explicitly

## Scope

This is a new direct wire format experiment:

- not `BLX6`-compatible
- no shared dictionary factoring yet
- focused on candidate-space expansion first

# BLINX-11

Date: March 30, 2026

`BLINX-11` attacks the candidate-admission bottleneck directly.

## Mutation

`BLINX-10` already expanded the candidate pool by allowing small support sets.
`BLINX-11` keeps the same round wire format, but changes which contexts are
admitted in the first place:

- keep contexts with support size up to `16`
- classify them into lanes by entropy and dominance
- admit `exact`, `tight`, `structured`, and `broad` lanes separately
- use a simple one-byte branch code per removed center

So this branch tries to stop failing before scheduling even matters. If a
context is low-entropy but not exact, it can still be admitted and coded.

## Intended Read

If the main bottleneck is “exact uniqueness is too strict,” `BLINX-11` should:

- admit materially more candidate contexts
- remove bytes on surfaces where `BLINX-9/10` still found `0` rounds
- show whether low-entropy, small-support contexts are worth paying for

## Scope

This is still a direct wire-format experiment:

- not `BLX6`-compatible
- still no shared dictionary factoring
- focused on candidate formation, not scheduler or packer polish

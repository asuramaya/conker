# Giddy-Up Architecture

`giddy_up` is the boundary layer between two different legality domains:

- `BLINX`: noncausal oracle and lossless structure discovery
- `Conker`: causal next-token runtime and submission path

That split creates three kinds of code:

1. Oracle analysis

- bidirectional context support
- leave-one-out support
- future-context and self-inclusion uplift

2. Bridge feature definitions

- causal confidence features derived from prefix-only distributions
- no right-context access at runtime

3. Runtime adapters

- replay helpers for auditing saved checkpoints
- optional dependencies on sibling `Conker` repos

What stays out of this repo:

- full BLINX codecs
- full Conker training/runtime stacks
- legality auditors like `conker-detect`

Practical contract:

- `BLINX` can import `giddy_up.oracle` and `giddy_up.attack`
- `Conker` can import `giddy_up.features`
- `Conker` should treat oracle outputs as offline teacher/probe data, never as live eval-time context

Companion repos:

- [`asuramaya/blinx`](https://github.com/asuramaya/blinx)
- [`asuramaya/conker`](https://github.com/asuramaya/conker)

# Absorbed Repos

This page records the collapse of the former sibling repos into `conker`.

As of March 30, 2026, the canonical private working tree is:

- `conker/` for the causal line
- `blinx/` for the absorbed noncausal codec/oracle line
- `giddy-up/` for the absorbed bridge/oracle-export line

The old sibling repo roots were:

- `https://github.com/asuramaya/blinx.git`
- `https://github.com/asuramaya/giddy-up.git`

Those trees are now preserved locally under this repo:

- [`../blinx`](../blinx/README.md)
- [`../giddy-up`](../giddy-up/README.md)

What this means operationally:

- all current work should land in this single `conker` repo
- absorbed subtree docs remain readable in place
- `conker-detect` and `conker-ledger` stay separate, because they serve audit and
  validity roles outside the core runtime tree

What was intentionally kept:

- branch notes
- scripts
- source trees
- local outputs and experiment scraps
- the current uncommitted branch work that had not yet been pushed

What was intentionally not preserved:

- nested `.git` directories from the absorbed repos
- separate canonical status for `BLINX` and `Giddy-Up` as independent repos

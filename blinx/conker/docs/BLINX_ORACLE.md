# BLINX Oracle Analysis

`BLINX-ORACLE` is an analysis-only pass over local BLINX files.

It measures how often a byte is structurally determined by its bidirectional context:

- left context: `radius` bytes
- right context: `radius` bytes
- oracle target: the center byte

The goal is not compression.
The goal is to measure how much local determinism exists in the repository surface.

Current output tracks:

- exact deterministic fraction
- globally unique fraction
- small candidate-set fractions for `k <= 2`, `4`, and `8`
- mean and max branching factor

Runnable entrypoint:

- [`run_blinx_oracle_analysis.py`](../scripts/run_blinx_oracle_analysis.py)
- [`run_blinx_oracle_attack.py`](../scripts/run_blinx_oracle_attack.py)

Current March 28, 2026 read from the radius-8 sweep:

- exact deterministic fraction rises quickly with radius and is already about `0.9638` by radius `4`
- globally unique centers stay much rarer and only reach about `0.2379` by radius `8`
- small candidate-set fractions are the stronger signal:
  - mean `candidate <= 4` fraction is about `0.9613` at radius `2`
  - about `0.9984` at radius `4`

Interpretation:

- exact uniqueness is not the right bridge target
- small candidate-set size is the durable oracle label
- BLINX remains useful as analysis and teacher signal even though it is not a profitable codec

## Attack Read

The next question is not just "how much structure exists?"
It is "how much of that structure survives adversarial accounting?"

The attack pass measures three failure modes:

- self-inclusion:
  - score each file against a corpus map that excludes that file itself
- future-context dependence:
  - compare bidirectional leave-one-out support with left-only leave-one-out support
- rulebook lower bound:
  - estimate whether removed bytes can even pay for the rulebook and mask side data

March 28, 2026 corpus summary over `172` files / `935,419` bytes:

- radius `2`
  - mean bidirectional leave-one-out `candidate <= 4`: `0.9236`
  - mean left-only leave-one-out `candidate <= 4`: `0.1713`
  - future-context uplift: `+0.7522`
  - mean naive net removed bytes: `-848`
- radius `3`
  - mean bidirectional leave-one-out `candidate <= 4`: `0.8904`
  - mean left-only leave-one-out `candidate <= 4`: `0.4970`
  - self-inclusion uplift: `+0.1066`
  - mean naive net removed bytes: `-1561.6`
- radius `4`
  - mean bidirectional leave-one-out `candidate <= 4`: `0.8380`
  - mean left-only leave-one-out `candidate <= 4`: `0.6937`
  - self-inclusion uplift: `+0.1604`
  - mean naive net removed bytes: `-1689.2`

Interpretation:

- the oracle signal is real, but a lot of it is future-only
- same-file self-inclusion can materially inflate the strongest small candidate-set numbers
- even after finding deterministic structure, the local rulebooks still fail break-even badly

So BLINX survives as an oracle, but only after a hard split:

- leave-one-out for honest analysis
- left-only proxies for anything causal
- no direct codec claim from raw removable fraction

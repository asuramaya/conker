# Source

Compression-branch model and training code lives here.

Use this folder for:

- minimal model definitions
- trainer/eval code tied to the compression framing
- tokenizer-aware runners that are intended to survive into submissions
- separate branch scaffolds when they are still small and self-contained, like `BLINX-1`
- control-first codec branches that stay self-contained, like `BLINX-4`
- payload-first codec branches that tighten side-data accounting, like `BLINX-5`
- explicit ablation wrappers when a branch splits into named payload variants, like `BLINX-5a/b/c`
- dictionary-factoring codec branches that test cross-round reuse, like `BLINX-6`
- typed-pruning codec branches that try to preserve reusable context layers, like `BLINX-7`

Avoid copying the full exploratory archive here unless a line has clearly survived ablation.

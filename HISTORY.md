# History

`Conker` started as a submission-facing subtree inside `carving_machine_v3`.

The split happened for three reasons:

1. the compression line had become its own research program
2. the useful runtime core was much smaller than the lab archive around it
3. the branch history mattered enough to preserve directly, not just through scattered notes

## Major Milestones

- `Conker-1` / `Conker-2`
  - early compressor-native branches and official-data bridge setup
- `Conker-3`
  - reservoir-bank shaping, local/linear mixing, quantization, and the first serious under-cap frontier
- `Conker-4b`
  - residual exact experts over a frozen base
  - established the additive-correction interface
- `Conker-5`
  - tandem training of the inherited base plus residual exact stack
  - first clearly legal full-holdout score deep below the old public leaderboard
  - submitted as non-record PR [#998](https://github.com/openai/parameter-golf/pull/998)
- `Conker-6`
  - mask-based cache branch
  - produced spectacular but invalid results via a trained non-causal side channel
  - directly motivated the external `conker-detect` tool
- `Conker-7`
  - legal causal student trained with a future-aware teacher during training only
  - current active frontier branch in this repo

## Extraction Notes

- the original `conker/docs/` files are preserved as branch records
- a lighter root-level `docs/` layer was added here to make the standalone repo easier to navigate
- artifact backlogs were intentionally not copied wholesale; use the lab tree or `conker-detect` when you need backlog-wide analysis


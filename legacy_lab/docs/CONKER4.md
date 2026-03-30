`Conker-4` is the first explicit expert-mixing branch in `conker/`.

Current design:
- expert 1: live `Conker-3` neural branch
- expert 2: exact 1-token backoff over earlier matching contexts in the same sequence
- expert 3: exact 2-token backoff over earlier matching contexts in the same sequence
- expert 4: decayed recency prior over previously seen tokens
- mixer: tiny learned probability mixer over per-expert entropy / max-prob / support features

Principle:
- keep the reservoir as the smooth expert
- add exact discrete experts instead of forcing the reservoir to solve exact-match compression alone
- spend almost no additional learned bytes on the new branch

Initial goal:
- answer whether a heterogeneous expert ensemble can produce a step change relative to the best single `Conker-3` branch
- kill it quickly if the exact experts do not buy bits on official golf data

First official-data probe, seed `42`, against the matching `Conker-3` reference:

- `Conker-3 window4 10x 1000 staticgate`: fp16 `2.0865`, `int6 2.1081`, `int4 2.2601`

Initial `Conker-4` outcomes:

- learned per-token MLP mixer: unstable, `NaN`
- support-weighted mixer: unstable, `NaN`
- support-weighted mixer with neural expert logically frozen: unstable, `NaN`
- support-weighted mixer with neural expert actually frozen out of the trainable set (`132,100` trainable params): still unstable, `NaN`
- zero-step static ensemble eval: fp16 `3.7207`, `int6 3.7619`, `int4 3.7724`
- low-LR (`1e-4`) frozen-neural run: still unstable, `NaN`

Current read:

- the first exact-expert mixture is not a hidden win waiting for a mixer
- the naive probability-space ensemble is actively bad before training
- the instability is not just “training the neural expert too hard”; it persists even when only the tiny ensemble head is trainable

So the first `Conker-4` implementation is a negative result:

- exact 1-token backoff, exact 2-token backoff, and recency counts as currently constructed do not integrate cleanly with the live `Conker-3` branch
- if `Conker-4` returns, it should probably do so as a residual or calibration expert family, not as a direct probability mixer over these raw count experts

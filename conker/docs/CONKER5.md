`Conker-5` is the first pure learned-discriminator attempt:

- frozen `Conker-3` substrate
- no hand-coded exact/support experts
- learned residual heads
- learned gates over frozen reservoir/base features

Seed-42 probe on official golf data, `window4 / 10x / 1000 / half_life=16 / osc=87.5% / staticgate`:

- `Conker-5`, `8` heads, rank `8`, residual cap `2.0`, `lr=5e-4`:
  - fp16 `3.4012 bpb`
  - `int6 3.4113`
  - `int4 3.4111`
  - train time `34.8s`
  - payload `~0.595 MB`

Read:

- this abstraction is alive as code, but dead as a frontier branch
- replacing all hand-coded selectivity at once loses the sparse exactness that made `Conker-4b` work
- the problem is not capacity; it is lack of ownership / selectivity
- the next learned direction should keep explicit sparse structure and learn overlap management, not erase the sparse structure entirely

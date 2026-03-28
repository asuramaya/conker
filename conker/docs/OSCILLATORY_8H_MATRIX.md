# Oscillatory 8h Matrix

Goal: spend one laptop-bound batch on the questions that matter for the packed `Conker-3` branch.

## Questions

1. Does oscillatory scaling keep buying packed score through under-cap scales?
2. Near cap, does `75%` or `87.5%` oscillatory mass actually win?
3. Are extra bytes better spent on more oscillatory modes or on a smaller, more pack-friendly local path?
4. Does the near-cap oscillatory winner replicate beyond seed `42`?

## Rows

### Packed scaling pilots, seed `42`

- `window4`, `1500`, `half_life_max=16`, periods `4..64`
- oscillatory fractions: `0.75`, `0.875`
- scales: `12x`, `16x`, `18x`
- metrics: fp16 `bpb`, `uniform_int6`, `uniform_int4`, train time

### Local-path byte-allocation probes, seed `42`

On the likely packed winner (`0.875`, `18x`, `1500`):

- `local_hidden_mult=0.75`
- `local_hidden_mult=0.50`
- `local_scale_override=0.20`

### Replication rows

- `0.75`, `16x`, seeds `43/44`
- `0.875`, `16x`, seeds `43/44`
- `0.75`, `18x`, seeds `43/44`
- `0.875`, `18x`, seeds `43/44`

## Decision Rule

- rank by packed score first: `int6`, then `int4`
- use fp16 only as supporting context
- use train time as a tie-breaker, not the main score

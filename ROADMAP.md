# Roadmap

## Frozen

- `Conker-5` is the last clearly legal tandem baseline
- `Conker-6` is closed as a submission line
- `conker-detect` owns the external audit/tooling surface

## Active

- `Conker-7` warm-start future-teacher refinement of the legal tandem student

Current confirmed anchor:
- full-holdout fp16 `0.5283 bpb`
- full-holdout int6 `0.5315 bpb`

## Next Work

1. break the deterministic warm-start replication path with seeded train-stream offsets
2. keep the teacher narrow and weak
3. teach confirmation/gating signals before teaching broader token distributions
4. continue auditing every promising bridge win with fresh-process full eval before treating it as real

## Risks

- bridge-only improvements can still die on full eval
- teacher schedules can destabilize even when local bridge metrics look better
- legality checks must run on trained checkpoints, not fresh init


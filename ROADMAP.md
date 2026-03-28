# Roadmap

## Frozen

- old `Conker-5/7` frontier is invalidated
- `Conker-6` is closed as a submission line
- `Conker-8` and `Conker-9` are honest dead ends at the strict floor
- `conker-detect` owns the external audit/tooling surface
- `conker-ledger` owns the external backlog/validity-bundle surface

## Active

- `Conker-10` memory-first restart

Current clean anchor:
- strict `Conker-4b` full-holdout fp16 `2.0971 bpb`
- strict `Conker-4b` full-holdout int6 `2.1055 bpb`

Current active pilot:
- `Conker-10` bridge fp16 `2.2397`
- memory-only falsification `6.0892`
- fixed heavy-memory blend `2.8436`

## Next Work

1. improve packed memory construction before adding more controller complexity
2. test packed prior + score-first online cache
3. keep the controller subordinate to memory, not the other way around
4. continue auditing every promising branch for both structural legality and artifact-boundary integrity before treating it as real

## Risks

- bridge-only improvements can still die on full eval
- memory tables can be weak even when they are densely populated
- legality checks must run on trained checkpoints, not fresh init
- artifact-size reporting can still lie if regenerated substrate crosses the boundary

# Current Frontier

## Legal Anchor

Current confirmed `Conker-7` full-holdout row:

- warm-start from legal tandem `Conker-5`
- bidirectional training-only teacher
- narrow teacher: `exact2 + exact3`
- `teacher_weight=0.10`
- `1000` warm-start fine-tune steps
- fp16 `0.5283 bpb`
- int6 `0.5315 bpb`
- int6 artifact `4,153,894` bytes

## Baseline

Replicated legal tandem `Conker-5` row:

- `seq_len=256`
- `steps=1500`
- fp16 `~0.5503 bpb`
- int6 `~0.5540 bpb`

## Important Caveat

Several later `Conker-7` bridge improvements looked better locally and then failed honest full-split eval with `NaN`. In this repo, full fresh-process eval outranks bridge numbers.

## Related

- Branch detail: [Conker-7](./branches/CONKER7.md)
- Validity notes: [Validity](./validity.md)


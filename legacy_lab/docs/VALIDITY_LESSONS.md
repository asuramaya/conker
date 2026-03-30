`Conker` validity lessons from the tandem `0.57x` regime:

1. Report full held-out scans for any serious claim.
   - Slice evals are useful for search.
   - They are not publication-grade.
   - The saved seed-43 tandem checkpoint moved from `0.5648` on a `204,800`-token slice to `0.5716` on the full `62,021,632`-token validation split.

2. Save checkpoints and re-evaluate them in a fresh process.
   - This kills a large class of bridge bugs and hidden mutable-state bugs.
   - The tandem branch survived this check.

3. Evaluate the packed artifact, not just the fp16 checkpoint.
   - If the legal artifact is `int6`, the real score is the packed `int6` score.
   - For the saved seed-43 tandem checkpoint:
     - full held-out fp16: `0.5716 bpb`
     - full held-out `int6`: `0.5752 bpb`
   - The packed number is the submission-shaped number.

4. Use actual artifact bytes, not estimated payload bytes, for legality.
   - Helper estimates can be wrong.
   - The only trustworthy byte count is the actual serialized artifact size on disk.
   - Current saved seed-43 packed artifact:
     - `artifact_bytes_zlib = 11,874,832`

5. Attack sequence structure directly.
   - Reverse and shuffle checks are fast sanity tests.
   - A real sequential model should collapse on both.
   - The saved tandem checkpoint did:
     - reverse: `2.2799 bpb`
     - shuffle: `2.3015 bpb`

6. Distinguish train/val leakage from genuine generalization.
   - Verify the bridge uses validation shards for reported `bpb`.
   - Verify stateful exact experts do not carry train-time tables into eval.
   - Verify train/val overlap is not blatant.

7. Do not trust single-seed miracles.
   - Replicate any absurd score across seeds before treating it as real.
   - The tandem `256/1000 @ 5e-4` branch replicated to `0.5615 bpb` on bridge slices before the full-split pass.

8. Keep search metrics and claim metrics separate.
   - Search metrics:
     - small held-out slices
     - post-train quant slice eval
   - Claim metrics:
     - fresh-process full held-out eval
     - packed full held-out eval
     - real artifact bytes

9. Prefer boring protocols over optimistic ones.
   - fixed data split
   - fixed transform (`none`)
   - fixed token count reported
   - checkpoint re-eval
   - no hidden adaptive state between runs

10. The lesson from absurd scores is not “celebrate harder.”
    - It is:
      - tighten evaluation
      - tighten packaging
      - then keep only the numbers that survive both

Current conservative benchmark for the tandem branch:
- bridge-slice replicated mean:
  - `0.5615 bpb`
- fresh-process held-out slice:
  - `0.5648 bpb`
- fresh-process full held-out fp16:
  - `0.5716 bpb`
- fresh-process full held-out `int6`:
  - `0.5752 bpb`

That is the ladder future `Conker-5` claims should satisfy.

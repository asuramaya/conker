# Negative Results

This file is the compact version of the `Conker` research trail after the March 28, 2026 reset. It exists so the repo tells the whole story, not only the current surviving anchor.

## Conker-3: Reservoir First

What worked:

- oscillatory modes were real
- static bank gating was real
- longer training helped

What did not become the main engine:

- more half-life fiddling
- input-projection geometry tricks
- local-path shrink
- naive train-through-pack
- tiny TTT routing probes

Read:

- the reservoir line became a strong smooth predictor
- it did not become the final compressor
- exact-history correction was the missing lever

## Conker-4: Direct Expert Mixture

What was tried:

- direct probability-space mixing of the neural branch with exact-count experts

What happened:

- `NaN`
- unstable mixers
- even frozen-neural variants were bad

Read:

- raw expert competition was the wrong interface
- exact experts should correct a strong base, not fight it over the whole distribution

## Conker-4b: Residual Exact Experts

What worked:

- sparse additive residual experts
- `exact2` as the spine
- `exact3` as a real gain
- delimiter / markup / attr / number / special support experts
- gate-only learned selection over fixed sparse maps

What failed:

- broad lexical class experts
- hard ownership routing
- most broad support slices

Read:

- exact-history residual correction is real
- but the frozen-base frontier still plateaued above the later tandem line
- after the freeze bug was fixed, strict `Conker-4b` became the clean anchor again around `2.10 bpb`

## Conker-5: Pure Learned Discriminators

What was tried:

- remove explicit sparse experts
- use only learned residual heads and learned gates

What happened:

- weak score, about `3.4 bpb`

Read:

- the learned part was not the problem
- erasing sparse exact selectivity was the problem
- learned overlap management works only on top of explicit sparse structure

## Conker-5 Tandem: Historical Frontier, Now Invalidated

What changed:

- stop importing a frozen base
- train the base and sparse residual stack together

What happened:

- full held-out score dropped into the `0.57` range
- longer tandem training pushed the line further
- later audit showed the branch inherited accidentally trainable structural buffers from `Conker-4b`

Read:

- this looked like the first real step change on the legal path
- in hindsight it was mostly a contaminated structural-control line, not a clean legal frontier

## Conker-6: The Illegal Mirage

What looked amazing:

- trained causal-mask cache branch near `0.072 bpb`

What actually happened:

- trained-model legality check failed
- learned diagonal and upper-triangle mask entries leaked future information
- normalization also failed on the trained model
- strict-lower legal reboot collapsed to about `5.75 bpb`

Read:

- the branch was not a legal compressor
- but it was still useful:
  - it exposed how valuable future information is
  - it motivated the teacher/student framing of `Conker-7`

The side-channel tooling stays in the package because invalid branches are still worth auditing and understanding.

## Conker-7: Distill The Lesson, Inherit The Same Bug

What worked:

- narrow training-only teacher
- `exact2 + exact3`
- weak teacher weight
- warm-start from old tandem `Conker-5`

What failed:

- broad future lexical teacher
- stronger teacher weights on honest eval
- some better-looking bridge rows that collapsed to `NaN` on full held-out evaluation

Read:

- bridge improvements are not enough
- full held-out evaluation is the only score that counts
- future-aware training did move the branch locally
- but the saved `0.5283 / 0.5315` row inherited the same invalid structural surface as old tandem

## Conker-8: Explicit Legal Structure Rebuild

What was tried:

- strict lower-triangular lag profile learning
- explicit within-support mask weighting
- patched strict `Conker-4b` underneath

What happened:

- first strict pilot `2.0600 bpb`
- removing learned lag or learned support weights leaves it at `~2.0598`
- stronger support-mask amplitudes diverge to `NaN`

Read:

- legal weighted-mask rebuilding is effectively inert
- the old hidden gain was not recoverable by simply making the masks explicit

## Conker-9: Legal Lag Controller

What was tried:

- fixed causal lag buckets
- small controller choosing which horizon to trust

What happened:

- first pilot `2.0600 bpb`

Read:

- the controller learned a real lag policy
- horizon selection alone does not move the strict floor

## Conker-10: Memory-First Restart

What was tried:

- packed unigram, bigram, and hashed trigram memory
- normalized posterior backoff
- learned mixer with the neural base

What happened:

- first pilot `2.2397 bpb`
- memory-only baseline `6.0892`
- fixed heavy-memory blend `2.8436`

Read:

- the first packed-memory construction is weak
- the problem is not just that the learned mixer was too conservative
- the next honest branch is better memory construction or packed prior + score-first online cache

## Global Lesson

The line is now clear:

- invalid branches are still valuable as falsifiers and teachers
- legal branches must survive:
  - strict causality
  - full normalization
  - score-before-update discipline
  - single-pass evaluation
  - fresh-process full held-out evaluation
  - honest artifact-boundary accounting

Everything else is only search noise.

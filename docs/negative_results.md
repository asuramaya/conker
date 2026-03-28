# Negative Results

This file is the compact version of the `Conker` research trail. It exists so the package tells the whole story, not only the current local winner.

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

## Conker-5 Tandem: Legal Frontier

What changed:

- stop importing a frozen base
- train the base and sparse residual stack together

What happened:

- legal full held-out score dropped into the `0.57` range
- longer tandem training pushed the legal frontier further

Read:

- this was the first real step change on the legal path
- tandem training was more important than many earlier architectural tweaks

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

## Conker-7: Distill The Lesson, Keep Eval Causal

What worked:

- narrow training-only teacher
- `exact2 + exact3`
- weak teacher weight
- warm-start from legal tandem `Conker-5`

What failed:

- broad future lexical teacher
- stronger teacher weights on honest eval
- some better-looking bridge rows that collapsed to `NaN` on full held-out evaluation

Read:

- bridge improvements are not enough
- full held-out evaluation is the only score that counts
- future-aware training helps, but only when the teacher stays narrow and the student already has a strong legal base

## Global Lesson

The line is now clear:

- illegal oracle branches are useful as teachers and falsifiers
- legal branches must survive:
  - strict causality
  - full normalization
  - score-before-update discipline
  - single-pass evaluation
  - fresh-process full held-out evaluation

Everything else is only search noise.

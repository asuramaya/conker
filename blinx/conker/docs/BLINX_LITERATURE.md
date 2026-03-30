# BLINX Literature Positioning

**March 29, 2026**

This note maps `BLINX-1..5` to the closest external lines of work.

It is a positioning note, not a claim that `BLINX` literally implements any one paper.

## Core Read

`BLINX` sits at the intersection of three older ideas:

- antidictionary / forbidden-word compression
- grammar / dictionary compression
- iterative masked reconstruction from bidirectional context

The first two are the closest classical compression relatives.
The third is the closest modern modeling analogy.

## BLINX-1

Code:

- [blinx1.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/src/blinx1.py)

Closest line:

- antidictionary compression

Reason:

- `BLINX-1` removes center bytes only when the surrounding local context uniquely determines them, then transmits side data for exact replay
- that is closest in spirit to "forbidden word" or antidictionary coding, where omission is justified by structural impossibility or determinism

Closest references:

- Crochemore et al., "Text Compression Using Antidictionaries"
  - https://citeseerx.ist.psu.edu/document?doi=4d92d39a928212816f164e39b6ae30daaf9444e9&repid=rep1&type=pdf

Closest modern analogy:

- bidirectional masked reconstruction, not as a lossless codec but as a prediction paradigm
- [BERT](https://research.google/pubs/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding/)
- [Mask-Predict](https://aclanthology.org/D19-1633/)
- [MaskGIT](https://arxiv.org/abs/2202.04200)

Important difference:

- those modern systems are approximate generators or denoisers
- `BLINX-1` is exact and lossless

## BLINX-2

Code:

- [blinx2.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/src/blinx2.py)

Closest line:

- grammar and dictionary compression

Reason:

- `BLINX-2` finds repeated phrases, assigns rule ids, and rewrites the stream in terms of raw chunks plus rule references
- that is directly in the space of grammar compressors and offline dictionary methods

Closest references:

- SEQUITUR / hierarchical grammar inference
  - https://researchcommons.waikato.ac.nz/entities/publication/86f76fd2-cf2b-4346-b3ba-0553fa83c941
- Re-Pair / recursive pair replacement
  - https://arxiv.org/abs/1704.08558

## BLINX-3

Code:

- [blinx3.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/src/blinx3.py)

Closest line:

- hybrid antidictionary-plus-grammar compression

Reason:

- `BLINX-3` keeps the `BLINX-1` local deterministic removal rule
- but compresses the cross-round dictionary keys with a pair grammar before final packing
- that makes it closest to a hybrid of antidictionary removal and Re-Pair-style rulebook compression

Important note:

- this hybrid is the clearest local novelty in the branch
- it does not map neatly onto one canonical named compressor

## BLINX-4

Code:

- [blinx4.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/src/blinx4.py)

Closest line:

- policy / modeling layer over a fixed coder

Reason:

- `BLINX-4` does not invent a new primitive codec
- it adds a phase controller over radius choice, dictionary cap, acceptance threshold, and grammar-budget search
- that makes it closer to the "separate modeling from coding" tradition than to a named legacy compressor

Closest reference:

- Moffat, Neal, Witten, "Arithmetic Coding Revisited"
  - https://glizen.com/radfordneal/ac.abstract.html

Important difference:

- arithmetic coding papers are about probabilistic coding infrastructure
- `BLINX-4` is a policy layer over deterministic rule extraction

So the relationship is structural, not algorithmic identity.

## BLINX-5

Code:

- [blinx5.py](/Users/asuramaya/Code/carving_machine_v3/blinx/conker/src/blinx5.py)

Closest line:

- compressed set / posting-list representation

Reason:

- `BLINX-5` attacks the side-data bottleneck by changing how removed positions are stored
- the branch now compares dense packed bitsets against sparse delta-coded position lists and keeps the cheaper payload

Closest references:

- Roaring bitmaps
  - https://lemire.me/en/publication/arxiv170907821/
- Partitioned Elias-Fano indexes
  - https://arpi.unipi.it/handle/11568/753933

Important difference:

- `BLINX-5` is not an index structure
- it borrows the same core idea: choose a position-set representation that matches sparsity and clustering

## Working Summary

The strongest compact description is:

- `BLINX-1`: antidictionary-style lossless removal
- `BLINX-2`: grammar / rulebook compression
- `BLINX-3`: antidictionary removal plus grammar-compressed side dictionaries
- `BLINX-4`: controller over codec policy
- `BLINX-5`: adaptive position-set payload encoding

So the branch is best read as:

- classical compression first
- modern masked-denoising intuition second
- repo-specific policy and payload experiments on top

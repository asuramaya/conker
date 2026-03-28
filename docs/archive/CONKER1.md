# Conker-1

`Conker-1` is the first compressor-native architecture in `conker/`.

It is not a direct promotion of the archival winners. It is a compression-shaped rebuild around the two BPE survivors that kept different strengths:

- `fast+mid-delay`: best frozen BPE compressor on the held-out bridge
- `v6_silenced`: best BPE hard-shift / boundary-stability branch

## Architecture

`Conker-1` is a two-expert causal mixer:

- expert A: `fast+mid-delay`
- expert B: `v6_silenced`
- mixer: tiny MLP over per-token logit statistics from both experts
- output: weighted logit blend plus a small learned residual bias

The design goal is simple:

- let `fast+mid-delay` carry ordinary compression
- let `v6_silenced` contribute when its stability prior is useful
- do not hard-route
- do not revive decorative side channels

## Why This Is The Right Pivot

The current `conker` evidence says:

- archive char-level winners do not transfer cleanly to BPE
- BPE itself split the frontier by objective
- compression literature rewards mixing specialized predictors more than forcing one branch to dominate

So `Conker-1` is the first model here that is explicitly shaped like a compressor, not a lab winner.

## What Counts As Success

`Conker-1` passes only if it does at least one of these on the official golf data:

- beats both single frozen experts on validation token loss / bits-per-token
- keeps comparable loss while using less trainable structure than the two branches separately
- becomes a better base for later legal test-time adaptation

If it fails, kill it quickly and return to the smaller single-expert survivor.
